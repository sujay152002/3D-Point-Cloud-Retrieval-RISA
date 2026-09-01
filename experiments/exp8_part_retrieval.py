"""Experiment 8 — Part-Aware Retrieval via 3D SIFT-style Descriptors + Inverted Index

Novel contribution: part-conditioned retrieval using local geometric descriptors.
Unlike exp1 (global embeddings, class-level retrieval), this experiment asks:
  "Retrieve all 3D shapes that share a specific geometric part with the query"
  e.g., "find all airplanes with a long wing", "find all chairs with armrests"

Pipeline:
  1. Detect keypoints per shape using ISS-style anisotropy saliency
  2. Compute rotation-invariant local descriptors (FPFH approximation via
     PCA-aligned spherical histograms — no Open3D dependency required)
  3. Build a visual vocabulary (k-means) over all training descriptors
  4. Build inverted index: word → {shape_id: count}
  5. Discover attribute words per part type using ShapeNet part labels
  6. Evaluate part-aware retrieval with TF-IDF scoring

Two retrieval modes:
  A) Global embedding baseline (from the trained models in exp1)
  B) Part-aware inverted index (3D SIFT) — this is the new contribution

Evaluation metric:
  part-mAP@5: relevance = same class AND overlapping part distribution
  (measured as IoU between the query's dominant-part profile and each result's)

Usage:
    cd /home/grad/smenon/retrieval_paper
    python experiments/exp8_part_retrieval.py [--vocab_size 512] [--n_keypoints 64]
    python experiments/exp8_part_retrieval.py --attribute_query wing --query_class airplane
"""

import argparse
import json
import os
import sys, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset, SHAPE_NET_PART_SEG_CLASSES
from core.trainer import unpack_encoder_output

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024

# ── ShapeNet semantic part names ───────────────────────────────────────────────
# Matches the 16 categories and their per-category part indices in datasets.py
# part_id = SHAPE_NET_PART_SEG_CLASSES[class_id][local_part_idx]
SHAPENET_CLASS_NAMES = [
    "airplane", "bag", "cap", "car", "chair",
    "earphone", "guitar", "knife", "lamp", "laptop",
    "motorbike", "mug", "pistol", "rocket", "skateboard", "table",
]

# Human-readable part names per class (same order as SHAPE_NET_PART_SEG_CLASSES)
SHAPENET_PART_NAMES = {
    "airplane":   ["body", "wing", "tail", "engine"],
    "bag":        ["body", "handle"],
    "cap":        ["brim", "crown"],
    "car":        ["body", "wheel", "hood", "roof"],
    "chair":      ["back", "seat", "leg", "arm"],
    "earphone":   ["earcup", "headband", "data_cord"],
    "guitar":     ["head", "neck", "body"],
    "knife":      ["blade", "handle"],
    "lamp":       ["base", "shade", "tube", "bulb"],
    "laptop":     ["lid", "base"],
    "motorbike":  ["gas_tank", "seat", "wheel", "handle", "light", "engine"],
    "mug":        ["body", "handle"],
    "pistol":     ["barrel", "handle", "trigger_guard"],
    "rocket":     ["nose", "body", "fin"],
    "skateboard": ["deck", "wheel", "truck"],
    "table":      ["top", "leg", "support"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — 3D Keypoint Detection (ISS-style anisotropy saliency)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    """
    Brute-force k-NN on CPU. (N, 3) → (N, k) indices.
    Fast enough for N=1024, k≤64 in pure numpy.
    """
    diff  = points[:, None, :] - points[None, :, :]   # (N, N, 3)
    dists = np.sum(diff ** 2, axis=-1)                 # (N, N)
    np.fill_diagonal(dists, np.inf)
    return np.argpartition(dists, k, axis=1)[:, :k]   # (N, k)


def detect_keypoints(points: np.ndarray, k: int = 24,
                     n_keypoints: int = 64) -> np.ndarray:
    """
    ISS-style keypoint detection via eigenvalue anisotropy.

    Computes local covariance at each point → eigenvalues λ1 ≥ λ2 ≥ λ3.
    Saliency = (λ1 - λ3) / (λ1 + 1e-8)  — high for elongated structures
    like wings, handles, legs.

    Returns the top n_keypoints point indices, sorted by saliency desc.
    """
    N = points.shape[0]
    if N <= n_keypoints:
        return np.arange(N)

    knn_idx = compute_knn_indices(points, k)           # (N, k)
    saliency = np.zeros(N, dtype=np.float32)

    for i in range(N):
        neighbors = points[knn_idx[i]] - points[i]    # (k, 3) centered
        cov = (neighbors.T @ neighbors) / k            # (3, 3)
        eigvals = np.linalg.eigvalsh(cov)              # ascending: λ0≤λ1≤λ2
        # Anisotropy: elongated structures (wings) have λ2 >> λ0
        saliency[i] = (eigvals[2] - eigvals[0]) / (eigvals[2] + 1e-8)

    top_idx = np.argpartition(saliency, -n_keypoints)[-n_keypoints:]
    # Sort by descending saliency
    top_idx = top_idx[np.argsort(saliency[top_idx])[::-1]]
    return top_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Rotation-Invariant Local Descriptor (FPFH approximation)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_lrf(center: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """
    Build a Local Reference Frame via weighted PCA.
    Returns (3, 3) rotation matrix: rows are [x_axis, y_axis, z_axis].
    Sign is disambiguated so the z-axis points away from the centroid
    (similar to SHOT's LRF disambiguation).
    """
    centered = neighbors - center                      # (M, 3)
    weights  = 1.0 / (np.linalg.norm(centered, axis=1) + 1e-8)
    W        = np.diag(weights / weights.sum())
    cov      = centered.T @ W @ centered               # (3, 3)

    _, vecs = np.linalg.eigh(cov)                     # cols are eigvecs, ascending
    # eigh returns ascending; we want descending (principal first)
    vecs = vecs[:, ::-1]                              # (3, 3): cols = [v2, v1, v0]

    # Sign disambiguation: majority of neighborhood points have positive projection
    for i in range(3):
        if np.sum(centered @ vecs[:, i]) < 0:
            vecs[:, i] *= -1

    # Build right-handed frame
    x = vecs[:, 0]
    y = vecs[:, 1]
    z = np.cross(x, y)
    z /= np.linalg.norm(z) + 1e-8
    y = np.cross(z, x)
    y /= np.linalg.norm(y) + 1e-8

    return np.stack([x, y, z], axis=0)               # (3, 3)


def compute_fpfh_descriptor(points: np.ndarray,
                             keypoint_idx: np.ndarray,
                             k: int = 24,
                             n_bins: int = 11) -> np.ndarray:
    """
    FPFH-style rotation-invariant local descriptor per keypoint.

    For each keypoint p:
      1. Find k-NN neighborhood
      2. Build LRF (Local Reference Frame) via weighted PCA
      3. For each neighbor q, compute the Darboux frame angles:
           α = angle between query normal and (p→q) direction
           φ = angle of neighbor projected onto LRF x-axis
           θ = angle between neighbor normal and (p→q) direction
         (simplified here: we use the LRF axes directly)
      4. Histogram each angle → concatenate → 33-dim descriptor

    The full FPFH also adds a simplified PFH over the neighborhood;
    here we use a clean 3 × n_bins = 33-dim histogram (matching standard FPFH).

    Returns (n_keypoints, 3 * n_bins) descriptor matrix.
    """
    knn_idx  = compute_knn_indices(points, k)
    n_kp     = len(keypoint_idx)
    desc_dim = 3 * n_bins
    descs    = np.zeros((n_kp, desc_dim), dtype=np.float32)

    for di, kp in enumerate(keypoint_idx):
        center    = points[kp]
        nb_idx    = knn_idx[kp]
        neighbors = points[nb_idx]                    # (k, 3)

        # Build LRF — makes the descriptor rotation-invariant
        lrf = _build_lrf(center, neighbors)           # (3, 3)

        # Project neighbor directions into LRF
        directions = neighbors - center               # (k, 3)
        dists      = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
        unit_dirs  = directions / dists               # (k, 3)
        proj       = unit_dirs @ lrf.T                # (k, 3) in LRF coords

        # Darboux-frame angles in LRF
        # alpha: angle with LRF z-axis (elevation)
        alpha = np.arccos(np.clip(proj[:, 2], -1, 1))          # [0, π]
        # phi:   azimuth angle in LRF xy-plane
        phi   = np.arctan2(proj[:, 1], proj[:, 0])             # [-π, π]
        # theta: angle of point-to-neighbor with LRF x-axis (torsion)
        theta = np.arctan2(proj[:, 2], proj[:, 0] + 1e-8)      # [-π, π]

        # Build normalized histograms for each angle
        h_alpha, _ = np.histogram(alpha, bins=n_bins, range=(0, np.pi))
        h_phi,   _ = np.histogram(phi,   bins=n_bins, range=(-np.pi, np.pi))
        h_theta, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))

        # Concatenate and L2-normalize (SIFT-style: clamp at 0.2, re-normalize)
        desc = np.concatenate([h_alpha, h_phi, h_theta]).astype(np.float32)
        norm = np.linalg.norm(desc) + 1e-8
        desc /= norm
        desc  = np.clip(desc, 0, 0.2)
        desc /= np.linalg.norm(desc) + 1e-8

        descs[di] = desc

    return descs                                      # (n_kp, 33)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Visual Vocabulary + Inverted Index
# ═══════════════════════════════════════════════════════════════════════════════

class InvertedIndex3D:
    """
    3D SIFT Bag-of-Words with TF-IDF inverted index.

    Usage:
        idx = InvertedIndex3D(vocab_size=512)
        idx.build_vocabulary(all_desc_matrix)  # (total_kps, 33)
        for shape_id, descs in enumerate(all_descs):
            idx.add_shape(shape_id, descs)
        results = idx.query(query_descs, top_k=10)
        results = idx.query_by_attribute(word_ids, top_k=10)  # attribute search
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size  = vocab_size
        self.kmeans      = MiniBatchKMeans(
            n_clusters=vocab_size, random_state=42,
            batch_size=min(4096, vocab_size * 8),
            n_init=3, max_iter=100,
        )
        # word_id → {shape_id: count}
        self.index: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # shape_id → total descriptor count (for TF normalization)
        self.shape_tf: dict[int, int] = {}
        self._fitted = False

    def build_vocabulary(self, all_descriptors: np.ndarray) -> None:
        """
        all_descriptors: (N_total, desc_dim) — pooled from all training shapes.
        Runs k-means to build the visual word vocabulary.
        """
        print(f"  Building vocabulary: {self.vocab_size} words "
              f"from {len(all_descriptors):,} descriptors...", flush=True)
        self.kmeans.fit(all_descriptors)
        self._fitted = True
        print(f"  Vocabulary ready.", flush=True)

    def add_shape(self, shape_id: int, descriptors: np.ndarray) -> None:
        """
        descriptors: (K, desc_dim) — all keypoint descriptors for this shape.
        Quantizes to words and updates the inverted index.
        """
        if not self._fitted:
            raise RuntimeError("Call build_vocabulary() before add_shape()")
        words = self.kmeans.predict(descriptors)      # (K,)
        self.shape_tf[shape_id] = len(words)
        for w in words:
            self.index[int(w)][shape_id] += 1

    def query(self, query_descriptors: np.ndarray, top_k: int = 10,
              exclude_id: int = -1) -> list[tuple[int, float]]:
        """
        TF-IDF retrieval given a descriptor matrix (K, desc_dim).
        Returns [(shape_id, score), ...] sorted by descending score.
        """
        words  = self.kmeans.predict(query_descriptors)
        N      = len(self.shape_tf)
        scores = defaultdict(float)

        for w in words:
            df = len(self.index[int(w)])
            if df == 0:
                continue
            idf = np.log((N + 1) / (df + 1))          # smoothed IDF
            for sid, count in self.index[int(w)].items():
                if sid == exclude_id:
                    continue
                tf = count / self.shape_tf[sid]
                scores[sid] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def query_by_attribute(self, attribute_word_ids: list[int],
                           top_k: int = 10) -> list[tuple[int, float]]:
        """
        Attribute-conditioned retrieval: given a pre-discovered set of word IDs
        that characterize a geometric part (e.g., "wing"), retrieve all shapes
        that strongly exhibit that feature.
        """
        N      = len(self.shape_tf)
        scores = defaultdict(float)
        for w in attribute_word_ids:
            df = len(self.index[int(w)])
            if df == 0:
                continue
            idf = np.log((N + 1) / (df + 1))
            for sid, count in self.index[int(w)].items():
                tf = count / self.shape_tf[sid]
                scores[sid] += tf * idf
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def word_assignments(self, descriptors: np.ndarray) -> np.ndarray:
        """Return word-id per descriptor (K,)."""
        return self.kmeans.predict(descriptors)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Attribute Word Discovery from ShapeNet Part Labels
# ═══════════════════════════════════════════════════════════════════════════════

def discover_attribute_words(
    index: InvertedIndex3D,
    all_descs_by_shape: list[np.ndarray],
    all_part_labels_by_shape: list[np.ndarray],
    all_class_labels: np.ndarray,
    class_id: int,
    part_name: str,
    top_n: int = 30,
) -> list[int]:
    """
    Find which vocabulary words are most characteristic of a named part.

    For each shape of class_id, extract descriptors whose keypoints land
    on part_name points, collect their word assignments, and rank words
    by how often they appear exclusively in that part vs. other parts.

    Returns a list of top_n word IDs that represent this attribute.
    """
    class_names = SHAPENET_CLASS_NAMES
    part_names  = SHAPENET_PART_NAMES.get(class_names[class_id], [])
    if part_name not in part_names:
        raise ValueError(f"Part '{part_name}' not in class '{class_names[class_id]}'. "
                         f"Valid parts: {part_names}")

    local_part_idx = part_names.index(part_name)
    # Global part label ID for this (class, local_part_idx)
    global_part_id = SHAPE_NET_PART_SEG_CLASSES[class_id][local_part_idx]

    # Accumulate word frequencies: part vs. not-part
    part_word_counts    = defaultdict(int)
    nonpart_word_counts = defaultdict(int)

    for shape_id, (descs, part_lbls, cls) in enumerate(
        zip(all_descs_by_shape, all_part_labels_by_shape, all_class_labels)
    ):
        if int(cls) != class_id:
            continue
        if descs is None or part_lbls is None:
            continue

        # descs here is actually a tuple (desc_matrix, kp_point_labels)
        kp_part_lbls = descs[1]                        # (K,) part label at each KP
        desc_matrix  = descs[0]
        words        = index.word_assignments(desc_matrix)  # (K,)

        for w, lbl in zip(words, kp_part_lbls):
            if int(lbl) == global_part_id:
                part_word_counts[int(w)] += 1
            else:
                nonpart_word_counts[int(w)] += 1

    # Score each word by: count_in_part / (count_in_part + count_not_part)
    # — high score = word is distinctive of this part
    all_words = set(part_word_counts.keys()) | set(nonpart_word_counts.keys())
    word_scores = {}
    for w in all_words:
        p  = part_word_counts.get(w, 0)
        np_ = nonpart_word_counts.get(w, 0)
        word_scores[w] = p / (p + np_ + 1e-8)

    top_words = sorted(word_scores.keys(), key=lambda w: -word_scores[w])[:top_n]
    print(f"  Attribute words for '{part_name}' in '{class_names[class_id]}': "
          f"{top_words[:10]}... (showing 10/{top_n})", flush=True)
    return top_words


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — Part-Aware Retrieval Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

# Build a global_part_id → "class/part" name lookup from SHAPE_NET_PART_SEG_CLASSES
def _build_part_id_to_name() -> dict[int, str]:
    lookup = {}
    for cls_name, part_ids in zip(SHAPENET_CLASS_NAMES, SHAPE_NET_PART_SEG_CLASSES):
        part_names = SHAPENET_PART_NAMES.get(cls_name, [])
        for local_idx, global_id in enumerate(part_ids):
            label = part_names[local_idx] if local_idx < len(part_names) else f"part{local_idx}"
            lookup[global_id] = f"{cls_name}/{label}"
    return lookup

_PART_ID_TO_NAME = _build_part_id_to_name()


def part_prob_profile(part_dist: np.ndarray) -> list[tuple[str, float]]:
    """
    Convert a (n_parts,) distribution vector into a ranked list of
    (part_name, probability) tuples, most probable first, zero-prob parts excluded.
    """
    ranked = [
        (_PART_ID_TO_NAME.get(i, f"part{i}"), float(part_dist[i]))
        for i in range(len(part_dist))
        if part_dist[i] > 0
    ]
    ranked.sort(key=lambda x: -x[1])
    return ranked


def part_distribution(part_labels: np.ndarray, n_parts: int) -> np.ndarray:
    """
    Compute the fraction of points belonging to each part.
    Returns a (n_parts,) probability vector.
    """
    counts = np.bincount(part_labels.astype(np.int32),
                         minlength=n_parts).astype(np.float32)
    total  = counts.sum()
    return counts / (total + 1e-8)


def part_iou(dist_a: np.ndarray, dist_b: np.ndarray,
             threshold: float = 0.1) -> float:
    """
    Part-distribution IoU: treats parts present above threshold as a set,
    computes set-level Jaccard similarity.
    """
    a = dist_a > threshold
    b = dist_b > threshold
    intersection = (a & b).sum()
    union        = (a | b).sum()
    return float(intersection / (union + 1e-8))


def compute_part_map_at_k(
    query_ids: list[int],
    db_ids: list[int],
    retrieval_fn,           # callable(query_id) → [(db_id, score), ...]
    class_labels: np.ndarray,
    part_dists: np.ndarray, # (N, n_parts)
    k: int = 5,
    part_iou_threshold: float = 0.3,
    return_profiles: bool = False,
) -> float | tuple[float, list[dict]]:
    """
    Part-aware mAP@k.

    Relevance is stricter than class-only retrieval:
      relevant = same class AND part_iou(query, result) >= part_iou_threshold

    If return_profiles=True, also returns per-query dicts with ranked part
    probability profiles for the query and each retrieved result.
    """
    aps = []
    profiles = []
    for q_id in query_ids:
        results = retrieval_fn(q_id)[:k]
        q_cls   = class_labels[q_id]
        q_dist  = part_dists[q_id]

        rel = []
        result_profiles = []
        for r_id, score in results:
            if r_id == q_id:
                continue
            same_class = (class_labels[r_id] == q_cls)
            iou        = part_iou(q_dist, part_dists[r_id])
            rel.append(float(same_class and iou >= part_iou_threshold))
            if return_profiles:
                result_profiles.append({
                    "shape_id":    r_id,
                    "score":       round(float(score), 4),
                    "relevant":    bool(same_class and iou >= part_iou_threshold),
                    "part_iou":    round(iou, 4),
                    "part_probs":  part_prob_profile(part_dists[r_id]),
                })

        rel = np.array(rel[:k])
        if rel.sum() == 0:
            aps.append(0.0)
        else:
            precision_at_k = rel.cumsum() / np.arange(1, len(rel) + 1)
            aps.append(float((precision_at_k * rel).sum() / rel.sum()))

        if return_profiles:
            profiles.append({
                "query_id":      q_id,
                "query_class":   SHAPENET_CLASS_NAMES[q_cls] if q_cls < len(SHAPENET_CLASS_NAMES) else str(q_cls),
                "query_parts":   part_prob_profile(q_dist),
                "ap":            round(aps[-1], 4),
                "results":       result_profiles,
            })

    map_score = float(np.mean(aps)) if aps else 0.0
    return (map_score, profiles) if return_profiles else map_score


def compute_class_map_at_k(
    query_ids: list[int],
    retrieval_fn,
    class_labels: np.ndarray,
    k: int = 5,
) -> float:
    """
    Standard class-level mAP@k (same as exp1, for baseline comparison).
    """
    aps = []
    for q_id in query_ids:
        results = retrieval_fn(q_id)[:k]
        q_cls   = class_labels[q_id]
        rel     = np.array([
            float(class_labels[r_id] == q_cls)
            for r_id, _ in results
            if r_id != q_id
        ][:k])
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        precision_at_k = rel.cumsum() / np.arange(1, len(rel) + 1)
        aps.append(float((precision_at_k * rel).sum() / rel.sum()))
    return float(np.mean(aps)) if aps else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — Global Embedding Baseline (from exp1 models)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_encoder(name: str):
    from models import (
        DGCNNEncoder, PointNet2Encoder,
        RotationInvariantSparseAttention, VNNEncoder, DiPVNetEncoder,
        RINet,
    )
    from models.transformer import PointTransformerEncoder
    registry = {
        "PointNet++":       PointNet2Encoder,
        "DGCNN":            DGCNNEncoder,
        "VNN":              VNNEncoder,
        "DiPVNet":          DiPVNetEncoder,
        "PointTransformer": PointTransformerEncoder,
        "RINet":            RINet,
    }
    if name == "RISA":
        return RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    return registry[name]()


class ProxyAnchorLoss(torch.nn.Module):
    """
    Proxy-Anchor Loss (Kim et al., CVPR 2020).

    Maintains one learnable proxy per class in embedding space.
    For each batch, pulls embeddings toward their class proxy and
    pushes them away from all other proxies, with per-proxy margins.

    This directly optimises the embedding space for retrieval — unlike
    cross-entropy which only requires class separability, proxies force
    tight intra-class clusters and large inter-class margins.

    Reference: https://arxiv.org/abs/2003.13911
    """
    def __init__(self, num_classes: int, embed_dim: int,
                 margin: float = 0.1, alpha: float = 32.0):
        super().__init__()
        self.proxies = torch.nn.Parameter(
            torch.randn(num_classes, embed_dim)
        )
        torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.num_classes = num_classes
        self.margin      = margin
        self.alpha       = alpha

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings : (B, D) — L2-normalised
        labels     : (B,)   — class indices
        """
        import torch.nn.functional as F
        P = F.normalize(self.proxies, dim=1)           # (C, D)
        E = F.normalize(embeddings,   dim=1)           # (B, D)

        # Cosine similarities: (B, C)
        sim = E @ P.T

        # One-hot positive mask: (B, C)
        pos_mask = torch.zeros_like(sim)
        pos_mask.scatter_(1, labels.unsqueeze(1), 1.0)
        neg_mask = 1.0 - pos_mask

        # Per-proxy positive and negative terms (sum over batch dimension)
        # Positive: proxies that have at least one positive in batch
        pos_exp = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_exp = torch.exp( self.alpha * (sim + self.margin)) * neg_mask

        # Only include proxies that have positives in this batch
        with_pos = (pos_mask.sum(0) > 0)              # (C,)

        loss_pos = (torch.log(1 + pos_exp.sum(0)) * with_pos).sum()
        loss_neg = (torch.log(1 + neg_exp.sum(0)) * with_pos).sum()

        return (loss_pos + loss_neg) / with_pos.sum().clamp(min=1)


def quick_train_encoder(encoder, dataset_name: str,
                        epochs: int = 20, seed: int = 42,
                        loss: str = "proxy_anchor",
                        batch_size: int = 16) -> None:
    """
    Train encoder with SO(3) augmentation.

    loss='classification' : cross-entropy on a linear head (original)
    loss='proxy_anchor'   : Proxy-Anchor metric learning loss (Kim et al. 2020)
                            Directly optimises embedding space for retrieval.
    batch_size            : reduce for large models (DiPVNet, RISA) to avoid OOM.
    """
    import torch.nn as nn, torch.optim as optim
    torch.manual_seed(seed)
    kwargs = {"variant": "OBJ_ONLY"} if dataset_name == "scanobjectnn" else {}
    try:
        ds = get_dataset(dataset_name, split="train", root=DATA_ROOT,
                         num_points=N_POINTS, **kwargs)
    except Exception as e:
        print(f"    [train] Cannot load {dataset_name}: {e} — skipping")
        return

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    encoder.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 64, device=DEVICE)
        z, _  = unpack_encoder_output(encoder(dummy))
    enc_dim  = z.shape[-1]
    num_cls  = int(getattr(getattr(ds, "dataset", ds), "num_classes", 16))

    if loss == "proxy_anchor":
        criterion = ProxyAnchorLoss(num_cls, enc_dim,
                                    margin=0.1, alpha=32.0).to(DEVICE)
        opt = optim.Adam(
            list(encoder.parameters()) + list(criterion.parameters()),
            lr=1e-4, weight_decay=1e-4,
        )
        # Separate higher LR for proxies (common practice)
        opt = optim.Adam([
            {"params": encoder.parameters(),    "lr": 1e-4},
            {"params": criterion.parameters(),  "lr": 1e-3},
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        head = None
    else:
        head      = nn.Linear(enc_dim, num_cls).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        opt       = optim.Adam(
            list(encoder.parameters()) + list(head.parameters()), lr=1e-3
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    def _rand_rot(B):
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
        Q[:, :, 0:1] *= torch.linalg.det(Q).sign().view(B, 1, 1)
        return Q

    encoder.train()
    if head is not None:
        head.train()

    for ep in range(epochs):
        for batch in loader:
            pts = batch[0] if isinstance(batch, (tuple, list)) else batch
            lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                pts, lbl = batch[0], batch[1]
            pts = pts.float().to(DEVICE)
            lbl = lbl.long().to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)
            pts = _rand_rot(pts.shape[0]) @ pts
            opt.zero_grad()
            z, _ = unpack_encoder_output(encoder(pts))
            if loss == "proxy_anchor":
                criterion(z, lbl).backward()
            else:
                criterion(head(z), lbl).backward()
            opt.step()
        print(f"    epoch {ep+1}/{epochs}", flush=True)
        scheduler.step()

    encoder.eval()
    if head is not None:
        del head
    del opt, criterion, loader, ds
    torch.cuda.empty_cache()


@torch.no_grad()
def encode_all(encoder, points_list: list[np.ndarray],
               batch_size: int = 16) -> np.ndarray:
    """
    Encode a list of (3, N) numpy arrays → (M, D) embedding matrix.
    """
    all_enc = []
    for i in range(0, len(points_list), batch_size):
        batch_np = points_list[i:i + batch_size]
        # points_list items are (3, N) — stack to (B, 3, N)
        pts = torch.from_numpy(np.stack(batch_np, axis=0)).float().to(DEVICE)
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu().numpy())
    return np.concatenate(all_enc, axis=0)            # (M, D)


def global_embedding_retrieval_fn(query_id: int,
                                   embeddings: np.ndarray,
                                   k: int = 5) -> list[tuple[int, float]]:
    """Cosine similarity retrieval in global embedding space."""
    q    = embeddings[query_id]
    q    = q / (np.linalg.norm(q) + 1e-8)
    db   = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = db @ q                                     # (M,)
    sims[query_id] = -np.inf                          # exclude self
    top_ids = np.argpartition(sims, -k)[-k:]
    top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]
    return [(int(i), float(sims[i])) for i in top_ids]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7 — Main Data Extraction Helper
# ═══════════════════════════════════════════════════════════════════════════════

def extract_shapes_from_dataset(dataset) -> tuple:
    """
    Extract all shapes from a dataset object.
    Returns:
      points_list    : list of (3, N) arrays
      class_labels   : (M,) int array
      part_labels_list: list of (N,) arrays or None
    """
    points_list      = []
    class_labels_lst = []
    part_labels_list = []
    has_parts        = False

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, (tuple, list)):
            pts = sample[0]                           # (3, N) tensor
            cls = sample[1]
            prt = sample[2] if len(sample) >= 3 else None
        else:
            pts = sample
            cls = torch.tensor(0)
            prt = None

        points_list.append(pts.numpy() if isinstance(pts, torch.Tensor) else pts)
        class_labels_lst.append(int(cls))
        if prt is not None:
            part_labels_list.append(
                prt.numpy() if isinstance(prt, torch.Tensor) else prt
            )
            has_parts = True
        else:
            part_labels_list.append(None)

    class_labels = np.array(class_labels_lst, dtype=np.int32)
    if not has_parts:
        part_labels_list = None

    return points_list, class_labels, part_labels_list


def build_descriptors_for_shape(
    pts_3n: np.ndarray,          # (3, N)
    part_lbl: np.ndarray | None, # (N,) or None
    n_keypoints: int = 64,
    k_nn: int = 24,
    n_bins: int = 11,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run keypoint detection + FPFH descriptor computation for one shape.
    Returns:
      desc_matrix  : (n_kp, desc_dim)
      kp_part_lbls : (n_kp,) part labels at keypoints, or None
    """
    pts = pts_3n.T                                    # (N, 3)
    # Ensure unit-sphere normalization
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    pts /= scale

    kp_idx = detect_keypoints(pts, k=k_nn, n_keypoints=n_keypoints)
    descs  = compute_fpfh_descriptor(pts, kp_idx, k=k_nn, n_bins=n_bins)

    kp_part_lbls = None
    if part_lbl is not None:
        kp_part_lbls = part_lbl[kp_idx]

    return descs, kp_part_lbls


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    print(f"Device: {DEVICE}")
    print(f"Loading ShapeNet test set ({N_POINTS} pts per shape)...")

    test_ds = get_dataset("shapenet", split="test",
                          root=DATA_ROOT, num_points=N_POINTS)

    print("Extracting shapes...")
    points_list, class_labels, part_labels_list = extract_shapes_from_dataset(test_ds)
    N = len(points_list)
    print(f"  {N} test shapes, {len(np.unique(class_labels))} classes", flush=True)

    # ── Compute part distributions for eval ────────────────────────────────────
    n_total_parts = 50                                # ShapeNet has 50 part types
    if part_labels_list is not None:
        print("Computing part distributions...")
        part_dists = np.stack([
            part_distribution(pl, n_parts=n_total_parts)
            if pl is not None else np.zeros(n_total_parts)
            for pl in part_labels_list
        ])                                            # (N, 50)
    else:
        print("WARNING: No part labels found, part-mAP will be 0.")
        part_dists = np.zeros((N, n_total_parts))

    # ── 3D SIFT descriptor extraction (with optional cache) ───────────────────
    import pickle
    cache_hit = False
    if args.sift_cache and os.path.exists(args.sift_cache):
        print(f"\nLoading 3D SIFT descriptors from cache: {args.sift_cache}")
        with open(args.sift_cache, "rb") as f:
            cache = pickle.load(f)
        all_descs_with_labels = cache["all_descs_with_labels"]
        all_desc_flat         = cache["all_desc_flat"]
        cache_hit = True
        print(f"  Loaded {len(all_descs_with_labels)} shapes from cache.", flush=True)
    else:
        print(f"\nExtracting 3D SIFT descriptors "
              f"(n_keypoints={args.n_keypoints}, k_nn={args.k_nn}, "
              f"n_bins={args.n_bins})...")
        all_descs_with_labels = []
        all_desc_flat         = []

        for i, (pts, prt) in enumerate(zip(points_list, part_labels_list or [None]*N)):
            descs, kp_part_lbls = build_descriptors_for_shape(
                pts, prt,
                n_keypoints=args.n_keypoints,
                k_nn=args.k_nn,
                n_bins=args.n_bins,
            )
            all_descs_with_labels.append((descs, kp_part_lbls))
            all_desc_flat.append(descs)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N} shapes processed...", flush=True)

        if args.sift_cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.sift_cache)), exist_ok=True)
            with open(args.sift_cache, "wb") as f:
                pickle.dump({"all_descs_with_labels": all_descs_with_labels,
                             "all_desc_flat": all_desc_flat}, f)
            print(f"  Saved descriptor cache to {args.sift_cache}", flush=True)

    all_desc_matrix = np.concatenate(all_desc_flat, axis=0)  # (N*K, desc_dim)
    print(f"  Total descriptors: {len(all_desc_matrix):,}", flush=True)

    # ── Build vocabulary + inverted index ─────────────────────────────────────
    print(f"\nBuilding inverted index (vocab_size={args.vocab_size})...")
    inv_index = InvertedIndex3D(vocab_size=args.vocab_size)
    inv_index.build_vocabulary(all_desc_matrix)

    for shape_id, (descs, _) in enumerate(all_descs_with_labels):
        inv_index.add_shape(shape_id, descs)
    print(f"  Indexed {N} shapes.", flush=True)

    # ── Attribute word discovery ───────────────────────────────────────────────
    attribute_words = {}
    if part_labels_list is not None and args.discover_attributes:
        print("\nDiscovering attribute words from part labels...")
        for cls_name, part_list in SHAPENET_PART_NAMES.items():
            if cls_name not in SHAPENET_CLASS_NAMES:
                continue
            cls_id = SHAPENET_CLASS_NAMES.index(cls_name)
            for part_name in part_list:
                key = f"{cls_name}/{part_name}"
                try:
                    words = discover_attribute_words(
                        inv_index,
                        all_descs_with_labels,
                        part_labels_list,
                        class_labels,
                        class_id=cls_id,
                        part_name=part_name,
                        top_n=args.attribute_top_n,
                    )
                    attribute_words[key] = words
                except Exception as e:
                    print(f"  Skipped {key}: {e}")

    # ── Demo: attribute query ──────────────────────────────────────────────────
    if args.attribute_query and args.query_class:
        key = f"{args.query_class}/{args.attribute_query}"
        if key in attribute_words:
            words = attribute_words[key]
            print(f"\nAttribute query: '{key}'")
            results = inv_index.query_by_attribute(words, top_k=10)
            print(f"  Top-10 results (shape_id, score):")
            for sid, score in results:
                cls_name = SHAPENET_CLASS_NAMES[class_labels[sid]] \
                    if class_labels[sid] < len(SHAPENET_CLASS_NAMES) else "?"
                print(f"    shape {sid:5d}  class={cls_name:<12s}  score={score:.4f}")
        else:
            print(f"\nAttribute '{key}' not in discovered words. "
                  f"Run with --discover_attributes to enable.")

    # ── Evaluation: 3D SIFT vs Global Embedding ────────────────────────────────
    # Subsample query set for speed (use first args.n_queries shapes per class)
    print(f"\nEvaluating retrieval (n_queries_per_class={args.n_queries_per_class})...")
    query_ids = []
    for cls_id in np.unique(class_labels):
        cls_mask = np.where(class_labels == cls_id)[0]
        n_q      = min(args.n_queries_per_class, len(cls_mask))
        query_ids.extend(cls_mask[:n_q].tolist())
    query_ids = sorted(query_ids)
    print(f"  {len(query_ids)} total queries", flush=True)

    # ── 3D SIFT retrieval functions ────────────────────────────────────────────
    def sift_retrieval_fn(q_id: int) -> list[tuple[int, float]]:
        descs, _ = all_descs_with_labels[q_id]
        return inv_index.query(descs, top_k=args.k_eval, exclude_id=q_id)

    # ── Evaluate 3D SIFT ───────────────────────────────────────────────────────
    sift_class_map = compute_class_map_at_k(
        query_ids, sift_retrieval_fn, class_labels, k=args.k_eval
    )
    results_dict = {
        "method": "3D-SIFT-InvIndex",
        "vocab_size":        args.vocab_size,
        "n_keypoints":       args.n_keypoints,
        "class_mAP@5":       round(sift_class_map, 4),
    }

    sift_profiles = []
    if part_labels_list is not None:
        sift_part_map, sift_profiles = compute_part_map_at_k(
            query_ids, list(range(N)), sift_retrieval_fn,
            class_labels, part_dists, k=args.k_eval,
            part_iou_threshold=args.part_iou_threshold,
            return_profiles=True,
        )
        results_dict["part_mAP@5"] = round(sift_part_map, 4)

    print(f"\n{'='*60}")
    print(f"3D SIFT + Inverted Index  (vocab={args.vocab_size})")
    print(f"  class-mAP@5  = {results_dict['class_mAP@5']:.4f}")
    if "part_mAP@5" in results_dict:
        print(f"  part-mAP@5   = {results_dict['part_mAP@5']:.4f}  "
              f"(IoU threshold={args.part_iou_threshold})")
        print(f"  Sample query part profiles (first 3 queries):")
        for qp in sift_profiles[:3]:
            print(f"    query {qp['query_id']} ({qp['query_class']})  AP={qp['ap']:.4f}")
            print(f"      query parts: " + ", ".join(f"{n}={p:.2f}" for n, p in qp['query_parts'][:4]))
            for i, r in enumerate(qp['results'][:3]):
                probs_str = ", ".join(f"{n}={p:.2f}" for n, p in r['part_probs'][:4])
                print(f"      result {i+1}: shape {r['shape_id']}  iou={r['part_iou']:.2f}  rel={r['relevant']}  [{probs_str}]")

    # ── Global embedding baselines ─────────────────────────────────────────────
    # Large models (DiPVNet, RISA) need a smaller batch size to fit on GPU.
    _SMALL_BATCH_MODELS = {"DiPVNet", "RISA", "VNN", "PointTransformer", "RINet"}
    _BATCH_SIZE = {m: (4 if m in _SMALL_BATCH_MODELS else 16) for m in args.models}

    baseline_results = {}
    for model_name in args.models:
        print(f"\n{'='*60}\nGlobal embedding baseline: {model_name}  [{args.loss}]")
        encoder = None
        embeddings = None
        try:
            torch.cuda.empty_cache()
            encoder = _build_encoder(model_name).to(DEVICE)
            quick_train_encoder(encoder, "shapenet",
                                epochs=args.epochs, seed=args.seed,
                                loss=args.loss,
                                batch_size=_BATCH_SIZE[model_name])
            embeddings = encode_all(encoder, points_list,
                                    batch_size=min(8, _BATCH_SIZE[model_name]))
        except Exception as e:
            print(f"  Error building/training {model_name}: {e} — skipping")
            continue
        finally:
            if encoder is not None:
                encoder.cpu()
                del encoder
                encoder = None
            torch.cuda.empty_cache()

        def make_global_fn(embs):
            def fn(q_id):
                return global_embedding_retrieval_fn(q_id, embs, k=args.k_eval)
            return fn

        global_fn = make_global_fn(embeddings)

        g_class_map = compute_class_map_at_k(
            query_ids, global_fn, class_labels, k=args.k_eval
        )
        g_part_map = None
        g_profiles = []
        if part_labels_list is not None:
            g_part_map, g_profiles = compute_part_map_at_k(
                query_ids, list(range(N)), global_fn,
                class_labels, part_dists, k=args.k_eval,
                part_iou_threshold=args.part_iou_threshold,
                return_profiles=True,
            )

        baseline_results[model_name] = {
            "class_mAP@5": round(g_class_map, 4),
        }
        if g_part_map is not None:
            baseline_results[model_name]["part_mAP@5"] = round(g_part_map, 4)
            baseline_results[model_name]["part_profiles"] = g_profiles

        print(f"  class-mAP@5 = {g_class_map:.4f}")
        if g_part_map is not None:
            print(f"  part-mAP@5  = {g_part_map:.4f}")
            print(f"  Sample query part profiles (first 3 queries):")
            for qp in g_profiles[:3]:
                print(f"    query {qp['query_id']} ({qp['query_class']})  AP={qp['ap']:.4f}")
                print(f"      query parts: " + ", ".join(f"{n}={p:.2f}" for n, p in qp['query_parts'][:4]))
                for i, r in enumerate(qp['results'][:3]):
                    probs_str = ", ".join(f"{n}={p:.2f}" for n, p in r['part_probs'][:4])
                    print(f"      result {i+1}: shape {r['shape_id']}  iou={r['part_iou']:.2f}  rel={r['relevant']}  [{probs_str}]")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Method':<22} {'class-mAP@5':>12} {'part-mAP@5':>12}")
    print(f"{'-'*46}")
    print(f"{'3D-SIFT-InvIndex':<22} "
          f"{results_dict['class_mAP@5']:>12.4f} "
          f"{results_dict.get('part_mAP@5', float('nan')):>12.4f}")
    for m, r in baseline_results.items():
        print(f"{m:<22} {r['class_mAP@5']:>12.4f} "
              f"{r.get('part_mAP@5', float('nan')):>12.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "sift_inverted_index":  results_dict,
        "sift_part_profiles":   sift_profiles,
        "global_baselines":     baseline_results,
        "attribute_words":      {k: v for k, v in attribute_words.items()},
        "config": vars(args),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Exp 8: 3D SIFT + Inverted Index vs Global Embeddings "
                    "for Part-Aware Retrieval on ShapeNet"
    )
    # Descriptor settings
    parser.add_argument("--n_keypoints",  type=int,   default=64,
                        help="Keypoints per shape (default: 64)")
    parser.add_argument("--k_nn",         type=int,   default=24,
                        help="k-NN for descriptor computation (default: 24)")
    parser.add_argument("--n_bins",       type=int,   default=11,
                        help="Histogram bins per angle (descriptor dim = 3*n_bins)")
    # Vocabulary + index settings
    parser.add_argument("--vocab_size",   type=int,   default=512,
                        help="Visual vocabulary size (default: 512)")
    # Evaluation
    parser.add_argument("--k_eval",               type=int,   default=5)
    parser.add_argument("--n_queries_per_class",  type=int,   default=50,
                        help="Max query shapes per class (for speed)")
    parser.add_argument("--part_iou_threshold",   type=float, default=0.3,
                        help="IoU threshold for part-relevant retrieval")
    # Attribute discovery
    parser.add_argument("--discover_attributes",  action="store_true", default=True,
                        help="Discover attribute words from part labels")
    parser.add_argument("--attribute_top_n",      type=int,   default=30)
    parser.add_argument("--attribute_query",      type=str,   default="wing",
                        help="Part name to query (e.g., wing, leg, handle)")
    parser.add_argument("--query_class",          type=str,   default="airplane",
                        help="Shape class for attribute query")
    # Baselines
    parser.add_argument("--models",  nargs="+",
                        default=["PointNet++", "DGCNN", "DiPVNet", "RINet", "RISA"],
                        help="Global embedding models to compare against")
    parser.add_argument("--loss",    type=str, default="proxy_anchor",
                        choices=["classification", "proxy_anchor"],
                        help="Training loss: 'classification' (cross-entropy) "
                             "or 'proxy_anchor' (metric learning, better for retrieval)")
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--seed",    type=int, default=42)
    # Cache
    parser.add_argument("--sift_cache", type=str,
                        default="outputs/exp8_sift_cache.npz",
                        help="Path to cache SIFT descriptors (.npz). "
                             "Loaded if exists, saved otherwise.")
    # Output
    parser.add_argument("--out",  default="outputs/exp8_part_retrieval.json")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
