"""Experiment 9 — RISA-SIFT: Using RISA Per-Point Features as 3D SIFT Descriptors

Instead of hand-crafted FPFH descriptors (exp8), we use RISA's learned per-point
features as descriptors in the same inverted index pipeline.

Pipeline:
  1. Train RISA with proxy-anchor loss on ShapeNet
  2. Detect keypoints per shape using ISS-style anisotropy saliency (same as exp8)
  3. Extract RISA per-point features at keypoint locations  [B, feat_dim, N]
  4. Build visual vocabulary (k-means) over all keypoint features
  5. Build inverted index: word -> {shape_id: count}
  6. Evaluate part-aware retrieval with TF-IDF scoring

Three-way comparison:
  A) 3D-SIFT-InvIndex  : hand-crafted FPFH descriptors (exp8 baseline)
  B) RISA-InvIndex     : learned RISA per-point features as descriptors (new)
  C) RISA-Global       : RISA global embedding + cosine similarity (exp8 baseline)

Hypothesis: RISA per-point features are richer and more discriminative than FPFH,
so RISA-InvIndex should outperform 3D-SIFT-InvIndex on part-mAP@5.

Usage:
    python3 experiments/exp9_risa_sift.py
    python3 experiments/exp9_risa_sift.py --epochs 20 --vocab_size 256
"""

import argparse
import json
import os
import pickle
import sys
import io

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

SHAPENET_CLASS_NAMES = [
    "airplane", "bag", "cap", "car", "chair",
    "earphone", "guitar", "knife", "lamp", "laptop",
    "motorbike", "mug", "pistol", "rocket", "skateboard", "table",
]

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
# Part 1 — Keypoint Detection (reused from exp8)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    diff  = points[:, None, :] - points[None, :, :]
    dists = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.argpartition(dists, k, axis=1)[:, :k]


def detect_keypoints(points: np.ndarray, k: int = 24, n_keypoints: int = 64) -> np.ndarray:
    """ISS-style keypoint detection via eigenvalue anisotropy."""
    N = points.shape[0]
    if N <= n_keypoints:
        return np.arange(N)
    knn_idx  = compute_knn_indices(points, k)
    saliency = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neighbors   = points[knn_idx[i]] - points[i]
        cov         = (neighbors.T @ neighbors) / k
        eigvals     = np.linalg.eigvalsh(cov)
        saliency[i] = (eigvals[2] - eigvals[0]) / (eigvals[2] + 1e-8)
    top_idx = np.argpartition(saliency, -n_keypoints)[-n_keypoints:]
    top_idx = top_idx[np.argsort(saliency[top_idx])[::-1]]
    return top_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — FPFH Descriptors (baseline, reused from exp8)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_lrf(center: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    centered = neighbors - center
    weights  = 1.0 / (np.linalg.norm(centered, axis=1) + 1e-8)
    W        = np.diag(weights / weights.sum())
    cov      = centered.T @ W @ centered
    _, vecs  = np.linalg.eigh(cov)
    vecs     = vecs[:, ::-1]
    for i in range(3):
        if np.sum(centered @ vecs[:, i]) < 0:
            vecs[:, i] *= -1
    x = vecs[:, 0]
    y = vecs[:, 1]
    z = np.cross(x, y); z /= np.linalg.norm(z) + 1e-8
    y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-8
    return np.stack([x, y, z], axis=0)


def compute_fpfh_descriptor(points: np.ndarray, keypoint_idx: np.ndarray,
                             k: int = 24, n_bins: int = 11) -> np.ndarray:
    knn_idx  = compute_knn_indices(points, k)
    n_kp     = len(keypoint_idx)
    descs    = np.zeros((n_kp, 3 * n_bins), dtype=np.float32)
    for di, kp in enumerate(keypoint_idx):
        center    = points[kp]
        neighbors = points[knn_idx[kp]]
        lrf       = _build_lrf(center, neighbors)
        directions = neighbors - center
        unit_dirs  = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
        proj       = unit_dirs @ lrf.T
        alpha = np.arccos(np.clip(proj[:, 2], -1, 1))
        phi   = np.arctan2(proj[:, 1], proj[:, 0])
        theta = np.arctan2(proj[:, 2], proj[:, 0] + 1e-8)
        h_a, _ = np.histogram(alpha, bins=n_bins, range=(0, np.pi))
        h_p, _ = np.histogram(phi,   bins=n_bins, range=(-np.pi, np.pi))
        h_t, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
        desc   = np.concatenate([h_a, h_p, h_t]).astype(np.float32)
        desc  /= np.linalg.norm(desc) + 1e-8
        desc   = np.clip(desc, 0, 0.2)
        desc  /= np.linalg.norm(desc) + 1e-8
        descs[di] = desc
    return descs


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Inverted Index (reused from exp8)
# ═══════════════════════════════════════════════════════════════════════════════

class InvertedIndex3D:
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.kmeans     = MiniBatchKMeans(
            n_clusters=vocab_size, random_state=42,
            batch_size=min(4096, vocab_size * 8),
            n_init=3, max_iter=100,
        )
        self.index: dict     = defaultdict(lambda: defaultdict(int))
        self.shape_tf: dict  = {}
        self._fitted         = False

    def build_vocabulary(self, all_descriptors: np.ndarray) -> None:
        print(f"  Building vocabulary: {self.vocab_size} words "
              f"from {len(all_descriptors):,} descriptors...", flush=True)
        self.kmeans.fit(all_descriptors)
        self._fitted = True
        print("  Vocabulary ready.", flush=True)

    def add_shape(self, shape_id: int, descriptors: np.ndarray) -> None:
        words = self.kmeans.predict(descriptors)
        self.shape_tf[shape_id] = len(words)
        for w in words:
            self.index[int(w)][shape_id] += 1

    def query(self, query_descriptors: np.ndarray, top_k: int = 10,
              exclude_id: int = -1) -> list:
        words  = self.kmeans.predict(query_descriptors)
        N      = len(self.shape_tf)
        scores = defaultdict(float)
        for w in words:
            df = len(self.index[int(w)])
            if df == 0:
                continue
            idf = np.log((N + 1) / (df + 1))
            for sid, count in self.index[int(w)].items():
                if sid == exclude_id:
                    continue
                scores[sid] += (count / self.shape_tf[sid]) * idf
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    def word_assignments(self, descriptors: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(descriptors)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Part evaluation utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _build_part_id_to_name() -> dict:
    lookup = {}
    for cls_name, part_ids in zip(SHAPENET_CLASS_NAMES, SHAPE_NET_PART_SEG_CLASSES):
        part_names = SHAPENET_PART_NAMES.get(cls_name, [])
        for local_idx, global_id in enumerate(part_ids):
            label = part_names[local_idx] if local_idx < len(part_names) else f"part{local_idx}"
            lookup[global_id] = f"{cls_name}/{label}"
    return lookup

_PART_ID_TO_NAME = _build_part_id_to_name()


def part_prob_profile(part_dist: np.ndarray) -> list:
    ranked = [
        (_PART_ID_TO_NAME.get(i, f"part{i}"), float(part_dist[i]))
        for i in range(len(part_dist)) if part_dist[i] > 0
    ]
    ranked.sort(key=lambda x: -x[1])
    return ranked


def part_distribution(part_labels: np.ndarray, n_parts: int) -> np.ndarray:
    counts = np.bincount(part_labels.astype(np.int32), minlength=n_parts).astype(np.float32)
    return counts / (counts.sum() + 1e-8)


def part_iou(dist_a: np.ndarray, dist_b: np.ndarray, threshold: float = 0.1) -> float:
    a = dist_a > threshold
    b = dist_b > threshold
    return float((a & b).sum() / ((a | b).sum() + 1e-8))


def compute_map_at_k(query_ids, retrieval_fn, class_labels, part_dists=None,
                     k=5, part_iou_threshold=0.3, return_profiles=False):
    """Compute class-mAP@k and optionally part-mAP@k with profiles."""
    class_aps, part_aps, profiles = [], [], []
    for q_id in query_ids:
        results = retrieval_fn(q_id)[:k]
        q_cls   = class_labels[q_id]
        q_dist  = part_dists[q_id] if part_dists is not None else None

        class_rel, part_rel, result_profiles = [], [], []
        for r_id, score in results:
            if r_id == q_id:
                continue
            same_cls = (class_labels[r_id] == q_cls)
            class_rel.append(float(same_cls))
            if q_dist is not None:
                iou = part_iou(q_dist, part_dists[r_id])
                part_rel.append(float(same_cls and iou >= part_iou_threshold))
                if return_profiles:
                    result_profiles.append({
                        "shape_id":   r_id,
                        "score":      round(float(score), 4),
                        "relevant":   bool(same_cls and iou >= part_iou_threshold),
                        "part_iou":   round(iou, 4),
                        "part_probs": part_prob_profile(part_dists[r_id]),
                    })

        def _ap(rel):
            rel = np.array(rel[:k])
            if rel.sum() == 0:
                return 0.0
            prec = rel.cumsum() / np.arange(1, len(rel) + 1)
            return float((prec * rel).sum() / rel.sum())

        class_aps.append(_ap(class_rel))
        if q_dist is not None:
            part_aps.append(_ap(part_rel))
        if return_profiles:
            profiles.append({
                "query_id":    q_id,
                "query_class": SHAPENET_CLASS_NAMES[q_cls] if q_cls < len(SHAPENET_CLASS_NAMES) else str(q_cls),
                "query_parts": part_prob_profile(q_dist) if q_dist is not None else [],
                "ap":          round(class_aps[-1], 4),
                "results":     result_profiles,
            })

    class_map = float(np.mean(class_aps)) if class_aps else 0.0
    part_map  = float(np.mean(part_aps))  if part_aps  else None
    return (class_map, part_map, profiles) if return_profiles else (class_map, part_map)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — RISA Training + Per-Point Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class ProxyAnchorLoss(torch.nn.Module):
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32.0):
        super().__init__()
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embed_dim))
        torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.margin = margin
        self.alpha  = alpha

    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=1)
        E = F.normalize(embeddings,   dim=1)
        sim      = E @ P.T
        pos_mask = torch.zeros_like(sim)
        pos_mask.scatter_(1, labels.unsqueeze(1), 1.0)
        neg_mask = 1.0 - pos_mask
        pos_exp  = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_exp  = torch.exp( self.alpha * (sim + self.margin)) * neg_mask
        with_pos = pos_mask.sum(0) > 0
        loss_pos = (torch.log(1 + pos_exp.sum(0)) * with_pos).sum()
        loss_neg = (torch.log(1 + neg_exp.sum(0)) * with_pos).sum()
        return (loss_pos + loss_neg) / with_pos.sum().clamp(min=1)


def build_risa(feat_dim: int = 256, checkpoint_dir: str = None):
    from models import RotationInvariantSparseAttention
    enc = RotationInvariantSparseAttention(
        encoding_out_dim=512, features_out_dim=feat_dim,
        model_dim=feat_dim, num_blocks=4,
    )
    if checkpoint_dir:
        import glob
        pattern = str(Path(checkpoint_dir) / "*_RotationInvariantSparseAttention.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            enc.load_state_dict(torch.load(matches[-1], map_location="cpu"))
            print(f"  Loaded RISA checkpoint: {matches[-1]}")
        else:
            print(f"  WARNING: no RISA checkpoint found in {checkpoint_dir}")
    return enc


def train_risa(encoder, epochs: int = 40, seed: int = 42, batch_size: int = 4) -> None:
    import torch.optim as optim
    torch.manual_seed(seed)
    ds = get_dataset("shapenet", split="train", root=DATA_ROOT, num_points=N_POINTS)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder.eval()
    with torch.no_grad():
        dummy    = torch.randn(2, 3, 64, device=DEVICE)
        z, _     = unpack_encoder_output(encoder(dummy))
    enc_dim  = z.shape[-1]
    num_cls  = int(getattr(getattr(ds, "dataset", ds), "num_classes", 16))

    criterion = ProxyAnchorLoss(num_cls, enc_dim).to(DEVICE)
    opt = optim.Adam([
        {"params": encoder.parameters(),   "lr": 1e-4},
        {"params": criterion.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def _rand_rot(B):
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
        Q[:, :, 0:1] *= torch.linalg.det(Q).sign().view(B, 1, 1)
        return Q

    encoder.train()
    for ep in range(epochs):
        for batch in loader:
            pts = batch[0] if isinstance(batch, (tuple, list)) else batch
            lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
            pts = pts.float().to(DEVICE)
            lbl = lbl.long().to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)
            pts = _rand_rot(pts.shape[0]) @ pts
            opt.zero_grad()
            z, _ = unpack_encoder_output(encoder(pts))
            criterion(z, lbl).backward()
            opt.step()
        print(f"    epoch {ep+1}/{epochs}", flush=True)
        scheduler.step()
    encoder.eval()
    del opt, criterion, loader, ds
    torch.cuda.empty_cache()


@torch.inference_mode()
def extract_risa_point_features(encoder, points_list: list, batch_size: int = 4) -> list:
    """
    Run RISA on each shape and return per-point features.
    Returns list of (N, feat_dim) arrays, one per shape.
    """
    all_feats = []
    for i in range(0, len(points_list), batch_size):
        batch_np = points_list[i:i + batch_size]
        pts = torch.from_numpy(np.stack(batch_np, axis=0)).float().to(DEVICE)
        _, features = encoder(pts)          # features: (B, feat_dim, N)
        features = features.permute(0, 2, 1).cpu().numpy()  # (B, N, feat_dim)
        for f in features:
            all_feats.append(f)             # (N, feat_dim)
        if (i + batch_size) % 500 < batch_size:
            print(f"  {min(i+batch_size, len(points_list))}/{len(points_list)} shapes encoded...", flush=True)
    return all_feats


@torch.inference_mode()
def encode_all_global(encoder, points_list: list, batch_size: int = 4) -> np.ndarray:
    all_enc = []
    for i in range(0, len(points_list), batch_size):
        pts = torch.from_numpy(np.stack(points_list[i:i+batch_size], axis=0)).float().to(DEVICE)
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu().numpy())
    return np.concatenate(all_enc, axis=0)


def global_retrieval_fn(query_id, embeddings, k=5):
    q    = embeddings[query_id] / (np.linalg.norm(embeddings[query_id]) + 1e-8)
    db   = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = db @ q
    sims[query_id] = -np.inf
    top  = np.argpartition(sims, -k)[-k:]
    top  = top[np.argsort(sims[top])[::-1]]
    return [(int(i), float(sims[i])) for i in top]


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_shapes(dataset):
    points_list, class_labels_lst, part_labels_list = [], [], []
    has_parts = False
    for idx in range(len(dataset)):
        sample = dataset[idx]
        pts = sample[0]; cls = sample[1]
        prt = sample[2] if len(sample) >= 3 else None
        points_list.append(pts.numpy() if isinstance(pts, torch.Tensor) else pts)
        class_labels_lst.append(int(cls))
        if prt is not None:
            part_labels_list.append(prt.numpy() if isinstance(prt, torch.Tensor) else prt)
            has_parts = True
        else:
            part_labels_list.append(None)
    class_labels = np.array(class_labels_lst, dtype=np.int32)
    return points_list, class_labels, (part_labels_list if has_parts else None)


def normalize_pts(pts_3n: np.ndarray) -> np.ndarray:
    pts = pts_3n.T
    pts = pts - pts.mean(axis=0)
    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    return pts


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    print(f"Device: {DEVICE}")
    print(f"Loading ShapeNet test set ({N_POINTS} pts per shape)...")
    test_ds = get_dataset("shapenet", split="test", root=DATA_ROOT, num_points=N_POINTS)
    points_list, class_labels, part_labels_list = extract_shapes(test_ds)
    N = len(points_list)
    print(f"  {N} test shapes, {len(np.unique(class_labels))} classes", flush=True)

    # ── Part distributions ─────────────────────────────────────────────────────
    n_total_parts = 50
    if part_labels_list is not None:
        part_dists = np.stack([
            part_distribution(pl, n_total_parts) if pl is not None else np.zeros(n_total_parts)
            for pl in part_labels_list
        ])
    else:
        part_dists = np.zeros((N, n_total_parts))

    # ── Query set ──────────────────────────────────────────────────────────────
    query_ids = []
    for cls_id in np.unique(class_labels):
        cls_mask = np.where(class_labels == cls_id)[0]
        query_ids.extend(cls_mask[:min(args.n_queries_per_class, len(cls_mask))].tolist())
    query_ids = sorted(query_ids)
    print(f"\nEvaluating {len(query_ids)} queries ({args.n_queries_per_class} per class)", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Method A — 3D-SIFT-InvIndex (FPFH, exp8 baseline)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nMethod A: 3D-SIFT-InvIndex (FPFH descriptors)")

    sift_cache = ROOT / args.sift_cache
    if sift_cache.exists():
        print(f"  Loading FPFH cache: {sift_cache}")
        with open(sift_cache, "rb") as f:
            cache = pickle.load(f)
        sift_descs_with_labels = cache["all_descs_with_labels"]
        sift_desc_flat         = cache["all_desc_flat"]
    else:
        print(f"  Extracting FPFH descriptors (n_keypoints={args.n_keypoints})...")
        sift_descs_with_labels, sift_desc_flat = [], []
        for i, (pts, prt) in enumerate(zip(points_list, part_labels_list or [None]*N)):
            p   = normalize_pts(pts)
            kp  = detect_keypoints(p, k=args.k_nn, n_keypoints=args.n_keypoints)
            d   = compute_fpfh_descriptor(p, kp, k=args.k_nn, n_bins=args.n_bins)
            kpl = prt[kp] if prt is not None else None
            sift_descs_with_labels.append((d, kpl))
            sift_desc_flat.append(d)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{N}...", flush=True)
        sift_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(sift_cache, "wb") as f:
            pickle.dump({"all_descs_with_labels": sift_descs_with_labels,
                         "all_desc_flat": sift_desc_flat}, f)

    sift_all_desc = np.concatenate(sift_desc_flat, axis=0)
    print(f"  Total FPFH descriptors: {len(sift_all_desc):,}", flush=True)

    sift_index = InvertedIndex3D(vocab_size=args.vocab_size)
    sift_index.build_vocabulary(sift_all_desc)
    for sid, (d, _) in enumerate(sift_descs_with_labels):
        sift_index.add_shape(sid, d)

    def sift_fn(q_id):
        d, _ = sift_descs_with_labels[q_id]
        return sift_index.query(d, top_k=args.k_eval, exclude_id=q_id)

    sift_class_map, sift_part_map, sift_profiles = compute_map_at_k(
        query_ids, sift_fn, class_labels, part_dists,
        k=args.k_eval, part_iou_threshold=args.part_iou_threshold,
        return_profiles=True,
    )
    print(f"  class-mAP@{args.k_eval} = {sift_class_map:.4f}")
    print(f"  part-mAP@{args.k_eval}  = {sift_part_map:.4f}")
    results["3D-SIFT-InvIndex"] = {
        "class_mAP@5": round(sift_class_map, 4),
        "part_mAP@5":  round(sift_part_map,  4),
        "profiles":    sift_profiles,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Train RISA (shared for Methods B and C)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nLoading RISA (feat_dim={args.feat_dim})...")
    encoder = build_risa(feat_dim=args.feat_dim, checkpoint_dir=args.checkpoint_dir).to(DEVICE)
    if not args.checkpoint_dir:
        print(f"Training RISA for {args.epochs} epochs...")
        train_risa(encoder, epochs=args.epochs, seed=args.seed)

    # ══════════════════════════════════════════════════════════════════════════
    # Method B — RISA-InvIndex (RISA per-point features as descriptors)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nMethod B: RISA-InvIndex (per-point features as descriptors)")
    print("  Extracting RISA per-point features for all shapes...")
    risa_point_feats = extract_risa_point_features(encoder, points_list, batch_size=4)

    # At each shape, select keypoint indices and take those feature vectors
    risa_descs_with_labels = []
    risa_desc_flat         = []
    for i, (pts, prt, feats) in enumerate(zip(points_list, part_labels_list or [None]*N, risa_point_feats)):
        p   = normalize_pts(pts)
        kp  = detect_keypoints(p, k=args.k_nn, n_keypoints=args.n_keypoints)
        d   = feats[kp]                    # (n_keypoints, feat_dim) — RISA features at keypoints
        kpl = prt[kp] if prt is not None else None
        risa_descs_with_labels.append((d, kpl))
        risa_desc_flat.append(d)

    risa_all_desc = np.concatenate(risa_desc_flat, axis=0)
    print(f"  Total RISA descriptors: {len(risa_all_desc):,}  dim={risa_all_desc.shape[1]}", flush=True)

    risa_index = InvertedIndex3D(vocab_size=args.vocab_size)
    risa_index.build_vocabulary(risa_all_desc)
    for sid, (d, _) in enumerate(risa_descs_with_labels):
        risa_index.add_shape(sid, d)

    def risa_inv_fn(q_id):
        d, _ = risa_descs_with_labels[q_id]
        return risa_index.query(d, top_k=args.k_eval, exclude_id=q_id)

    risa_inv_class_map, risa_inv_part_map, risa_inv_profiles = compute_map_at_k(
        query_ids, risa_inv_fn, class_labels, part_dists,
        k=args.k_eval, part_iou_threshold=args.part_iou_threshold,
        return_profiles=True,
    )
    print(f"  class-mAP@{args.k_eval} = {risa_inv_class_map:.4f}")
    print(f"  part-mAP@{args.k_eval}  = {risa_inv_part_map:.4f}")
    results["RISA-InvIndex"] = {
        "class_mAP@5": round(risa_inv_class_map, 4),
        "part_mAP@5":  round(risa_inv_part_map,  4),
        "profiles":    risa_inv_profiles,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Method C — RISA-Global (global embedding + cosine similarity)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\nMethod C: RISA-Global (global embedding retrieval)")
    global_embs = encode_all_global(encoder, points_list, batch_size=4)

    def risa_global_fn(q_id):
        return global_retrieval_fn(q_id, global_embs, k=args.k_eval)

    risa_g_class_map, risa_g_part_map, risa_g_profiles = compute_map_at_k(
        query_ids, risa_global_fn, class_labels, part_dists,
        k=args.k_eval, part_iou_threshold=args.part_iou_threshold,
        return_profiles=True,
    )
    print(f"  class-mAP@{args.k_eval} = {risa_g_class_map:.4f}")
    print(f"  part-mAP@{args.k_eval}  = {risa_g_part_map:.4f}")
    results["RISA-Global"] = {
        "class_mAP@5": round(risa_g_class_map, 4),
        "part_mAP@5":  round(risa_g_part_map,  4),
        "profiles":    risa_g_profiles,
    }

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Method':<22} {'class-mAP@5':>12} {'part-mAP@5':>12}")
    print(f"{'-'*46}")
    for method, r in results.items():
        print(f"{method:<22} {r['class_mAP@5']:>12.4f} {r['part_mAP@5']:>12.4f}")

    # ── Sample part profiles ───────────────────────────────────────────────────
    print(f"\nSample part profiles — RISA-InvIndex (first 3 queries):")
    for qp in risa_inv_profiles[:3]:
        print(f"  query {qp['query_id']} ({qp['query_class']})  AP={qp['ap']:.4f}")
        print(f"    query parts: " + ", ".join(f"{n}={p:.2f}" for n, p in qp['query_parts'][:4]))
        for i, r in enumerate(qp['results'][:3]):
            probs_str = ", ".join(f"{n}={p:.2f}" for n, p in r['part_probs'][:4])
            print(f"    result {i+1}: shape {r['shape_id']}  iou={r['part_iou']:.2f}  rel={r['relevant']}  [{probs_str}]")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"\nResults saved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exp 9: RISA per-point features as 3D SIFT descriptors"
    )
    parser.add_argument("--n_keypoints",         type=int,   default=64)
    parser.add_argument("--k_nn",                type=int,   default=24)
    parser.add_argument("--n_bins",              type=int,   default=11)
    parser.add_argument("--vocab_size",          type=int,   default=512)
    parser.add_argument("--feat_dim",            type=int,   default=256,
                        help="RISA per-point feature dimension")
    parser.add_argument("--k_eval",              type=int,   default=5)
    parser.add_argument("--n_queries_per_class", type=int,   default=50)
    parser.add_argument("--part_iou_threshold",  type=float, default=0.3)
    parser.add_argument("--epochs",              type=int,   default=40)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--sift_cache",          type=str,
                        default="outputs/exp8_sift_cache.npz",
                        help="Reuse exp8 FPFH cache if available")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Load saved RISA checkpoint instead of training")
    parser.add_argument("--out",                 type=str,
                        default="outputs/exp9_risa_sift.json")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
