"""Experiment 10 — Geometric Prior Point Selection Ablation

Question: which rotation-invariant geometric prior best selects the K most
informative points to attend over, and does selecting K << N points improve
retrieval over attending all N?

All conditions use identical RISA architecture and training (proxy anchor,
SO(3) aug, ModelNet40). The only difference is which K points enter the
attention blocks. Features are always computed over all N points first
(cheap, no-grad) — selection happens before feature_embed.

Conditions
----------
full_N          : baseline — all N points, standard RISA
random_K        : random K points — lower bound
fps_K           : farthest point sampling on XYZ — spatial coverage
eigenentropy_K  : top-K by -sum(e_i * log(e_i)) — geometrically complex points
surface_var_K   : top-K by e3 / (e1+e2+e3) — high local variation
curvature_K     : top-K by (e3-e1)/e3 (anisotropy) — elongated/curved structures
salient_K       : top-K by enc_token cross-attention weights (block 0) — learned saliency
aggregate_K         : FPS centroids + max-pool over ball-query neighborhood — lossless compression
knn_dist_entropy_K  : top-K by entropy of k-NN distance distribution — geometrically diverse neighborhoods

Usage:
    cd /home/grad/smenon/retrieval_paper
    python experiments/exp10_prior_selection.py [--K 256] [--epochs 100]
"""

import argparse
import json
import sys, io
if hasattr(sys.stdout, 'fileno'):
    try:
        sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
    except Exception:
        pass
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import (
    encode_dataset, encode_rotated, recall_at_k, mean_ap_at_k,
    rotation_by_angle,
)
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
        pos_mask = torch.zeros_like(sim).scatter_(1, labels.unsqueeze(1), 1.0)
        neg_mask = 1.0 - pos_mask
        with_pos  = pos_mask.sum(0) > 0
        pos_exp   = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_exp   = torch.exp( self.alpha * (sim + self.margin)) * neg_mask
        loss_pos  = (torch.log(1 + pos_exp.sum(0)) * with_pos).sum()
        loss_neg  = (torch.log(1 + neg_exp.sum(0)) * with_pos).sum()
        return (loss_pos + loss_neg) / with_pos.sum().clamp(min=1)

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ANGLES    = [0, 15, 30, 45, 60, 90, 120, 150, 180]

PRIOR_NAMES = [
    "full_N",
    "random_K",
    "fps_K",
    "eigenentropy_K",
    "surface_var_K",
    "curvature_K",
    "salient_K",
    "aggregate_K",
    "knn_dist_entropy_K",
]


# ── Point selection functions ─────────────────────────────────────────────────
# All take features [B, 64, N] and evals [B, N, 3] (from GeometricInvariantExtractor)
# and return indices [B, K] of selected points.

def select_random(features, evals, K):
    B, _, N = features.shape
    idx = torch.stack([torch.randperm(N, device=features.device)[:K] for _ in range(B)])
    return idx  # [B, K]


def select_fps(xyz, K):
    """FPS on XYZ coordinates. xyz: [B, 3, N] → [B, K]"""
    B, _, N = xyz.shape
    device = xyz.device
    xyz_T = xyz.permute(0, 2, 1)  # [B, N, 3]

    selected = torch.zeros(B, K, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(K):
        selected[:, i] = farthest
        centroid = xyz_T[torch.arange(B, device=device), farthest].unsqueeze(1)  # [B, 1, 3]
        d = ((xyz_T - centroid) ** 2).sum(dim=-1)  # [B, N]
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=-1)

    return selected  # [B, K]


def select_knn_dist_entropy(xyz, K, k_nb=16, eps=1e-8):
    """Top-K by entropy of the k-NN distance distribution.

    Points with high entropy in their neighbor-distance histogram have
    geometrically diverse neighborhoods — maximally informative for the
    neighbor-distance MLP in GeometricInvariantExtractor.

    Computes only the k_nb nearest distances per point (O(N * k_nb))
    rather than the full N×N matrix.

    xyz  : [B, 3, N]
    k_nb : neighborhood size for distance distribution
    returns [B, K]
    """
    B, _, N = xyz.shape
    xyz_T = xyz.permute(0, 2, 1)                                    # [B, N, 3]
    # chunk over points to avoid O(N^2) memory
    knn_dists = []
    chunk = 64
    for i in range(0, N, chunk):
        diff = xyz_T[:, i:i+chunk, :].unsqueeze(2) - xyz_T.unsqueeze(1)  # [B, chunk, N, 3]
        d    = (diff ** 2).sum(dim=-1)                              # [B, chunk, N]
        # mask self: point i+j has global index i+j in the N dimension
        for j in range(d.shape[1]):
            d[:, j, i + j] = float('inf')
        knn_dists.append(d.topk(k_nb, dim=-1, largest=False).values)       # [B, chunk, k_nb]
    knn_dists = torch.cat(knn_dists, dim=1)                         # [B, N, k_nb]
    # use actual distances (sqrt) so the distribution isn't dominated by far neighbors
    knn_dists = knn_dists.sqrt().clamp(min=eps)
    p = knn_dists / knn_dists.sum(dim=-1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=-1)                             # [B, N]
    return entropy.topk(K, dim=-1).indices                          # [B, K]


def select_eigenentropy(evals, K, eps=1e-8):
    """Top-K by eigenentropy = -sum(p_i * log(p_i)) where p_i = e_i / sum(e).
    evals: [B, N, 3] → [B, K]"""
    e = evals.clamp(min=eps)
    p = e / e.sum(dim=-1, keepdim=True)          # normalise to probability
    entropy = -(p * p.log()).sum(dim=-1)          # [B, N]
    return entropy.topk(K, dim=-1).indices        # [B, K]


def select_surface_var(evals, K, eps=1e-8):
    """Top-K by surface variation = e3 / (e1+e2+e3). evals: [B, N, 3] → [B, K]"""
    e_sum = evals.sum(dim=-1).clamp(min=eps)
    sv = evals[:, :, 2] / e_sum  # [B, N]
    return sv.topk(K, dim=-1).indices  # [B, K]


def select_curvature(evals, K, eps=1e-8):
    """Top-K by PCA curvature = e1 / (e1+e2+e3) — smallest eigenvalue ratio.
    High values indicate locally curved / corner-like regions.
    evals: [B, N, 3] sorted ascending → e1 <= e2 <= e3. [B, N, 3] → [B, K]"""
    e_sum = evals.sum(dim=-1).clamp(min=eps)
    curvature = evals[:, :, 0] / e_sum           # [B, N]
    return curvature.topk(K, dim=-1).indices      # [B, K]


def aggregate_fps_ballquery(features, xyz, K):
    """Aggregate N points into K tokens via FPS centroids + ball-query max-pool.

    1. FPS on XYZ → K centroid indices
    2. For each centroid, find its M=N//K nearest neighbors (ball query)
    3. Max-pool their 64-dim features → one token per centroid

    features : [B, 64, N]
    xyz      : [B, 3,  N]
    returns  : aggregated [B, 64, K], centroid_idx [B, K], pos_enc_idx [B, K]
    """
    B, D, N = features.shape
    M = max(1, N // K)          # neighbors per centroid
    device = xyz.device

    # 1. FPS → centroid indices [B, K]
    centroid_idx = select_fps(xyz, K)                          # [B, K]
    b_idx = torch.arange(B, device=device).unsqueeze(1)        # [B, 1]

    xyz_T     = xyz.permute(0, 2, 1)                           # [B, N, 3]
    centroids = xyz_T[b_idx, centroid_idx]                     # [B, K, 3]

    # 2. For each centroid find M nearest neighbors among all N points
    diff = centroids.unsqueeze(2) - xyz_T.unsqueeze(1)         # [B, K, N, 3]
    dist = (diff ** 2).sum(dim=-1)                             # [B, K, N]
    nb_idx = dist.topk(M, dim=-1, largest=False).indices       # [B, K, M]

    # 3. Gather features and max-pool
    feats_T  = features.permute(0, 2, 1)                       # [B, N, 64]
    b_idx_k  = torch.arange(B, device=device).view(B,1,1).expand(B, K, M)
    nb_feats = feats_T[b_idx_k, nb_idx]                        # [B, K, M, 64]
    agg      = nb_feats.max(dim=2).values                      # [B, K, 64]

    return agg.permute(0, 2, 1), centroid_idx                  # [B, 64, K], [B, K]


def select_salient(features, indices, pos_enc, feature_embed, block, enc_token, K):
    """Top-K by enc_token cross-attention weights from block 0.
    Runs feature_embed + one attention block under no_grad.
    features: [B, 64, N], returns [B, K]"""
    with torch.no_grad():
        B, D, N = features.shape
        f = features.permute(0, 2, 1).reshape(B * N, D)
        f = feature_embed(f)
        f = f.reshape(B, N, -1).permute(0, 2, 1)  # [B, C, N]
        _, _, attn_enc = block(
            f, enc_token=enc_token, indices=indices,
            pos_enc=pos_enc, update_points=True, return_weights=True,
        )  # attn_enc: [B, 1, N]
    scores = attn_enc.squeeze(1)  # [B, N]
    return scores.topk(K, dim=-1).indices  # [B, K]


# ── RISA with prior-based point selection ─────────────────────────────────────

class RISAWithPrior(nn.Module):
    """RISA encoder with a pluggable point selection prior.

    GeometricInvariantExtractor runs over all N points (unchanged).
    Before feature_embed, K points are selected by the prior.
    Attention runs only over those K points → O(K²) instead of O(N²).

    prior: one of PRIOR_NAMES
    K:     number of points to select (ignored for full_N)
    """

    name = "risa"  # overridden per condition at runtime

    def set_prototypes(self, prototypes): self._prototypes = prototypes
    def set_labels(self, labels): self._labels = labels

    def __init__(
        self,
        prior: str = "full_N",
        K: int = 256,
        num_local: int = 32,
        num_global: int = 32,
        num_blocks: int = 4,
        model_dim: int = 256,
        features_out_dim: int = 256,
        encoding_out_dim: int = 512,
    ):
        super().__init__()
        assert prior in PRIOR_NAMES, f"Unknown prior: {prior}"

        self.prior            = prior
        self.K                = K
        self.model_dim        = model_dim
        self.encoding_out_dim = encoding_out_dim
        self._prototypes      = None
        self._labels          = None

        from models.risa import (
            GeometricInvariantExtractor,
            SparsePointCloudAttention,
        )

        self.feature_extractor = GeometricInvariantExtractor(
            sparse     = True,
            num_global = num_global,
            num_local  = num_local,
        )
        feat_dim = self.feature_extractor.feature_dim  # 64

        self.feature_embed = nn.Sequential(
            nn.Linear(feat_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        # Attention blocks — sparse kNN within the selected K points
        self.blocks = nn.ModuleList([
            SparsePointCloudAttention(
                model_dim    = model_dim,
                out_dim      = model_dim,
                num_heads    = 4,
                attn_augment = "weight",
            )
            for _ in range(num_blocks)
        ])

        self.encoding_token = nn.Parameter(torch.empty(1, model_dim, 1))
        nn.init.xavier_uniform_(self.encoding_token)

        self.features_post = nn.Sequential(
            nn.Conv1d(model_dim, features_out_dim, 1),
            nn.BatchNorm1d(features_out_dim),
            nn.GELU(),
        )
        self.encoding_post = nn.Sequential(
            nn.Conv1d(model_dim, encoding_out_dim, 1),
            nn.BatchNorm1d(encoding_out_dim),
            nn.GELU(),
        )

    def _select(self, features, evals, xyz, indices, pos_enc):
        """Select K point indices using self.prior. Returns [B, K]."""
        B, _, N = features.shape
        K = min(self.K, N)

        if self.prior == "full_N":
            return None  # signal to use all points

        if self.prior == "random_K":
            return select_random(features, evals, K)

        if self.prior == "fps_K":
            return select_fps(xyz, K)

        if self.prior == "eigenentropy_K":
            return select_eigenentropy(evals, K)

        if self.prior == "surface_var_K":
            return select_surface_var(evals, K)

        if self.prior == "curvature_K":
            return select_curvature(evals, K)

        if self.prior == "knn_dist_entropy_K":
            return select_knn_dist_entropy(xyz, K)

        if self.prior == "salient_K":
            enc_token = self.encoding_token.expand(B, -1, -1).to(device=xyz.device)
            return select_salient(features, indices, pos_enc, self.feature_embed, self.blocks[0], enc_token, K)

    def forward(self, x):
        B, _, N = x.shape

        # Full-N feature extraction (no grad, cheap)
        features, indices, pos_enc = self.feature_extractor(x)
        evals = features[:, 1:4, :].permute(0, 2, 1)  # [B, N, 3]

        if self.prior == "aggregate_K":
            K = min(self.K, N)
            with torch.no_grad():
                agg_features, centroid_idx = aggregate_fps_ballquery(features, x, K)  # [B, 64, K], [B, K]
            b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
            # use centroid's own pos_enc as representative for the cluster
            pos_enc_mean = pos_enc.mean(dim=2)                           # [B, N, 9]
            pos_enc_K    = pos_enc_mean[b_idx, centroid_idx]             # [B, K, 9]
            pos_enc_K    = pos_enc_K.unsqueeze(2).expand(-1, -1, K, -1) # [B, K, K, 9]
            indices_K    = torch.arange(K, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, K, -1)
            features_in  = agg_features
            attn_indices = indices_K
            attn_pos_enc = pos_enc_K
        else:
            # Select K point indices (or None for full_N)
            sel_idx = self._select(features, evals, x, indices, pos_enc)  # [B, K] or None
            self._labels = None

            if sel_idx is not None:
                K = sel_idx.shape[1]
                b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
                features_K   = features.permute(0, 2, 1)[b_idx, sel_idx].permute(0, 2, 1)
                indices_K    = torch.arange(K, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, K, -1)
                pos_enc_mean = pos_enc.mean(dim=2)
                pos_enc_K    = pos_enc_mean[b_idx, sel_idx]
                pos_enc_K    = pos_enc_K.unsqueeze(2).expand(-1, -1, K, -1)
                features_in  = features_K
                attn_indices = indices_K
                attn_pos_enc = pos_enc_K
            else:
                features_in  = features
                attn_indices = indices
                attn_pos_enc = pos_enc

        # Embed features
        B2, D, M = features_in.shape
        f = features_in.permute(0, 2, 1).reshape(B2 * M, D)
        f = self.feature_embed(f)
        f = f.reshape(B2, M, self.model_dim).permute(0, 2, 1)  # [B, C, M]

        enc_token = self.encoding_token.expand(B2, -1, -1).to(dtype=x.dtype)

        for block in self.blocks:
            f, enc_token = block(
                f,
                enc_token     = enc_token,
                indices       = attn_indices,
                pos_enc       = attn_pos_enc,
                update_points = True,
            )

        f        = self.features_post(f)
        encoding = self.encoding_post(enc_token)
        return encoding, f


# ── Training ──────────────────────────────────────────────────────────────────

def train_condition(encoder, dataset_name, epochs, seed, batch_size=4):
    torch.manual_seed(seed)
    ds = get_dataset(dataset_name, split="train", root=DATA_ROOT, num_points=N_POINTS)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    with torch.no_grad():
        dummy = torch.randn(2, 3, 64, device=DEVICE)
        z, _  = unpack_encoder_output(encoder(dummy))
    enc_dim = z.shape[-1]
    num_cls = int(getattr(getattr(ds, "dataset", ds), "num_classes", 40))

    criterion = ProxyAnchorLoss(num_cls, enc_dim, margin=0.1, alpha=32.0).to(DEVICE)
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
        ep_loss, n = 0.0, 0
        for batch in loader:
            pts = batch[0] if isinstance(batch, (tuple, list)) else batch
            lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
            pts = pts.float().to(DEVICE)
            lbl = lbl.long().to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)
            pts = _rand_rot(pts.shape[0]) @ pts
            opt.zero_grad()
            if encoder.prior == "prototype_K":
                encoder.set_labels(lbl)
            z, _ = unpack_encoder_output(encoder(pts))
            loss = criterion(z, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n += 1
        scheduler.step()
        print(f"    epoch {ep+1}/{epochs}  loss={ep_loss/max(n,1):.4f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    encoder.eval()
    del criterion, opt, scheduler, loader, ds
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="modelnet40")
    parser.add_argument("--K",        type=int, default=256,
                        help="Points to select per shape (ignored for full_N)")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--priors",   nargs="+", default=PRIOR_NAMES,
                        help="Subset of priors to run")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dim",  type=int, default=256)
    parser.add_argument("--out",      default="outputs/exp10_prior_selection.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}  K={args.K}  epochs={args.epochs}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    results = {}
    tb = None
    if _TB_AVAILABLE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb = _TBWriter(log_dir=f"runs/exp10_prior_selection_{ts}")

    for prior in args.priors:
        if prior not in PRIOR_NAMES:
            print(f"  Unknown prior '{prior}', skipping")
            continue

        print(f"\n{'='*60}\nPrior: {prior}  (K={args.K if prior != 'full_N' else N_POINTS})")

        encoder = RISAWithPrior(
            prior            = prior,
            K                = args.K,
            num_local        = 32,
            num_global       = 32,
            num_blocks       = 4,
            model_dim        = args.model_dim,
            features_out_dim = args.model_dim,
            encoding_out_dim = args.model_dim * 2,
        ).to(DEVICE)
        encoder.name = prior

        if prior == "prototype_K":
            train_ds = get_dataset(args.dataset, split="train", root=DATA_ROOT, num_points=N_POINTS)
            print("  Building class prototypes...", flush=True)
            prototypes = build_class_prototypes(encoder.feature_extractor, train_ds, DEVICE)
            encoder.set_prototypes(prototypes)
            print(f"  Prototypes built: {prototypes.shape}", flush=True)

        train_condition(encoder, args.dataset, args.epochs, args.seed, batch_size=args.batch_size)

        db_enc, db_lbl = encode_dataset(encoder, test_ds)

        angle_results = []
        for theta in ANGLES:
            if theta == 0:
                q_enc = db_enc
            else:
                R     = rotation_by_angle(theta, axis="y", device=DEVICE)
                q_enc = encode_rotated(encoder, test_ds, R)

            sim  = q_enc @ db_enc.T
            r1   = recall_at_k(sim, db_lbl, db_lbl, k=1)
            r5   = recall_at_k(sim, db_lbl, db_lbl, k=5)
            map5 = mean_ap_at_k(sim, db_lbl, db_lbl, k=5)
            angle_results.append({"angle": theta, "r1": r1, "r5": r5, "map5": map5})
            print(f"  θ={theta:5.1f}°  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}", flush=True)

            if tb is not None:
                tb.add_scalar(f"{prior}/R@1",   r1,   theta)
                tb.add_scalar(f"{prior}/R@5",   r5,   theta)
                tb.add_scalar(f"{prior}/mAP@5", map5, theta)

        results[prior] = angle_results
        del encoder
        torch.cuda.empty_cache()

    if tb is not None:
        tb.close()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"PRIOR SELECTION SUMMARY — {args.dataset} — K={args.K}")
    print(f"{'Prior':<20} {'0°':>6} {'45°':>6} {'90°':>6} {'180°':>6}  mAP@5(0°)")
    print("-" * 60)
    key_angles = {0, 45, 90, 180}
    for prior, angle_res in results.items():
        row = {r["angle"]: r for r in angle_res}
        r1s  = [row.get(a, {}).get("r1",   float("nan")) for a in sorted(key_angles)]
        map0 = row.get(0, {}).get("map5", float("nan"))
        print(f"{prior:<20} {r1s[0]:>6.3f} {r1s[1]:>6.3f} {r1s[2]:>6.3f} {r1s[3]:>6.3f}  {map0:>6.3f}")


if __name__ == "__main__":
    main()
