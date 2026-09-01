"""Experiment 7 — Proper Metric Learning Training Comparison

Trains every model with ProxyAnchor loss (retrieval-optimised objective)
instead of the cross-entropy proxy used in exp1-6.

ProxyAnchor is chosen because:
  - Works correctly at small batch sizes (batch_size=8) — no large batch needed
  - Directly optimises the embedding space for nearest-neighbour retrieval
  - Current standard in retrieval papers (image + 3D)

For each model we report:
  - R@1 / R@5 / mAP@5 at 0°  (canonical)
  - R@1 / R@5 / mAP@5 at 90° (rotated)
  - R@1 / R@5 / mAP@5 at 180° (rotated)
  - Drop = R@1(0°) - R@1(180°)

The key comparison:
  Non-invariant models: high R@1 at 0°, large drop at 180°
  RISA:                 high R@1 at 0°, ~zero drop at 180°

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp7_proper_training.py [--dataset modelnet40] [--epochs 30]
"""

import argparse
import json
import sys
from pathlib import Path

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

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
EVAL_ANGLES = [0, 90, 180]


# ── ProxyAnchor Loss ──────────────────────────────────────────────────────────

class ProxyAnchorLoss(nn.Module):
    """
    ProxyAnchor Loss (Kim et al., CVPR 2020).

    One learnable proxy per class. For each batch:
      - Pull embeddings toward their class proxy
      - Push embeddings away from all other proxies

    Works at small batch sizes because gradients flow through proxies,
    not through pairwise sample distances.
    """

    def __init__(self, num_classes: int, embed_dim: int, margin: float = 0.1, alpha: float = 32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.margin = margin
        self.alpha  = alpha

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalise both embeddings and proxies
        e = F.normalize(embeddings, dim=1)          # (B, D)
        p = F.normalize(self.proxies, dim=1)        # (C, D)

        sim = e @ p.T                               # (B, C)

        # Positive mask: sim[i,c] is positive if label[i] == c
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        pos_mask[torch.arange(len(labels)), labels] = True
        neg_mask = ~pos_mask

        # Per-proxy positive and negative terms
        pos_exp = torch.exp(-self.alpha * (sim - self.margin))
        neg_exp = torch.exp( self.alpha * (sim + self.margin))

        # Only sum over proxies that have at least one positive in the batch
        with_pos = pos_mask.any(dim=0)              # (C,)

        pos_term = (pos_exp * pos_mask.float()).sum(dim=0)   # (C,)
        neg_term = (neg_exp * neg_mask.float()).sum(dim=0)   # (C,)

        loss_pos = torch.log(1 + pos_term[with_pos]).mean()
        loss_neg = torch.log(1 + neg_term[with_pos]).mean()

        return loss_pos + loss_neg


# ── Training ──────────────────────────────────────────────────────────────────

def proxy_anchor_train(encoder, dataset_name, num_classes, epochs=30,
                       batch_size=8, seed=42, lr=1e-4, proxy_lr=1e-3):
    torch.manual_seed(seed)
    kwargs = {"variant": "OBJ_ONLY"} if dataset_name == "scanobjectnn" else {}
    try:
        ds = get_dataset(dataset_name, split="train", root=DATA_ROOT,
                         num_points=N_POINTS, **kwargs)
    except Exception as e:
        print(f"    [train] Cannot load {dataset_name}: {e} — skipping"); return

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Get embedding dim from a dummy forward pass
    encoder.eval()
    with torch.no_grad():
        z, _ = unpack_encoder_output(encoder(torch.randn(2, 3, 64, device=DEVICE)))
    embed_dim = z.shape[-1]

    criterion = ProxyAnchorLoss(num_classes, embed_dim).to(DEVICE)

    # Separate LR for proxies (higher) vs encoder (lower) — standard practice
    opt = optim.AdamW([
        {"params": encoder.parameters(),   "lr": lr},
        {"params": criterion.parameters(), "lr": proxy_lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def _rand_rot(B):
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
        Q[:, :, 0:1] *= torch.linalg.det(Q).sign().view(B, 1, 1)
        return Q

    encoder.train(); criterion.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for batch in loader:
            pts = batch[0] if isinstance(batch, (tuple, list)) else batch
            lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                pts, lbl, _ = batch
            pts = pts.float().to(DEVICE)
            lbl = lbl.long().to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)

            # SO(3) augmentation
            pts = _rand_rot(pts.shape[0]) @ pts

            opt.zero_grad()
            z, _ = unpack_encoder_output(encoder(pts))
            loss  = criterion(z, lbl)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        print(f"    epoch {ep+1}/{epochs}  loss={epoch_loss/n_batches:.4f}", flush=True)

    encoder.eval()
    del criterion, opt, scheduler


# ── Model registry ────────────────────────────────────────────────────────────

def _build_registry():
    from models import (
        DGCNNEncoder, PointMambaEncoder, PointNet2Encoder,
        RSCNNEncoder, VNNEncoder, DiPVNetEncoder,
    )
    from models.transformer import PointTransformerEncoder
    return {
        "PointNet++":       PointNet2Encoder,
        "DGCNN":            DGCNNEncoder,
        "RS-CNN":           RSCNNEncoder,
        "PointTransformer": PointTransformerEncoder,
        "PointMamba":       PointMambaEncoder,
        "VNN":              VNNEncoder,
        "DiPVNet":          DiPVNetEncoder,
        "RISA":             None,   # built separately
    }


def _build_encoder(name, cls):
    if name == "RISA":
        from models import RotationInvariantSparseAttention
        return RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    return cls()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="modelnet40")
    parser.add_argument("--models",   nargs="+", default=None)
    parser.add_argument("--epochs",   type=int, default=30)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="outputs/exp7_proper_training.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    # Get num_classes for ProxyAnchor
    _base      = getattr(test_ds, "dataset", test_ds)
    num_classes = int(getattr(_base, "num_classes", 40))
    print(f"Dataset: {args.dataset}  |  classes: {num_classes}  |  test: {len(test_ds)}")

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls).to(DEVICE)

        proxy_anchor_train(
            encoder, args.dataset, num_classes,
            epochs=args.epochs, seed=args.seed,
        )

        db_enc, db_lbl = encode_dataset(encoder, test_ds)

        angle_results = []
        for theta in EVAL_ANGLES:
            if theta == 0:
                q_enc = db_enc
            else:
                R     = rotation_by_angle(theta, axis="y", device=DEVICE)
                q_enc = encode_rotated(encoder, test_ds, R)

            r1   = recall_at_k(q_enc, db_enc, db_lbl, db_lbl, k=1)
            r5   = recall_at_k(q_enc, db_enc, db_lbl, db_lbl, k=5)
            map5 = mean_ap_at_k(q_enc, db_enc, db_lbl, db_lbl, k=5)
            angle_results.append({"angle": theta, "r1": r1, "r5": r5, "map5": map5})
            print(f"  θ={theta:5.1f}°  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}",
                  flush=True)

        results[model_name] = angle_results
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"PROPER TRAINING COMPARISON — {args.dataset} — ProxyAnchor Loss")
    print(f"{'Model':<18} {'R@1 (0°)':>9} {'R@1 (90°)':>10} {'R@1 (180°)':>11} {'Drop':>8}  {'Invariant?':>10}")
    print("-" * 72)
    for model_name, angle_res in results.items():
        row  = {r["angle"]: r["r1"] for r in angle_res}
        r0   = row.get(0,   float("nan"))
        r90  = row.get(90,  float("nan"))
        r180 = row.get(180, float("nan"))
        drop = r0 - r180
        inv  = "✓" if drop < 0.05 else "✗"
        print(f"{model_name:<18} {r0:>9.3f} {r90:>10.3f} {r180:>11.3f} {drop:>8.3f}  {inv:>10}")
    print("="*72)
    print("Drop < 0.05 = effectively rotation-invariant")


if __name__ == "__main__":
    main()
