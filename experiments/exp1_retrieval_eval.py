"""Experiment 1 — Rotation-Stratified Retrieval Benchmark

For each model x dataset:
  1. Encode the full test set at canonical orientation → database
  2. Query with each shape rotated by angle θ (swept 0° → 180°)
  3. Measure Recall@1, Recall@5, mAP@5 at each angle

This is the core experiment of the paper. The key figure is
Recall@1 vs rotation angle for all models on all datasets.

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp1_retrieval_eval.py [--datasets modelnet40 shapenet] [--n_angles 19]
"""

import argparse
import json
import sys, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024

DATASETS = ["synthetic", "modelnet40", "shapenet", "scanobjectnn"]


# ── Model registry ────────────────────────────────────────────────────────────

def _build_registry():
    from models import (
        DGCNNEncoder, PointMambaEncoder, PointNet2Encoder,
        RotationInvariantSparseAttention, RSCNNEncoder,
        VNNEncoder, DiPVNetEncoder,
    )
    from models.transformer import PointTransformerEncoder
    return {
        "PointNet++"      : PointNet2Encoder,
        "PointMamba"      : PointMambaEncoder,
        "RS-CNN"          : RSCNNEncoder,
        "PointTransformer": PointTransformerEncoder,
        "DGCNN"           : DGCNNEncoder,
        "VNN"             : VNNEncoder,
        "DiPVNet"         : DiPVNetEncoder,
        "RISA"            : RotationInvariantSparseAttention,
    }


def _build_encoder(name, cls, checkpoint_dir=None):
    if name == "RISA":
        from models import RotationInvariantSparseAttention
        enc = RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    else:
        enc = cls()
    if checkpoint_dir:
        import glob
        # match by class name substring, e.g. DGCNNEncoder, PointNet2Encoder
        pattern = str(Path(checkpoint_dir) / f"*_{enc.__class__.__name__}.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            ckpt = matches[-1]  # latest
            enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  Loaded checkpoint: {ckpt}")
        else:
            print(f"  WARNING: no checkpoint found for {name} in {checkpoint_dir}, using random weights")
    return enc


# ── Rotation helpers ──────────────────────────────────────────────────────────

def random_rotation(device):
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotation_by_angle(angle_deg: float, axis: str = "y", device="cpu"):
    """Fixed-axis rotation by a specific angle (for the sweep experiment)."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "y":
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    elif axis == "x":
        R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    else:
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    return R.to(device)


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(encoder, dataset, batch_size=8):
    """Encode all shapes in dataset. Returns (N, D) encodings and (N,) labels."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_enc, all_lbl = [], []
    for batch in loader:
        pts = batch[0] if isinstance(batch, (tuple, list)) else batch
        lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0])
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            pts, lbl, _ = batch
        pts = pts.float().to(DEVICE)
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu())
        all_lbl.append(lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl))
    return torch.cat(all_enc), torch.cat(all_lbl).long()


@torch.no_grad()
def encode_rotated(encoder, dataset, R: torch.Tensor, batch_size=8):
    """Encode all shapes after applying rotation R. Returns (N, D) encodings."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_enc = []
    for batch in loader:
        pts = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            pts = batch[0]
        pts = pts.float().to(DEVICE)
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        pts = R.to(DEVICE) @ pts          # apply rotation
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu())
    return torch.cat(all_enc)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(query_enc, db_enc, query_lbl, db_lbl, k=1):
    """Recall@k: fraction of queries whose top-k results contain a same-class item."""
    q = F.normalize(query_enc, dim=-1)
    d = F.normalize(db_enc,    dim=-1)
    sims   = q @ d.T                          # (Q, N)
    topk   = sims.topk(k + 1, dim=-1).indices # +1 to exclude self
    hits   = 0
    for i in range(len(query_lbl)):
        retrieved = topk[i]
        # exclude the query itself (exact match by index)
        retrieved = retrieved[retrieved != i][:k]
        if (db_lbl[retrieved] == query_lbl[i]).any():
            hits += 1
    return hits / len(query_lbl)


def mean_ap_at_k(query_enc, db_enc, query_lbl, db_lbl, k=5):
    """mAP@k across all queries."""
    q = F.normalize(query_enc, dim=-1)
    d = F.normalize(db_enc,    dim=-1)
    sims = q @ d.T
    aps  = []
    for i in range(len(query_lbl)):
        row = sims[i].clone()
        row[i] = -1e9                          # exclude self
        topk_idx = row.topk(k).indices
        rel = (db_lbl[topk_idx] == query_lbl[i]).float()
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        precisions = rel.cumsum(0) / torch.arange(1, k + 1, dtype=torch.float)
        aps.append((precisions * rel).sum().item() / min(k, int(rel.sum().item())))
    return float(np.mean(aps))


# ── Quick training ────────────────────────────────────────────────────────────

def quick_train(encoder, dataset_name, epochs=20, batch_size=8, seed=42):
    import torch.nn as nn, torch.optim as optim

    torch.manual_seed(seed)
    kwargs = {"variant": "OBJ_ONLY"} if dataset_name == "scanobjectnn" else {}
    try:
        ds = get_dataset(dataset_name, split="train", root=DATA_ROOT,
                         num_points=N_POINTS, **kwargs)
    except Exception as e:
        print(f"    [train] Cannot load {dataset_name}: {e} — skipping")
        return

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    encoder.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 64, device=DEVICE)
        z, _  = unpack_encoder_output(encoder(dummy))
    enc_dim = z.shape[-1]

    _base   = getattr(ds, "dataset", ds)
    num_cls = int(getattr(_base, "num_classes", 40))
    head    = nn.Linear(enc_dim, num_cls).to(DEVICE)
    opt     = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

    def _rand_rot(B):
        """Batch of random SO(3) rotation matrices (B, 3, 3)."""
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
        signs = torch.linalg.det(Q).sign().view(B, 1, 1)
        Q[:, :, 0:1] *= signs
        return Q

    encoder.train(); head.train()
    for ep in range(epochs):
        for batch in loader:
            pts = batch[0] if isinstance(batch, (tuple, list)) else batch
            lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                pts, lbl, _ = batch
            pts = pts.float().to(DEVICE)
            lbl = lbl.long().to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)
            # Batched SO(3) augmentation — no Python loop
            pts = _rand_rot(pts.shape[0]) @ pts
            opt.zero_grad()
            z, _ = unpack_encoder_output(encoder(pts))
            crit(head(z), lbl).backward()
            opt.step()
        print(f"    epoch {ep+1}/{epochs}", flush=True)
    encoder.eval()
    del head, opt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", default=DATASETS)
    parser.add_argument("--models",    nargs="+", default=None,
                        help="Subset of models to run, e.g. --models RISA VNN")
    parser.add_argument("--epochs",    type=int,  default=20)
    parser.add_argument("--n_angles",  type=int,  default=19,
                        help="Number of angles from 0 to 180 (inclusive)")
    parser.add_argument("--n_random",  type=int,  default=5,
                        help="Random SO(3) rotations per angle for averaging")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Load saved checkpoints from this dir instead of training from scratch")
    parser.add_argument("--out",       default="outputs/exp1_retrieval.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}\n")

    angles   = np.linspace(0, 180, args.n_angles).tolist()
    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results  = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        results[model_name] = {}

        for ds_name in args.datasets:
            print(f"\n  Dataset: {ds_name}")
            kwargs = {"variant": "OBJ_ONLY"} if ds_name == "scanobjectnn" else {}

            try:
                test_ds = get_dataset(ds_name, split="test", root=DATA_ROOT,
                                      num_points=N_POINTS, **kwargs)
            except Exception as e:
                print(f"  Cannot load {ds_name}: {e} — skipping")
                continue

            encoder = _build_encoder(model_name, model_cls, args.checkpoint_dir).to(DEVICE)
            if not args.checkpoint_dir:
                quick_train(encoder, ds_name, epochs=args.epochs, seed=args.seed)

            # Build database from canonical (unrotated) encodings
            print("  Building database...", flush=True)
            db_enc, db_lbl = encode_dataset(encoder, test_ds)

            angle_results = []
            for theta in angles:
                r1_vals, r5_vals, map5_vals = [], [], []
                for _ in range(args.n_random):
                    if theta == 0:
                        q_enc = db_enc
                    else:
                        # Random axis rotation at fixed angle for robustness
                        R = rotation_by_angle(theta, axis="y", device=DEVICE)
                        q_enc = encode_rotated(encoder, test_ds, R)
                    r1   = recall_at_k(q_enc, db_enc, db_lbl, db_lbl, k=1)
                    r5   = recall_at_k(q_enc, db_enc, db_lbl, db_lbl, k=5)
                    map5 = mean_ap_at_k(q_enc, db_enc, db_lbl, db_lbl, k=5)
                    r1_vals.append(r1); r5_vals.append(r5); map5_vals.append(map5)

                angle_results.append({
                    "angle":  theta,
                    "r1":     float(np.mean(r1_vals)),
                    "r5":     float(np.mean(r5_vals)),
                    "map5":   float(np.mean(map5_vals)),
                })
                print(f"    θ={theta:6.1f}°  R@1={np.mean(r1_vals):.3f}  "
                      f"R@5={np.mean(r5_vals):.3f}  mAP@5={np.mean(map5_vals):.3f}",
                      flush=True)

            results[model_name][ds_name] = angle_results
            del encoder
            torch.cuda.empty_cache()

    if _TB_AVAILABLE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb = _TBWriter(log_dir=f"runs/experiment1/{ts}")
        for model_name, ds_dict in results.items():
            for ds_name, angle_results in ds_dict.items():
                for entry in angle_results:
                    step = int(entry["angle"])
                    tb.add_scalar(f"{model_name}/{ds_name}/R@1",   entry["r1"],   step)
                    tb.add_scalar(f"{model_name}/{ds_name}/R@5",   entry["r5"],   step)
                    tb.add_scalar(f"{model_name}/{ds_name}/mAP@5", entry["map5"], step)
        tb.close()
        print(f"TensorBoard logs → runs/experiment1/{ts}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
