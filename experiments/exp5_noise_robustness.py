"""Experiment 5 — Retrieval Under Joint Rotation + Noise

Tests retrieval robustness when queries are both rotated AND corrupted
with Gaussian noise (simulating real sensor conditions).

Noise levels: σ ∈ {0.00, 0.01, 0.02, 0.05}
Rotation:     random SO(3)

RISA's PCA eigenvalue features are noise-robust by construction
(eigenvalues of the covariance matrix are stable under small perturbations).
This experiment quantifies that advantage.

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp5_noise_robustness.py [--dataset modelnet40]
"""

import argparse
import json
import sys, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import (
    encode_dataset, recall_at_k, mean_ap_at_k,
    random_rotation, quick_train,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024

NOISE_LEVELS = [0.00, 0.01, 0.02, 0.05]
N_TRIALS     = 5   # random rotation trials per noise level


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


def _build_encoder(name, cls):
    if name == "RISA":
        from models import RotationInvariantSparseAttention
        return RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    return cls()


@torch.no_grad()
def encode_rotated_noisy(encoder, dataset, noise_std: float, batch_size=32):
    """Encode all shapes after random SO(3) rotation + Gaussian noise."""
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

        R   = random_rotation(DEVICE)
        pts = R @ pts

        if noise_std > 0:
            pts = pts + torch.randn_like(pts) * noise_std

        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu())
    return torch.cat(all_enc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="modelnet40")
    parser.add_argument("--models",       nargs="+", default=None,
                        help="Subset of models to run, e.g. --models RISA")
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--noise_levels", nargs="+", type=float, default=NOISE_LEVELS)
    parser.add_argument("--n_trials",     type=int, default=N_TRIALS)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--out",          default="outputs/exp5_noise_robustness.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results  = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls).to(DEVICE)
        quick_train(encoder, args.dataset, epochs=args.epochs, seed=args.seed)

        # Canonical database (no rotation, no noise)
        db_enc, db_lbl = encode_dataset(encoder, test_ds)

        noise_results = []
        for sigma in args.noise_levels:
            r1_trials, map5_trials = [], []
            for t in range(args.n_trials):
                torch.manual_seed(args.seed + t)
                q_enc = encode_rotated_noisy(encoder, test_ds, noise_std=sigma)
                r1    = recall_at_k(q_enc, db_enc, db_lbl, db_lbl, k=1)
                map5  = mean_ap_at_k(q_enc, db_enc, db_lbl, db_lbl, k=5)
                r1_trials.append(r1)
                map5_trials.append(map5)

            r1_mean   = float(np.mean(r1_trials))
            r1_std    = float(np.std(r1_trials))
            map5_mean = float(np.mean(map5_trials))
            noise_results.append({
                "sigma":     sigma,
                "r1_mean":   r1_mean,
                "r1_std":    r1_std,
                "map5_mean": map5_mean,
            })
            print(f"  σ={sigma:.2f}  R@1={r1_mean:.3f}±{r1_std:.3f}  "
                  f"mAP@5={map5_mean:.3f}", flush=True)

        results[model_name] = noise_results
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"NOISE ROBUSTNESS — {args.dataset} — Recall@1 (rotation + noise)")
    header = f"{'Model':<22}" + "".join(f"  σ={s:.2f}" for s in args.noise_levels)
    print(header)
    print("-" * len(header))
    for model_name, noise_res in results.items():
        row = {r["sigma"]: r["r1_mean"] for r in noise_res}
        vals = "".join(f"  {row.get(s, float('nan')):>6.3f}" for s in args.noise_levels)
        print(f"{model_name:<22}{vals}")


if __name__ == "__main__":
    main()
