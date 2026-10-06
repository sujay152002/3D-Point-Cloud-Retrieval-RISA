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
_fd = sys.stdout.fileno()
if _fd >= 0:
    sys.stdout = io.TextIOWrapper(open(_fd, "wb", 0), write_through=True)
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
    train_ir,
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
    """Encode all shapes after independent per-sample SO(3) rotation + Gaussian noise."""
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

        B = pts.shape[0]
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
        Q[:, :, 0] *= torch.linalg.det(Q).sign().view(B, 1)
        pts = Q @ pts

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
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save/load checkpoints. Saves as <dir>/<model>_<dataset>.pt")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if a checkpoint already exists")
    parser.add_argument("--out",          default="outputs/exp5_noise_robustness.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)
    val_ds  = get_dataset(args.dataset, split="val",  root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results  = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls).to(DEVICE)
        ckpt_path = (
            str(Path(args.checkpoint_dir) / f"{model_name}_{args.dataset}.pt")
            if args.checkpoint_dir else None
        )
        train_ir(encoder, args.dataset, epochs=args.epochs, seed=args.seed,
                 checkpoint_path=ckpt_path, force_retrain=args.force_retrain)

        # gallery = test set (canonical), query = val set (disjoint)
        db_enc, db_lbl = encode_dataset(encoder, test_ds)
        q0_enc, q_lbl  = encode_dataset(encoder, val_ds)

        noise_results = []
        for sigma in args.noise_levels:
            r1_trials, r5_trials, r10_trials, map5_trials, map10_trials = [], [], [], [], []
            for t in range(args.n_trials):
                torch.manual_seed(args.seed + t)
                q_enc = encode_rotated_noisy(encoder, val_ds, noise_std=sigma)
                q_n   = F.normalize(q_enc,  dim=1)
                db_n  = F.normalize(db_enc, dim=1)
                sim   = q_n @ db_n.T
                r1_trials.append(recall_at_k(sim, q_lbl, db_lbl, k=1))
                r5_trials.append(recall_at_k(sim, q_lbl, db_lbl, k=5))
                r10_trials.append(recall_at_k(sim, q_lbl, db_lbl, k=10))
                map5_trials.append(mean_ap_at_k(sim, q_lbl, db_lbl, k=5))
                map10_trials.append(mean_ap_at_k(sim, q_lbl, db_lbl, k=10))

            noise_results.append({
                "sigma":      sigma,
                "r1_mean":    float(np.mean(r1_trials)),
                "r1_std":     float(np.std(r1_trials)),
                "r5_mean":    float(np.mean(r5_trials)),
                "r10_mean":   float(np.mean(r10_trials)),
                "map5_mean":  float(np.mean(map5_trials)),
                "map10_mean": float(np.mean(map10_trials)),
            })
            print(f"  σ={sigma:.2f}  R@1={noise_results[-1]['r1_mean']:.3f}±{noise_results[-1]['r1_std']:.3f}"
                  f"  R@10={noise_results[-1]['r10_mean']:.3f}"
                  f"  mAP@10={noise_results[-1]['map10_mean']:.3f}", flush=True)

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
