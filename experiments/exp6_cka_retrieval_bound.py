"""Experiment 6 — Empirical Validation of the μCKA → Recall@1 Bound

The paper's theoretical claim:
    Any encoder with μCKA < 1 under SO(3) has a provable upper bound
    on worst-case Recall@1.

This experiment empirically validates that relationship by:
  1. Computing μCKA for each model (from existing invariance eval results)
  2. Computing worst-case Recall@1 (minimum over all rotation angles)
  3. Plotting μCKA vs worst-case Recall@1 — should show a tight correlation

If the correlation is strong (R² > 0.9), it empirically supports the
theoretical claim that μCKA is a sufficient predictor of retrieval robustness.

This is the experiment that connects your existing invariance results
(Table 4 in the original paper) to the new retrieval benchmark.

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp6_cka_retrieval_bound.py [--dataset modelnet40]
"""

import argparse
import json
import sys, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.metrics import compute_cka
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import (
    encode_dataset, encode_rotated, recall_at_k,
    rotation_by_angle, quick_train,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
N_CKA_PAIRS = 16
ANGLES      = [0, 30, 60, 90, 120, 150, 180]


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


def random_rotation(device):
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@torch.no_grad()
def compute_mu_cka_for_encoder(encoder, samples, n_pairs=16):
    """Mean CKA between original and rotated encodings."""
    pts = samples.to(DEVICE)
    z0, _ = unpack_encoder_output(encoder(pts))
    cka_vals = []
    for _ in range(n_pairs):
        R       = random_rotation(DEVICE)
        pts_rot = R @ pts
        z1, _   = unpack_encoder_output(encoder(pts_rot))
        cka_vals.append(compute_cka(z0, z1))
    return float(np.mean(cka_vals))


def load_samples(dataset_name, n=16, seed=0):
    kwargs = {"variant": "OBJ_ONLY"} if dataset_name == "scanobjectnn" else {}
    ds  = get_dataset(dataset_name, split="test", root=DATA_ROOT,
                      num_points=N_POINTS, **kwargs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    pts_list = []
    for i in idx:
        item = ds[int(i)]
        pts  = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(item, (tuple, list)) and len(item) == 3:
            pts = item[0]
        if not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.float()
        if pts.shape[0] != 3:
            pts = pts.T
        pts_list.append(pts)
    return torch.stack(pts_list)


def plot_bound(model_names, mu_ckas, worst_r1s, dataset_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (name, cka, r1) in enumerate(zip(model_names, mu_ckas, worst_r1s)):
        ax.scatter(cka, r1, s=80, zorder=3)
        ax.annotate(name, (cka, r1), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    # Fit line
    if len(mu_ckas) > 2:
        z    = np.polyfit(mu_ckas, worst_r1s, 1)
        xfit = np.linspace(min(mu_ckas) - 0.05, 1.0, 100)
        ax.plot(xfit, np.polyval(z, xfit), "k--", alpha=0.4, linewidth=1)
        r, p = pearsonr(mu_ckas, worst_r1s)
        rho, _ = spearmanr(mu_ckas, worst_r1s)
        ax.set_title(
            f"μCKA vs Worst-case Recall@1 — {dataset_name}\n"
            f"Pearson r={r:.3f}  Spearman ρ={rho:.3f}  (p={p:.3f})",
            fontsize=10,
        )

    ax.set_xlabel("μCKA (rotation invariance)", fontsize=11)
    ax.set_ylabel("Worst-case Recall@1 (min over angles)", fontsize=11)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="modelnet40")
    parser.add_argument("--models",     nargs="+", default=None,
                        help="Subset of models to run, e.g. --models RISA")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--n_cka",      type=int, default=N_CKA_PAIRS)
    parser.add_argument("--n_samples",  type=int, default=16,
                        help="Shapes used for μCKA computation")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        default="outputs/exp6_cka_bound.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)
    samples = load_samples(args.dataset, n=args.n_samples, seed=args.seed)

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results  = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls).to(DEVICE)
        quick_train(encoder, args.dataset, epochs=args.epochs, seed=args.seed)

        # μCKA
        mu_cka = compute_mu_cka_for_encoder(encoder, samples, n_pairs=args.n_cka)
        print(f"  μCKA = {mu_cka:.4f}")

        # Worst-case Recall@1 across angles
        db_enc, db_lbl = encode_dataset(encoder, test_ds)
        r1_per_angle   = []
        for theta in ANGLES:
            if theta == 0:
                q_enc = db_enc
            else:
                R     = rotation_by_angle(theta, axis="y", device=DEVICE)
                q_enc = encode_rotated(encoder, test_ds, R)
            r1 = recall_at_k(q_enc, db_enc, db_lbl, db_lbl, k=1)
            r1_per_angle.append(r1)
            print(f"  θ={theta:5.1f}°  R@1={r1:.3f}", flush=True)

        worst_r1 = float(min(r1_per_angle))
        print(f"  Worst-case R@1 = {worst_r1:.4f}")

        results[model_name] = {
            "mu_cka":       mu_cka,
            "worst_r1":     worst_r1,
            "r1_per_angle": {str(a): v for a, v in zip(ANGLES, r1_per_angle)},
        }
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Plot
    names    = list(results.keys())
    mu_ckas  = [results[n]["mu_cka"]   for n in names]
    worst_r1 = [results[n]["worst_r1"] for n in names]
    plot_bound(
        names, mu_ckas, worst_r1,
        dataset_name=args.dataset,
        out_path=ROOT / "plots" / f"exp6_cka_bound_{args.dataset}.png",
    )

    # Correlation stats
    if len(names) > 2:
        r, p   = pearsonr(mu_ckas, worst_r1)
        rho, _ = spearmanr(mu_ckas, worst_r1)
        print(f"\nCorrelation: Pearson r={r:.3f}  Spearman ρ={rho:.3f}  p={p:.4f}")
        results["_correlation"] = {"pearson_r": r, "spearman_rho": rho, "p_value": p}
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
