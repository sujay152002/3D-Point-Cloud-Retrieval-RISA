"""Experiment 6 — Per-Angle CKA Rotation Invariance Curves

For each model, sample N_OBJECTS objects, encode them at 0° to get a baseline
embedding matrix Z0, then encode the same objects at each rotation angle to get
Zθ. Compute CKA(Z0, Zθ) at each angle.

CKA = 1.0 means the entire embedding structure is perfectly preserved under
that rotation. CKA dropping means the model is scrambling the relationships
between objects as the angle increases.

This is stronger than cosine similarity because it captures whether the
*relative structure* between objects is preserved, not just individual vectors.

Output:
  - outputs/exp6_cka_curves.json          — raw CKA curves per model
  - plots/exp6_cka_curves_{dataset}.png   — line plot, one curve per model

Usage:
    python experiments/exp6_cka_curves.py [--dataset modelnet40]
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.metrics import compute_cka
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import train_ir, rotation_by_angle

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ANGLES    = list(range(0, 181, 10))  # 0, 10, 20, ... 180
N_OBJECTS = 50                        # objects to sample (across all classes)


def _build_registry():
    from models import (
        DGCNNEncoder, PointMambaEncoder, PointNet2Encoder,
        RSCNNEncoder, VNNEncoder, DiPVNetEncoder,
    )
    from models.transformer import PointTransformerEncoder
    return {
        "PointNet++":       PointNet2Encoder,
        "DGCNN":            DGCNNEncoder,
        "VNN":              VNNEncoder,
        "DiPVNet":          DiPVNetEncoder,
        "PointTransformer": PointTransformerEncoder,
        "PointMamba":       PointMambaEncoder,
        "RS-CNN":           RSCNNEncoder,
        "RISA":             None,
    }


def _build_encoder(name, cls):
    if name == "RISA":
        from models import RotationInvariantSparseAttention
        return RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    return cls()


def sample_objects(dataset, n, seed=42):
    """Randomly sample n objects from the dataset, return stacked tensor [n, 3, N]."""
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    pts_list = []
    for i in idxs:
        item = dataset[int(i)]
        pts  = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(pts, torch.Tensor):
            pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.float()
        if pts.shape[0] != 3:
            pts = pts.T
        pts_list.append(pts)
    return torch.stack(pts_list)  # [n, 3, N_POINTS]


@torch.no_grad()
def compute_cka_curve(encoder, pts, angles):
    """
    Encode pts at 0° → Z0, then encode at each angle → Zθ.
    Return CKA(Z0, Zθ) for each angle.

    pts: [n, 3, N_POINTS] on CPU
    Returns: list of floats, one per angle
    """
    encoder.eval()
    pts = pts.to(DEVICE)

    # Baseline at 0°
    z0, _ = unpack_encoder_output(encoder(pts))
    z0 = z0.squeeze(-1) if z0.dim() == 3 else z0  # [n, D]

    cka_vals = []
    for theta in angles:
        if theta == 0:
            cka_vals.append(1.0)
            continue
        R      = rotation_by_angle(theta, axis="y", device=DEVICE)
        pts_r  = R @ pts                                        # [n, 3, N]
        zr, _  = unpack_encoder_output(encoder(pts_r))
        zr     = zr.squeeze(-1) if zr.dim() == 3 else zr       # [n, D]
        cka_vals.append(float(compute_cka(z0, zr)))

    return cka_vals


def plot_curves(results, angles, dataset_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    invariant = {"RISA", "DiPVNet", "VNN"}
    colors    = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (model_name, curve), color in zip(results.items(), colors):
        ls = "-"  if model_name in invariant else "--"
        lw = 2.2  if model_name in invariant else 1.5
        ax.plot(angles, curve, label=model_name, color=color,
                linestyle=ls, linewidth=lw, marker="o", markersize=3)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Rotation angle (°)", fontsize=12)
    ax.set_ylabel("CKA(Z₀, Zθ)", fontsize=12)
    ax.set_title(
        f"CKA vs Rotation Angle — {dataset_name}\n"
        f"(n={N_OBJECTS} objects, averaged embedding structure)",
        fontsize=11
    )
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="modelnet40")
    parser.add_argument("--models",   nargs="+", default=None)
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--n_obj",    type=int, default=N_OBJECTS)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save/load checkpoints. Saves as <dir>/<model>_<dataset>.pt")
    parser.add_argument("--force_retrain",  action="store_true")
    parser.add_argument("--out",      default="outputs/exp6_cka_curves.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    print(f"Sampling {args.n_obj} objects from {args.dataset} test set...")
    pts = sample_objects(test_ds, n=args.n_obj, seed=args.seed)
    print(f"  Sampled shape: {pts.shape}")

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}

    results = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls).to(DEVICE)
        ckpt_path = (
            str(Path(args.checkpoint_dir) / f"{model_name}_{args.dataset}.pt")
            if args.checkpoint_dir else None
        )
        train_ir(encoder, args.dataset, epochs=args.epochs, seed=args.seed,
                 checkpoint_path=ckpt_path, force_retrain=args.force_retrain)

        curve = compute_cka_curve(encoder, pts, ANGLES)
        results[model_name] = {"angles": ANGLES, "cka": curve}

        # Print at key angles
        for theta, cka in zip(ANGLES, curve):
            if theta % 30 == 0:
                print(f"  θ={theta:3d}°  CKA={cka:.4f}", flush=True)

        del encoder
        torch.cuda.empty_cache()

    # Save
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Plot
    plot_curves(
        {m: v["cka"] for m, v in results.items()},
        ANGLES, args.dataset,
        ROOT / "plots" / f"exp6_cka_curves_{args.dataset}.png"
    )

    # Summary table
    print(f"\n{'='*60}")
    print(f"CKA CURVES SUMMARY — {args.dataset}")
    print(f"{'Model':<18} {'CKA@0°':>8} {'CKA@90°':>9} {'CKA@180°':>10} {'drop':>8}")
    print("-" * 55)
    for model_name, v in results.items():
        curve = v["cka"]
        c0   = curve[ANGLES.index(0)]
        c90  = curve[ANGLES.index(90)]
        c180 = curve[ANGLES.index(180)]
        drop = c0 - c180
        print(f"{model_name:<18} {c0:>8.4f} {c90:>9.4f} {c180:>10.4f} {drop:>8.4f}")


if __name__ == "__main__":
    main()
