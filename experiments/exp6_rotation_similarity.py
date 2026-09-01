"""Experiment 6 — Per-Object Rotation Similarity Curves

For each model, take 50 objects per class, rotate each from 0° to 180°,
measure cosine similarity between the rotated embedding and the 0° embedding,
and average across all objects.

This directly answers: "does this model think a rotated object is still
the same object?" A perfectly invariant model gives a flat line at 1.0.
A rotation-sensitive model shows a curve that drops as angle increases.

Output:
  - outputs/exp6_rotation_similarity.json  — raw similarity curves
  - plots/exp6_rotation_similarity.png     — line plot, one curve per model

Usage:
    python experiments/exp6_rotation_similarity.py [--dataset modelnet40]
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import quick_train, rotation_by_angle

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ANGLES    = list(range(0, 181, 10))   # 0, 10, 20, ... 180
N_OBJECTS = 50                         # objects per class to average over


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


def sample_objects(dataset, n_per_class, seed=42):
    """Sample n_per_class objects from each class, return list of (pts, label)."""
    rng = np.random.default_rng(seed)
    # Group indices by label
    by_class = {}
    for i in range(len(dataset)):
        item = dataset[i]
        lbl  = int(item[1]) if isinstance(item, (tuple, list)) else 0
        by_class.setdefault(lbl, []).append(i)

    selected = []
    for lbl, indices in sorted(by_class.items()):
        chosen = rng.choice(indices, size=min(n_per_class, len(indices)), replace=False)
        for idx in chosen:
            item = dataset[int(idx)]
            pts  = item[0] if isinstance(item, (tuple, list)) else item
            if not isinstance(pts, torch.Tensor):
                pts = torch.tensor(pts, dtype=torch.float32)
            pts = pts.float()
            if pts.shape[0] != 3:
                pts = pts.T
            selected.append((pts, lbl))
    return selected


@torch.no_grad()
def compute_similarity_curve(encoder, objects, angles):
    """
    For each object, encode at 0° and at each angle, compute cosine similarity.
    Returns mean similarity curve averaged over all objects: [n_angles]
    """
    encoder.eval()
    all_sims = []  # [n_objects, n_angles]

    for pts, _ in objects:
        pts = pts.unsqueeze(0).to(DEVICE)  # [1, 3, N]

        # Baseline embedding at 0°
        z0, _ = unpack_encoder_output(encoder(pts))
        z0 = F.normalize(z0, dim=-1)  # [1, D]

        sims = []
        for theta in angles:
            if theta == 0:
                sim = 1.0
            else:
                R      = rotation_by_angle(theta, axis="y", device=DEVICE)
                pts_r  = R @ pts
                zr, _  = unpack_encoder_output(encoder(pts_r))
                zr     = F.normalize(zr, dim=-1)
                sim    = float((z0 * zr).sum(dim=-1).mean().item())
            sims.append(sim)
        all_sims.append(sims)

    return np.mean(all_sims, axis=0).tolist()  # [n_angles]


def plot_curves(results, angles, dataset_name, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Style: invariant models get solid lines, others dashed
    invariant = {"RISA", "DiPVNet", "VNN"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (model_name, curve), color in zip(results.items(), colors):
        ls = "-" if model_name in invariant else "--"
        lw = 2.0 if model_name in invariant else 1.5
        ax.plot(angles, curve, label=model_name, color=color, linestyle=ls, linewidth=lw)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Rotation angle (°)", fontsize=12)
    ax.set_ylabel("Mean cosine similarity to 0° embedding", fontsize=12)
    ax.set_title(f"Rotation Similarity Curves — {dataset_name}\n"
                 f"({N_OBJECTS} objects/class averaged)", fontsize=11)
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
    parser.add_argument("--n_obj",    type=int, default=N_OBJECTS,
                        help="Objects per class to average over")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="outputs/exp6_rotation_similarity.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    print(f"Sampling {args.n_obj} objects/class from {args.dataset} test set...")
    objects = sample_objects(test_ds, n_per_class=args.n_obj, seed=args.seed)
    print(f"  Total objects: {len(objects)}")

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}

    results = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls).to(DEVICE)
        quick_train(encoder, args.dataset, epochs=args.epochs, seed=args.seed)

        curve = compute_similarity_curve(encoder, objects, ANGLES)
        results[model_name] = {"angles": ANGLES, "similarity": curve}

        # Print summary at key angles
        for theta, sim in zip(ANGLES, curve):
            if theta % 30 == 0:
                print(f"  θ={theta:3d}°  sim={sim:.4f}", flush=True)

        del encoder
        torch.cuda.empty_cache()

    # Save JSON
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Plot
    plot_curves(
        {m: v["similarity"] for m, v in results.items()},
        ANGLES, args.dataset,
        ROOT / "plots" / f"exp6_rotation_similarity_{args.dataset}.png"
    )

    # Summary table
    print(f"\n{'='*65}")
    print(f"ROTATION SIMILARITY — {args.dataset}")
    print(f"{'Model':<18} {'sim@0°':>8} {'sim@90°':>9} {'sim@180°':>10} {'drop':>8}")
    print("-" * 55)
    for model_name, v in results.items():
        curve = v["similarity"]
        s0   = curve[ANGLES.index(0)]
        s90  = curve[ANGLES.index(90)]
        s180 = curve[ANGLES.index(180)]
        drop = s0 - s180
        print(f"{model_name:<18} {s0:>8.4f} {s90:>9.4f} {s180:>10.4f} {drop:>8.4f}")


if __name__ == "__main__":
    main()
