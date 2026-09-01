"""Experiment 2 — Semantic Collapse Visualization

For each model, collect encodings of N classes x R rotations.
Plot t-SNE showing how rotation scatters encodings across class boundaries.

Key figure: 2-column grid (RISA vs best baseline) showing tight clusters
per class under rotation (RISA) vs scattered/overlapping clusters (baseline).

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp2_semantic_collapse.py [--model_a RISA --model_b PointNet++]
"""

import argparse
import json
import sys, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024

# Visually distinct colors for up to 10 classes
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _build_encoder(name):
    from models import (
        DGCNNEncoder, PointMambaEncoder, PointNet2Encoder,
        RotationInvariantSparseAttention, RSCNNEncoder,
        VNNEncoder, DiPVNetEncoder,
    )
    from models.transformer import PointTransformerEncoder
    registry = {
        "PointNet++"      : PointNet2Encoder,
        "PointMamba"      : PointMambaEncoder,
        "RS-CNN"          : RSCNNEncoder,
        "PointTransformer": PointTransformerEncoder,
        "DGCNN"           : DGCNNEncoder,
        "VNN"             : VNNEncoder,
        "DiPVNet"         : DiPVNetEncoder,
        "RISA"            : RotationInvariantSparseAttention,
    }
    cls = registry[name]
    if name == "RISA":
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


def quick_train(encoder, dataset_name, epochs=20, seed=42):
    import torch.nn as nn, torch.optim as optim

    torch.manual_seed(seed)
    kwargs = {"variant": "OBJ_ONLY"} if dataset_name == "scanobjectnn" else {}
    try:
        ds = get_dataset(dataset_name, split="train", root=DATA_ROOT, num_points=N_POINTS, **kwargs)
    except Exception as e:
        print(f"  [train] {e} — skipping training"); return

    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    encoder.eval()
    with torch.no_grad():
        z, _ = unpack_encoder_output(encoder(torch.randn(2, 3, 64, device=DEVICE)))
    enc_dim = z.shape[-1]
    _base   = getattr(ds, "dataset", ds)
    num_cls = int(getattr(_base, "num_classes", 40))
    head    = nn.Linear(enc_dim, num_cls).to(DEVICE)
    opt     = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    crit    = nn.CrossEntropyLoss()

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
            B = pts.shape[0]
            Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
            Q[:, :, 0] *= torch.linalg.det(Q).sign().view(B, 1)
            pts = Q @ pts
            opt.zero_grad()
            z, _ = unpack_encoder_output(encoder(pts))
            crit(head(z), lbl).backward()
            opt.step()
        print(f"    epoch {ep+1}/{epochs}", flush=True)
    encoder.eval()
    del head, opt


@torch.no_grad()
def collect_encodings(encoder, dataset, n_classes=8, n_shapes_per_class=5, n_rotations=32):
    """
    For each of n_classes, pick n_shapes_per_class shapes and encode them
    under n_rotations random SO(3) rotations.

    Returns:
        encodings : (n_classes * n_shapes_per_class * n_rotations, D)
        labels    : (n_classes * n_shapes_per_class * n_rotations,)  — class index
        shape_ids : same shape — which shape within the class
    """
    # Group dataset indices by class
    all_pts, all_lbl = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        pts  = item[0] if isinstance(item, (tuple, list)) else item
        lbl  = item[1] if isinstance(item, (tuple, list)) else 0
        if isinstance(item, (tuple, list)) and len(item) == 3:
            pts, lbl, _ = item
        all_lbl.append(int(lbl) if not isinstance(lbl, int) else lbl)
        all_pts.append(pts)

    unique_classes = sorted(set(all_lbl))[:n_classes]
    class_to_idx   = {c: [i for i, l in enumerate(all_lbl) if l == c] for c in unique_classes}

    rng = np.random.default_rng(0)
    encodings, labels, shape_ids = [], [], []

    for class_rank, cls in enumerate(unique_classes):
        idxs = rng.choice(class_to_idx[cls],
                          size=min(n_shapes_per_class, len(class_to_idx[cls])),
                          replace=False)
        for shape_rank, idx in enumerate(idxs):
            pts = all_pts[idx]
            if not isinstance(pts, torch.Tensor):
                pts = torch.tensor(pts, dtype=torch.float32)
            pts = pts.float()
            if pts.shape[0] != 3:
                pts = pts.T
            pts = pts.unsqueeze(0).to(DEVICE)   # (1, 3, N)

            for _ in range(n_rotations):
                R   = random_rotation(DEVICE)
                rpt = R @ pts
                enc, _ = unpack_encoder_output(encoder(rpt))
                encodings.append(enc.squeeze(0).cpu())
                labels.append(class_rank)
                shape_ids.append(shape_rank)

    return (torch.stack(encodings).numpy(),
            np.array(labels),
            np.array(shape_ids))


def plot_tsne(encodings_a, labels_a, name_a,
              encodings_b, labels_b, name_b,
              dataset_name, out_path, n_classes):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Semantic Collapse under SO(3) — {dataset_name}", fontsize=14, fontweight="bold")

    for ax, enc, lbl, title in [
        (axes[0], encodings_a, labels_a, name_a),
        (axes[1], encodings_b, labels_b, name_b),
    ]:
        tsne  = TSNE(n_components=2, perplexity=30, random_state=0, max_iter=1000)
        proj  = tsne.fit_transform(enc)

        for c in range(n_classes):
            mask = lbl == c
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=CLASS_COLORS[c % len(CLASS_COLORS)],
                       s=8, alpha=0.6, label=f"class {c}")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=7, loc="best",
                  ncol=2 if n_classes > 5 else 1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a",    default="RISA",
                        help="Invariant model (left panel)")
    parser.add_argument("--model_b",    default="PointNet++",
                        help="Baseline model (right panel)")
    parser.add_argument("--dataset",    default="modelnet40")
    parser.add_argument("--n_classes",  type=int, default=8)
    parser.add_argument("--n_shapes",   type=int, default=5,
                        help="Shapes per class")
    parser.add_argument("--n_rotations",type=int, default=32,
                        help="SO(3) rotations per shape")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    kwargs   = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds  = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                           num_points=N_POINTS, **kwargs)

    results = {}
    for model_name in [args.model_a, args.model_b]:
        print(f"\n{'='*50}\nModel: {model_name}")
        encoder = _build_encoder(model_name).to(DEVICE)
        quick_train(encoder, args.dataset, epochs=args.epochs, seed=args.seed)

        print(f"  Collecting encodings ({args.n_classes} classes × "
              f"{args.n_shapes} shapes × {args.n_rotations} rotations)…")
        enc, lbl, _ = collect_encodings(
            encoder, test_ds,
            n_classes=args.n_classes,
            n_shapes_per_class=args.n_shapes,
            n_rotations=args.n_rotations,
        )
        results[model_name] = (enc, lbl)
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / "plots" / f"exp2_semantic_collapse_{args.dataset}.png"
    print("\nRunning t-SNE and plotting…")
    plot_tsne(
        results[args.model_a][0], results[args.model_a][1], args.model_a,
        results[args.model_b][0], results[args.model_b][1], args.model_b,
        dataset_name=args.dataset,
        out_path=out_path,
        n_classes=args.n_classes,
    )

    # Save raw encodings for replotting
    save = {
        args.model_a: {"encodings": results[args.model_a][0].tolist(),
                       "labels":    results[args.model_a][1].tolist()},
        args.model_b: {"encodings": results[args.model_b][0].tolist(),
                       "labels":    results[args.model_b][1].tolist()},
    }
    out_json = ROOT / "outputs" / f"exp2_semantic_collapse_{args.dataset}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(save, f)
    print(f"Raw encodings saved → {out_json}")


if __name__ == "__main__":
    main()
