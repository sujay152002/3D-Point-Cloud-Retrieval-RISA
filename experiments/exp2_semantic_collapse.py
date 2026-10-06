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
_fd = sys.stdout.fileno()
if _fd >= 0:
    sys.stdout = io.TextIOWrapper(open(_fd, "wb", 0), write_through=True)
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

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


def _build_encoder(name, checkpoint_dir=None):
    from models import (
        DGCNNEncoder, PointMambaEncoder, PointNet2Encoder,
        RotationInvariantSparseAttention, RSCNNEncoder,
        VNNEncoder, DiPVNetEncoder, RINet,
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
        "RINet"           : RINet,
        "RISA"            : RotationInvariantSparseAttention,
    }
    if name == "RISA":
        enc = RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    else:
        enc = registry[name]()
    if checkpoint_dir:
        import glob
        pattern = str(Path(checkpoint_dir) / f"*_{enc.__class__.__name__}.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            enc.load_state_dict(torch.load(matches[-1], map_location="cpu"))
            print(f"  Loaded checkpoint: {matches[-1]}")
        else:
            print(f"  WARNING: no checkpoint found for {name} in {checkpoint_dir}")
    return enc


def random_rotation(device):
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _train(encoder, dataset_name, epochs, seed, checkpoint_path=None, force_retrain=False):
    from experiments.exp1_retrieval_eval import train_ir
    train_ir(encoder, dataset_name, epochs=epochs, seed=seed,
             checkpoint_path=checkpoint_path, force_retrain=force_retrain)


@torch.inference_mode()
def collect_encodings(encoder, dataset, n_classes=8, n_shapes_per_class=5, n_rotations=32, seed=0):
    """
    For each of n_classes, pick n_shapes_per_class shapes and encode them
    under n_rotations random SO(3) rotations.

    Returns:
        encodings : (n_classes * n_shapes_per_class * n_rotations, D)
        labels    : (n_classes * n_shapes_per_class * n_rotations,)  — class index
        shape_ids : same shape — which shape within the class
    """
    encoder.eval()
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

    rng = np.random.default_rng(seed)
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


def cluster_metrics(enc, lbl, n_classes):
    """K-means cluster assignments → NMI and AMI against true labels."""
    km = KMeans(n_clusters=n_classes, random_state=0, n_init=10)
    pred = km.fit_predict(enc)
    nmi = normalized_mutual_info_score(lbl, pred, average_method="arithmetic")
    ami = adjusted_mutual_info_score(lbl, pred)
    return nmi, ami


def plot_tsne_all(results_dict, dataset_name, out_path, n_classes):
    models = list(results_dict.keys())
    n_cols = min(3, len(models))
    n_rows = (len(models) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle(f"Semantic Collapse under SO(3) — {dataset_name}", fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for ax, model_name in zip(axes, models):
        enc, lbl, nmi, ami = results_dict[model_name]
        tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed, max_iter=1000)
        proj = tsne.fit_transform(enc)
        for c in range(n_classes):
            mask = lbl == c
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=CLASS_COLORS[c % len(CLASS_COLORS)],
                       s=8, alpha=0.6, label=f"class {c}")
        ax.set_title(f"{model_name}\nNMI={nmi:.3f}  AMI={ami:.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2 if n_classes > 5 else 1)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["PointNet++", "DGCNN", "DiPVNet", "RINet", "RISA"],
                        help="Models to visualize")
    parser.add_argument("--dataset",       default="shapenet")
    parser.add_argument("--n_classes",     type=int, default=8)
    parser.add_argument("--n_shapes",      type=int, default=5,
                        help="Shapes per class")
    parser.add_argument("--n_rotations",   type=int, default=32,
                        help="SO(3) rotations per shape")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--force_retrain",  action="store_true",
                        help="Retrain even if a checkpoint already exists, overwriting it")
    parser.add_argument("--epochs",        type=int, default=20)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test", root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    results = {}
    for model_name in args.models:
        print(f"\n{'='*50}\nModel: {model_name}")
        encoder = _build_encoder(model_name).to(DEVICE)
        ckpt_path = (
            str(Path(args.checkpoint_dir) / f"{model_name}_{args.dataset}.pt")
            if args.checkpoint_dir else None
        )
        _train(encoder, args.dataset, epochs=args.epochs, seed=args.seed,
               checkpoint_path=ckpt_path, force_retrain=args.force_retrain)

        print(f"  Collecting encodings ({args.n_classes} classes × "
              f"{args.n_shapes} shapes × {args.n_rotations} rotations)…")
        enc, lbl, _ = collect_encodings(
            encoder, test_ds,
            n_classes=args.n_classes,
            n_shapes_per_class=args.n_shapes,
            n_rotations=args.n_rotations,
            seed=args.seed,
        )
        nmi, ami = cluster_metrics(enc, lbl, args.n_classes)
        print(f"  NMI={nmi:.3f}  AMI={ami:.3f}")
        results[model_name] = (enc, lbl, nmi, ami)
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / "plots" / f"exp2_semantic_collapse_{args.dataset}.png"
    print("\nRunning t-SNE and plotting…")
    plot_tsne_all(results, dataset_name=args.dataset, out_path=out_path, n_classes=args.n_classes)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {}
    for m in results:
        enc_arr, lbl_arr, nmi, ami = results[m]
        enc_path = out_dir / f"exp2_{args.dataset}_{m}_enc.npy"
        lbl_path = out_dir / f"exp2_{args.dataset}_{m}_lbl.npy"
        np.save(enc_path, enc_arr)
        np.save(lbl_path, lbl_arr)
        save[m] = {"encodings_path": str(enc_path), "labels_path": str(lbl_path),
                   "nmi": nmi, "ami": ami}
    out_json = out_dir / f"exp2_semantic_collapse_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(save, f, indent=2)
    print(f"Results saved → {out_json}")


if __name__ == "__main__":
    main()
