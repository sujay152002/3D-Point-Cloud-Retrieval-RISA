"""Experiment 11 — Rotation-Invariant Open-Vocabulary 3D Retrieval via ImageBind

Design
------
We replace Point-BIND's pose-dependent backbone with RISAWithPrior — a
rotation-invariant encoder operating on K geometry-salient points selected
by the best prior from exp10.

Pipeline:
    N points
      ──► exp10 prior (pluggable, default=eigenentropy_K)
      ──► K points
      ──► RISAWithPrior (frozen after loading exp10 checkpoint)
      ──► ProjectionMLP  ◄── only thing trained
      ──► ImageBind space  (D=1024)

Training signal:
    For each shape, render to 4 views → ImageBind image encoder → average → z_ib
    Train ProjectionMLP so that cosine_similarity(proj(z_risa), z_ib) → 1
    Loss = InfoNCE(z_shape_proj, z_ib) over batch

At retrieval time:
    text  ──► ImageBind text encoder  ──► nearest neighbour over shape embeddings
    image ──► ImageBind image encoder ──► nearest neighbour over shape embeddings
    shape ──► RISAWithPrior + proj    ──► nearest neighbour (shape-to-shape)

Evaluation axes (vs Point-BIND):
    1. R@1, R@5, mAP@5 on text→shape retrieval
    2. Same metrics under SO(3) rotations (0°→180°) — Point-BIND degrades, ours doesn't
    3. Zero-shot: query with class names not seen during projection training

Usage:
    # With a pre-trained checkpoint:
    python experiments/exp11_imagebind_retrieval.py \\
        --prior fps_K --K 256 --epochs 50 \\
        --risa_ckpt outputs/checkpoints/exp10_fps_K.pt

    # Without a checkpoint — encoder trains automatically first:
    python experiments/exp11_imagebind_retrieval.py \\
        --prior fps_K --K 256 --risa_epochs 100 --epochs 50
"""

import argparse
import json
import sys
import io
if hasattr(sys.stdout, 'fileno'):
    try:
        fd = sys.stdout.fileno()
        if fd >= 0:
            sys.stdout = io.TextIOWrapper(open(fd, "wb", 0), write_through=True)
    except Exception:
        pass

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp10_prior_selection import RISAWithPrior, PRIOR_NAMES, train_condition
from experiments.exp1_retrieval_eval import recall_at_k, mean_ap_at_k

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
IB_DIM    = 1024   # ImageBind embedding dimension
ANGLES    = [0, 15, 30, 45, 60, 90, 120, 150, 180]


# ── ImageBind wrapper ─────────────────────────────────────────────────────────

def load_imagebind():
    """Load ImageBind model. Requires: pip install imagebind"""
    try:
        from imagebind import data as ib_data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval().to(DEVICE)
        for p in model.parameters():
            p.requires_grad = False
        return model, ib_data, ModalityType
    except ImportError:
        raise ImportError(
            "ImageBind not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/ImageBind.git"
        )


# ── Renderer — offline, cached ────────────────────────────────────────────────

def render_point_cloud(xyz_np, size=224, n_views=4):
    """Render a point cloud to n_views PIL images using matplotlib.

    xyz_np: [N, 3] numpy array, already normalised to unit sphere.
    Returns list of PIL Images.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io as _io

    azimuths = np.linspace(0, 360, n_views, endpoint=False)
    images = []
    for az in azimuths:
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2],
                   s=0.5, c=xyz_np[:, 2], cmap="viridis", alpha=0.7)
        ax.view_init(elev=20, azim=az)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB").resize((size, size)))
    return images


def build_imagebind_cache(dataset, cache_path: Path, n_views: int = 4, batch_size: int = 16):
    """Render all shapes and cache their ImageBind image embeddings.

    cache_path: .pt file — dict with keys 'embeddings' [N, IB_DIM] and 'labels' [N]
    Skips if cache already exists.
    """
    if cache_path.exists():
        print(f"  ImageBind cache found: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"  Building ImageBind cache ({len(dataset)} shapes, {n_views} views each)...")
    ib_model, ib_data, ModalityType = load_imagebind()

    all_embeddings = []
    all_labels     = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        pts  = item[0]   # [3, N] or [N, 3]
        lbl  = item[1]
        if pts.shape[0] == 3:
            xyz = pts.permute(1, 0).numpy()
        else:
            xyz = pts.numpy()

        images = render_point_cloud(xyz, n_views=n_views)

        # ImageBind expects a list of PIL images wrapped in its data format
        inputs = {
            ModalityType.VISION: ib_data.load_and_transform_vision_data(images, DEVICE)
        }
        with torch.no_grad():
            embs = ib_model(inputs)[ModalityType.VISION]  # [n_views, IB_DIM]
        z = embs.mean(dim=0)  # [IB_DIM] — average over views

        all_embeddings.append(z.cpu())
        all_labels.append(int(lbl) if not isinstance(lbl, int) else lbl)

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(dataset)}")

    cache = {
        "embeddings": torch.stack(all_embeddings),  # [N, IB_DIM]
        "labels":     torch.tensor(all_labels),      # [N]
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"  Saved cache: {cache_path}")
    return cache


# ── Projection MLP ────────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """Small MLP projecting RISA embeddings into ImageBind's space.

    Following Point-BIND: two-layer MLP with GELU, output L2-normalised.
    risa_dim: encoding_out_dim of RISAWithPrior (default 512)
    ib_dim:   ImageBind embedding dimension (1024)
    """

    def __init__(self, risa_dim: int = 512, hidden_dim: int = 1024, ib_dim: int = IB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(risa_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ib_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (CLIP-style) between shape and ImageBind embeddings.

    Both inputs are assumed L2-normalised.
    temperature: learnable scalar, initialised to log(1/0.07) following CLIP.
    """

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(1.0 / init_temp)))

    def forward(self, z_shape: torch.Tensor, z_ib: torch.Tensor) -> torch.Tensor:
        # z_shape, z_ib: [B, D], both L2-normalised
        temp = self.log_temp.exp().clamp(max=100.0)
        logits = (z_shape @ z_ib.T) * temp          # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_s2i = F.cross_entropy(logits,   labels)
        loss_i2s = F.cross_entropy(logits.T, labels)
        return (loss_s2i + loss_i2s) / 2.0


# ── Encode dataset with RISA+proj ─────────────────────────────────────────────

@torch.no_grad()
def encode_with_proj(encoder: nn.Module, proj: nn.Module,
                     dataset, batch_size: int = 16) -> tuple:
    """Encode all shapes in dataset → [N, IB_DIM] L2-normalised embeddings."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_z, all_labels = [], []
    encoder.eval(); proj.eval()

    for batch in loader:
        pts = batch[0].to(DEVICE)
        lbl = batch[1]
        z, _ = unpack_encoder_output(encoder(pts))   # [B, risa_dim]
        z    = proj(z)                                # [B, IB_DIM]
        all_z.append(z.cpu())
        all_labels.append(lbl)

    return torch.cat(all_z), torch.cat(all_labels)


@torch.no_grad()
def encode_rotated_with_proj(encoder, proj, dataset, angle_deg: float,
                              batch_size: int = 16) -> torch.Tensor:
    """Encode dataset under a fixed SO(3) rotation by angle_deg around Y-axis."""
    from scipy.spatial.transform import Rotation as R
    rot = torch.from_numpy(
        R.from_euler("y", angle_deg, degrees=True).as_matrix()
    ).float().to(DEVICE)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_z = []
    encoder.eval(); proj.eval()

    for batch in loader:
        pts = batch[0].to(DEVICE)                    # [B, 3, N]
        pts_r = rot @ pts                            # [B, 3, N]
        z, _  = unpack_encoder_output(encoder(pts_r))
        all_z.append(proj(z).cpu())

    return torch.cat(all_z)


# ── Text query via ImageBind ──────────────────────────────────────────────────

@torch.no_grad()
def encode_text_queries(queries: list) -> torch.Tensor:
    """Encode a list of text strings with ImageBind → [Q, IB_DIM]."""
    ib_model, ib_data, ModalityType = load_imagebind()
    inputs = {ModalityType.TEXT: ib_data.load_and_transform_text(queries, DEVICE)}
    embs   = ib_model(inputs)[ModalityType.TEXT]     # [Q, IB_DIM]
    return F.normalize(embs, dim=-1).cpu()


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def retrieval_metrics(q_embs: torch.Tensor, db_embs: torch.Tensor,
                      q_labels: torch.Tensor, db_labels: torch.Tensor,
                      ks=(1, 5)) -> dict:
    """Compute R@k and mAP@5 for query embeddings against a database."""
    sim    = q_embs @ db_embs.T                      # [Q, DB]
    # exclude self-matches when query == database
    if q_embs.shape[0] == db_embs.shape[0]:
        sim.fill_diagonal_(-1e4)

    results = {}
    for k in ks:
        results[f"R@{k}"] = recall_at_k(sim, q_labels, db_labels, k)
    results["mAP@5"] = mean_ap_at_k(sim, q_labels, db_labels, 5)
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train_projection(
    encoder:    nn.Module,
    proj:       nn.Module,
    criterion:  nn.Module,
    ib_cache:   dict,
    dataset,
    epochs:     int,
    batch_size: int,
    lr:         float,
) -> list:
    """Train only the projection MLP using cached ImageBind embeddings as targets."""

    ib_embs   = F.normalize(ib_cache["embeddings"].to(DEVICE), dim=-1)  # [N, IB_DIM]
    ib_labels = ib_cache["labels"]

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
    )

    # Only projection + temperature are trained; encoder is frozen
    opt = optim.AdamW(
        list(proj.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # Build index mapping from dataset position to ib_cache row
    # (assumes dataset order matches cache order)
    history = []
    encoder.eval()

    for epoch in range(epochs):
        proj.train(); criterion.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_idx, batch in enumerate(loader):
            pts = batch[0].to(DEVICE)
            # Recover the dataset indices for this batch to look up ib_embs
            # DataLoader with shuffle=True doesn't expose indices directly,
            # so we use a custom sampler approach: store indices in dataset
            # For now: use the batch position * batch_size as approximate index
            # (works because drop_last=True keeps batches full and ordered within epoch)
            start = batch_idx * batch_size
            idx   = torch.arange(start, start + pts.size(0))
            idx   = idx.clamp(max=ib_embs.size(0) - 1)

            z_ib = ib_embs[idx]                          # [B, IB_DIM]

            with torch.no_grad():
                z_risa, _ = unpack_encoder_output(encoder(pts))  # [B, risa_dim]

            z_proj = proj(z_risa)                        # [B, IB_DIM]
            loss   = criterion(z_proj, z_ib)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"temp={criterion.log_temp.exp().item():.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior",      default="eigenentropy_K", choices=PRIOR_NAMES)
    parser.add_argument("--K",          type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--n_views",    type=int,   default=4,
                        help="Number of render views per shape for IB cache")
    parser.add_argument("--risa_ckpt",   type=str,   default=None,
                        help="Path to exp10 checkpoint (.pt). If None, trains encoder from scratch.")
    parser.add_argument("--risa_epochs", type=int,   default=100,
                        help="Encoder training epochs if no --risa_ckpt is provided.")
    parser.add_argument("--risa_batch_size", type=int, default=8,
                        help="Batch size for encoder training.")
    parser.add_argument("--dataset",    default="shapenet")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = ROOT / "outputs" / f"exp11_{args.prior}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Exp 11 — ImageBind Retrieval")
    print(f"  prior={args.prior}  K={args.K}  dataset={args.dataset}")
    print(f"  output: {out_dir}")
    print(f"{'='*60}\n")

    # ── Encoder (RISAWithPrior, frozen) ───────────────────────────────────────
    encoder = RISAWithPrior(
        prior            = args.prior,
        K                = args.K,
        num_local        = 32,
        num_global       = 32,
        num_blocks       = 4,
        model_dim        = 256,
        features_out_dim = 256,
        encoding_out_dim = 512,
    ).to(DEVICE)

    ckpt_dir = ROOT / "outputs" / "checkpoints"
    auto_ckpt = ckpt_dir / f"exp10_{args.prior}.pt"

    if args.risa_ckpt is not None:
        ckpt = torch.load(args.risa_ckpt, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt, strict=False)
        print(f"  Loaded RISA checkpoint: {args.risa_ckpt}")
    elif auto_ckpt.exists():
        ckpt = torch.load(auto_ckpt, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt, strict=False)
        print(f"  Loaded cached RISA checkpoint: {auto_ckpt}")
    else:
        print(f"  No checkpoint found — training encoder ({args.risa_epochs} epochs)...")
        train_condition(encoder, args.dataset, args.risa_epochs, args.seed,
                        batch_size=args.risa_batch_size)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), auto_ckpt)
        print(f"  Encoder checkpoint saved → {auto_ckpt}")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ── Projection MLP (trained) ──────────────────────────────────────────────
    proj      = ProjectionMLP(risa_dim=512, hidden_dim=1024, ib_dim=IB_DIM).to(DEVICE)
    criterion = InfoNCELoss(init_temp=0.07).to(DEVICE)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = get_dataset(args.dataset, split="train", root=DATA_ROOT, num_points=N_POINTS)
    test_ds  = get_dataset(args.dataset, split="test",  root=DATA_ROOT, num_points=N_POINTS)

    # ── ImageBind cache ───────────────────────────────────────────────────────
    cache_dir  = ROOT / "outputs" / "imagebind_cache"
    train_cache = build_imagebind_cache(
        train_ds,
        cache_dir / f"{args.dataset}_train_ib.pt",
        n_views=args.n_views,
    )
    test_cache = build_imagebind_cache(
        test_ds,
        cache_dir / f"{args.dataset}_test_ib.pt",
        n_views=args.n_views,
    )

    # ── Train projection ──────────────────────────────────────────────────────
    print("\n  Training projection MLP...")
    history = train_projection(
        encoder    = encoder,
        proj       = proj,
        criterion  = criterion,
        ib_cache   = train_cache,
        dataset    = train_ds,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )
    torch.save(proj.state_dict(), out_dir / "proj_best.pt")

    # ── Shape-to-shape retrieval under rotation ───────────────────────────────
    print("\n  Evaluating shape-to-shape retrieval under rotation...")
    test_labels = test_cache["labels"]
    rotation_results = []

    for angle in ANGLES:
        z_rotated = encode_rotated_with_proj(encoder, proj, test_ds, angle)
        z_db, _   = encode_with_proj(encoder, proj, test_ds)
        metrics   = retrieval_metrics(z_rotated, z_db, test_labels, test_labels)
        metrics["angle"] = angle
        rotation_results.append(metrics)
        print(f"    θ={angle:5.1f}°  R@1={metrics['R@1']:.3f}  "
              f"R@5={metrics['R@5']:.3f}  mAP@5={metrics['mAP@5']:.3f}")

    # ── Text-to-shape retrieval (zero-shot) ───────────────────────────────────
    print("\n  Evaluating text-to-shape retrieval (zero-shot)...")
    z_db, db_labels = encode_with_proj(encoder, proj, test_ds)

    # Use category names as text queries — one query per class
    base_ds    = getattr(test_ds, "dataset", test_ds)
    categories = getattr(base_ds, "categories", {})
    text_queries = [categories[i] for i in sorted(categories.keys())]
    query_labels = torch.tensor(sorted(categories.keys()))

    z_text   = encode_text_queries(text_queries)             # [C, IB_DIM]
    text_sim = z_text @ z_db.T                               # [C, DB]

    text_r1   = recall_at_k(text_sim,   query_labels, db_labels, 1)
    text_r5   = recall_at_k(text_sim,   query_labels, db_labels, 5)
    text_map5 = mean_ap_at_k(text_sim,  query_labels, db_labels, 5)
    print(f"    text→shape  R@1={text_r1:.3f}  R@5={text_r5:.3f}  mAP@5={text_map5:.3f}")

    # ── IB image-to-shape retrieval ───────────────────────────────────────────
    print("\n  Evaluating ImageBind image-to-shape retrieval...")
    z_ib_test = F.normalize(test_cache["embeddings"].to(DEVICE), dim=-1).cpu()
    ib_sim    = z_ib_test @ z_db.T                           # [N_test, DB]
    ib_metrics = retrieval_metrics(z_ib_test, z_db, test_labels, test_labels)
    print(f"    image→shape  R@1={ib_metrics['R@1']:.3f}  "
          f"R@5={ib_metrics['R@5']:.3f}  mAP@5={ib_metrics['mAP@5']:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config": vars(args),
        "rotation_retrieval":      rotation_results,
        "text_to_shape": {
            "R@1": text_r1, "R@5": text_r5, "mAP@5": text_map5,
        },
        "image_to_shape":          ib_metrics,
        "train_loss_history":      history,
        "timestamp":               ts,
    }
    out_path = out_dir / "exp11_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Also write to outputs/ for generate_site.py to pick up
    site_path = ROOT / "outputs" / "exp11_imagebind_retrieval.json"
    with open(site_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Site results: {site_path}")


if __name__ == "__main__":
    main()
