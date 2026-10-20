"""Experiment 11b — Projection MLP with Stronger Text Alignment

Identical to exp11 except for three changes to the projection training:

1. Multi-template text augmentation
   10 prompt templates per class are encoded with ImageBind text encoder and
   stored as [C, T, IB_DIM]. During each training batch a template is sampled
   randomly per sample, so the shape embedding must align to the *distribution*
   of text descriptions, not a single anchor.

2. Per-sample random template selection
   Each shape in a batch independently draws a random template index, giving
   T^B diversity across B samples without needing extra forward passes.

3. text_lambda=2.0 (was 0.5)
   Text alignment loss now outweighs image loss, pulling shapes close enough
   to text embeddings for free-form text queries at retrieval time.

Everything else (RISA encoder, ProjectionMLP architecture, ImageBind cache,
evaluation protocol) is unchanged from exp11.

Usage
-----
    python experiments/exp11b_imagebind_retrieval.py \\
        --prior aggregate_K --K 256 --epochs 50

    # Explicit checkpoints:
    python experiments/exp11b_imagebind_retrieval.py \\
        --risa_ckpt outputs/checkpoints/exp10_aggregate_K.pt \\
        --text_lambda 2.0 --epochs 50
"""

import argparse
import json
import random
import sys
import os

from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

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
from experiments.exp10_prior_selection import RISAWithPrior, PRIOR_NAMES, set_seed
from experiments.exp1_retrieval_eval import recall_at_k, mean_ap_at_k
from experiments.exp11_imagebind_retrieval import (
    ProjectionMLP, InfoNCELoss, IB_DIM,
    build_imagebind_cache, encode_with_proj, encode_rotated_with_proj,
    load_imagebind,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ANGLES    = [0, 15, 30, 45, 60, 90, 120, 150, 180]

# ── Text templates ────────────────────────────────────────────────────────────
# 10 diverse phrasings — covers class names, descriptions, and visual cues.
# More templates = better coverage of ImageBind's text embedding space.

TEXT_TEMPLATES = [
    "a 3D model of a {}",
    "a point cloud of a {}",
    "a {} shape",
    "a rendering of a {}",
    "a {}",
    "this is a {}",
    "a CAD model of a {}",
    "a {} object",
    "an example of a {}",
    "a three-dimensional {} shape",
]


# ── Text embedding precomputation ─────────────────────────────────────────────

@torch.no_grad()
def encode_text_queries(queries: list) -> torch.Tensor:
    """Encode a list of text strings with ImageBind → [Q, IB_DIM] normalised."""
    ib_model, ib_data, ModalityType = load_imagebind()
    inputs = {ModalityType.TEXT: ib_data.load_and_transform_text(queries, DEVICE)}
    embs   = ib_model(inputs)[ModalityType.TEXT]
    return F.normalize(embs, dim=-1).cpu()


def precompute_text_embs(categories: dict) -> torch.Tensor:
    """Precompute ImageBind text embeddings for all classes × all templates.

    Returns
    -------
    text_embs : [C, T, IB_DIM]  — L2-normalised, on CPU
        C = number of categories, T = number of templates
    """
    C = len(categories)
    T = len(TEXT_TEMPLATES)
    text_embs = torch.zeros(C, T, IB_DIM)

    print(f"  Precomputing text embeddings: {C} classes × {T} templates …")
    for t_idx, template in enumerate(TEXT_TEMPLATES):
        queries = [template.format(categories[i]) for i in sorted(categories.keys())]
        embs    = encode_text_queries(queries)   # [C, IB_DIM]
        text_embs[:, t_idx, :] = embs
        print(f"    template {t_idx+1:2d}/{T}  \"{template}\"")

    print(f"  Text embeddings shape: {text_embs.shape}")
    return text_embs   # [C, T, IB_DIM]


# ── Training ──────────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return (*item, i) if isinstance(item, tuple) else (item, i)


def train_projection(
    encoder,
    proj,
    criterion,
    ib_cache,
    dataset,
    epochs:      int,
    batch_size:  int,
    lr:          float,
    text_embs:   torch.Tensor,   # [C, T, IB_DIM] — multi-template, L2-normalised
    text_lambda: float = 2.0,
) -> list:
    """Train projection MLP with dual InfoNCE: image targets + multi-template text targets.

    image loss : InfoNCE(z_proj, z_ib_image)
    text loss  : InfoNCE(z_proj, z_text[label, random_template])
    total loss : image_loss + text_lambda * text_loss

    Per batch, each sample independently draws a random template index, giving
    diverse text targets without extra ImageBind forward passes.
    """
    ib_embs   = F.normalize(ib_cache["embeddings"].to(DEVICE), dim=-1)  # [N, IB_DIM]
    ib_labels = ib_cache["labels"].to(DEVICE)                            # [N]
    text_embs = text_embs.to(DEVICE)                                     # [C, T, IB_DIM]
    T = text_embs.shape[1]

    loader = DataLoader(
        _IndexedDataset(dataset), batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    opt = optim.AdamW(
        list(proj.parameters()) + list(criterion.parameters()) +
        [p for p in encoder.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    history = []
    encoder.eval()
    use_gumbel = (hasattr(encoder, 'gumbel_selector') and
                  encoder.gumbel_selector is not None)

    for epoch in range(epochs):
        proj.train(); criterion.train()
        if use_gumbel:
            encoder.gumbel_selector.train()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            pts = batch[0].to(DEVICE)
            idx = batch[-1]                          # dataset indices [B]
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)

            z_ib  = ib_embs[idx]                     # [B, IB_DIM]

            with torch.no_grad():
                z_risa, _ = unpack_encoder_output(encoder(pts))

            z_proj = proj(z_risa)

            # image InfoNCE
            loss = criterion(z_proj, z_ib)

            # text InfoNCE — random template per sample
            lbl        = ib_labels[idx]              # [B] class indices
            t_idx      = torch.randint(0, T, (len(lbl),), device=DEVICE)  # [B]
            z_text_tgt = text_embs[lbl, t_idx, :]   # [B, IB_DIM]
            loss = loss + text_lambda * criterion(z_proj, z_text_tgt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        if use_gumbel:
            tau = max(1.0 * (0.1 ** (epoch / max(epochs - 1, 1))), 0.1)
            encoder.gumbel_selector.set_tau(tau)

        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"temp={criterion.log_temp.exp().item():.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    return history


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def retrieval_metrics(sim, q_labels, db_labels, ks=(1, 5)):
    results = {}
    for k in ks:
        results[f"R@{k}"] = recall_at_k(sim, q_labels, db_labels, k)
    results["mAP@5"] = mean_ap_at_k(sim, q_labels, db_labels, 5)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior",       default="aggregate_K", choices=PRIOR_NAMES)
    parser.add_argument("--K",           type=int,   default=256)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--n_views",     type=int,   default=4)
    parser.add_argument("--text_lambda", type=float, default=2.0,
                        help="Weight for text InfoNCE loss (default 2.0, was 0.5 in exp11)")
    parser.add_argument("--risa_ckpt",   type=str,   default=None)
    parser.add_argument("--proj_ckpt",   type=str,   default=None,
                        help="If set, skip training and evaluate only.")
    parser.add_argument("--dataset",     default="shapenet")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"  seed={args.seed}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"exp11b_{args.prior}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Exp 11b — ImageBind Retrieval (stronger text alignment)")
    print(f"  prior={args.prior}  K={args.K}  dataset={args.dataset}")
    print(f"  text_lambda={args.text_lambda}  templates={len(TEXT_TEMPLATES)}")
    print(f"  output: {out_dir}")
    print(f"{'='*60}\n")

    # ── Encoder ───────────────────────────────────────────────────────────────
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

    ckpt_dir  = ROOT / "outputs" / "checkpoints"
    auto_ckpt = ckpt_dir / f"exp10_{args.prior}.pt"
    risa_ckpt = args.risa_ckpt or str(auto_ckpt)

    if not Path(risa_ckpt).exists():
        print(f"  Error: RISA checkpoint not found at {risa_ckpt}")
        sys.exit(1)

    state = torch.load(risa_ckpt, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(state, strict=False)
    print(f"  Loaded RISA checkpoint: {risa_ckpt}")

    for p in encoder.parameters():
        p.requires_grad = False
    if args.prior == "gumbel_K" and encoder.gumbel_selector is not None:
        for p in encoder.gumbel_selector.parameters():
            p.requires_grad = True
    encoder.eval()

    # ── Projection MLP ────────────────────────────────────────────────────────
    proj      = ProjectionMLP(risa_dim=512, hidden_dim=1024, ib_dim=IB_DIM).to(DEVICE)
    criterion = InfoNCELoss(init_temp=0.07).to(DEVICE)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = get_dataset(args.dataset, split="train", root=DATA_ROOT, num_points=N_POINTS)
    test_ds  = get_dataset(args.dataset, split="test",  root=DATA_ROOT, num_points=N_POINTS)

    # ── ImageBind cache ───────────────────────────────────────────────────────
    cache_dir   = ROOT / "outputs" / "imagebind_cache"
    train_cache = build_imagebind_cache(
        train_ds, cache_dir / f"{args.dataset}_train_ib.pt", n_views=args.n_views,
    )
    test_cache  = build_imagebind_cache(
        test_ds,  cache_dir / f"{args.dataset}_test_ib.pt",  n_views=args.n_views,
    )

    # ── Train or load projection ──────────────────────────────────────────────
    auto_proj_ckpt = ckpt_dir / f"exp11b_{args.prior}_proj.pt"

    if args.proj_ckpt and Path(args.proj_ckpt).exists():
        proj.load_state_dict(
            torch.load(args.proj_ckpt, map_location=DEVICE, weights_only=True)
        )
        print(f"  Loaded projection checkpoint: {args.proj_ckpt}")
        history = []
    else:
        # Precompute multi-template text embeddings [C, T, IB_DIM]
        base_train_ds = getattr(train_ds, "dataset", train_ds)
        train_cats    = getattr(base_train_ds, "categories", {})
        text_embs     = precompute_text_embs(train_cats)   # [C, T, IB_DIM]

        print(f"\n  Training projection MLP …")
        history = train_projection(
            encoder      = encoder,
            proj         = proj,
            criterion    = criterion,
            ib_cache     = train_cache,
            dataset      = train_ds,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            lr           = args.lr,
            text_embs    = text_embs,
            text_lambda  = args.text_lambda,
        )
        torch.save(proj.state_dict(), out_dir / "proj_best.pt")
        torch.save(proj.state_dict(), auto_proj_ckpt)
        print(f"  Projection checkpoint saved → {auto_proj_ckpt}")

    # ── Shape-to-shape retrieval under rotation ───────────────────────────────
    print("\n  Evaluating shape-to-shape retrieval under rotation…")
    val_ds          = get_dataset(args.dataset, split="val", root=DATA_ROOT, num_points=N_POINTS)
    z_db, db_labels = encode_with_proj(encoder, proj, test_ds)
    z_q0, q_labels  = encode_with_proj(encoder, proj, val_ds)
    db_n            = F.normalize(z_db, dim=-1)
    rotation_results = []

    for angle in ANGLES:
        z_q     = z_q0 if angle == 0 else encode_rotated_with_proj(encoder, proj, val_ds, angle)
        sim     = F.normalize(z_q, dim=-1) @ db_n.T
        metrics = retrieval_metrics(sim, q_labels, db_labels)
        metrics["angle"] = angle
        rotation_results.append(metrics)
        print(f"    θ={angle:5.1f}°  R@1={metrics['R@1']:.3f}  "
              f"R@5={metrics['R@5']:.3f}  mAP@5={metrics['mAP@5']:.3f}")

    # ── Text-to-shape retrieval ───────────────────────────────────────────────
    print("\n  Evaluating text-to-shape retrieval (zero-shot, all templates)…")
    base_ds    = getattr(test_ds, "dataset", test_ds)
    categories = getattr(base_ds, "categories", {})

    # Evaluate once per template + once with averaged embedding
    text_results_by_template = []
    z_db_n    = F.normalize(z_db, dim=-1)
    q_labels_t = torch.tensor(sorted(categories.keys()))

    all_template_embs = []
    for t_idx, template in enumerate(TEXT_TEMPLATES):
        queries  = [template.format(categories[i]) for i in sorted(categories.keys())]
        z_text   = encode_text_queries(queries)           # [C, IB_DIM]
        all_template_embs.append(z_text)
        text_sim = z_text @ z_db_n.T
        r1   = recall_at_k(text_sim,  q_labels_t, db_labels, 1)
        r5   = recall_at_k(text_sim,  q_labels_t, db_labels, 5)
        map5 = mean_ap_at_k(text_sim, q_labels_t, db_labels, 5)
        text_results_by_template.append({"template": template, "R@1": r1, "R@5": r5, "mAP@5": map5})
        print(f'    [{t_idx+1:2d}] R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}  "{template}"')

    # Averaged embedding across all templates
    z_text_avg = F.normalize(torch.stack(all_template_embs).mean(dim=0), dim=-1)
    text_sim   = z_text_avg @ z_db_n.T
    r1_avg   = recall_at_k(text_sim,  q_labels_t, db_labels, 1)
    r5_avg   = recall_at_k(text_sim,  q_labels_t, db_labels, 5)
    map5_avg = mean_ap_at_k(text_sim, q_labels_t, db_labels, 5)
    print(f"    [avg] R@1={r1_avg:.3f}  R@5={r5_avg:.3f}  mAP@5={map5_avg:.3f}  (averaged over all templates)")

    # ── ImageBind image-to-shape retrieval ────────────────────────────────────
    print("\n  Evaluating ImageBind image-to-shape retrieval…")
    val_cache  = build_imagebind_cache(
        val_ds, cache_dir / f"{args.dataset}_val_ib.pt", n_views=args.n_views,
    )
    z_ib_q    = F.normalize(val_cache["embeddings"], dim=-1)
    ib_q_lbl  = val_cache["labels"]
    ib_sim    = z_ib_q @ z_db_n.T
    ib_r1     = recall_at_k(ib_sim,  ib_q_lbl, db_labels, 1)
    ib_r5     = recall_at_k(ib_sim,  ib_q_lbl, db_labels, 5)
    ib_map5   = mean_ap_at_k(ib_sim, ib_q_lbl, db_labels, 5)
    print(f"    image→shape  R@1={ib_r1:.3f}  R@5={ib_r5:.3f}  mAP@5={ib_map5:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config":                   vars(args),
        "templates":                TEXT_TEMPLATES,
        "rotation_retrieval":       rotation_results,
        "text_to_shape_per_template": text_results_by_template,
        "text_to_shape_avg_template": {"R@1": r1_avg, "R@5": r5_avg, "mAP@5": map5_avg},
        "image_to_shape":           {"R@1": ib_r1, "R@5": ib_r5, "mAP@5": ib_map5},
        "train_loss_history":       history,
        "timestamp":                ts,
    }
    out_path = out_dir / "exp11b_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    site_path = ROOT / "outputs" / "exp11b_imagebind_retrieval.json"
    with open(site_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Site results: {site_path}")


if __name__ == "__main__":
    main()
