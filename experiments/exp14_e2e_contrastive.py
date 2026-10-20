"""Experiment 14 — End-to-End InfoNCE with RISA + Gumbel-K Prior on Objaverse

Motivation
----------
Exp11/11b/13 all freeze the RISA encoder and train only a projection MLP on
top of pretrained features. The prior is selected by a separate geometric
heuristic or learned on a different objective (proxy-anchor on ShapeNet).

The hypothesis here: jointly optimising RISA *and* the projection with a
single InfoNCE loss against text targets lets the Gumbel-K selector learn
which points matter for *text alignment*, not just geometric reconstruction.
At large dataset scale (Objaverse ~800k objects) overfitting is not a concern
and end-to-end gradients reach the feature extractor.

Architecture
------------
    point cloud [3, N]
        → RISA encoder with fps_K prior  (unfrozen)
        → z_risa  [encoding_out_dim]
        → ProjectionMLP (encoding_out_dim → text_dim)
        → z_proj  [text_dim]   ←→   InfoNCE   ←→   z_text [text_dim]

Text encoder: sentence-transformers all-mpnet-base-v2 (768-dim, frozen)
  - Encodes Objaverse tags/descriptions per object
  - Text embeddings are pre-computed once and cached on disk

Training details
----------------
  - Gumbel temperature annealed 1.0 → 0.1 over training (same schedule as exp10)
  - Differential learning rates: RISA encoder 1e-5, projection head 1e-4
  - Gradient clipping at 1.0 for RISA, 5.0 for projection
  - Val loss tracked each epoch; best checkpoint saved
  - Large batch preferred (InfoNCE quality scales with batch size)

Evaluation
----------
  1. Val InfoNCE loss curve
  2. If test split available: text→shape R@1/R@5/mAP@5
  3. Zero-shot transfer: encode ShapeNet/ModelNet40 test set (no fine-tuning)
     and measure shape-to-shape R@1/R@5/mAP@5 — the key generalisation check

Usage
-----
    python experiments/exp14_e2e_contrastive.py

    # Resume from checkpoint:
    python experiments/exp14_e2e_contrastive.py \\
        --resume outputs/checkpoints/exp14_fps_K_best.pt

    # Warm-start from exp10 fps_K RISA checkpoint:
    python experiments/exp14_e2e_contrastive.py \\
        --risa_ckpt outputs/checkpoints/exp10_fps_K.pt

    # Custom settings:
    python experiments/exp14_e2e_contrastive.py \\
        --K 256 --epochs 30 --batch_size 128 --lr_risa 1e-5 --lr_proj 1e-4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
from experiments.exp10_prior_selection import RISAWithPrior, set_seed
from experiments.exp1_retrieval_eval import recall_at_k, mean_ap_at_k

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
TEXT_DIM  = 768   # all-mpnet-base-v2 output dimension


# ── Text encoder (sentence-transformers) ──────────────────────────────────────

def load_sentence_transformer(model_name: str = "all-mpnet-base-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not found — installing…")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "sentence-transformers==3.4.1", "--user", "-q",
        ])
        from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.eval()
    return model


@torch.inference_mode()
def encode_texts(texts: list, st_model, batch_size: int = 512) -> torch.Tensor:
    """Encode a list of strings → [N, TEXT_DIM] L2-normalised on CPU."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        embs  = st_model.encode(
            chunk, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


# ── Text embedding cache ──────────────────────────────────────────────────────

def build_text_cache(dataset, st_model, cache_path: Path,
                     split_name: str = "train") -> torch.Tensor:
    """Pre-compute and cache text embeddings for all shapes in dataset.

    Handles both plain datasets and torch.utils.data.Subset: always returns
    a tensor aligned to dataset[0..len(dataset)-1].

    cache_path : if exists and aligned, loaded from disk; otherwise recomputed.
    Returns [N, TEXT_DIM] L2-normalised float32 tensor on CPU.
    """
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        raw_ds  = dataset.dataset
        indices = list(dataset.indices)
    else:
        raw_ds  = dataset
        indices = list(range(len(raw_ds)))

    N = len(indices)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.shape[0] == N and cached.shape[1] == TEXT_DIM:
            print(f"  Loaded text cache ({N} embeddings) ← {cache_path}")
            return cached
        print(f"  Cache size mismatch ({cached.shape[0]} vs {N}) — recomputing…")

    texts = [raw_ds[i][2] for i in indices]   # index 2 is the text string
    print(f"  Encoding {N} texts for {split_name} split…")
    embs = encode_texts(texts, st_model)       # [N, TEXT_DIM]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, cache_path)
    print(f"  Text cache saved → {cache_path}")
    return embs


# ── Projection MLP ────────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """Two-layer MLP: RISA encoding_out_dim → TEXT_DIM.

    Identical structure to exp13 but adapts to any in_dim so it works
    regardless of encoding_out_dim setting.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = TEXT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE with a learnable log-inverse-temperature parameter.

    CLIP convention: log_inv_temp = log(1/T), clamped to T ∈ [0.01, 100].
    """

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_inv_temp = nn.Parameter(torch.tensor(np.log(1.0 / init_temp)))

    @property
    def temperature(self) -> float:
        return 1.0 / self.log_inv_temp.clamp(
            min=np.log(0.01), max=np.log(100.0)
        ).exp().item()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        inv_temp = self.log_inv_temp.clamp(
            min=np.log(0.01), max=np.log(100.0)
        ).exp()
        logits  = (z_a @ z_b.T) * inv_temp
        labels  = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits,   labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2.0


# ── DataLoader helpers ────────────────────────────────────────────────────────

def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _text_collate_fn(batch):
    """Custom collate that handles the str text field returned by ObjaverseDataset.

    batch items: (pts_t [3,N], lbl_t, text str, dataset_idx int)
    Returns: pts [B,3,N], lbls [B], texts list[str], idxs [B]
    """
    pts   = torch.stack([b[0] for b in batch])
    lbls  = torch.stack([b[1] for b in batch])
    texts = [b[2] for b in batch]
    idxs  = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return pts, lbls, texts, idxs


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the position index i alongside each item."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return (*item, i) if isinstance(item, tuple) else (item, i)


# ── Encoding helpers ──────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_dataset_shapes(encoder, proj, dataset, batch_size: int = 32):
    """Encode all shapes → [N, TEXT_DIM] L2-normalised + labels."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
        ),
    )
    all_z, all_labels = [], []
    encoder.eval(); proj.eval()
    for pts, lbls in loader:
        pts = pts.to(DEVICE)
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        z, _ = unpack_encoder_output(encoder(pts))
        all_z.append(F.normalize(proj(z), dim=-1).cpu())
        all_labels.append(lbls)
    return torch.cat(all_z), torch.cat(all_labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    encoder,
    proj,
    criterion,
    train_ds,
    val_ds,
    train_text_embs: torch.Tensor,   # [N_train, TEXT_DIM]
    val_text_embs:   torch.Tensor,   # [N_val,   TEXT_DIM]
    epochs:          int,
    batch_size:      int,
    lr_risa:         float,
    lr_proj:         float,
    ckpt_dir:        Path,
) -> tuple:
    """End-to-end training loop.

    Differential learning rates: RISA encoder uses lr_risa, projection +
    temperature use lr_proj.

    Returns (train_history, val_history) — lists of per-epoch average losses.
    """
    train_text_embs = train_text_embs.to(DEVICE)
    val_text_embs   = val_text_embs.to(DEVICE)

    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size    = batch_size,
        shuffle       = True,
        drop_last     = True,
        num_workers   = 4,
        pin_memory    = True,
        collate_fn    = _text_collate_fn,
        worker_init_fn= _worker_init_fn,
    )
    val_loader = DataLoader(
        _IndexedDataset(val_ds),
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = True,
        collate_fn  = _text_collate_fn,
    )

    # Differential learning rates — encoder gets 10× lower lr than projection
    opt = optim.AdamW([
        {"params": encoder.parameters(),   "lr": lr_risa,  "weight_decay": 1e-4},
        {"params": proj.parameters(),      "lr": lr_proj,  "weight_decay": 1e-4},
        {"params": criterion.parameters(), "lr": lr_proj,  "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-7
    )

    best_val_loss  = float("inf")
    best_ckpt_path = ckpt_dir / "exp14_fps_K_best.pt"
    train_history, val_history = [], []

    for epoch in range(epochs):
        encoder.train(); proj.train(); criterion.train()
        total_loss, n_batches = 0.0, 0

        for pts, _, texts, idxs in train_loader:
            pts       = pts.to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)

            z_text = train_text_embs[idxs]   # [B, TEXT_DIM] — pre-cached, no grad

            z_risa, _ = unpack_encoder_output(encoder(pts))
            z_proj    = proj(z_risa)
            loss      = criterion(z_proj, z_text)

            opt.zero_grad()
            loss.backward()
            # Separate gradient clipping: tighter for encoder, looser for head
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            nn.utils.clip_grad_norm_(proj.parameters(),    5.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_avg = total_loss / max(n_batches, 1)
        train_history.append(train_avg)

        # Validation
        encoder.eval(); proj.eval(); criterion.eval()
        val_loss, val_n = 0.0, 0
        with torch.inference_mode():
            for pts, _, texts, idxs in val_loader:
                pts = pts.to(DEVICE)
                if pts.shape[1] != 3:
                    pts = pts.permute(0, 2, 1)
                z_text = val_text_embs[idxs]
                z_risa, _ = unpack_encoder_output(encoder(pts))
                val_loss += criterion(proj(z_risa), z_text).item()
                val_n    += 1
        val_avg = val_loss / max(val_n, 1)
        val_history.append(val_avg)

        is_best = val_avg < best_val_loss
        if is_best:
            best_val_loss = val_avg
            torch.save(
                {"encoder": encoder.state_dict(), "proj": proj.state_dict()},
                best_ckpt_path,
            )

        print(
            f"  epoch {epoch+1:3d}/{epochs}  "
            f"loss={train_avg:.4f}  val={val_avg:.4f}  "
            f"temp={criterion.temperature:.3f}  "
            f"lr_risa={opt.param_groups[0]['lr']:.2e}"
            f"{'  ★' if is_best else ''}"
        )

    # Restore best
    if best_ckpt_path.exists():
        state = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(state["encoder"])
        proj.load_state_dict(state["proj"])
        print(f"  Restored best checkpoint (val={best_val_loss:.4f}) ← {best_ckpt_path}")

    encoder.eval(); proj.eval()
    return train_history, val_history


# ── Zero-shot transfer evaluation ─────────────────────────────────────────────

def eval_zero_shot(encoder, proj, dataset_name: str, st_model,
                   batch_size: int = 32) -> dict:
    """Encode a held-out dataset (e.g. ShapeNet or ModelNet40) with no fine-tuning.

    Queries: val split text descriptions (class name template)
    Gallery: test split shape embeddings

    Returns dict with R@1, R@5, mAP@5 at angle=0.
    """
    print(f"\n  Zero-shot transfer → {dataset_name}…")
    try:
        test_ds = get_dataset(dataset_name, split="test",
                              root=DATA_ROOT, num_points=N_POINTS)
        val_ds  = get_dataset(dataset_name, split="val",
                              root=DATA_ROOT, num_points=N_POINTS)
    except FileNotFoundError as e:
        print(f"  Skipping {dataset_name}: {e}")
        return {}

    base_test = getattr(test_ds, "dataset", test_ds)
    categories = getattr(base_test, "categories", {})

    # Gallery: encode test shapes
    gallery_z, gallery_labels = encode_dataset_shapes(encoder, proj, test_ds, batch_size)
    gallery_z = F.normalize(gallery_z, dim=-1)

    # Queries: class-name text → shape  (one query per class)
    if categories:
        query_texts  = [f"a 3D model of a {categories[i]}" for i in sorted(categories)]
        query_labels = torch.tensor(sorted(categories.keys()), dtype=torch.long)
    else:
        # Objaverse-style: use val shapes as text queries
        print(f"  No categories — using val shapes as shape queries")
        q_z, q_labels = encode_dataset_shapes(encoder, proj, val_ds, batch_size)
        q_z = F.normalize(q_z, dim=-1)
        sim  = q_z @ gallery_z.T
        r1   = recall_at_k(sim, q_labels, gallery_labels, 1)
        r5   = recall_at_k(sim, q_labels, gallery_labels, 5)
        map5 = mean_ap_at_k(sim, q_labels, gallery_labels, 5)
        print(f"    shape→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
        return {"R@1": r1, "R@5": r5, "mAP@5": map5, "mode": "shape2shape"}

    q_embs = encode_texts(query_texts, st_model)    # [C, TEXT_DIM]
    q_embs = F.normalize(q_embs, dim=-1)
    sim    = q_embs @ gallery_z.T
    r1     = recall_at_k(sim,  query_labels, gallery_labels, 1)
    r5     = recall_at_k(sim,  query_labels, gallery_labels, 5)
    map5   = mean_ap_at_k(sim, query_labels, gallery_labels, 5)
    print(f"    text→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
    return {"R@1": r1, "R@5": r5, "mAP@5": map5, "mode": "text2shape"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 14 — End-to-end InfoNCE, RISA fps_K, Objaverse"
    )
    parser.add_argument("--K",          type=int,   default=256,
                        help="Gumbel-K points selected per shape")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=128,
                        help="Large batches = more negatives per InfoNCE step")
    parser.add_argument("--lr_risa",    type=float, default=1e-5,
                        help="Learning rate for RISA encoder (unfrozen)")
    parser.add_argument("--lr_proj",    type=float, default=1e-4,
                        help="Learning rate for projection MLP + temperature")
    parser.add_argument("--risa_ckpt",  type=str,   default=None,
                        help="Warm-start RISA from this checkpoint (optional)")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Resume full model from exp14 checkpoint dict")
    parser.add_argument("--dataset",    default="objaverse")
    parser.add_argument("--st_model",   default="all-mpnet-base-v2")
    parser.add_argument("--zero_shot",  nargs="*",
                        default=["shapenet", "modelnet40"],
                        help="Datasets to evaluate zero-shot transfer on")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"exp14_fps_K_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to both terminal and log file
    log_path = ROOT / "outputs" / "exp14_run.log"
    log_f    = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, msg):
            for s in self.streams:
                try: s.write(msg)
                except OSError: pass
        def flush(self):
            for s in self.streams:
                try: s.flush()
                except OSError: pass
        def reconfigure(self, **kw): pass

    sys.stdout = _Tee(sys.__stdout__, log_f)
    print(f"  log → {log_path}")

    print(f"\n{'='*60}")
    print(f"  Exp 14 — End-to-End InfoNCE (RISA fps_K + Objaverse)")
    print(f"  K={args.K}  epochs={args.epochs}  batch={args.batch_size}")
    print(f"  lr_risa={args.lr_risa}  lr_proj={args.lr_proj}")
    print(f"  text encoder: {args.st_model}")
    print(f"  dataset: {args.dataset}")
    print(f"  output: {out_dir}")
    print(f"{'='*60}\n")

    # ── RISA encoder (fps_K, unfrozen) ───────────────────────────────────────
    encoder = RISAWithPrior(
        prior            = "fps_K",
        K                = args.K,
        num_local        = 32,
        num_global       = 32,
        num_blocks       = 4,
        model_dim        = 256,
        features_out_dim = 256,
        encoding_out_dim = 512,
    ).to(DEVICE)

    ckpt_dir = ROOT / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(state["encoder"])
        print(f"  Resumed encoder from: {args.resume}")
    elif args.risa_ckpt and Path(args.risa_ckpt).exists():
        state = torch.load(args.risa_ckpt, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(state, strict=False)
        print(f"  Warm-started RISA from: {args.risa_ckpt}")
    else:
        # Try the standard exp10 fps_K checkpoint as a warm start
        auto_ckpt = ckpt_dir / "exp10_fps_K.pt"
        if auto_ckpt.exists():
            state = torch.load(auto_ckpt, map_location=DEVICE, weights_only=True)
            encoder.load_state_dict(state, strict=False)
            print(f"  Warm-started RISA from exp10 checkpoint: {auto_ckpt}")
        else:
            print("  No RISA checkpoint found — training from random init")

    # All encoder parameters trainable
    for p in encoder.parameters():
        p.requires_grad = True

    n_enc  = sum(p.numel() for p in encoder.parameters())
    print(f"  RISA encoder parameters: {n_enc:,}")

    # ── Projection MLP ────────────────────────────────────────────────────────
    proj      = ProjectionMLP(in_dim=512, hidden_dim=1024, out_dim=TEXT_DIM).to(DEVICE)
    criterion = InfoNCELoss(init_temp=0.07).to(DEVICE)

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        proj.load_state_dict(state["proj"])
        print(f"  Resumed projection from: {args.resume}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = get_dataset(args.dataset, split="train",
                           root=DATA_ROOT, num_points=N_POINTS)
    val_ds   = get_dataset(args.dataset, split="val",
                           root=DATA_ROOT, num_points=N_POINTS)
    print(f"  Train: {len(train_ds)} shapes   Val: {len(val_ds)} shapes")

    # ── Sentence-transformer ──────────────────────────────────────────────────
    print(f"  Loading sentence-transformer: {args.st_model}")
    st_model = load_sentence_transformer(args.st_model)
    print(f"  ✓ sentence-transformer ready")

    # ── Text embedding cache ──────────────────────────────────────────────────
    cache_dir = ROOT / "outputs" / "text_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_text_embs = build_text_cache(
        train_ds, st_model,
        cache_dir / f"{args.dataset}_train_text.pt", split_name="train"
    )
    val_text_embs = build_text_cache(
        val_ds, st_model,
        cache_dir / f"{args.dataset}_val_text.pt",   split_name="val"
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n  Training end-to-end…")
    train_history, val_history = train(
        encoder          = encoder,
        proj             = proj,
        criterion        = criterion,
        train_ds         = train_ds,
        val_ds           = val_ds,
        train_text_embs  = train_text_embs,
        val_text_embs    = val_text_embs,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr_risa          = args.lr_risa,
        lr_proj          = args.lr_proj,
        ckpt_dir         = ckpt_dir,
    )

    # Save final checkpoint
    final_ckpt = ckpt_dir / "exp14_fps_K_final.pt"
    torch.save({"encoder": encoder.state_dict(), "proj": proj.state_dict()}, final_ckpt)
    print(f"  Final checkpoint saved → {final_ckpt}")

    # ── Zero-shot transfer evaluation ─────────────────────────────────────────
    zero_shot_results = {}
    for ds_name in (args.zero_shot or []):
        zero_shot_results[ds_name] = eval_zero_shot(
            encoder, proj, ds_name, st_model, batch_size=args.batch_size
        )

    # ── Objaverse val: text→shape retrieval ──────────────────────────────────
    objaverse_val_results = {}
    try:
        test_ds = get_dataset(args.dataset, split="test",
                              root=DATA_ROOT, num_points=N_POINTS)
        print("\n  Objaverse text→shape retrieval (val text → test shapes)…")
        gallery_z, gallery_labels = encode_dataset_shapes(
            encoder, proj, test_ds, args.batch_size
        )
        gallery_z = F.normalize(gallery_z, dim=-1)

        val_text_embs_n = F.normalize(val_text_embs, dim=-1).to("cpu")

        # val labels
        from torch.utils.data import Subset
        if isinstance(val_ds, Subset):
            raw_val     = val_ds.dataset
            val_indices = list(val_ds.indices)
        else:
            raw_val     = val_ds
            val_indices = list(range(len(val_ds)))
        val_labels  = torch.tensor(
            [int(raw_val.labels[i]) for i in val_indices], dtype=torch.long
        )

        sim  = val_text_embs_n @ gallery_z.T
        r1   = recall_at_k(sim,  val_labels, gallery_labels, 1)
        r5   = recall_at_k(sim,  val_labels, gallery_labels, 5)
        map5 = mean_ap_at_k(sim, val_labels, gallery_labels, 5)
        print(f"    text→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
        objaverse_val_results = {"R@1": r1, "R@5": r5, "mAP@5": map5}
    except FileNotFoundError:
        print("  No Objaverse test split — skipping test retrieval")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config":              vars(args),
        "train_loss_history":  train_history,
        "val_loss_history":    val_history,
        "objaverse_retrieval": objaverse_val_results,
        "zero_shot_transfer":  zero_shot_results,
        "timestamp":           ts,
    }
    out_path = out_dir / "exp14_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    site_path = ROOT / "outputs" / "exp14_e2e_contrastive.json"
    with open(site_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Site results → {site_path}")

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
