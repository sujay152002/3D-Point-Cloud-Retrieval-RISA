"""Experiment 15 — End-to-End InfoNCE with Part-Composition Text Targets

Motivation
----------
Exp14 trains RISA end-to-end against Objaverse object-level text (tags +
descriptions).  This learns whole-object semantics but misses part-level
structure: the encoder has no signal about *which parts* compose a shape.

Exp13 uses per-point part labels from ShapeNet to generate part-composition
descriptions ("a chair with a back, a seat, and 4 legs"), but freezes the
RISA encoder and trains only a projection MLP — limiting the encoder's
ability to adapt to text-alignment gradients.

Exp15 combines the two ideas: **end-to-end** RISA training (exp14) with
**part-composition text targets** (exp13).  The encoder is unfrozen and the
InfoNCE loss backpropagates through both the projection MLP *and* the RISA
encoder, so the Gumbel-K selector and feature extractor learn which points
matter for *part-level* text alignment — not just geometric reconstruction
or whole-object description matching.

The part-composition descriptions provide a richer, more fine-grained
semantic signal than object-level tags: two chairs with different part
compositions ("4 legs + back" vs "a seat + 2 arms + 4 legs") get distinct
text targets, pushing the encoder to discriminate structural detail.
This also acts as a regulariser — the encoder must understand internal
structure, not just memorise class-level appearance.

Architecture
------------
    point cloud [3, N]
        → RISA encoder with fps_K prior  (unfrozen)
        → z_risa  [encoding_out_dim]
        → ProjectionMLP (encoding_out_dim → text_dim)
        → z_proj  [text_dim]   ←→   InfoNCE   ←→   z_text [text_dim]

Text targets (from part labels):
  - For each shape, generate part-composition description via connected-
    component counting on per-point part labels.
  - Multi-view: 1 class-bearing view ("a chair with a back, seat, 4 legs")
    + N class-agnostic views ("something with a back, seat, 4 legs" etc.)
    configurable via --n_agnostic (default 1).
  - Each batch samples one random view per shape — text augmentation
    that teaches invariance to phrasing while keeping part composition
    as the discriminative signal.

Text encoder: sentence-transformers all-mpnet-base-v2 (768-dim, frozen)
  - Encodes part-composition descriptions per shape
  - Text embeddings are pre-computed and cached on disk

Dataset: ShapeNet (16 classes, per-point part labels, 50 part IDs)
  - train: ~12k shapes (80% of shapenet_train.npz)
  - val:   ~3k shapes (carved from train deterministically)
  - test:  3,370 shapes (shapenet_test.npz)

Training details
----------------
  - Gumbel temperature annealed 1.0 → 0.1 over training
  - Differential learning rates: RISA encoder lr_risa, projection lr_proj
  - Gradient clipping: 1.0 for encoder, 5.0 for projection
  - Val loss tracked each epoch; best checkpoint saved
  - Early stopping with patience

Evaluation
----------
  1. Val InfoNCE loss curve
  2. Part-query retrieval: structural text queries ("something with 4 legs")
     → part_hit_rate, class_hit_rate across the test gallery
  3. Standard text→shape R@1, R@5, mAP@5 (class-level retrieval)
  4. Rotation-stratified retrieval (val→test, 9 angles)
  5. Zero-shot transfer to ModelNet40 / Objaverse (if available)

Usage
-----
    # Default: ShapeNet, K=256, end-to-end
    python experiments/exp15_e2e_part_contrastive.py

    # Custom hyper-parameters
    python experiments/exp15_e2e_part_contrastive.py \\
        --K 256 --epochs 50 --batch_size 64 \\
        --lr_risa 1e-5 --lr_proj 1e-4 --patience 10

    # Freeze encoder (train projection only — ablation / comparison to exp13)
    python experiments/exp15_e2e_part_contrastive.py --freeze_encoder

    # Warm-start from exp10 RISA checkpoint
    python experiments/exp15_e2e_part_contrastive.py --risa_ckpt outputs/checkpoints/exp10_fps_K.pt

    # Resume from exp15 checkpoint
    python experiments/exp15_e2e_part_contrastive.py \\
        --resume outputs/checkpoints/exp15_fps_K_best.pt
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

from core.datasets import get_dataset, SHAPE_NET_PART_SEG_CLASSES
from core.trainer import unpack_encoder_output
from experiments.exp10_prior_selection import RISAWithPrior, set_seed
from experiments.exp1_retrieval_eval import recall_at_k, mean_ap_at_k

# ── Part-composition description generation (from exp13) ─────────────────────
from experiments.exp13_part_text_retrieval import (
    SHAPENET_CLASS_NAMES,
    SHAPENET_PART_NAMES,
    PARTNET_PART_NAMES,
    PART_ID_TO_INFO,
    COUNTABLE_PARTS,
    AN_PARTS,
    _article,
    _pluralise,
    _count_instances_cc,
    _build_part_phrases,
    _join_part_phrases,
    _AGOSTIC_PREFIXES,
    generate_description,
    generate_agnostic_description,
    generate_agnostic_descriptions_all,
    build_shape_descriptions_dual,
    evaluate_part_queries,
    PART_QUERIES,
    PARTNET_PART_QUERIES,
    load_sentence_transformer,
    encode_texts,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
TEXT_DIM  = 768   # all-mpnet-base-v2 output dimension


# ── Projection MLP ────────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """Two-layer MLP projecting RISA encoding → sentence-transformer dim.

    Identical structure to exp14 — LayerNorm + dropout for stability during
    end-to-end training.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 1024,
                 out_dim: int = TEXT_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE with a learnable log-inverse-temperature parameter.

    CLIP convention: log_inv_temp = log(1/T), clamped to T ∈ [temp_floor, 100].
    The temp_floor prevents the temperature from collapsing too low, which
    sharpens logits and accelerates memorisation / overfitting.
    """

    def __init__(self, init_temp: float = 0.07, temp_floor: float = 0.03):
        super().__init__()
        self.temp_floor = temp_floor
        self.log_inv_temp = nn.Parameter(torch.tensor(np.log(1.0 / init_temp)))

    @property
    def temperature(self) -> float:
        return 1.0 / self.log_inv_temp.clamp(
            min=np.log(1.0 / 100.0), max=np.log(1.0 / self.temp_floor)
        ).exp().item()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        inv_temp = self.log_inv_temp.clamp(
            min=np.log(1.0 / 100.0), max=np.log(1.0 / self.temp_floor)
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


def _part_text_collate_fn(batch):
    """Custom collate for ShapeNet items: (pts_t, lbl_t, part_lbl_t, text, idx).

    ShapeNet __getitem__ returns (pts [3,N], lbl, part_lbl) — 3 elements.
    _IndexedDataset appends idx → 4 elements.

    Returns: pts [B,3,N], lbls [B], idxs [B]
    """
    pts   = torch.stack([b[0] for b in batch])
    lbls  = torch.stack([b[1] for b in batch])
    idxs  = torch.tensor([b[-1] for b in batch], dtype=torch.long)
    return pts, lbls, idxs


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the position index alongside each item."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return (*item, i) if isinstance(item, tuple) else (item, i)


# ── Part-composition text cache ──────────────────────────────────────────────

def build_part_text_cache(dataset, st_model, cache_path: Path,
                          split_name: str = "train",
                          dataset_name: str = "shapenet",
                          n_agnostic: int = 1) -> torch.Tensor:
    """Pre-compute and cache part-composition text embeddings for all shapes.

    Uses exp13's build_shape_descriptions_dual() to generate:
      - 1 class-bearing description ("a chair with a back, seat, 4 legs")
      - up to 9 class-agnostic descriptions ("something with a back, seat, 4 legs" etc.)

    n_agnostic controls how many class-agnostic views to keep (default 1).
    If an existing cache has more views than requested, it is sliced down.

    Returns [N, 1 + n_agnostic, TEXT_DIM] — multi-view text embeddings.
    Each batch samples one random view per shape for text augmentation.
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
        if cached.dim() == 3 and cached.shape[0] == N and cached.shape[2] == TEXT_DIM:
            n_cached = cached.shape[1] - 1  # views minus class-bearing
            if n_cached >= n_agnostic:
                # Slice down to requested number of agnostic views
                text_embs = cached[:, :1 + n_agnostic, :]
                print(f"  Loaded text cache ({N} embeddings, {text_embs.shape[1]} views) ← {cache_path}")
                return text_embs
            else:
                print(f"  Cache has {n_cached} agnostic views, need {n_agnostic} — recomputing…")
        else:
            print(f"  Cache size mismatch — recomputing…")

    categories = getattr(raw_ds, "categories", {})

    # Generate dual descriptions (class-bearing + all class-agnostic variants)
    print(f"  Generating per-shape part descriptions for {split_name} set ({dataset_name})…")
    class_descs, agnostic_descs_all = build_shape_descriptions_dual(
        dataset, categories, dataset_name=dataset_name
    )
    n_available = len(agnostic_descs_all[0]) if agnostic_descs_all else 0
    n_agnostic = min(n_agnostic, n_available)
    # Slice to requested number of agnostic views
    agnostic_descs_all = [ads[:n_agnostic] for ads in agnostic_descs_all]
    print(f"  {n_agnostic} class-agnostic variant{'s' if n_agnostic != 1 else ''} per shape")

    # Print samples for verification
    shown = set()
    if isinstance(dataset, Subset):
        raw_indices = list(dataset.indices)
    else:
        raw_indices = list(range(len(raw_ds)))
    for pos, (raw_i, cd, ads) in enumerate(zip(raw_indices, class_descs, agnostic_descs_all)):
        cls = int(raw_ds.labels[raw_i])
        if cls not in shown:
            print(f"    [{categories.get(cls, f'class{cls}')}] {cd}")
            for ad in ads:
                print(f"            → {ad}")
            shown.add(cls)
        if len(shown) >= 4:
            break

    # Encode class-bearing descriptions
    print(f"  Encoding {N} class-bearing + {N * n_agnostic} class-agnostic descriptions…")
    class_embs = encode_texts(class_descs, st_model)                     # [N, D]

    # Flatten all agnostic variants, encode, reshape
    flat_agnostic = [d for shape_ads in agnostic_descs_all for d in shape_ads]
    agnostic_embs = encode_texts(flat_agnostic, st_model)                # [N * n_agnostic, D]
    agnostic_embs = agnostic_embs.view(N, n_agnostic, TEXT_DIM)          # [N, n_agnostic, D]

    # Stack: [N, 1 + n_agnostic, D]
    text_embs = torch.cat([
        class_embs.unsqueeze(1),      # [N, 1, D]
        agnostic_embs,                 # [N, n_agnostic, D]
    ], dim=1)
    print(f"  Text embeddings: {text_embs.shape}  (1 class + {n_agnostic} agnostic views)")

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(text_embs, cache_path)
    print(f"  Text cache saved → {cache_path}")

    return text_embs


# ── Objaverse rich-text cache ────────────────────────────────────────────────

def build_objaverse_text_cache(dataset, st_model, cache_path: Path,
                               split_name: str = "test") -> torch.Tensor:
    """Pre-compute and cache Objaverse per-object text embeddings.

    Objaverse shapes carry a `texts` field (tags/captions joined as a sentence)
    rather than structural part labels, so we encode those directly — one
    embedding per shape.  The result shape is [N, 1, TEXT_DIM] to match the
    multi-view convention used by build_part_text_cache, even though there is
    only a single view.

    Returns [N, 1, TEXT_DIM] tensor, cached to disk.
    """
    from torch.utils.data import Subset

    raw_ds  = dataset.dataset if isinstance(dataset, Subset) else dataset
    N = len(dataset)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.dim() == 3 and cached.shape[0] == N and cached.shape[2] == TEXT_DIM:
            print(f"  Loaded Objaverse text cache ({N} embeddings) ← {cache_path}")
            return cached
        else:
            print(f"  Objaverse text cache size mismatch — recomputing…")

    texts = getattr(raw_ds, "texts", None)
    if texts is None:
        raise AttributeError("Dataset has no 'texts' attribute — is this an Objaverse dataset?")

    # Resolve indices if Subset
    if isinstance(dataset, Subset):
        indices = list(dataset.indices)
        texts = [texts[i] for i in indices]
    else:
        texts = list(texts)

    # Show a few samples
    print(f"  Objaverse {split_name} texts ({N} objects):")
    for i in range(min(3, N)):
        print(f"    [{i}] {texts[i][:120]}")

    print(f"  Encoding {N} Objaverse descriptions…")
    embs = encode_texts(texts, st_model)          # [N, TEXT_DIM]

    # Wrap to [N, 1, TEXT_DIM] to match multi-view cache convention
    text_embs = embs.unsqueeze(1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(text_embs, cache_path)
    print(f"  Objaverse text cache saved → {cache_path}")

    return text_embs


# ── Encode dataset shapes ────────────────────────────────────────────────────

@torch.inference_mode()
def encode_dataset_shapes(encoder, proj, dataset, batch_size: int = 32):
    """Encode all shapes → [N, TEXT_DIM] L2-normalised + labels."""
    loader = DataLoader(
        _IndexedDataset(dataset), batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=_part_text_collate_fn,
    )
    all_z, all_labels = [], []
    encoder.eval(); proj.eval()
    for pts, lbls, idxs in loader:
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
    train_text_embs: torch.Tensor,   # [N_train, n_views, TEXT_DIM]
    val_text_embs:   torch.Tensor,   # [N_val,   n_views, TEXT_DIM]
    epochs:          int,
    batch_size:      int,
    lr_risa:         float,
    lr_proj:         float,
    ckpt_dir:        Path,
    patience:        int = 0,
    min_delta:       float = 0.005,
    freeze_encoder:  bool = False,
    warmup_epochs:   int = 0,
    ckpt_prefix:     str = "exp15_fps_K",
) -> tuple:
    """End-to-end training loop with part-composition text targets.

    Differential learning rates: RISA encoder uses lr_risa, projection +
    temperature use lr_proj.

    When freeze_encoder=True, the encoder is excluded from the optimizer
    and kept in eval mode throughout training (ablation: matches exp13 setup).

    warmup_epochs: freeze encoder for this many epochs first, then unfreeze.
    Ignored when freeze_encoder=True.

    patience + min_delta: early stopping fires when val loss hasn't improved
    by at least min_delta for `patience` consecutive epochs.

    Supports multi-view text embeddings: a random view is sampled per shape
    each batch (text augmentation that prevents memorisation).

    Returns (train_history, val_history) — lists of per-epoch average losses.
    """
    train_text_embs = train_text_embs.to(DEVICE)
    val_text_embs   = val_text_embs.to(DEVICE)

    n_views = train_text_embs.size(1)

    train_loader = DataLoader(
        _IndexedDataset(train_ds),
        batch_size    = batch_size,
        shuffle       = True,
        drop_last     = True,
        num_workers   = 4,
        pin_memory    = True,
        collate_fn    = _part_text_collate_fn,
        worker_init_fn= _worker_init_fn,
    )
    val_loader = DataLoader(
        _IndexedDataset(val_ds),
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = True,
        collate_fn  = _part_text_collate_fn,
    )

    # Build optimizer — always include encoder param group so we can
    # unfreeze mid-training (warmup).  Always start with lr_risa so the
    # CosineAnnealingLR schedule is computed from the correct base lr;
    # during warmup, requires_grad=False prevents gradients from flowing
    # regardless of the lr value.
    enc_lr_init = 0.0 if freeze_encoder else lr_risa
    opt = optim.AdamW([
        {"params": encoder.parameters(),   "lr": enc_lr_init, "weight_decay": 1e-4},
        {"params": proj.parameters(),      "lr": lr_proj,     "weight_decay": 1e-4},
        {"params": criterion.parameters(), "lr": lr_proj,     "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-7
    )

    best_val_loss  = float("inf")
    best_ckpt_path = ckpt_dir / f"{ckpt_prefix}_best.pt"
    train_history, val_history = [], []
    epochs_since_best = 0

    for epoch in range(epochs):
        # Warmup phase: keep encoder frozen, then unfreeze.
        # No need to manually set lr — the CosineAnnealingLR scheduler was
        # initialised with lr_risa as the base and has been annealing
        # throughout warmup.  When we unfreeze, the scheduler's current
        # lr is already at the correct (cosine-annealed) value.
        if not freeze_encoder and warmup_epochs > 0 and epoch == warmup_epochs:
            for p in encoder.parameters():
                p.requires_grad = True
            print(f"  epoch {epoch+1}: encoder unfrozen (warmup complete)")

        enc_is_frozen = freeze_encoder or (warmup_epochs > 0 and epoch < warmup_epochs)
        if enc_is_frozen:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
        else:
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad = True
        proj.train(); criterion.train()
        total_loss, n_batches = 0.0, 0

        for pts, _, idxs in train_loader:
            pts = pts.to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)

            # Sample one random view per shape each batch
            views = train_text_embs[idxs]                        # [B, n_views, D]
            view_idx = torch.randint(0, n_views, (views.size(0),),
                                      device=DEVICE)              # [B]
            z_text = views[torch.arange(views.size(0), device=DEVICE),
                            view_idx]                             # [B, D]

            z_risa, _ = unpack_encoder_output(encoder(pts))
            z_proj    = proj(z_risa)
            loss      = criterion(z_proj, z_text)

            opt.zero_grad()
            loss.backward()
            if not enc_is_frozen:
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            nn.utils.clip_grad_norm_(proj.parameters(),    5.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_avg = total_loss / max(n_batches, 1)
        train_history.append(train_avg)

        # Validation — average over all views for stability
        encoder.eval(); proj.eval(); criterion.eval()
        val_loss, val_n = 0.0, 0
        with torch.inference_mode():
            for pts, _, idxs in val_loader:
                pts = pts.to(DEVICE)
                if pts.shape[1] != 3:
                    pts = pts.permute(0, 2, 1)

                z_risa, _ = unpack_encoder_output(encoder(pts))
                z_proj    = proj(z_risa)

                views = val_text_embs[idxs]      # [B, n_views, D]
                v_loss = 0.0
                for v in range(n_views):
                    v_loss += criterion(z_proj, views[:, v, :]).item()
                val_loss += v_loss / n_views
                val_n += 1

        val_avg = val_loss / max(val_n, 1)
        val_history.append(val_avg)

        is_best = val_avg < (best_val_loss - min_delta)
        if is_best:
            best_val_loss = val_avg
            torch.save(
                {"encoder": encoder.state_dict(), "proj": proj.state_dict()},
                best_ckpt_path,
            )
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        print(
            f"  epoch {epoch+1:3d}/{epochs}  "
            f"loss={train_avg:.4f}  val={val_avg:.4f}  gap={val_avg-train_avg:+.4f}  "
            f"temp={criterion.temperature:.3f}  "
            f"lr_enc={opt.param_groups[0]['lr']:.2e}  "
            f"lr_proj={opt.param_groups[1]['lr']:.2e}"
            f"{'  ★' if is_best else ''}"
        )

        if patience > 0 and epochs_since_best >= patience:
            print(f"  Early stopping: no improvement for {patience} epochs")
            break

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
                   batch_size: int = 32,
                   cache_dir: Path = None) -> dict:
    """Encode a held-out dataset with no fine-tuning.

    Query strategy (in priority order):
      1. Per-object rich descriptions  — if the dataset exposes a `texts`
         attribute (e.g. Objaverse tags/captions).  Embeddings are pre-computed
         via build_objaverse_text_cache and cached to disk.  Each shape's own
         text is used as its query; retrieval is evaluated as text→shape with
         the matching shape as the single relevant item (R@1/R@5/mAP@5 over
         the test gallery, one query per gallery item).
      2. Class-name queries            — if `categories` is a real mapping
         (more than a single dummy class), one text query per class.
      3. Shape→shape fallback          — val shapes used as queries when
         neither rich texts nor real categories are available.

    Returns dict with R@1, R@5, mAP@5, and 'mode' key.
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

    gallery_z, gallery_labels = encode_dataset_shapes(encoder, proj, test_ds, batch_size)
    gallery_z = F.normalize(gallery_z, dim=-1)
    N_gallery = gallery_z.size(0)

    # ── Strategy 1: per-object rich descriptions (Objaverse texts) ──────────
    rich_texts = getattr(base_test, "texts", None)
    if rich_texts is not None:
        _cache_dir = cache_dir if cache_dir is not None else (ROOT / "outputs" / "text_cache")
        cache_path = _cache_dir / f"{dataset_name}_test_rich_text.pt"
        text_embs = build_objaverse_text_cache(
            test_ds, st_model, cache_path, split_name="test"
        )                                                    # [N_gallery, 1, TEXT_DIM]
        q_embs = F.normalize(text_embs[:, 0, :], dim=-1)    # [N_gallery, TEXT_DIM]

        # Each query's "correct" match is itself in the gallery
        gallery_ids = torch.arange(N_gallery, dtype=torch.long)

        sim   = q_embs @ gallery_z.T                         # [N_gallery, N_gallery]
        r1    = recall_at_k(sim,  gallery_ids, gallery_ids, 1)
        r5    = recall_at_k(sim,  gallery_ids, gallery_ids, 5)
        map5  = mean_ap_at_k(sim, gallery_ids, gallery_ids, 5)
        print(f"    text→shape (rich)  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
        return {"R@1": r1, "R@5": r5, "mAP@5": map5, "mode": "text2shape_rich"}

    # ── Strategy 2: one class-name query per category ────────────────────────
    # Treat the dummy single-class Objaverse mapping as absent.
    real_categories = {k: v for k, v in categories.items() if v != "object"}
    if real_categories:
        query_texts  = [f"a 3D model of a {real_categories[i]}"
                        for i in sorted(real_categories)]
        query_labels = torch.tensor(sorted(real_categories.keys()), dtype=torch.long)
        q_embs = encode_texts(query_texts, st_model)
        q_embs = F.normalize(q_embs, dim=-1)
        sim    = q_embs @ gallery_z.T
        r1     = recall_at_k(sim,  query_labels, gallery_labels, 1)
        r5     = recall_at_k(sim,  query_labels, gallery_labels, 5)
        map5   = mean_ap_at_k(sim, query_labels, gallery_labels, 5)
        print(f"    text→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
        return {"R@1": r1, "R@5": r5, "mAP@5": map5, "mode": "text2shape"}

    # ── Strategy 3: shape→shape fallback ────────────────────────────────────
    print(f"  No categories or rich texts — using val shapes as shape queries")
    q_z, q_labels = encode_dataset_shapes(encoder, proj, val_ds, batch_size)
    q_z = F.normalize(q_z, dim=-1)
    sim  = q_z @ gallery_z.T
    r1   = recall_at_k(sim, q_labels, gallery_labels, 1)
    r5   = recall_at_k(sim, q_labels, gallery_labels, 5)
    map5 = mean_ap_at_k(sim, q_labels, gallery_labels, 5)
    print(f"    shape→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
    return {"R@1": r1, "R@5": r5, "mAP@5": map5, "mode": "shape2shape"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 15 — End-to-End InfoNCE, Part-Composition Text, ShapeNet"
    )
    parser.add_argument("--K",          type=int,   default=256,
                        help="Gumbel-K points selected per shape")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=128,
                        help="Large batches = more negatives per InfoNCE step")
    parser.add_argument("--lr_risa",    type=float, default=1e-6,
                        help="Learning rate for RISA encoder (unfrozen)")
    parser.add_argument("--lr_proj",    type=float, default=1e-4,
                        help="Learning rate for projection MLP + temperature")
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--temp_floor", type=float, default=0.07,
                        help="Minimum temperature floor for InfoNCE")
    parser.add_argument("--dropout",    type=float, default=0.2,
                        help="Dropout rate in projection MLP")
    parser.add_argument("--freeze_encoder", action="store_true", default=False,
                        help="Freeze RISA encoder — train only projection MLP (ablation: matches exp13)")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Freeze encoder for this many epochs before end-to-end training")
    parser.add_argument("--min_delta",   type=float, default=0.005,
                        help="Minimum val loss improvement to reset early stopping counter")
    parser.add_argument("--risa_ckpt",  type=str,   default=None,
                        help="Warm-start RISA from this checkpoint (optional)")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Resume full model from exp15 checkpoint dict")
    parser.add_argument("--dataset",    default="shapenet",
                        help="Dataset with part labels (default: shapenet)")
    parser.add_argument("--n_agnostic", type=int,   default=1,
                        help="Number of class-agnostic text views per shape (default 1, max 9). "
                             "Reduces near-duplicate text targets that don't add regularization signal.")
    parser.add_argument("--st_model",   default="all-mpnet-base-v2")
    parser.add_argument("--zero_shot",  nargs="*",
                        default=["modelnet40"],
                        help="Datasets to evaluate zero-shot transfer on")
    parser.add_argument("--topk_eval",  type=int,   default=10,
                        help="Top-k for part-query evaluation")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"exp15_fps_K_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to both terminal and log file
    tag = "frozen" if args.freeze_encoder else "e2e"
    log_path = ROOT / "outputs" / f"exp15_{tag}_run_{ts}.log"
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
    print(f"  Exp 15 — End-to-End InfoNCE (Part-Composition Text)")
    print(f"  K={args.K}  epochs={args.epochs}  batch={args.batch_size}")
    print(f"  lr_risa={args.lr_risa}  lr_proj={args.lr_proj}")
    print(f"  dropout={args.dropout}  temp_floor={args.temp_floor}  patience={args.patience}  min_delta={args.min_delta}")
    print(f"  freeze_encoder={args.freeze_encoder}  warmup_epochs={args.warmup_epochs}")
    print(f"  text encoder: {args.st_model}  n_agnostic={args.n_agnostic}")
    print(f"  dataset: {args.dataset}  (part-composition targets)")
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
        print("  No warm-start — training RISA from random init")

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        print("  Encoder frozen — training projection MLP only")
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        print("  Encoder unfrozen — end-to-end training")

    n_enc  = sum(p.numel() for p in encoder.parameters())
    print(f"  RISA encoder parameters: {n_enc:,}")

    # ── Projection MLP ────────────────────────────────────────────────────────
    proj      = ProjectionMLP(in_dim=512, hidden_dim=1024, out_dim=TEXT_DIM,
                              dropout=args.dropout).to(DEVICE)
    criterion = InfoNCELoss(init_temp=0.07, temp_floor=args.temp_floor).to(DEVICE)

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        proj.load_state_dict(state["proj"])
        print(f"  Resumed projection from: {args.resume}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = get_dataset(args.dataset, split="train",
                           root=DATA_ROOT, num_points=N_POINTS)
    val_ds   = get_dataset(args.dataset, split="val",
                           root=DATA_ROOT, num_points=N_POINTS)
    test_ds  = get_dataset(args.dataset, split="test",
                           root=DATA_ROOT, num_points=N_POINTS)
    print(f"  Train: {len(train_ds)} shapes   Val: {len(val_ds)} shapes   Test: {len(test_ds)} shapes")

    # ── Sentence-transformer ──────────────────────────────────────────────────
    print(f"  Loading sentence-transformer: {args.st_model}")
    st_model = load_sentence_transformer(args.st_model)
    print(f"  ✓ sentence-transformer ready")

    # ── Text embedding cache ───────────────────────────────────────────────────
    # Objaverse: use per-object rich texts already in the dataset.
    # Part-labelled datasets (ShapeNet, PartNet): generate part-composition
    # descriptions from part_labels via build_part_text_cache.
    cache_dir = ROOT / "outputs" / "text_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_train = getattr(train_ds, "dataset", train_ds)
    has_rich_texts  = getattr(base_train, "texts",       None) is not None
    has_part_labels = getattr(base_train, "part_labels", None) is not None

    if has_rich_texts:
        print(f"  Dataset '{args.dataset}' has rich texts — using object descriptions as targets")
        train_text_embs = build_objaverse_text_cache(
            train_ds, st_model,
            cache_dir / f"{args.dataset}_train_rich_text.pt", split_name="train",
        )
        val_text_embs = build_objaverse_text_cache(
            val_ds, st_model,
            cache_dir / f"{args.dataset}_val_rich_text.pt", split_name="val",
        )
    elif has_part_labels:
        print(f"  Dataset '{args.dataset}' has part labels — using part-composition descriptions as targets")
        train_text_embs = build_part_text_cache(
            train_ds, st_model,
            cache_dir / f"{args.dataset}_train_part_text.pt", split_name="train",
            dataset_name=args.dataset,
            n_agnostic=args.n_agnostic,
        )
        val_text_embs = build_part_text_cache(
            val_ds, st_model,
            cache_dir / f"{args.dataset}_val_part_text.pt", split_name="val",
            dataset_name=args.dataset,
            n_agnostic=args.n_agnostic,
        )
    else:
        print(f"  Error: dataset '{args.dataset}' has neither 'texts' nor 'part_labels'.")
        sys.exit(1)

    # ── Training ──────────────────────────────────────────────────────────────
    if args.freeze_encoder:
        print(f"\n  Training projection MLP (encoder frozen)…")
    else:
        print(f"\n  Training end-to-end ({'rich object texts' if has_rich_texts else 'part-composition targets'})…")
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
        patience         = args.patience,
        min_delta        = args.min_delta,
        freeze_encoder   = args.freeze_encoder,
        warmup_epochs    = args.warmup_epochs,
        ckpt_prefix      = "exp15_fps_K",
    )

    # Save final checkpoint
    final_ckpt = ckpt_dir / "exp15_fps_K_final.pt"
    torch.save({"encoder": encoder.state_dict(), "proj": proj.state_dict()}, final_ckpt)
    print(f"  Final checkpoint saved → {final_ckpt}")

    # ── Part-query retrieval evaluation ────────────────────────────────────────
    base_test = getattr(test_ds, "dataset", test_ds)
    categories = getattr(base_train, "categories", {})

    gallery_z, gallery_labels = encode_dataset_shapes(
        encoder, proj, test_ds, args.batch_size
    )
    gallery_z = F.normalize(gallery_z, dim=-1)

    if getattr(base_test, "part_labels", None) is not None:
        print(f"\n  Part-query retrieval (test gallery, top-{args.topk_eval})…")
        gallery_indices = torch.arange(len(base_test))
        queries = PARTNET_PART_QUERIES if args.dataset.lower() == "partnet" else PART_QUERIES
        part_query_results = evaluate_part_queries(
            queries         = queries,
            gallery_z       = gallery_z,
            gallery_labels  = gallery_labels,
            gallery_indices = gallery_indices,
            categories      = categories,
            st_model        = st_model,
            raw_ds          = base_test,
            topk            = args.topk_eval,
            dataset_name    = args.dataset,
        )
    else:
        print(f"\n  Part-query retrieval skipped (no part_labels in '{args.dataset}')")
        part_query_results = {}

    # ── Standard text→shape retrieval (class-level) ───────────────────────────
    print(f"\n  Standard text→shape retrieval (class-level, test gallery)…")
    if categories:
        query_texts  = [f"a 3D model of a {categories[i]}" for i in sorted(categories)]
        query_labels = torch.tensor(sorted(categories.keys()), dtype=torch.long)
        q_embs = encode_texts(query_texts, st_model)
        q_embs = F.normalize(q_embs, dim=-1)
        sim    = q_embs @ gallery_z.T
        r1     = recall_at_k(sim,  query_labels, gallery_labels, 1)
        r5     = recall_at_k(sim,  query_labels, gallery_labels, 5)
        map5   = mean_ap_at_k(sim, query_labels, gallery_labels, 5)
        print(f"    text→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")
        class_retrieval = {"R@1": r1, "R@5": r5, "mAP@5": map5}
    else:
        class_retrieval = {}

    # ── Rotation-stratified retrieval (val→test, 9 angles) ───────────────────
    print(f"\n  Rotation-stratified retrieval (val→test, 9 angles)…")
    from experiments.exp1_retrieval_eval import rotation_by_angle

    z_q0, q_labels = encode_dataset_shapes(encoder, proj, val_ds, args.batch_size)
    z_q0 = F.normalize(z_q0, dim=-1)

    rotation_results = []
    for angle in [0, 15, 30, 45, 60, 90, 120, 150, 180]:
        if angle == 0:
            z_q = z_q0
        else:
            R_mat  = rotation_by_angle(angle, axis="y", device=DEVICE)
            val_loader = DataLoader(
                _IndexedDataset(val_ds), batch_size=args.batch_size, shuffle=False,
                num_workers=2, pin_memory=True, collate_fn=_part_text_collate_fn,
            )
            all_z = []
            encoder.eval(); proj.eval()
            with torch.inference_mode():
                for pts, _, idxs in val_loader:
                    pts = pts.to(DEVICE)
                    if pts.shape[1] != 3:
                        pts = pts.permute(0, 2, 1)
                    pts = R_mat @ pts
                    z_risa, _ = unpack_encoder_output(encoder(pts))
                    all_z.append(F.normalize(proj(z_risa), dim=-1).cpu())
            z_q = torch.cat(all_z)

        sim  = z_q @ gallery_z.T
        r1   = recall_at_k(sim, q_labels, gallery_labels, 1)
        r5   = recall_at_k(sim, q_labels, gallery_labels, 5)
        map5 = mean_ap_at_k(sim, q_labels, gallery_labels, 5)
        rotation_results.append({"angle": angle, "R@1": r1, "R@5": r5, "mAP@5": map5})
        print(f"    angle={angle:3d}°  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")

    # ── Zero-shot transfer evaluation ─────────────────────────────────────────
    zero_shot_results = {}
    for ds_name in (args.zero_shot or []):
        zero_shot_results[ds_name] = eval_zero_shot(
            encoder, proj, ds_name, st_model,
            batch_size=args.batch_size,
            cache_dir=cache_dir,
        )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config":              vars(args),
        "train_loss_history":  train_history,
        "val_loss_history":    val_history,
        "part_query_retrieval": part_query_results,
        "class_retrieval":     class_retrieval,
        "rotation_results":    rotation_results,
        "zero_shot_transfer":  zero_shot_results,
        "timestamp":           ts,
    }
    out_path = out_dir / "exp15_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    site_path = ROOT / "outputs" / "exp15_e2e_part_contrastive.json"
    with open(site_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Site results → {site_path}")

    log_f.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
