"""Experiment 12 — Interactive Retrieval Interface

Loads the trained exp11 models (RISA encoder + ProjectionMLP) and exposes
an interactive CLI for querying the ShapeNet gallery.

Supported query modes
---------------------
  text  : free-form text string  → nearest shapes in the gallery
  shape : pick a shape by index  → nearest shapes (shape-to-shape)
  info  : show gallery statistics
  quit  : exit

Usage
-----
    python experiments/exp12_interactive_retrieval.py

    # Explicit checkpoint paths (defaults to exp11 aggregate_K):
    python experiments/exp12_interactive_retrieval.py \\
        --risa_ckpt  outputs/checkpoints/exp10_aggregate_K.pt \\
        --proj_ckpt  outputs/checkpoints/exp11_aggregate_K_proj.pt \\
        --prior aggregate_K --K 256 --dataset shapenet --topk 5

    # Use the exp11b checkpoint (stronger text alignment):
    python experiments/exp12_interactive_retrieval.py --exp11b
    # or explicitly:
    python experiments/exp12_interactive_retrieval.py \\
        --proj_ckpt outputs/checkpoints/exp11b_aggregate_K_proj.pt

    # Use the exp13 checkpoint (part-composition text retrieval):
    python experiments/exp12_interactive_retrieval.py --exp13
    # then try queries like:
    #   text something with 4 legs
    #   text a shape with wings
    #   text an object with wheels
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp10_prior_selection import RISAWithPrior, PRIOR_NAMES, set_seed
from experiments.exp11_imagebind_retrieval import ProjectionMLP as ProjectionMLP11, IB_DIM, encode_with_proj
from experiments.exp13_part_text_retrieval import (
    ProjectionMLP as ProjectionMLP13, ST_DIM,
    load_sentence_transformer, encode_texts,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(prior: str, K: int, risa_ckpt: str, proj_ckpt: str, exp13: bool = False):
    """Load RISA encoder + ProjectionMLP from checkpoints.

    exp13=True uses ProjectionMLP13 (512→768, sentence-transformer space).
    exp13=False uses ProjectionMLP11 (512→1024, ImageBind space).
    """
    encoder = RISAWithPrior(
        prior            = prior,
        K                = K,
        num_local        = 32,
        num_global       = 32,
        num_blocks       = 4,
        model_dim        = 256,
        features_out_dim = 256,
        encoding_out_dim = 512,
    ).to(DEVICE)

    enc_state = torch.load(risa_ckpt, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(enc_state, strict=False)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"  ✓ Loaded RISA encoder  ← {risa_ckpt}")

    if exp13:
        proj = ProjectionMLP13(in_dim=512, hidden_dim=1024, out_dim=ST_DIM).to(DEVICE)
    else:
        proj = ProjectionMLP11(risa_dim=512, hidden_dim=1024, ib_dim=IB_DIM).to(DEVICE)

    proj_state = torch.load(proj_ckpt, map_location=DEVICE, weights_only=True)
    proj.load_state_dict(proj_state)
    for p in proj.parameters():
        p.requires_grad = False
    proj.eval()
    print(f"  ✓ Loaded ProjectionMLP ← {proj_ckpt}")

    return encoder, proj


# ── Gallery builder ───────────────────────────────────────────────────────────

def build_gallery(encoder, proj, dataset, split_name: str, exp13: bool = False):
    """Encode all shapes → [N, D] normalised embeddings."""
    print(f"  Building gallery from {split_name} split ({len(dataset)} shapes)…")
    if exp13:
        from experiments.exp13_part_text_retrieval import encode_dataset as encode_dataset13
        z, labels = encode_dataset13(encoder, proj, dataset, batch_size=32)
    else:
        z, labels = encode_with_proj(encoder, proj, dataset, batch_size=32)
    z = F.normalize(z, dim=-1)
    print(f"  ✓ Gallery ready: {z.shape}")
    return z, labels


# ── ImageBind text encoder ────────────────────────────────────────────────────

def load_imagebind():
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


@torch.no_grad()
def encode_text_imagebind(query: str, ib_model, ib_data, ModalityType) -> torch.Tensor:
    """Encode a single text string with ImageBind → [1, IB_DIM] normalised."""
    inputs = {ModalityType.TEXT: ib_data.load_and_transform_text([query], DEVICE)}
    emb    = ib_model(inputs)[ModalityType.TEXT]
    return F.normalize(emb, dim=-1).cpu()


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query_emb: torch.Tensor, gallery_z: torch.Tensor,
             gallery_labels: torch.Tensor, categories: dict,
             topk: int, exclude_idx: int = None):
    """
    query_emb : [1, D] or [D]  — already L2-normalised
    Returns list of dicts: {rank, idx, label, class_name, score}
    """
    q    = F.normalize(query_emb.view(1, -1), dim=-1)
    sims = (q @ gallery_z.T).squeeze(0)

    if exclude_idx is not None:
        sims[exclude_idx] = -1.0

    top_scores, top_idx = torch.topk(sims, k=min(topk, len(sims)))

    results = []
    for rank, (idx, score) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
        lbl  = int(gallery_labels[idx])
        name = categories.get(lbl, f"class_{lbl}")
        results.append({"rank": rank, "idx": idx, "label": lbl,
                        "class_name": name, "score": score})
    return results


def print_results(results: list, header: str):
    print(f"\n  {header}")
    print(f"  {'Rank':<5}  {'Score':<8}  {'Class':<20}  {'Gallery idx'}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*20}  {'-'*11}")
    for r in results:
        print(f"  {r['rank']:<5}  {r['score']:.4f}    {r['class_name']:<20}  {r['idx']}")
    print()


# ── Interactive loop ──────────────────────────────────────────────────────────

BANNER_IB = """
╔══════════════════════════════════════════════════════════╗
║   Exp 12 — Interactive 3D Retrieval (RISA + ImageBind)  ║
╚══════════════════════════════════════════════════════════╝

Commands
────────
  text  <query>        Search by text   e.g.  text  a 3d model of a chair
  shape <index>        Search by shape  e.g.  shape 42
  info                 Show gallery stats and available classes
  help                 Show this message
  quit / exit          Exit
"""

BANNER_ST = """
╔═══════════════════════════════════════════════════════════════╗
║   Exp 12 — Interactive 3D Retrieval (RISA + Part-Text exp13) ║
╚═══════════════════════════════════════════════════════════════╝

Commands
────────
  text  <query>        Search by text   e.g.  text  something with 4 legs
                                               text  a shape with wings
                                               text  an object with wheels
  shape <index>        Search by shape  e.g.  shape 42
  info                 Show gallery stats and available classes
  help                 Show this message
  quit / exit          Exit
"""


def run_interactive(gallery_z, gallery_labels, categories, topk,
                    text_encoder, encoder, proj, test_ds, exp13: bool = False):
    print(BANNER_ST if exp13 else BANNER_IB)

    while True:
        try:
            raw = input("retrieval> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        # ── quit ──────────────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            print("  Bye!")
            break

        # ── help ──────────────────────────────────────────────────────────────
        elif cmd in ("help", "h", "?"):
            print(BANNER_ST if exp13 else BANNER_IB)

        # ── info ──────────────────────────────────────────────────────────────
        elif cmd == "info":
            mode = "exp13 (part-text, sentence-transformers)" if exp13 \
                   else "exp11/11b (ImageBind)"
            print(f"\n  Gallery size : {len(gallery_z)} shapes")
            print(f"  Embedding dim: {gallery_z.shape[1]}")
            print(f"  Top-k        : {topk}")
            print(f"  Mode         : {mode}")
            print(f"  Device       : {DEVICE}")
            counts = {}
            for lbl in gallery_labels.tolist():
                counts[lbl] = counts.get(lbl, 0) + 1
            print(f"\n  Classes ({len(categories)}):")
            for cid in sorted(categories):
                print(f"    [{cid:2d}]  {categories[cid]:<20}  {counts.get(cid, 0)} shapes")
            print()

        # ── text query ────────────────────────────────────────────────────────
        elif cmd == "text":
            if not arg:
                print("  Usage: text <query string>")
                continue
            print(f"  Encoding: \"{arg}\"")
            if exp13:
                # Use sentence-transformer
                q_emb = encode_texts([arg], text_encoder)   # [1, ST_DIM]
            else:
                # Use ImageBind
                ib_model, ib_data, ModalityType = text_encoder
                q_emb = encode_text_imagebind(arg, ib_model, ib_data, ModalityType)
            results = retrieve(q_emb, gallery_z, gallery_labels, categories, topk)
            print_results(results, f'Text query: "{arg}"  →  Top {topk}')

        # ── shape query ───────────────────────────────────────────────────────
        elif cmd == "shape":
            if not arg:
                print("  Usage: shape <index>   (0-indexed into test split)")
                continue
            try:
                idx = int(arg)
            except ValueError:
                print(f"  Error: '{arg}' is not a valid integer index.")
                continue
            if idx < 0 or idx >= len(test_ds):
                print(f"  Error: index {idx} out of range [0, {len(test_ds)-1}].")
                continue

            item     = test_ds[idx]
            pts      = item[0].unsqueeze(0).to(DEVICE)
            lbl      = int(item[1])
            cls_name = categories.get(lbl, f"class_{lbl}")

            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)

            with torch.no_grad():
                z_risa, _ = unpack_encoder_output(encoder(pts))
                q_emb     = proj(z_risa).cpu()

            results = retrieve(q_emb, gallery_z, gallery_labels, categories,
                               topk, exclude_idx=idx)
            print_results(results,
                          f"Shape query: index={idx}  class={cls_name}  →  Top {topk}")

        else:
            print(f"  Unknown command: '{cmd}'.  Type 'help' for usage.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 12 — Interactive retrieval using trained exp11/exp13 models"
    )
    parser.add_argument("--prior",      default="aggregate_K", choices=PRIOR_NAMES)
    parser.add_argument("--K",          type=int, default=256)
    parser.add_argument("--dataset",    default="shapenet")
    parser.add_argument("--topk",       type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument(
        "--risa_ckpt", default=None,
        help="Path to RISA encoder checkpoint."
    )
    parser.add_argument(
        "--proj_ckpt", default=None,
        help="Path to ProjectionMLP checkpoint."
    )
    parser.add_argument(
        "--exp11b", action="store_true",
        help="Use exp11b checkpoint (stronger text alignment, ImageBind space)."
    )
    parser.add_argument(
        "--exp13", action="store_true",
        help="Use exp13 checkpoint (part-composition text, sentence-transformer space)."
    )
    args = parser.parse_args()

    set_seed(args.seed)

    ckpt_dir  = ROOT / "outputs" / "checkpoints"
    risa_ckpt = args.risa_ckpt or str(ckpt_dir / f"exp10_{args.prior}.pt")

    if args.proj_ckpt:
        proj_ckpt = args.proj_ckpt
    elif args.exp13:
        proj_ckpt = str(ckpt_dir / f"exp13_{args.prior}_proj.pt")
    elif args.exp11b:
        proj_ckpt = str(ckpt_dir / f"exp11b_{args.prior}_proj.pt")
    else:
        proj_ckpt = str(ckpt_dir / f"exp11_{args.prior}_proj.pt")

    for path, name in [(risa_ckpt, "RISA checkpoint"), (proj_ckpt, "Projection checkpoint")]:
        if not Path(path).exists():
            print(f"  Error: {name} not found at {path}")
            sys.exit(1)

    if args.exp13:
        mode_str = "exp13 (part-composition, sentence-transformers)"
    elif args.exp11b:
        mode_str = "exp11b (stronger text, ImageBind)"
    else:
        mode_str = "exp11 (ImageBind)"

    print(f"\n  Device : {DEVICE}")
    print(f"  Prior  : {args.prior}  K={args.K}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode   : {mode_str}\n")

    # Load RISA + projection
    encoder, proj = load_models(args.prior, args.K, risa_ckpt, proj_ckpt,
                                 exp13=args.exp13)

    # Load dataset
    test_ds    = get_dataset(args.dataset, split="test", root=DATA_ROOT, num_points=N_POINTS)
    base_ds    = getattr(test_ds, "dataset", test_ds)
    categories = getattr(base_ds, "categories", {})

    # Build gallery
    gallery_z, gallery_labels = build_gallery(encoder, proj, test_ds, "test",
                                               exp13=args.exp13)

    # Load appropriate text encoder
    if args.exp13:
        print("  Loading sentence-transformer (all-mpnet-base-v2)…")
        text_encoder = load_sentence_transformer("all-mpnet-base-v2")
        print("  ✓ sentence-transformer ready\n")
    else:
        print("  Loading ImageBind…")
        text_encoder = load_imagebind()   # returns (model, ib_data, ModalityType)
        print("  ✓ ImageBind ready\n")

    # Start interactive loop
    run_interactive(
        gallery_z      = gallery_z,
        gallery_labels = gallery_labels,
        categories     = categories,
        topk           = args.topk,
        text_encoder   = text_encoder,
        encoder        = encoder,
        proj           = proj,
        test_ds        = test_ds,
        exp13          = args.exp13,
    )


if __name__ == "__main__":
    main()
