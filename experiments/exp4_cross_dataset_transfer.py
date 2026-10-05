"""Experiment 4 — Cross-Dataset Transfer Retrieval

Train on ModelNet40 (CAD models), build retrieval database from ModelNet40,
then query with ScanObjectNN (real-world scans) at arbitrary rotations.

This tests whether invariance generalises across domain shift.
RISA's geometric features (PCA eigenvalues, pairwise distances) are
domain-agnostic by construction — they should transfer better than
coordinate-based baselines.

Shared classes between ModelNet40 and ScanObjectNN:
    chair, table, sofa, bed, monitor, desk, toilet, bathtub

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp4_cross_dataset_transfer.py
"""

import argparse
import json
import sys, io
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import (
    recall_at_k, mean_ap_at_k, rotation_by_angle,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ANGLES    = [0, 30, 60, 90, 120, 150, 180]

# ModelNet40 class names that overlap with ScanObjectNN
MN40_CATEGORIES = {
    0: "airplane", 1: "bathtub", 2: "bed", 3: "bench", 4: "bookshelf",
    5: "bottle", 6: "bowl", 7: "car", 8: "chair", 9: "cone",
    10: "cup", 11: "curtain", 12: "desk", 13: "door", 14: "dresser",
    15: "flower_pot", 16: "glass_box", 17: "guitar", 18: "keyboard", 19: "lamp",
    20: "laptop", 21: "mantel", 22: "monitor", 23: "night_stand", 24: "person",
    25: "piano", 26: "plant", 27: "radio", 28: "range_hood", 29: "sink",
    30: "sofa", 31: "stairs", 32: "stool", 33: "table", 34: "tent",
    35: "toilet", 36: "tv_stand", 37: "vase", 38: "wardrobe", 39: "xbox",
}

SCANOBJ_CATEGORIES = {
    0: "bag", 1: "bin", 2: "box", 3: "cabinet", 4: "chair",
    5: "desk", 6: "display", 7: "door", 8: "shelf", 9: "table",
    10: "bed", 11: "pillow", 12: "sink", 13: "sofa", 14: "toilet",
}

# Map ScanObjectNN class → ModelNet40 class (shared semantic categories)
SCANOBJ_TO_MN40 = {
    4:  8,   # chair  → chair
    5:  12,  # desk   → desk
    9:  33,  # table  → table
    10: 2,   # bed    → bed
    12: 29,  # sink   → sink
    13: 30,  # sofa   → sofa
    14: 35,  # toilet → toilet
}


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


def _build_encoder(name, cls, perceiver=True):
    if name == "RISA":
        from models import RotationInvariantSparseAttention
        return RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
            perceiver=perceiver,
        )
    return cls()


@torch.no_grad()
def encode_subset(encoder, dataset, valid_indices, batch_size=32):
    """Encode only the shapes at valid_indices. Returns (M, D) and remapped labels."""
    subset = torch.utils.data.Subset(dataset, valid_indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_enc, all_lbl = [], []
    for batch in loader:
        pts = batch[0] if isinstance(batch, (tuple, list)) else batch
        lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0])
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            pts, lbl, _ = batch
        pts = pts.float().to(DEVICE)
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu())
        all_lbl.append(lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl))
    return torch.cat(all_enc), torch.cat(all_lbl).long()


@torch.no_grad()
def encode_subset_rotated(encoder, dataset, valid_indices, R, batch_size=32):
    subset = torch.utils.data.Subset(dataset, valid_indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_enc = []
    for batch in loader:
        pts = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            pts = batch[0]
        pts = pts.float().to(DEVICE)
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        pts = R.to(DEVICE) @ pts
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu())
    return torch.cat(all_enc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to run, e.g. --models RISA")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--perceiver", action="store_true", default=False,
                        help="Use Perceiver-style RISA (only first block updates point features)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint to load (skips training)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training entirely (use with --checkpoint)")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                        help="Path to save the best encoder checkpoint (e.g. outputs/checkpoints/exp4_best.pt)")
    parser.add_argument("--out",    default="outputs/exp4_cross_dataset.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")
    print("Loading datasets…")

    mn40_test  = get_dataset("modelnet40",   split="test", root=DATA_ROOT, num_points=N_POINTS)
    scan_test  = get_dataset("scanobjectnn", split="test", root=DATA_ROOT, num_points=N_POINTS,
                             variant="OBJ_ONLY")

    # Find ScanObjectNN indices for shared classes and remap labels to MN40 space
    scan_labels = []
    for i in range(len(scan_test)):
        item = scan_test[i]
        lbl  = int(item[1]) if isinstance(item, (tuple, list)) else 0
        scan_labels.append(lbl)

    shared_scan_idx    = [i for i, l in enumerate(scan_labels) if l in SCANOBJ_TO_MN40]
    remapped_scan_lbl  = torch.tensor(
        [SCANOBJ_TO_MN40[scan_labels[i]] for i in shared_scan_idx], dtype=torch.long
    )

    # Find MN40 indices for shared classes
    mn40_labels = []
    for i in range(len(mn40_test)):
        item = mn40_test[i]
        lbl  = int(item[1]) if isinstance(item, (tuple, list)) else 0
        mn40_labels.append(lbl)

    shared_mn40_classes = set(SCANOBJ_TO_MN40.values())
    shared_mn40_idx     = [i for i, l in enumerate(mn40_labels) if l in shared_mn40_classes]
    remapped_mn40_lbl   = torch.tensor([mn40_labels[i] for i in shared_mn40_idx], dtype=torch.long)

    print(f"  MN40 shared shapes:    {len(shared_mn40_idx)}")
    print(f"  ScanObjNN shared shapes: {len(shared_scan_idx)}")

    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results  = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        encoder = _build_encoder(model_name, model_cls, perceiver=args.perceiver).to(DEVICE)

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=DEVICE)
            missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
            if missing or unexpected:
                print(f"  [checkpoint] missing={len(missing)} unexpected={len(unexpected)} keys")
            else:
                print(f"  [checkpoint] Loaded {args.checkpoint}")

        arch_tag = f"{model_name}_{'perceiver' if args.perceiver else 'original'}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb = _TBWriter(log_dir=f"runs/exp4/{ts}_{arch_tag}") if _TB_AVAILABLE else None

        if not args.skip_train:
            import torch.nn as nn, torch.optim as optim
            torch.manual_seed(args.seed)
            ds     = get_dataset("modelnet40", split="train", root=DATA_ROOT, num_points=N_POINTS)
            val_ds = get_dataset("modelnet40", split="test",  root=DATA_ROOT, num_points=N_POINTS)
            loader     = torch.utils.data.DataLoader(ds,     batch_size=4, shuffle=True,  drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=False)
            # Pre-build val subset loaders for cross-dataset retrieval eval
            _mn40_val_subset  = torch.utils.data.Subset(mn40_test,  shared_mn40_idx)
            _scan_val_subset  = torch.utils.data.Subset(scan_test,  shared_scan_idx)
            _mn40_val_loader  = torch.utils.data.DataLoader(_mn40_val_subset, batch_size=32, shuffle=False)
            _scan_val_loader  = torch.utils.data.DataLoader(_scan_val_subset, batch_size=32, shuffle=False)
            encoder.eval()
            with torch.no_grad():
                z, _ = unpack_encoder_output(encoder(torch.randn(2, 3, 64, device=DEVICE)))
            head = nn.Linear(z.shape[-1], 40).to(DEVICE)
            opt  = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
            sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
            crit = nn.CrossEntropyLoss()
            def _rand_rot(B):
                Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
                Q[:, :, 0:1] *= torch.linalg.det(Q).sign().view(B, 1, 1)
                return Q
            encoder.train(); head.train()
            best_val_r1 = -1.0
            best_encoder_state = None
            for ep in range(args.epochs):
                ep_loss, ep_correct, n_batches, n_samples = 0.0, 0, 0, 0
                for batch in loader:
                    pts = batch[0].float().to(DEVICE)
                    lbl = batch[1].long().to(DEVICE)
                    if pts.shape[1] != 3: pts = pts.permute(0, 2, 1)
                    pts = _rand_rot(pts.shape[0]) @ pts
                    opt.zero_grad()
                    z, _ = unpack_encoder_output(encoder(pts))
                    logits = head(z)
                    loss = crit(logits, lbl)
                    loss.backward(); opt.step()
                    ep_loss    += loss.item()
                    ep_correct += (logits.argmax(dim=-1) == lbl).sum().item()
                    n_batches  += 1
                    n_samples  += lbl.shape[0]
                sch.step()
                avg_loss  = ep_loss / max(n_batches, 1)
                train_acc = ep_correct / max(n_samples, 1)
                lr        = sch.get_last_lr()[0]
                print(f"    epoch {ep+1}/{args.epochs}  train_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  lr={lr:.2e}", flush=True)
                if tb:
                    tb.add_scalar("train/loss", avg_loss,  ep + 1)
                    tb.add_scalar("train/acc",  train_acc, ep + 1)
                    tb.add_scalar("train/lr",   lr,        ep + 1)

                if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
                    encoder.eval(); head.eval()
                    val_loss, val_correct, val_n_batches, val_n_samples = 0.0, 0, 0, 0
                    with torch.no_grad():
                        for batch in val_loader:
                            pts = batch[0].float().to(DEVICE)
                            lbl = batch[1].long().to(DEVICE)
                            if pts.shape[1] != 3: pts = pts.permute(0, 2, 1)
                            pts = _rand_rot(pts.shape[0]) @ pts  # match train distribution
                            z, _ = unpack_encoder_output(encoder(pts))
                            logits = head(z)
                            val_loss    += crit(logits, lbl).item()
                            val_correct += (logits.argmax(dim=-1) == lbl).sum().item()
                            val_n_batches  += 1
                            val_n_samples  += lbl.shape[0]
                    val_loss /= max(val_n_batches, 1)
                    val_acc   = val_correct / max(val_n_samples, 1)

                    # Cross-dataset retrieval: ScanObjNN queries → MN40 database
                    with torch.no_grad():
                        db_enc_val = torch.cat([unpack_encoder_output(encoder(
                            b[0].float().to(DEVICE) if b[0].shape[1] == 3 else b[0].float().permute(0,2,1).to(DEVICE)
                        ))[0].cpu() for b in _mn40_val_loader])
                        q_enc_val  = torch.cat([unpack_encoder_output(encoder(
                            b[0].float().to(DEVICE) if b[0].shape[1] == 3 else b[0].float().permute(0,2,1).to(DEVICE)
                        ))[0].cpu() for b in _scan_val_loader])
                    val_r1  = recall_at_k(q_enc_val, db_enc_val, remapped_scan_lbl, remapped_mn40_lbl, k=1)
                    val_r5  = recall_at_k(q_enc_val, db_enc_val, remapped_scan_lbl, remapped_mn40_lbl, k=5)
                    val_map = mean_ap_at_k(q_enc_val, db_enc_val, remapped_scan_lbl, remapped_mn40_lbl, k=5)

                    is_best = val_r1 > best_val_r1
                    if is_best:
                        best_val_r1 = val_r1
                        best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                    print(f"    epoch {ep+1}/{args.epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_R@1={val_r1:.4f}  val_R@5={val_r5:.4f}  val_mAP@5={val_map:.4f}{' ★' if is_best else ''}", flush=True)
                    if tb:
                        tb.add_scalar("val/loss",   val_loss, ep + 1)
                        tb.add_scalar("val/acc",    val_acc,  ep + 1)
                        tb.add_scalar("val/R@1",    val_r1,   ep + 1)
                        tb.add_scalar("val/R@5",    val_r5,   ep + 1)
                        tb.add_scalar("val/mAP@5",  val_map,  ep + 1)
                    encoder.train(); head.train()

            # Restore best checkpoint before eval
            if best_encoder_state is not None:
                encoder.load_state_dict(best_encoder_state)
                print(f"  Restored best encoder (val_R@1={best_val_r1:.4f})")
            if args.save_checkpoint and best_encoder_state is not None:
                ckpt_path = Path(args.save_checkpoint)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_encoder_state, ckpt_path)
                print(f"  Saved best checkpoint → {ckpt_path}")
            encoder.eval(); del head, opt, sch, loader, val_loader, ds, val_ds
        else:
            print("  Skipping training (--skip_train set)")

        # Database: MN40 shared classes at canonical orientation
        print("  Building MN40 database (shared classes)…")
        db_enc, db_lbl = encode_subset(encoder, mn40_test, shared_mn40_idx)
        db_lbl = remapped_mn40_lbl

        angle_results = []
        for theta in ANGLES:
            if theta == 0:
                q_enc = encode_subset(encoder, scan_test, shared_scan_idx)[0]
            else:
                R     = rotation_by_angle(theta, axis="y", device=DEVICE)
                q_enc = encode_subset_rotated(encoder, scan_test, shared_scan_idx, R)

            q_lbl = remapped_scan_lbl
            r1    = recall_at_k(q_enc, db_enc, q_lbl, db_lbl, k=1)
            r5    = recall_at_k(q_enc, db_enc, q_lbl, db_lbl, k=5)
            map5  = mean_ap_at_k(q_enc, db_enc, q_lbl, db_lbl, k=5)
            angle_results.append({"angle": theta, "r1": r1, "r5": r5, "map5": map5})
            print(f"  θ={theta:5.1f}°  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}", flush=True)
            if tb:
                tb.add_scalar("eval/R@1",   r1,   int(theta))
                tb.add_scalar("eval/R@5",   r5,   int(theta))
                tb.add_scalar("eval/mAP@5", map5, int(theta))

        if tb:
            tb.close()
        results[model_name] = angle_results
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("CROSS-DATASET TRANSFER — MN40 (train) → ScanObjectNN (query)")
    print(f"{'Model':<22} {'R@1 (0°)':>10} {'R@1 (90°)':>10} {'R@1 (180°)':>11} {'Drop':>8}")
    print("-" * 65)
    for model_name, angle_res in results.items():
        row = {r["angle"]: r["r1"] for r in angle_res}
        r0, r90, r180 = row.get(0, 0), row.get(90, 0), row.get(180, 0)
        drop = r0 - r180
        print(f"{model_name:<22} {r0:>10.3f} {r90:>10.3f} {r180:>11.3f} {drop:>8.3f}")


if __name__ == "__main__":
    main()
