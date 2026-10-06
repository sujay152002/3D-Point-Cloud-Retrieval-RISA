"""Experiment 1 — Rotation-Stratified Retrieval Benchmark

Training: Proxy Anchor loss with SO(3) augmentation — correct IR objective.
Evaluation: proper query/gallery split (query ∩ gallery = ∅).
  Gallery : full test set at canonical orientation
  Query   : held-out val set, queried at rotation angles 0°→180°

Metrics: R@1, R@5, R@10, mAP@5, mAP@10 at each angle.

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp1_retrieval_eval.py [--datasets modelnet40 shapenet] [--n_angles 19]
"""

import argparse
import json
import sys, io
_fd = sys.stdout.fileno()
if _fd >= 0:
    sys.stdout = io.TextIOWrapper(open(_fd, "wb", 0), write_through=True)
from pathlib import Path
from datetime import datetime

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

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024

DATASETS = ["synthetic", "modelnet40", "shapenet", "scanobjectnn"]


# ── Model registry ────────────────────────────────────────────────────────────

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


def _build_encoder(name, cls, checkpoint_dir=None):
    if name == "RISA":
        from models import RotationInvariantSparseAttention
        enc = RotationInvariantSparseAttention(
            encoding_out_dim=512, features_out_dim=256,
            model_dim=256, num_blocks=4,
        )
    else:
        enc = cls()
    if checkpoint_dir:
        import glob
        # match by class name substring, e.g. DGCNNEncoder, PointNet2Encoder
        pattern = str(Path(checkpoint_dir) / f"*_{enc.__class__.__name__}.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            ckpt = matches[-1]  # latest
            enc.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"  Loaded checkpoint: {ckpt}")
        else:
            print(f"  WARNING: no checkpoint found for {name} in {checkpoint_dir}, using random weights")
    return enc


# ── Rotation helpers ──────────────────────────────────────────────────────────

def random_rotation(device):
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotation_by_angle(angle_deg: float, axis: str = "y", device="cpu"):
    """Fixed-axis rotation by a specific angle (for the sweep experiment)."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "y":
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    elif axis == "x":
        R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    else:
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    return R.to(device)


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_dataset(encoder, dataset, batch_size=8):
    """Encode all shapes in dataset. Returns (N, D) encodings and (N,) labels."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
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


@torch.inference_mode()
def encode_rotated(encoder, dataset, R: torch.Tensor, batch_size=8):
    """Encode all shapes after applying rotation R. Returns (N, D) encodings."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_enc = []
    for batch in loader:
        pts = batch[0] if isinstance(batch, (tuple, list)) else batch
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            pts = batch[0]
        pts = pts.float().to(DEVICE)
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        pts = R.to(DEVICE) @ pts          # apply rotation
        enc, _ = unpack_encoder_output(encoder(pts))
        all_enc.append(enc.cpu())
    return torch.cat(all_enc)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(sim: torch.Tensor, query_lbl: torch.Tensor,
                db_lbl: torch.Tensor, k: int = 1) -> float:
    """Recall@k: fraction of queries with a same-class item in top-k.

    sim: [Q, DB] precomputed similarity matrix (query rows, gallery cols).
    Query and gallery are separate — no self-exclusion needed.
    """
    topk_idx = sim.topk(k, dim=-1).indices          # [Q, k]
    hits = 0
    for i in range(len(query_lbl)):
        if (db_lbl[topk_idx[i]] == query_lbl[i]).any():
            hits += 1
    return hits / len(query_lbl)


def mean_ap_at_k(sim: torch.Tensor, query_lbl: torch.Tensor,
                 db_lbl: torch.Tensor, k: int = 5) -> float:
    """mAP@k across all queries.

    AP@k = sum(P@i * rel@i for i in 1..k) / min(R, k)
    where R = total relevant items in the gallery for this query.
    This is the standard IR definition (not capped at retrieved rel count).

    sim: [Q, DB] precomputed similarity matrix.
    """
    aps = []
    for i in range(len(query_lbl)):
        topk_idx = sim[i].topk(k).indices
        rel = (db_lbl[topk_idx] == query_lbl[i]).float()
        # Total relevant in full gallery — correct denominator
        n_relevant = int((db_lbl == query_lbl[i]).sum().item())
        if n_relevant == 0:
            aps.append(0.0)
            continue
        precisions = rel.cumsum(0) / torch.arange(1, k + 1, dtype=torch.float)
        aps.append((precisions * rel).sum().item() / min(n_relevant, k))
    return float(np.mean(aps))


# ── Proxy Anchor loss ────────────────────────────────────────────────────────

class ProxyAnchorLoss(torch.nn.Module):
    """Proxy Anchor loss (Kim et al., CVPR 2020) — proper IR training objective."""
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32.0):
        super().__init__()
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embed_dim))
        torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.margin = margin
        self.alpha  = alpha

    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=1)
        E = F.normalize(embeddings,   dim=1)
        sim      = E @ P.T
        pos_mask = torch.zeros_like(sim).scatter_(1, labels.unsqueeze(1), 1.0)
        neg_mask = 1.0 - pos_mask
        with_pos  = pos_mask.sum(0) > 0
        pos_exp   = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_exp   = torch.exp( self.alpha * (sim + self.margin)) * neg_mask
        loss_pos  = (torch.log(1 + pos_exp.sum(0)) * with_pos).sum()
        loss_neg  = (torch.log(1 + neg_exp.sum(0)) * with_pos).sum()
        return (loss_pos + loss_neg) / with_pos.sum().clamp(min=1)


# ── IR training with Proxy Anchor ─────────────────────────────────────────────

def train_ir(encoder, dataset_name, epochs=100, batch_size=32, seed=42,
             checkpoint_path=None, force_retrain=False):
    """Train encoder with Proxy Anchor loss + SO(3) aug.

    Args:
        checkpoint_path:  if given and the file exists, load and skip training
                          (unless force_retrain=True). If file does not exist,
                          train and save the best checkpoint to that path.
        force_retrain:    ignore existing checkpoint and retrain from scratch,
                          overwriting the checkpoint on completion.
    """
    import torch.optim as optim
    from pathlib import Path as _Path

    if checkpoint_path is not None and not force_retrain:
        ckpt = _Path(checkpoint_path)
        if ckpt.exists():
            encoder.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            encoder.eval()
            print(f"    [train_ir] Loaded checkpoint {ckpt} — skipping training")
            return

    # If force_retrain and checkpoint exists, version the output path
    save_path = _Path(checkpoint_path) if checkpoint_path is not None else None
    if save_path is not None and force_retrain and save_path.exists():
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.with_stem(f"{save_path.stem}_{ts}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    kwargs = {"variant": "OBJ_ONLY"} if dataset_name == "scanobjectnn" else {}
    try:
        ds     = get_dataset(dataset_name, split="train", root=DATA_ROOT,
                             num_points=N_POINTS, **kwargs)
        val_ds = get_dataset(dataset_name, split="val",   root=DATA_ROOT,
                             num_points=N_POINTS, **kwargs)
    except Exception as e:
        print(f"    [train_ir] Cannot load {dataset_name}: {e} — skipping")
        return

    loader     = torch.utils.data.DataLoader(
        ds,     batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    encoder.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 64, device=DEVICE)
        z, _  = unpack_encoder_output(encoder(dummy))
    enc_dim = z.shape[-1]

    _base   = getattr(ds, "dataset", ds)
    num_cls = int(getattr(_base, "num_classes", 40))

    criterion = ProxyAnchorLoss(num_cls, enc_dim, margin=0.1, alpha=32.0).to(DEVICE)
    opt = optim.Adam([
        {"params": encoder.parameters(),   "lr": 1e-4},
        {"params": criterion.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def _rand_rot(B):
        Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
        Q[:, :, 0:1] *= torch.linalg.det(Q).sign().view(B, 1, 1)
        return Q

    best_val_loss   = float("inf")
    best_state      = None

    encoder.train()
    for ep in range(epochs):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            pts = batch[0] if isinstance(batch, (tuple, list)) else batch
            lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                pts, lbl, _ = batch
            pts = pts.float().to(DEVICE)
            lbl = lbl.long().to(DEVICE)
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)
            pts = _rand_rot(pts.shape[0]) @ pts
            opt.zero_grad()
            z, _ = unpack_encoder_output(encoder(pts))
            loss  = criterion(z, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()

        # Val loss every epoch for checkpoint tracking
        val_loss, val_n = 0.0, 0
        encoder.eval()
        with torch.inference_mode():
            for batch in val_loader:
                pts = batch[0] if isinstance(batch, (tuple, list)) else batch
                lbl = batch[1] if isinstance(batch, (tuple, list)) else torch.zeros(pts.shape[0], dtype=torch.long)
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    pts, lbl, _ = batch
                pts = pts.float().to(DEVICE)
                lbl = lbl.long().to(DEVICE)
                if pts.shape[1] != 3:
                    pts = pts.permute(0, 2, 1)
                z, _ = unpack_encoder_output(encoder(pts))
                val_loss += criterion(z, lbl).item()
                val_n    += 1
        val_loss /= max(val_n, 1)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        encoder.train()

        print(f"    epoch {ep+1:3d}/{epochs}  loss={total_loss/max(n_batches,1):.4f}  "
              f"val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
              f"{' ★' if is_best else ''}", flush=True)

    # Restore best and optionally save
    if best_state is not None:
        encoder.load_state_dict(best_state)
        print(f"    [train_ir] Restored best (val_loss={best_val_loss:.4f})")
    if save_path is not None and best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        print(f"    [train_ir] Saved checkpoint → {save_path}")

    encoder.eval()
    del criterion, opt, scheduler


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",  nargs="+", default=DATASETS)
    parser.add_argument("--models",    nargs="+", default=None,
                        help="Subset of models to run, e.g. --models RISA VNN")
    parser.add_argument("--epochs",    type=int,  default=20)
    parser.add_argument("--n_angles",  type=int,  default=19,
                        help="Number of angles from 0 to 180 (inclusive)")
    parser.add_argument("--n_random",  type=int,  default=5,
                        help="Random SO(3) rotations per angle for averaging")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save/load checkpoints. Saves as <dir>/<model>_<dataset>.pt")
    parser.add_argument("--force_retrain",  action="store_true",
                        help="Retrain even if a checkpoint already exists, overwriting it")
    parser.add_argument("--out",       default="outputs/exp1_retrieval.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"Device: {DEVICE}\n")

    angles   = np.linspace(0, 180, args.n_angles).tolist()
    registry = _build_registry()
    if args.models:
        registry = {k: v for k, v in registry.items() if k in args.models}
    results  = {}

    for model_name, model_cls in registry.items():
        print(f"\n{'='*60}\nModel: {model_name}")
        results[model_name] = {}

        for ds_name in args.datasets:
            print(f"\n  Dataset: {ds_name}")
            kwargs = {"variant": "OBJ_ONLY"} if ds_name == "scanobjectnn" else {}

            try:
                gallery_ds = get_dataset(ds_name, split="test", root=DATA_ROOT,
                                         num_points=N_POINTS, **kwargs)
            except Exception as e:
                print(f"  Cannot load {ds_name}: {e} — skipping")
                continue

            encoder = _build_encoder(model_name, model_cls).to(DEVICE)
            ckpt_path = (
                str(Path(args.checkpoint_dir) / f"{model_name}_{ds_name}.pt")
                if args.checkpoint_dir else None
            )
            train_ir(encoder, ds_name, epochs=args.epochs, seed=args.seed,
                     checkpoint_path=ckpt_path, force_retrain=args.force_retrain)

            # ── Proper query / gallery split ──────────────────────────────
            # Gallery: full test set at canonical orientation
            # Query:   held-out val set (never seen during training, disjoint from gallery)
            # Query ∩ Gallery = ∅  by construction via get_dataset splits
            print("  Building gallery (test set)...", flush=True)
            gallery_ds  = get_dataset(ds_name, split="test", root=DATA_ROOT,
                                      num_points=N_POINTS, **kwargs)
            query_ds    = get_dataset(ds_name, split="val",  root=DATA_ROOT,
                                      num_points=N_POINTS, **kwargs)

            db_enc,  db_lbl  = encode_dataset(encoder, gallery_ds)
            q_enc_0, q_lbl   = encode_dataset(encoder, query_ds)

            # Normalise once — reused across all angles
            db_enc_n = F.normalize(db_enc, dim=-1)

            angle_results = []
            for theta in angles:
                r1_vals, r5_vals, r10_vals, map5_vals, map10_vals = [], [], [], [], []
                for _ in range(args.n_random):
                    if theta == 0:
                        q_enc = q_enc_0
                    else:
                        # Sample a random SO(3) rotation with the fixed angle
                        # by composing a random axis with the given angle magnitude,
                        # giving uniform coverage of SO(3) at each angle stratum.
                        axis = torch.nn.functional.normalize(
                            torch.randn(3, device=DEVICE), dim=0
                        )
                        rad  = np.radians(theta)
                        c, s = float(np.cos(rad)), float(np.sin(rad))
                        # Rodrigues' rotation formula
                        K = torch.tensor([
                            [0,        -axis[2],  axis[1]],
                            [ axis[2],  0,        -axis[0]],
                            [-axis[1],  axis[0],   0],
                        ], device=DEVICE)
                        R = (c * torch.eye(3, device=DEVICE)
                             + s * K
                             + (1 - c) * torch.outer(axis, axis))
                        q_enc = encode_rotated(encoder, query_ds, R)
                    q_enc_n = F.normalize(q_enc, dim=-1)
                    sim     = q_enc_n @ db_enc_n.T          # [Q, DB]

                    r1_vals.append(recall_at_k(sim, q_lbl, db_lbl, k=1))
                    r5_vals.append(recall_at_k(sim, q_lbl, db_lbl, k=5))
                    r10_vals.append(recall_at_k(sim, q_lbl, db_lbl, k=10))
                    map5_vals.append(mean_ap_at_k(sim, q_lbl, db_lbl, k=5))
                    map10_vals.append(mean_ap_at_k(sim, q_lbl, db_lbl, k=10))

                angle_results.append({
                    "angle":  theta,
                    "r1":     float(np.mean(r1_vals)),
                    "r5":     float(np.mean(r5_vals)),
                    "r10":    float(np.mean(r10_vals)),
                    "map5":   float(np.mean(map5_vals)),
                    "map10":  float(np.mean(map10_vals)),
                })
                print(f"    θ={theta:6.1f}°  R@1={np.mean(r1_vals):.3f}  "
                      f"R@5={np.mean(r5_vals):.3f}  R@10={np.mean(r10_vals):.3f}  "
                      f"mAP@5={np.mean(map5_vals):.3f}  mAP@10={np.mean(map10_vals):.3f}",
                      flush=True)

            results[model_name][ds_name] = angle_results
            del encoder
            torch.cuda.empty_cache()
    if _TB_AVAILABLE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb = _TBWriter(log_dir=f"runs/experiment1/{ts}")
        for model_name, ds_dict in results.items():
            for ds_name, angle_results in ds_dict.items():
                for entry in angle_results:
                    step = int(entry["angle"])
                    tb.add_scalar(f"{model_name}/{ds_name}/R@1",   entry["r1"],   step)
                    tb.add_scalar(f"{model_name}/{ds_name}/R@5",   entry["r5"],   step)
                    tb.add_scalar(f"{model_name}/{ds_name}/mAP@5", entry["map5"], step)
        tb.close()
        print(f"TensorBoard logs → runs/experiment1/{ts}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
