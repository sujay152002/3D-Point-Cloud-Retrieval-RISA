"""Experiment 3 — Ablation: What Drives RISA's Retrieval Gain?

Compares four conditions to isolate whether architectural invariance
beats learned/test-time invariance:

  A. RISA                          — invariant by architecture
  B. PointNet++ + SO(3) train aug  — approximately invariant by training
  C. PointNet++ + TTA voting       — invariant at inference (12 rotations)
  D. RISA w/o invariant features   — sparse attention only, XYZ input (no PCA/quaternion)

If A >> B,C: architectural invariance beats learned invariance → key finding
If A >> D:   the geometric features (PCA/quaternion) are what matters, not just sparsity

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/exp3_ablation.py [--dataset modelnet40]
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset
from core.trainer import unpack_encoder_output
from experiments.exp1_retrieval_eval import (
    encode_dataset, encode_rotated, recall_at_k, mean_ap_at_k,
    train_ir,
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ANGLES    = [0, 15, 30, 45, 60, 90, 120, 150, 180]


# ── Condition D: RISA with XYZ input instead of invariant geometric features ──

class RISAXYZBaseline(nn.Module):
    """RISA architecture but replaces GeometricInvariantExtractor with raw XYZ.
    Keeps the sparse attention blocks identical — isolates the feature contribution."""

    name = "RISA-XYZ"

    def __init__(self, model_dim=128, num_blocks=2, encoding_out_dim=512):
        super().__init__()
        from models.risa import SparsePointCloudAttention
        self.model_dim        = model_dim
        self.encoding_out_dim = encoding_out_dim

        # Replace invariant features with raw XYZ (3-dim)
        self.feature_embed = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.blocks = nn.ModuleList([
            SparsePointCloudAttention(
                model_dim=model_dim, out_dim=model_dim, attn_augment="none"
            )
            for _ in range(num_blocks)
        ])
        self.encoding_token = nn.Parameter(torch.empty(1, model_dim, 1))
        nn.init.xavier_uniform_(self.encoding_token)

        self.features_post = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1),
            nn.BatchNorm1d(model_dim), nn.GELU(),
        )
        self.encoding_post = nn.Sequential(
            nn.Conv1d(model_dim, encoding_out_dim, 1),
            nn.BatchNorm1d(encoding_out_dim), nn.GELU(),
        )

    def forward(self, x):
        B, _, N = x.shape
        # x: (B, 3, N) — use raw XYZ as features
        features = self.feature_embed(
            x.permute(0, 2, 1).reshape(B * N, 3)
        ).reshape(B, N, self.model_dim).permute(0, 2, 1)   # (B, C, N)

        # Sparse k-NN attention (k=48) — same budget as RISA's num_local+num_global
        # Full N×N attention OOMs at N=1024, B=8 (8GB tensor)
        k = min(48, N - 1)
        xyz_T = x.permute(0, 2, 1)                          # (B, N, 3)
        dist  = torch.cdist(xyz_T, xyz_T)                   # (B, N, N)
        indices = dist.topk(k, dim=-1, largest=False).indices  # (B, N, k)
        enc_token = self.encoding_token.expand(B, -1, -1)

        for block in self.blocks:
            features, enc_token = block(features, enc_token=enc_token,
                                        indices=indices, pos_enc=None)

        features  = self.features_post(features)
        encoding  = self.encoding_post(enc_token)
        return encoding, features


# ── TTA wrapper ───────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_with_tta(encoder, dataset, n_votes=12, batch_size=8):
    """Test-time augmentation: average encodings over n_votes random rotations."""
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

        B = pts.shape[0]
        vote_enc = None
        for _ in range(n_votes):
            Q, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
            Q[:, :, 0] *= torch.linalg.det(Q).sign().view(B, 1)
            pts_r = Q @ pts
            enc, _ = unpack_encoder_output(encoder(pts_r))
            vote_enc = enc if vote_enc is None else vote_enc + enc
        all_enc.append((vote_enc / n_votes).cpu())
    return torch.cat(all_enc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default="modelnet40")
    parser.add_argument("--epochs",          type=int, default=20)
    parser.add_argument("--n_votes",         type=int, default=12,
                        help="TTA votes for condition C")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--checkpoint_dir",  type=str, default=None,
                        help="Directory to save/load checkpoints. Saves as <dir>/<cond>.pt")
    parser.add_argument("--force_retrain",   action="store_true",
                        help="Retrain even if a checkpoint already exists, overwriting it")
    parser.add_argument("--out",             default="outputs/exp3_ablation.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"Device: {DEVICE}")

    kwargs  = {"variant": "OBJ_ONLY"} if args.dataset == "scanobjectnn" else {}
    test_ds = get_dataset(args.dataset, split="test",  root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)
    val_ds  = get_dataset(args.dataset, split="val",   root=DATA_ROOT,
                          num_points=N_POINTS, **kwargs)

    from models import RotationInvariantSparseAttention, DGCNNEncoder

    conditions = {
        "A_RISA"           : RotationInvariantSparseAttention(
                                 encoding_out_dim=512, features_out_dim=256,
                                 model_dim=256, num_blocks=4),
        "B_DGCNN_SO3"      : DGCNNEncoder(),
        "C_DGCNN_TTA"      : DGCNNEncoder(),
        "D_RISA_XYZ"       : RISAXYZBaseline(
                                 model_dim=256, num_blocks=4, encoding_out_dim=512),
        "E_RISA_NO_TOKEN"  : RotationInvariantSparseAttention(
                                 encoding_out_dim=512, features_out_dim=256,
                                 model_dim=256, num_blocks=4,
                                 encoding_method="mean"),
    }

    results = {}

    for cond_name, encoder in conditions.items():
        print(f"\n{'='*60}\nCondition: {cond_name}")
        encoder = encoder.to(DEVICE)

        # All conditions trained with Proxy Anchor + SO(3) augmentation
        ckpt_path = (
            str(Path(ROOT / args.checkpoint_dir) / f"{cond_name}.pt")
            if args.checkpoint_dir else None
        )
        train_ir(encoder, args.dataset, epochs=args.epochs, seed=args.seed,
                 checkpoint_path=ckpt_path, force_retrain=args.force_retrain)

        # gallery = test set (canonical), query = val set (disjoint)
        db_enc, db_lbl = encode_dataset(encoder, test_ds)
        q0_enc, q_lbl  = encode_dataset(encoder, val_ds)

        angle_results = []
        db_enc_n = F.normalize(db_enc, dim=1)
        for theta in ANGLES:
            r1_vals, r5_vals, r10_vals, map5_vals, map10_vals = [], [], [], [], []
            n_trials = 1 if theta == 0 else 5  # average over 5 random axes per angle
            for _ in range(n_trials):
                if theta == 0:
                    if cond_name == "C_DGCNN_TTA":
                        q_enc = encode_with_tta(encoder, val_ds, n_votes=args.n_votes)
                    else:
                        q_enc = q0_enc
                else:
                    # Random-axis Rodrigues rotation at fixed angle magnitude
                    axis = F.normalize(torch.randn(3, device=DEVICE), dim=0)
                    rad  = np.radians(theta)
                    c, s = float(np.cos(rad)), float(np.sin(rad))
                    K = torch.tensor([
                        [0,        -axis[2],  axis[1]],
                        [ axis[2],  0,        -axis[0]],
                        [-axis[1],  axis[0],   0],
                    ], device=DEVICE)
                    R = (c * torch.eye(3, device=DEVICE)
                         + s * K
                         + (1 - c) * torch.outer(axis, axis))
                    if cond_name == "C_DGCNN_TTA":
                        loader = torch.utils.data.DataLoader(
                            val_ds, batch_size=8, shuffle=False, drop_last=False
                        )
                        all_enc = []
                        with torch.inference_mode():
                            for batch in loader:
                                pts = batch[0] if isinstance(batch, (tuple, list)) else batch
                                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                                    pts = batch[0]
                                pts = pts.float().to(DEVICE)
                                if pts.shape[1] != 3:
                                    pts = pts.permute(0, 2, 1)
                                pts = R @ pts
                                B = pts.shape[0]
                                vote_enc = None
                                for _ in range(args.n_votes):
                                    Qv, _ = torch.linalg.qr(torch.randn(B, 3, 3, device=DEVICE))
                                    Qv[:, :, 0] *= torch.linalg.det(Qv).sign().view(B, 1)
                                    pts_r = Qv @ pts
                                    enc, _ = unpack_encoder_output(encoder(pts_r))
                                    vote_enc = enc if vote_enc is None else vote_enc + enc
                                all_enc.append((vote_enc / args.n_votes).cpu())
                        q_enc = torch.cat(all_enc)
                    else:
                        q_enc = encode_rotated(encoder, val_ds, R)

                q_n = F.normalize(q_enc, dim=1)
                sim = q_n @ db_enc_n.T
                r1_vals.append(recall_at_k(sim, q_lbl, db_lbl, k=1))
                r5_vals.append(recall_at_k(sim, q_lbl, db_lbl, k=5))
                r10_vals.append(recall_at_k(sim, q_lbl, db_lbl, k=10))
                map5_vals.append(mean_ap_at_k(sim, q_lbl, db_lbl, k=5))
                map10_vals.append(mean_ap_at_k(sim, q_lbl, db_lbl, k=10))

            r1, r5, r10 = np.mean(r1_vals), np.mean(r5_vals), np.mean(r10_vals)
            map5, map10 = np.mean(map5_vals), np.mean(map10_vals)
            angle_results.append({"angle": theta, "r1": r1, "r5": r5, "r10": r10,
                                   "map5": map5, "map10": map10})
            print(f"  θ={theta:5.1f}°  R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}"
                  f"  mAP@5={map5:.3f}  mAP@10={map10:.3f}",
                  flush=True)

        results[cond_name] = angle_results
        del encoder
        torch.cuda.empty_cache()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY — {args.dataset} — Recall@1 at key angles")
    print(f"{'Condition':<22} {'0°':>6} {'45°':>6} {'90°':>6} {'180°':>6}")
    print("-" * 54)
    key_angles = {0, 45, 90, 180}
    for cond, angle_res in results.items():
        row = {r["angle"]: r["r1"] for r in angle_res}
        vals = [row.get(a, float("nan")) for a in sorted(key_angles)]
        print(f"{cond:<22} {vals[0]:>6.3f} {vals[1]:>6.3f} {vals[2]:>6.3f} {vals[3]:>6.3f}")


if __name__ == "__main__":
    main()
