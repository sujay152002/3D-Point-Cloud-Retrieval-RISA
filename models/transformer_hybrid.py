"""
HYBRID TRANSFORMER: SE3-style invariant attention on raw features
with skip connections for better geometry preservation.

Architecture:
  Input → Raw Features (CNN) → Invariant Attention → + Skip Connection → Output
         → Skip (xyz) ↗

This combines rotation-invariant attention pattern with raw geometry preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_invariant import (
    GeometricInvariantExtractor,
    MultiScaleInvariantAttention,
    farthest_point_sample,
    compute_local_frames,
)
from .transformer import PointTransformerBlock


class HybridEncoderBlock(nn.Module):
    """
    Single hybrid block with invariant attention.
    """
    def __init__(self, c, k=16, n_global=32):
        super().__init__()
        self.c = c
        self.k = k
        self.n_global = n_global

        self.q = nn.Linear(c, c)
        self.k_linear = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.q_global = nn.Linear(c, c)
        self.k_global = nn.Linear(c, c)

        self.pos_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.norm_attn = nn.LayerNorm(c)
        self.norm_ff = nn.LayerNorm(c)
        self.ff = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.GELU(),
            nn.Linear(c * 2, c)
        )

    def forward(self, f, p, eigenvecs=None, eigenvals=None, use_invariant_pos=True):
        B, C, N = f.shape
        f_t = f.transpose(1, 2)

        f_norm = self.norm_attn(f_t)
        q = self.q(f_norm)
        k_val = self.k_linear(f_norm)
        v = self.v(f_norm)

        with torch.no_grad():
            dist = torch.cdist(p.transpose(1, 2), p.transpose(1, 2))
            _, idx = dist.topk(self.k, dim=-1, largest=False)

        batch_idx = torch.arange(B, device=f.device).view(B, 1, 1).expand(-1, N, self.k)

        if use_invariant_pos and eigenvecs is not None:
            from .transformer_invariant import compute_pair_pos_encoding
            with torch.no_grad():
                f_pos_ij = compute_pair_pos_encoding(p, eigenvecs, eigenvals, idx).detach()
            pos_bias_local = self.pos_mlp(f_pos_ij).squeeze(-1)
        else:
            pos_bias_local = 0.0

        q_expanded = q.unsqueeze(2)
        k_gathered = k_val[batch_idx, idx]
        v_gathered = v[batch_idx, idx]

        attn_scores = (q_expanded * k_gathered).sum(dim=-1, keepdim=True) / (C ** 0.5) + pos_bias_local.unsqueeze(-1)
        attn = F.softmax(attn_scores, dim=2)
        local_out = (attn * v_gathered).sum(dim=2)

        fps_idx = farthest_point_sample(p.transpose(1, 2), self.n_global)
        fps_features = f_norm.gather(dim=1, index=fps_idx.unsqueeze(-1).expand(-1, -1, C))
        q_global = self.q_global(f_norm)
        k_global = self.k_global(fps_features)
        v_global = self.v(fps_features)
        global_attn = torch.bmm(q_global, k_global.transpose(1, 2)) / (C ** 0.5)
        global_attn = F.softmax(global_attn, dim=-1)
        global_out = torch.bmm(global_attn, v_global)

        combined = local_out + 0.5 * global_out
        f_t = f_t + combined

        f_t = f_t + self.ff(self.norm_ff(f_t))

        return f_t.transpose(1, 2)


class PointTransformerHybridEncoder(nn.Module):
    """
    Hybrid encoder: uses SE3-style invariant attention on raw features.
    Adds skip connections for better geometry preservation.
    """
    def __init__(self):
        super().__init__()
        
        # Raw feature path (preserves geometry)
        self.prep = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Skip connection: project raw xyz to same dimension
        self.skip_conv = nn.Conv1d(3, 128, 1)

        # SE3-style invariant feature extractor for attention pattern
        self.geometric_prep = GeometricInvariantExtractor(128, n_global=64)

        # Blocks with invariant attention on raw features
        self.blocks = nn.ModuleList([
            HybridEncoderBlock(128, k=16, n_global=32),
            HybridEncoderBlock(128, k=16, n_global=32)
        ])

        self.post = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) input point cloud
        
        Returns:
            (B, 512) global feature vector
        """
        # Raw features for reconstruction
        f_raw = self.prep(x)  # [B, 128, N]
        
        # Skip connection: embed raw xyz
        xyz_skip = self.skip_conv(x)  # [B, 128, N]

        # Get invariant features for attention pattern
        c = x.mean(dim=-1, keepdim=True)
        x_centered = x - c
        
        with torch.no_grad():
            eigenvecs, eigenvals = compute_local_frames(x_centered, k=16)
        
        f_inv = self.geometric_prep(x_centered)

        # Use invariant attention on raw features
        for block in self.blocks:
            f_raw = block(f_raw, x_centered, eigenvecs, eigenvals, use_invariant_pos=True)

        # Add skip connection for geometry preservation
        f_raw = f_raw + 0.3 * xyz_skip

        f = self.post(f_raw)
        z = torch.max(f, dim=-1)[0]

        return self.fc(z)

    def forward_seg(self, x):
        """Returns (global_feat [B,512], per_point_feat [B,512,N]) for segmentation."""
        f_raw = self.prep(x)
        
        xyz_skip = self.skip_conv(x)
        c = x.mean(dim=-1, keepdim=True)
        x_centered = x - c
        
        with torch.no_grad():
            eigenvecs, eigenvals = compute_local_frames(x_centered, k=16)
        
        f_inv = self.geometric_prep(x_centered)

        for block in self.blocks:
            f_raw = block(f_raw, x_centered, eigenvecs, eigenvals, use_invariant_pos=True)

        f_raw = f_raw + 0.3 * xyz_skip

        f = self.post(f_raw)
        z = torch.max(f, dim=-1)[0]
        z = self.fc(z)

        return z, f


PointTransformerHybrid = PointTransformerHybridEncoder