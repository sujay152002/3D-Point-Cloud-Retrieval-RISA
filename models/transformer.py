"""Point Transformer encoder for point cloud encoding.

Global self-attention with relative position encoding (delta-P).
Outputs a 512-dim global feature via max pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.pos_mlp = nn.Sequential(nn.Conv2d(3, 32, 1), nn.ReLU(), nn.Conv2d(32, c, 1))
        # Hard cap to avoid N×N positional conv blowing up GPU memory
        # For larger point sets we will skip the expensive positional encoding.
        self.max_points_for_posenc = 512

    def forward(self, f, p):
        B, C, N = f.shape
        f_t, p_t = f.permute(0, 2, 1), p.permute(0, 2, 1)
        q, k, v = self.q(f_t), self.k(f_t), self.v(f_t)
        # Full pairwise positional conv is O(N^2) in memory; skip for large N
        if N <= self.max_points_for_posenc:
            rel_p = p.unsqueeze(-1) - p.unsqueeze(2)
            pos_enc = self.pos_mlp(rel_p).sum(1)
        else:
            pos_enc = 0.0
        attn = F.softmax((torch.bmm(q, k.transpose(1, 2)) + pos_enc) / (C**0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2)
        return f + out

class PointTransformerEncoder(nn.Module):
    def __init__(self):
        """
        Point Transformer encoder with multi-scale attention.
        """
        super().__init__()
        
        # Input projection
        self.prep = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Transformer blocks (stack multiple for depth)
        self.blocks = nn.ModuleList([
            PointTransformerBlock(128),
            PointTransformerBlock(128)
        ])
        
        # Output projection
        self.post = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Final FC
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
        # Feature extraction: (B, 3, N) -> (B, 128, N)
        f = self.prep(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            f = block(f, x)
        
        # Global feature: (B, 128, N) -> (B, 512, N)
        f = self.post(f)
        
        # Global max pooling: (B, 512, N) -> (B, 512)
        z = torch.max(f, dim=-1)[0]
        
        # Final projection: (B, 512) -> (B, 512)
        return self.fc(z)