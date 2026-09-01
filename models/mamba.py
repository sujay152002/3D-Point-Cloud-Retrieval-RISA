"""Point Mamba encoder for point cloud encoding.

Applies a simplified selective state-space model (Mamba-style SSM) over
lexicographically sorted point sequences. Outputs a 512-dim global feature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointMambaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d * 3)
        # Initialize A with small values for stability
        self.A = nn.Parameter(torch.randn(d, d) * 0.01)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Simplified Mamba-style selective SSM.
        
        Args:
            x: (B, N, D) input features
        
        Returns:
            (B, N, D) output features
        """
        B, N, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outs = []
        
        for t in range(N):
            params = self.proj(x[:, t, :])
            b, c, delta = torch.split(params, D, dim=-1)
            
            # Selective state update with gating
            delta = torch.sigmoid(delta)  # Gate in [0,1]
            h = torch.tanh(torch.matmul(h, self.A) * delta + b)
            
            # Output with skip connection
            out = c * h
            outs.append(out.unsqueeze(1))
        
        result = torch.cat(outs, dim=1)
        return self.layer_norm(result + x)  # Residual connection

class PointMambaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mamba = PointMambaBlock(64)
        self.post_mamba_bn = nn.BatchNorm1d(64)
        self.out_proj = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512)
        )

    def _lexicographic_sort(self, pc):
        """
        Stable lexicographic sort (x, then y, then z).
        
        Args:
            pc: (B, N, 3) point cloud
        
        Returns:
            (B, N, 3) sorted point cloud
        """
        B, N, C = pc.shape
        
        # Sort by z (least significant)
        _, idx = torch.sort(pc[:, :, 2], dim=1, stable=True)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)
        pc = torch.gather(pc, 1, idx_exp)
        
        # Sort by y
        _, idx = torch.sort(pc[:, :, 1], dim=1, stable=True)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)
        pc = torch.gather(pc, 1, idx_exp)
        
        # Sort by x (most significant)
        _, idx = torch.sort(pc[:, :, 0], dim=1, stable=True)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)
        pc = torch.gather(pc, 1, idx_exp)
        
        return pc

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) input point cloud
        
        Returns:
            (B, 512) global feature vector
        """
        # x: (B, 3, N) -> (B, N, 3)
        x_transposed = x.permute(0, 2, 1)
        
        # Stable multi-key lexicographic sort
        x_sorted = self._lexicographic_sort(x_transposed)
        
        # Back to (B, 3, N)
        x_sorted = x_sorted.permute(0, 2, 1)
        
        # Feature extraction
        f = self.in_proj(x_sorted).transpose(1, 2)  # (B, N, 64)
        f = self.mamba(f)  # (B, N, 64)
        
        # Global max pooling
        z = torch.max(f, dim=1)[0]  # (B, 64)
        z = self.post_mamba_bn(z)
        
        return self.out_proj(z)  # (B, 512)