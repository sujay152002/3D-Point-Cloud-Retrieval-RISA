"""Polyformer: frame-distance relative position encoding for point transformers.

Experimental model using quaternion-based relative position representations
between point frames. Not part of the main benchmark pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def difference_vector_rel_pos(points: torch.Tensor) -> torch.Tensor:
    
    # Assumes fist 3 point semantic dimensions are x, y, z coordinates
    xyz = points[:, :3, :]
    rel_p = xyz.unsqueeze(-1) - xyz.unsqueeze(2)
    return rel_p

def frame_distance_rel_pos(points: torch.Tensor) -> torch.Tensor:
    """
    Args:
        points: (B, (>7), N) input point cloud
    
    Returns:
        (B, 5, N, N) relative position representation
    """

    dtype, device, (batch_size, dim_points, num_points) = points.dtype, points.device, points.shape

    assert dim_points >= 7, "Points must have at least 7 semantic dimensions"

    xyz = points[:, :3, :]
    distances = torch.norm(xyz.unsqueeze(-1) - xyz.unsqueeze(2), dim = 1, keepdim = True)

    # Assumes first 3 point semantic dimensions are x, y, z coordinates

    rel_pos_representations = torch.concatenate([
        torch.zeros(shape   = (batch_size, 3, num_points, num_points), dtype = dtype, device = device),
        torch.ones(shape    = (batch_size, 1, num_points, num_points), dtype = dtype, device = device),
        distances
    ], dim = 1)

    for b in range(batch_size):
        quaternions = Rotation.from_quat(points[b, 3:7, :])
        for i in range(num_points):
            for j in range(i + 1, num_points):
                rel_pos_representations[b, :4, i, j] = (quaternions[i].inv() * quaternions[j]).as_quat()

    return rel_pos_representations

class PointTransformerBlock(nn.Module):
    def __init__(
            self,
            c,
            pos_enc_dim = 3,
            pos_repr_function = difference_vector_rel_pos,
            max_points_for_posenc = 512
        ):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.pos_mlp = nn.Sequential(nn.Conv2d(pos_enc_dim, 32, 1), nn.ReLU(), nn.Conv2d(32, c, 1))
        # Hard cap to avoid N×N positional conv blowing up GPU memory
        # For larger point sets we will skip the expensive positional encoding.

        self.max_points_for_posenc = max_points_for_posenc
        self.get_rel_pos = pos_repr_function

    def forward(self, f, p):
        _B, C, N = f.shape
        f_t = f.permute(0, 2, 1)

        q, k, v = self.q(f_t), self.k(f_t), self.v(f_t)

        # Full pairwise positional conv is O(N^2) in memory; skip for large N
        if N <= self.max_points_for_posenc:
            rel_p   = self.get_rel_pos(p)
            pos_enc = self.pos_mlp(rel_p).sum(1)
        else:
            pos_enc = 0.0

        attn = F.softmax((torch.bmm(q, k.transpose(1, 2)) + pos_enc) / (C**0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2)
        return f + out

class PointTransformerEncoder(nn.Module):
    def __init__(self, dim_points : int = 3,):
        """
        Point Transformer encoder with multi-scale attention.
        """
        super().__init__()

        # Input projection
        self.prep = nn.Sequential(
            nn.Conv1d(dim_points, 128, 1),
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