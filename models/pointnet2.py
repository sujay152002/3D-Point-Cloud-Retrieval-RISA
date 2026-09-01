"""PointNet++ encoder for point cloud encoding.

Hierarchical local feature aggregation via set abstraction layers (SA1→SA2→SA3).
Outputs a 512-dim global feature via farthest point sampling and ball query.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    
    Args:
        src: (B, N, 3)
        dst: (B, M, 3)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint, seed=None):
    """
    Farthest Point Sampling

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample
        seed: optional seed for reproducibility (default: 42 for determinism)
    Returns:
        centroids: (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Set seed for reproducibility (using CPU generator then moving result)
    if seed is not None:
        torch.manual_seed(seed)
    
    # Use torch.randperm for deterministic random selection on any device
    # First point selection: use seed if provided, otherwise use first point
    if seed is not None:
        # Create a seeded random selection - generate on CPU then move to device
        farthest = torch.randint(0, N, (B,), generator=torch.Generator().manual_seed(seed))
        farthest = farthest.to(device)
    else:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    
    return centroids


def index_points(points, idx):
    """
    Index points according to idx
    
    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)
    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query
    
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
    Returns:
        group_idx: (B, S, nsample) grouped points index
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        """
        PointNet Set Abstraction Layer
        
        Args:
            npoint: Number of points to sample (None for global pooling)
            radius: Ball query radius
            nsample: Number of points in each local region
            in_channel: Input feature dimension
            mlp: List of MLP dimensions
            group_all: Whether to group all points (global pooling)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, 3, N) or (B, N, 3) point positions
            points: (B, C, N) point features (or None)
        
        Returns:
            new_xyz: (B, 3, S) sampled point positions
            new_points: (B, C', S) aggregated features
        """
        # Ensure xyz is (B, N, 3)
        if xyz.size(1) == 3:
            xyz = xyz.permute(0, 2, 1)  # (B, 3, N) -> (B, N, 3)
        
        B, N, C = xyz.shape
        
        if self.group_all:
            # Global pooling
            new_xyz = torch.zeros(B, 1, C, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, C)
            
            if points is not None:
                # points: (B, C_in, N) -> (B, N, C_in)
                points_t = points.permute(0, 2, 1)
                grouped_points = torch.cat([grouped_xyz, points_t.unsqueeze(1)], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            # Farthest point sampling
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            
            # Ball query
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            
            # Group points
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)  # Relative coordinates
            
            if points is not None:
                points_t = points.permute(0, 2, 1)  # (B, C_in, N) -> (B, N, C_in)
                grouped_points = index_points(points_t, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        
        # grouped_points: (B, npoint, nsample, C_in+3)
        # Permute to (B, C_in+3, nsample, npoint)
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        
        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        
        # Max pooling: (B, C_out, nsample, npoint) -> (B, C_out, npoint)
        new_points = torch.max(grouped_points, 2)[0]
        
        # Return: (B, 3, npoint), (B, C_out, npoint)
        return new_xyz.permute(0, 2, 1), new_points


class PointNet2Encoder(nn.Module):
    def __init__(self):
        """
        PointNet++ encoder for point cloud classification.
        
        Architecture:
        - SA1: 1024 points -> 512 points, [64, 64, 128] features
        - SA2: 512 points -> 128 points, [128, 128, 256] features  
        - SA3: Global pooling, [256, 512, 1024] features
        - FC: 1024 -> 512
        """
        super().__init__()
        
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,  # xyz coordinates
            mlp=[64, 64, 128],
            group_all=False
        )
        
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,  # features + xyz
            mlp=[128, 128, 256],
            group_all=False
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,  # features + xyz
            mlp=[256, 512, 1024],
            group_all=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        B, _, N = x.shape
        
        # SA1: (B, 3, N) -> (B, 3, 512), (B, 128, 512)
        l1_xyz, l1_points = self.sa1(x, None)
        
        # SA2: (B, 3, 512), (B, 128, 512) -> (B, 3, 128), (B, 256, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # SA3: (B, 3, 128), (B, 256, 128) -> (B, 3, 1), (B, 1024, 1)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global feature: (B, 1024, 1) -> (B, 1024)
        x = l3_points.squeeze(-1)
        
        # Final FC: (B, 1024) -> (B, 512)
        x = self.fc(x)
        
        return x