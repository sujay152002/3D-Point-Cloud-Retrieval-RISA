"""Synthetic point cloud generation and geometric transformation utilities.

Provides shape generators (sphere, cube, chair), normalization, rotation,
shuffling, sorting, and random SO(3) sampling used across all experiments.
"""
import torch
import numpy as np

def normalize_pc(pc):
    """Center and scale to unit sphere."""
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    scale = torch.max(torch.norm(pc, dim=-1, keepdim=True), dim=1, keepdim=True)[0]
    return pc / (scale + 1e-8)

def rotate_pc(pc, angle, axis='z'):
    device = pc.device
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)
    if axis == 'z':
        R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    elif axis == 'y':
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    else: # x
        R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    R = R.to(device)
    return torch.matmul(pc, R.T)

def random_rotation_3d(pc):
    """Apply random SO(3) rotation using QR decomposition.
    
    More efficient than sequential axis rotations and provides
    uniform sampling over SO(3).
    """
    device = pc.device
    # Generate random 3x3 matrix and QR decompose
    random_matrix = torch.randn(3, 3, dtype=torch.float32)
    Q, R = torch.linalg.qr(random_matrix)
    
    # Ensure proper rotation (det = 1, not -1 for reflection)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    
    Q = Q.to(device)
    return torch.matmul(pc, Q.T)

def shuffle_pc(pc):
    """Randomly shuffles point order."""
    idx = torch.randperm(pc.shape[1])
    return pc[:, idx, :]

def sort_pc(pc):
    """Sorts points lexicographically (x, then y, then z) to create canonical order."""
    # Stable multi-key sort to avoid precision issues
    B, N, C = pc.shape
    device = pc.device
    
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

def generate_chair(num_batch, n=1024):
    pts = []
    for _ in range(num_batch):
        n_seat = int(n * 0.4)
        n_back = int(n * 0.3)
        n_legs = n - n_seat - n_back
        seat = np.random.rand(n_seat, 3) * [1, 1, 0.1] - [0.5, 0.5, 0]
        back = np.random.rand(n_back, 3) * [1, 0.1, 1] - [0.5, 0.5, 0]
        legs = np.random.rand(n_legs, 3) * [1, 1, 0.5] - [0.5, 0.5, 0.5]
        pts.append(np.concatenate([seat, back, legs], axis=0))
    return torch.tensor(np.stack(pts), dtype=torch.float32)

def generate_sphere(num_batch, n=1024):
    v = torch.randn(num_batch, n, 3)
    return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

def generate_cube(num_batch, n=1024):
    return torch.rand(num_batch, n, 3) * 2 - 1