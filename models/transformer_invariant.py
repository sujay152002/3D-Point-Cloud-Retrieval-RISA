"""
MULTI-SCALE INVARIANT TRANSFORMER - Improved Version
Changes:
1. k-NN uses median instead of max
2. PCA-based 3 angles for rotation invariance
3. Multi-scale attention: local (k=32) + global via FPS (n_global=128)
4. Global tokens computed via FPS to capture whole-pointcloud structure
5. Per-pair local frame alignment positional encoding (10-dim f_pos_ij)
   - Local PCA frame per point from k-NN neighborhood
   - Relative rotation between frames encoded as quaternion (4 components)
   - Eigenvalue ratios (3) + distances (3) = 10-dim invariant per-pair feature
   - Computed ONCE at encoder entry and reused across all attention layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

torch.backends.cudnn.benchmark = True


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling with geometry-based tie-breaking.
    - Seed: point farthest from centroid (rotation-invariant, no index dependency)
    - Tie-break: when two points are equidistant from the selected set,
      prefer the one farthest from the global centroid (rotation-invariant secondary criterion)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Rotation-invariant seed: point farthest from mean centroid
    global_centroid = xyz.mean(dim=1, keepdim=True)                    # [B, 1, 3]
    d_from_centroid = torch.norm(xyz - global_centroid, dim=-1)        # [B, N]
    farthest = d_from_centroid.argmax(dim=-1)                          # [B]

    batch_idx = torch.arange(B, device=device)
    tie_break_eps = 1e-6  # threshold to detect ties in floating point

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)                    # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]

        # Find max distance value per batch
        max_dist = distance.max(dim=-1, keepdim=True)[0]               # [B, 1]

        # All points within eps of the max are considered tied
        tied = (distance >= max_dist - tie_break_eps)                  # [B, N] bool

        # Among tied points, pick the one farthest from global centroid
        # For non-tied points, set score to -inf so they are never picked
        tie_score = torch.where(tied, d_from_centroid,
                                torch.full_like(d_from_centroid, -1e10))
        farthest = tie_score.argmax(dim=-1)                            # [B]

    return centroids


def compute_local_frames(p, k=32):
    """
    Compute per-point local PCA frames from k-NN neighborhoods.
    Computed once at encoder entry and reused in all attention layers.
    Memory-efficient: processes points in chunks to avoid OOM.

    Args:
        p: [B, 3, N] centered point cloud
    Returns:
        eigenvecs: [B, N, 3, 3] local frames, columns are v1,v2,v3 descending
        eigenvals: [B, N, 3]    eigenvalues sorted descending
    """
    B, _, N = p.shape
    p_t = p.transpose(1, 2)                                        # [B, N, 3]
    k = min(k, N)

    all_eigenvecs = torch.zeros(B, N, 3, 3, device=p.device)
    all_eigenvals = torch.zeros(B, N, 3, device=p.device)

    # Process each batch item separately to avoid B*N*N distance matrix OOM
    for b in range(B):
        pts = p_t[b]                                               # [N, 3]
        dist_mat = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)  # [N, N]
        _, idx = dist_mat.topk(k, dim=-1, largest=False)           # [N, k]
        neighbors = pts[idx]                                       # [N, k, 3]
        neighbors_centered = neighbors - pts.unsqueeze(1)          # [N, k, 3]
        cov = torch.einsum('nki,nkj->nij', neighbors_centered, neighbors_centered) / k  # [N, 3, 3]
        cov = cov + torch.eye(3, device=p.device) * 1e-6
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)         # [N,3], [N,3,3]
        idx_sort = torch.argsort(eigenvalues, dim=-1, descending=True)
        eigenvalues = torch.gather(eigenvalues, 1, idx_sort)
        eigenvectors = torch.gather(
            eigenvectors, 1,
            idx_sort.unsqueeze(-1).expand(-1, -1, 3)
        )
        all_eigenvecs[b] = eigenvectors
        all_eigenvals[b] = eigenvalues

    return all_eigenvecs, all_eigenvals


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrices to unit quaternions [qx, qy, qz, qw].
    Args:
        R: [B, N, k, 3, 3]
    Returns:
        q: [B, N, k, 4]
    """
    shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)                                   # [M, 3, 3]
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]  # [M]
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=1e-10) / 4.0)
    qx = torch.sqrt(torch.clamp(1.0 + R_flat[:, 0, 0] - R_flat[:, 1, 1] - R_flat[:, 2, 2], min=1e-10) / 4.0)
    qy = torch.sqrt(torch.clamp(1.0 - R_flat[:, 0, 0] + R_flat[:, 1, 1] - R_flat[:, 2, 2], min=1e-10) / 4.0)
    qz = torch.sqrt(torch.clamp(1.0 - R_flat[:, 0, 0] - R_flat[:, 1, 1] + R_flat[:, 2, 2], min=1e-10) / 4.0)
    sign_x = torch.sign(R_flat[:, 2, 1] - R_flat[:, 1, 2])
    sign_y = torch.sign(R_flat[:, 0, 2] - R_flat[:, 2, 0])
    sign_z = torch.sign(R_flat[:, 1, 0] - R_flat[:, 0, 1])
    qx = qx * torch.where(sign_x >= 0, torch.ones_like(sign_x), -torch.ones_like(sign_x))
    qy = qy * torch.where(sign_y >= 0, torch.ones_like(sign_y), -torch.ones_like(sign_y))
    qz = qz * torch.where(sign_z >= 0, torch.ones_like(sign_z), -torch.ones_like(sign_z))
    q = torch.stack([qx, qy, qz, qw], dim=-1)                     # [M, 4]
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    return q.reshape(*shape, 4)


def compute_pair_pos_encoding(p, eigenvecs, eigenvals, idx):
    """
    Compute 10-dim per-pair invariant positional encoding f_pos_ij.

    Components:
      3 distances:        d_i, d_j, d_ij
      3 eigenvalue ratios: ev_i / ev_j  (local geometry comparison)
      4 quaternion:       relative rotation from frame_i to frame_j

    All rotation-invariant by construction.

    Args:
        p:         [B, 3, N]
        eigenvecs: [B, N, 3, 3]
        eigenvals: [B, N, 3]
        idx:       [B, N, k] k-NN indices
    Returns:
        f_pos_ij:  [B, N, k, 10]
    """
    B, _, N = p.shape
    k = idx.shape[-1]
    p_t = p.transpose(1, 2)                                        # [B, N, 3]
    batch_idx = torch.arange(B, device=p.device).view(B, 1, 1).expand(-1, N, k)

    # Distances
    d_i = torch.norm(p_t, dim=-1)                                  # [B, N]
    d_j = d_i.gather(1, idx.view(B, N * k)).view(B, N, k)         # [B, N, k]
    p_j = p_t[batch_idx, idx]                                      # [B, N, k, 3]
    d_ij = torch.norm(p_j - p_t.unsqueeze(2), dim=-1)             # [B, N, k]
    dist_feats = torch.stack(
        [d_i.unsqueeze(2).expand(-1, -1, k), d_j, d_ij], dim=-1   # [B, N, k, 3]
    )

    # Eigenvalue ratios
    ev_i = eigenvals.unsqueeze(2).expand(-1, -1, k, -1)           # [B, N, k, 3]
    ev_j = eigenvals.gather(
        1, idx.view(B, N * k).unsqueeze(-1).expand(-1, -1, 3)
    ).view(B, N, k, 3)                                             # [B, N, k, 3]
    ratio_feats = ev_i / (ev_j + 1e-8)                            # [B, N, k, 3]

    # Relative rotation R = V_i^T @ V_j
    V_i = eigenvecs.unsqueeze(2).expand(-1, -1, k, -1, -1)        # [B, N, k, 3, 3]
    V_j = eigenvecs.gather(
        1,
        idx.view(B, N * k).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
    ).view(B, N, k, 3, 3)                                          # [B, N, k, 3, 3]
    R_rel = torch.einsum('bnkji,bnkjl->bnkil', V_i, V_j)          # [B, N, k, 3, 3]
    quat = rotation_matrix_to_quaternion(R_rel)                    # [B, N, k, 4]

    return torch.cat([dist_feats, ratio_feats, quat], dim=-1)      # [B, N, k, 10]


def three_pca_angles_from_cov(cov):
    """
    Compute PCA eigenvalues and angles from covariance matrix.
    Returns 3 rotation-invariant angles and 2 eigenvalue ratios.
    """
    cov = cov + torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0) * 1e-6
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # [B, 3], [B, 3, 3]

    # Sort by eigenvalue (largest first)
    idx = torch.argsort(eigenvalues, dim=-1, descending=True)
    eigenvalues = torch.gather(eigenvalues, 1, idx)
    eigenvectors = torch.gather(eigenvectors, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    v1 = eigenvectors[:, :, 0]
    v2 = eigenvectors[:, :, 1]
    v3 = eigenvectors[:, :, 2]

    angle_12 = torch.sum(v1 * v2, dim=-1).clamp(-1, 1).acos()
    angle_23 = torch.sum(v2 * v3, dim=-1).clamp(-1, 1).acos()
    angle_13 = torch.sum(v1 * v3, dim=-1).clamp(-1, 1).acos()

    # Eigenvalue ratios: capture shape elongation/flatness (rotation-invariant)
    l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    ratio_12 = l1 / (l2 + 1e-8)  # elongation
    ratio_23 = l2 / (l3 + 1e-8)  # flatness

    return (angle_12.unsqueeze(-1), angle_23.unsqueeze(-1), angle_13.unsqueeze(-1),
            ratio_12.unsqueeze(-1), ratio_23.unsqueeze(-1))


class GeometricInvariantExtractor(nn.Module):
    """INVARIANT Feature Extractor using PCA angles + k-NN median + global context."""
    def __init__(self, out_channels, n_global=128):
        super().__init__()
        self.n_global = n_global
        
        # Feature MLP: expects 17 features
        # 1-4.  d_i, d_knn_mean, d_knn_median, d_knn_std       (per-point, spatially varying)
        # 5-7.  Local PCA angles (12, 23, 13)                   (global, shape descriptor)
        # 8-9.  log1p(d_i), d_i^2                               (per-point, spatially varying)
        # 10.   Global PCA angle                                 (global, shape descriptor)
        # 11-12. Local eigenvalue ratios                         (global, shape descriptor)
        # 13-14. Global eigenvalue ratios                        (global, shape descriptor)
        # 15-17. Projection of p_i onto global PCA axes v1,v2,v3 (per-point, spatially varying)
        self.mlp = nn.Sequential(
            nn.Linear(17, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
            nn.LayerNorm(out_channels)
        )

    def forward(self, x):
        B, _, N = x.shape
        c = x.mean(dim=-1, keepdim=True)  # [B, 3, 1]
        p = x - c  # Center the point cloud
        
        # 1. Basic distance features
        d_i = torch.norm(p, dim=1, keepdim=True)  # [B, 1, N]
        
        # 2. k-NN distance statistics (MEDIAN instead of max)
        # Process per batch item to avoid B*N*N OOM
        k = min(32, N)
        topk_dist_list = []
        p_t = p.transpose(1, 2)  # [B, N, 3]
        for b in range(B):
            dm = torch.cdist(p_t[b:b+1], p_t[b:b+1]).squeeze(0)  # [N, N]
            td, _ = dm.topk(k, dim=-1, largest=False)              # [N, k]
            topk_dist_list.append(td)
        topk_dist = torch.stack(topk_dist_list, dim=0)             # [B, N, k]
        d_knn_mean = topk_dist.mean(dim=-1, keepdim=True).transpose(1, 2)  # [B, 1, N]
        d_knn_median = topk_dist.median(dim=-1, keepdim=True)[0].transpose(1, 2)  # [B, 1, N]
        d_knn_std = topk_dist.std(dim=-1, keepdim=True).transpose(1, 2)
        
        # 3. Three PCA angles from entire point cloud (rotation invariant)
        # Center entire point cloud
        p_centered = p - p.mean(dim=-1, keepdim=True)  # [B, 3, N]
        # Covariance [B, 3, 3]: p_centered is [B, 3, N]
        cov = torch.bmm(p_centered, p_centered.transpose(1, 2)) / (N - 1)  # [B, 3, 3]
        angle_12, angle_23, angle_13, ratio_12, ratio_23 = three_pca_angles_from_cov(cov)

        angle_12_exp = angle_12.unsqueeze(-1).expand(-1, -1, N)
        angle_23_exp = angle_23.unsqueeze(-1).expand(-1, -1, N)
        angle_13_exp = angle_13.unsqueeze(-1).expand(-1, -1, N)
        ratio_12_exp = ratio_12.unsqueeze(-1).expand(-1, -1, N)
        ratio_23_exp = ratio_23.unsqueeze(-1).expand(-1, -1, N)
        
        # 4. Global shape features via FPS for capturing whole-pointcloud structure
        fps_idx = farthest_point_sample(p.transpose(1, 2), self.n_global)  # [B, n_global]
        
        # Compute global PCA angles from FPS points - use gather for proper indexing
        batch_idx = torch.arange(B, device=p.device).unsqueeze(1).expand(-1, self.n_global)  # [B, n_global]
        fps_points = p.transpose(1, 2).gather(dim=1, index=fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, n_global, 3]
        fps_centered = fps_points - fps_points.mean(dim=1, keepdim=True)
        fps_cov = torch.bmm(fps_centered.transpose(1, 2), fps_centered) / (self.n_global - 1)
        g_angle_12, g_angle_23, g_angle_13, g_ratio_12, g_ratio_23 = three_pca_angles_from_cov(fps_cov)

        g_angle_12_exp = g_angle_12.unsqueeze(-1).expand(-1, -1, N)
        g_ratio_12_exp = g_ratio_12.unsqueeze(-1).expand(-1, -1, N)
        g_ratio_23_exp = g_ratio_23.unsqueeze(-1).expand(-1, -1, N)
        
        # 5. Per-point projections onto global PCA axes (rotation-invariant AND spatially varying)
        eigenvalues_full, eigenvectors_full = torch.linalg.eigh(cov)   # [B,3], [B,3,3]
        idx_sort = torch.argsort(eigenvalues_full, dim=-1, descending=True)
        eigenvectors_full = torch.gather(
            eigenvectors_full, 1,
            idx_sort.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, 3, 3] sorted descending
        # proj_k = |p_i . v_k|  -- absolute value removes sign ambiguity of eigenvectors
        proj = torch.bmm(eigenvectors_full.transpose(1, 2), p_centered)  # [B, 3, N]
        proj_v1 = proj[:, 0:1, :].abs()   # [B, 1, N]
        proj_v2 = proj[:, 1:2, :].abs()   # [B, 1, N]
        proj_v3 = proj[:, 2:3, :].abs()   # [B, 1, N]

        feats = torch.cat([
            d_i,                                        # 1.  d_i
            d_knn_mean,                                # 2.  d_knn_mean
            d_knn_median,                              # 3.  d_knn_median
            d_knn_std,                                 # 4.  d_knn_std
            angle_12_exp,                              # 5.  Local PCA angle 12
            angle_23_exp,                              # 6.  Local PCA angle 23
            angle_13_exp,                              # 7.  Local PCA angle 13
            torch.log1p(d_i),                          # 8.  log distance
            (d_i ** 2),                                # 9.  squared distance
            g_angle_12_exp,                            # 10. Global PCA angle
            ratio_12_exp,                              # 11. Local eigenvalue ratio (elongation)
            ratio_23_exp,                              # 12. Local eigenvalue ratio (flatness)
            g_ratio_12_exp,                            # 13. Global eigenvalue ratio (elongation)
            g_ratio_23_exp,                            # 14. Global eigenvalue ratio (flatness)
            proj_v1,                                   # 15. Projection onto 1st PCA axis
            proj_v2,                                   # 16. Projection onto 2nd PCA axis
            proj_v3,                                   # 17. Projection onto 3rd PCA axis
        ], dim=1)  # [B, 17, N]

        BN = feats.shape[0] * feats.shape[2]
        f = self.mlp(feats.transpose(1, 2).reshape(BN, 17))
        return f.reshape(feats.shape[0], feats.shape[2], -1).transpose(1, 2).contiguous()


class MultiScaleInvariantAttention(nn.Module):
    """
    Multi-scale attention with pre-norm architecture for stable gradient flow.
    Pre-norm: LayerNorm -> Attention -> Residual -> LayerNorm -> FF -> Residual
    This prevents feature norm explosion and ensures gradients reach all layers.
    """
    def __init__(self, c, k=32, n_global=64):
        super().__init__()
        self.k = k
        self.n_global = n_global
        self.c = c

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

        # Pre-norm layers for stable gradient flow
        self.norm_attn = nn.LayerNorm(c)
        self.norm_ff = nn.LayerNorm(c)
        self.ff = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.GELU(),
            nn.Linear(c * 2, c)
        )

    def forward(self, f, p, eigenvecs, eigenvals):
        B, C, N = f.shape
        f_t = f.transpose(1, 2)                                    # [B, N, C]

        # Pre-norm before attention
        f_norm = self.norm_attn(f_t)                               # [B, N, C]
        q = self.q(f_norm)
        k_val = self.k_linear(f_norm)
        v = self.v(f_norm)

        # === LOCAL ATTENTION (k-NN) ===
        with torch.no_grad():
            dist = torch.cdist(p.transpose(1, 2), p.transpose(1, 2))
            _, idx = dist.topk(self.k, dim=-1, largest=False)      # [B, N, k]

        batch_idx = torch.arange(B, device=f.device).view(B, 1, 1).expand(-1, N, self.k)

        with torch.no_grad():
            f_pos_ij = compute_pair_pos_encoding(p, eigenvecs, eigenvals, idx).detach()
        pos_bias_local = self.pos_mlp(f_pos_ij).squeeze(-1)        # [B, N, k]

        q_expanded = q.unsqueeze(2)
        k_gathered = k_val[batch_idx, idx]
        v_gathered = v[batch_idx, idx]

        attn_scores = (q_expanded * k_gathered).sum(dim=-1, keepdim=True) / (C ** 0.5) + pos_bias_local.unsqueeze(-1)
        attn = F.softmax(attn_scores, dim=2)
        local_out = (attn * v_gathered).sum(dim=2)                 # [B, N, C]

        # === GLOBAL ATTENTION (FPS tokens) ===
        fps_idx = farthest_point_sample(p.transpose(1, 2), self.n_global)
        fps_features = f_norm.gather(dim=1, index=fps_idx.unsqueeze(-1).expand(-1, -1, C))
        q_global = self.q_global(f_norm)
        k_global = self.k_global(fps_features)
        v_global = self.v(fps_features)
        global_attn = torch.bmm(q_global, k_global.transpose(1, 2)) / (C ** 0.5)
        global_attn = F.softmax(global_attn, dim=-1)
        global_out = torch.bmm(global_attn, v_global)              # [B, N, C]

        # Residual 1: attention output
        combined = local_out + 0.5 * global_out
        f_t = f_t + combined                                       # [B, N, C]

        # Pre-norm + feed-forward + residual
        f_t = f_t + self.ff(self.norm_ff(f_t))                     # [B, N, C]

        return f_t.transpose(1, 2)                                 # [B, C, N]


class PointTransformerInvariantEncoder(nn.Module):
    """Multi-scale invariant transformer with FPS global context
    and per-pair local frame alignment positional encoding."""
    def __init__(self):
        super().__init__()
        self.geometric_prep = GeometricInvariantExtractor(128, n_global=64)
        self.blocks = nn.ModuleList([
            MultiScaleInvariantAttention(128, k=16, n_global=32),
            MultiScaleInvariantAttention(128, k=16, n_global=32)
        ])
        self.post = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc = nn.Linear(512, 512)
        self.fc_bn = nn.BatchNorm1d(512)  # bounds scale without collapsing to unit sphere

    def _encode(self, x):
        """Shared encode: computes local frames once, reuses across all blocks."""
        c = x.mean(dim=-1, keepdim=True)
        x_centered = x - c
        with torch.no_grad():
            eigenvecs, eigenvals = compute_local_frames(x_centered, k=16)
        f = self.geometric_prep(x_centered)   # [B, 128, N] — spatially varying
        f_geom = f.clone()                    # save raw geometric features
        for block in self.blocks:
            f = block(f, x_centered, eigenvecs, eigenvals)
        # Add geometric features back as skip connection
        # This ensures per-point features retain spatial variation
        f = f + f_geom
        return f, x_centered

    def _aggregate(self, f):
        """Aggregate per-point features to global descriptor.
        Uses max + mean pooling concatenated for richer gradient flow.
        Max pooling alone kills gradients for all non-max points.
        """
        f_max = torch.max(f, dim=-1)[0]   # [B, 512]
        f_mean = f.mean(dim=-1)            # [B, 512]
        return f_max + f_mean              # [B, 512] — both contribute gradients

    def forward(self, x):
        f, _ = self._encode(x)
        f = self.post(f)
        z = self._aggregate(f)
        return self.fc_bn(self.fc(z))

    def forward_seg(self, x):
        """Returns (global_feat [B,512], per_point_feat [B,512,N]) for segmentation."""
        f, _ = self._encode(x)
        f = self.post(f)                                            # [B, 512, N]
        z = self.fc_bn(self.fc(self._aggregate(f)))                 # [B, 512]
        return z, f


PointTransformerInvariant = PointTransformerInvariantEncoder
