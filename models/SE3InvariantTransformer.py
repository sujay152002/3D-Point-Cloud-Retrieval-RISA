"""
MULTI-SCALE INVARIANT TRANSFORMER - Improved Version
Changes:
1. k-NN uses median instead of max
2. PCA-based 3 angles for rotation invariance
3. Multi-scale attention: local (k=32) + global via FPS (num_global=128)
4. Global tokens computed via FPS to capture whole-pointcloud structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to unit quaternions using Shepperd's method.

    Fully PyTorch-native: no CPU/NumPy round-trip, no scipy dependency.
    Branchless via ``torch.where`` so it handles arbitrary batch shapes in a
    single vectorised pass.

    Shepperd's method selects whichever quaternion component (w, x, y, z) is
    largest for a given rotation, using that component as the stable denominator
    to recover the remaining three.  All four cases are computed in parallel;
    ``torch.where`` picks the numerically dominant branch per sample.

    The four sqrt arguments equal 4w², 4x², 4y², 4z² respectively for a valid
    SO(3) matrix, so they are analytically ≥ 0.  A small ``eps`` clamp guards
    against floating-point negatives that can appear after PCA eigenvector
    normalisation.

    Args:
        R: ``[..., 3, 3]`` rotation matrices (``det ≈ +1``).

    Returns:
        ``[..., 4]`` unit quaternions in ``(x, y, z, w)`` order, matching the
        scipy ``Rotation.as_quat()`` convention.
    """
    eps = 1e-10

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Denominators S = 4 * (largest component).  Computed for all four cases
    # simultaneously; only the selected case is numerically critical.
    S1 = torch.sqrt(torch.clamp(trace + 1.0,                                            min=eps)) * 2  # S = 4w
    S2 = torch.sqrt(torch.clamp(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2],
                                min=eps)) * 2  # S = 4x
    S3 = torch.sqrt(torch.clamp(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2],
                                min=eps)) * 2  # S = 4y
    S4 = torch.sqrt(torch.clamp(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1],
                                min=eps)) * 2  # S = 4z

    # Case 1: trace > 0  →  w is largest
    w1 = 0.25 * S1
    x1 = (R[..., 2, 1] - R[..., 1, 2]) / S1
    y1 = (R[..., 0, 2] - R[..., 2, 0]) / S1
    z1 = (R[..., 1, 0] - R[..., 0, 1]) / S1

    # Case 2: R00 > R11 and R00 > R22  →  x is largest
    w2 = (R[..., 2, 1] - R[..., 1, 2]) / S2
    x2 = 0.25 * S2
    y2 = (R[..., 0, 1] + R[..., 1, 0]) / S2
    z2 = (R[..., 0, 2] + R[..., 2, 0]) / S2

    # Case 3: R11 > R22  →  y is largest
    w3 = (R[..., 0, 2] - R[..., 2, 0]) / S3
    x3 = (R[..., 0, 1] + R[..., 1, 0]) / S3
    y3 = 0.25 * S3
    z3 = (R[..., 1, 2] + R[..., 2, 1]) / S3

    # Case 4: else  →  z is largest
    w4 = (R[..., 1, 0] - R[..., 0, 1]) / S4
    x4 = (R[..., 0, 2] + R[..., 2, 0]) / S4
    y4 = (R[..., 1, 2] + R[..., 2, 1]) / S4
    z4 = 0.25 * S4

    # Selection masks (priority: case 1 → 2 → 3 → 4)
    c1 = trace > 0
    c2 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    c3 = R[..., 1, 1] > R[..., 2, 2]

    x = torch.where(c1, x1, torch.where(c2, x2, torch.where(c3, x3, x4)))
    y = torch.where(c1, y1, torch.where(c2, y2, torch.where(c3, y3, y4)))
    z = torch.where(c1, z1, torch.where(c2, z2, torch.where(c3, z3, z4)))
    w = torch.where(c1, w1, torch.where(c2, w2, torch.where(c3, w3, w4)))

    q = torch.stack([x, y, z, w], dim=-1)           # [..., 4]  (x, y, z, w)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    return q

class GeometricInvariantExtractor(nn.Module):

    feature_dim = 16
    pos_enc_dim = 12

    _min_neighborhood_points = 16

    """INVARIANT Feature Extractor using PCA angles + k-NN median + global context."""

    def __init__(self, num_global = 128, num_local = 32, indices = list(range(feature_dim))):
        super().__init__()
        """
        Index selection rules:

        [0, 1, 2] # different distance scales
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # eigenvalue geometry
        [13, 14, 15] # neighborhood diffusion statistics

        Ablation rule:

        indices = []

        for dist_flag in (True, False):
            if dist_flag:
                indices.extend([0, 1, 2])

            for pca_flag in (True, False):
                if pca_flag:
                    indices.extend([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

                for diffusion_flag in (True, False):
                    if diffusion_flag:
                        indices.extend([13, 14, 15])

                    if len(indices) == 0:
                        continue
        """

        self.num_global = num_global
        self.num_local = max(self._min_neighborhood_points, num_local)
        self.indices = indices
        self.feature_dim = len(indices)

    def positional_encoding_features(
            self,
            xyz : torch.Tensor,
            evecs : torch.Tensor,
            evals : torch.Tensor,
            dist_pointwise : torch.Tensor,
            dist_pairwise : torch.Tensor,
            attended_indices : torch.Tensor
        ):

        eps = 1e-6
        batch_size, num_points, num_attended = attended_indices.shape
        device = xyz.device

        batch_idx = torch.arange(batch_size, device = device).view(batch_size, 1, 1).expand(-1, num_points, num_attended)
        point_idx = torch.arange(num_points, device = device).view(1, num_points, 1).expand(batch_size, -1, num_attended)

        # Gather per-point attended j values without building dense [N, N] tensors.
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        xyz_j = xyz_t[batch_idx, attended_indices]  # [B, N, A, 3]

        dist_pointwise_flat = dist_pointwise.squeeze(1)  # [B, N]
        distances_i = dist_pointwise_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_attended, -1)
        distances_j = dist_pointwise_flat[batch_idx, attended_indices].unsqueeze(-1)
        dist_ij = dist_pairwise[batch_idx, point_idx, attended_indices].unsqueeze(-1)

        # Eigenvalue Ratios
        evals   = evals #.permute(0, 2, 1)  # [B, N, 3]
        evals_i = evals.unsqueeze(2)
        evals_j = torch.clamp(evals[batch_idx, attended_indices], min = eps)
        rsig_ij = torch.sigmoid(torch.divide(evals_i, torch.clip(evals_j, min = eps)))

        # Direction Similarity
        directions_i = torch.divide(xyz_t, torch.clamp(dist_pointwise_flat.unsqueeze(-1), min = eps))
        directions_j = torch.divide(xyz_j, torch.clamp(distances_j, min = eps))
        dot_product  = torch.multiply(directions_i.unsqueeze(2), directions_j).sum(dim = -1, keepdim = True)
        cos_sim_ij   = torch.clamp(dot_product, min = -1, max = 1)
        angle_ij     = torch.acos(cos_sim_ij) / torch.pi

        # PCA Relative Quaternions

        evecs_i = evecs.unsqueeze(2).expand(-1, -1, attended_indices.shape[-1], -1, -1)
        evecs_i_flat = evecs_i.reshape(batch_size * num_points * attended_indices.shape[-1], 3, 3)

        evecs_jT = evecs[batch_idx, attended_indices].permute(0, 1, 2, 4, 3)
        evecs_jT_flat = evecs_jT.reshape(batch_size * num_points * attended_indices.shape[-1], 3, 3)
        
        R_ij = torch.bmm(evecs_i_flat, evecs_jT_flat)
        dq = _rotation_matrix_to_quaternion(R_ij)
        dq = dq.reshape(batch_size, num_points, attended_indices.shape[-1], 4)

        pos_enc_features = torch.cat([
            distances_i,
            distances_j,
            dist_ij,
            cos_sim_ij,
            angle_ij,
            rsig_ij,
            dq,
        ], dim = -1)

        return pos_enc_features

    @staticmethod
    def _seeded_exclusive_fps(
            xyz: torch.Tensor,
            dist_pairwise: torch.Tensor,
            npoint: int,
            neighbor_indices: torch.Tensor = None,
            return_points: bool = False
        ):
        """
        Run FPS independently per seed point and return sampled coordinates + indices.

        Args:
            xyz: [B, N, 3]
            dist_pairwise: [B, N, N] precomputed pairwise distances
            npoint: number of FPS samples per seed
            neighbor_indices: [B, N, K] local neighborhood per seed to exclude from FPS

        Returns:
            fps_points: [B, N, npoint, 3]
            fps_indices: [B, N, npoint]
        """
        B, N, _ = xyz.shape
        device = xyz.device

        if npoint <= 0:
            empty_idx = torch.empty(B, N, 0, dtype=torch.long, device=device)
            if return_points:
                empty_points = torch.empty(B, N, 0, 3, dtype=xyz.dtype, device=device)
                return empty_points, empty_idx
            return empty_idx

        valid_mask = torch.ones(B, N, N, dtype=torch.bool, device=device)
        if neighbor_indices is not None:
            valid_mask.scatter_(dim=2, index=neighbor_indices, value=False)

        min_available = int(valid_mask.sum(dim=-1).min().item())
        npoint = min(npoint, min_available)

        if npoint <= 0:
            empty_idx = torch.empty(B, N, 0, dtype=torch.long, device=device)
            if return_points:
                empty_points = torch.empty(B, N, 0, 3, dtype=xyz.dtype, device=device)
                return empty_points, empty_idx
            return empty_idx

        fps_indices = torch.zeros(B, N, npoint, dtype=torch.long, device=device)
        min_dist = torch.full((B, N, N), float("inf"), dtype=dist_pairwise.dtype, device=device)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        candidate_idx = torch.arange(N, device=device).view(1, 1, N)

        farthest = dist_pairwise.masked_fill(~valid_mask, -1.0).argmax(dim=-1)
        fps_indices[:, :, 0] = farthest

        for i in range(1, npoint):
            # Distances from all candidates to newly selected point per seed.
            new_dist = dist_pairwise[batch_idx, candidate_idx, farthest.unsqueeze(-1)]
            min_dist = torch.minimum(min_dist, new_dist)
            masked_min_dist = min_dist.masked_fill(~valid_mask, -1.0)
            farthest = masked_min_dist.argmax(dim=-1)
            fps_indices[:, :, i] = farthest

            # Prevent reselection of the same sampled point for each seed.
            valid_mask.scatter_(dim=2, index=farthest.unsqueeze(-1), value=False)

        if return_points:
            fps_points = xyz[batch_idx, fps_indices, :]
            return fps_points, fps_indices
        else:
            return fps_indices

    def forward(self, points : torch.Tensor):

        batch_size, _, num_points = points.shape
        device  = points.device
        dtype   = points.dtype

        eps = 1e-6

        if num_points < self.num_local:
            raise ValueError(
                f"Point cloud has only {num_points} points, but num_local={self.num_local} "
                f"neighbours are required. Pass num_local<={num_points} to the encoder constructor."
            )

        with torch.no_grad():

            # Point Cloud Preprocessing
            xyz     = points[:, :3, :].reshape(batch_size, 3, num_points)
            xyz     = xyz - xyz.mean(dim = -1, keepdim = True)
            xyz_T   = xyz.transpose(1, 2)  # [B, N, 3]

            # Distances
            dist_pointwise  = torch.norm(xyz, dim = 1, keepdim = True)
            dist_pairwise   = torch.cdist(xyz_T, xyz_T)  # [B, N, N]

            # Context

            attn_mask = torch.zeros(batch_size, num_points, num_points, device = device, dtype = torch.bool)

            if self.num_local == num_points:
                dist_neighborhood = dist_pairwise
                neighbor_indices = torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)

                # Local neighborhood is all points, attend to everything
                attn_mask.fill_(True)
                attended_indices = torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)

            else:

                dist_neighborhood, neighbor_indices = dist_pairwise.topk(self.num_local, dim = -1, largest = False)  # [B, N, self.num_local]

                if self.num_local + self.num_global < num_points:

                    fps_idx_by_seed = self._seeded_exclusive_fps(
                        xyz              = xyz_T,
                        dist_pairwise    = dist_pairwise,
                        neighbor_indices = neighbor_indices,
                        npoint           = self.num_global,
                        return_points    = False,
                    )

                    attended_indices = torch.cat([neighbor_indices, fps_idx_by_seed], dim = -1)
                    attn_mask.scatter_(dim = 2, index = attended_indices, value = True)
                
                else:
                    # Local neighborhood + global fps sample is all points, attend to everything
                    attn_mask.fill_(True)
                    attended_indices = torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)

            # Local Neighborhood

            neighborhood = torch.gather(
                input   = xyz_T.unsqueeze(1).expand(-1, num_points, -1, -1),  # [B, N, N, 3]
                dim     = 2,
                index   = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, N, K, 3]
            )

            # Neighborhood Diffusion Statistics

            d_knn_mean      = dist_neighborhood.mean(dim = -1, keepdim = True).transpose(1, 2)  # [B, 1, N]
            d_knn_median    = dist_neighborhood.median(dim = -1, keepdim = True)[0].transpose(1, 2)  # [B, 1, N]
            d_knn_std       = dist_neighborhood.std(dim = -1, keepdim = True).transpose(1, 2)

            neighborhood_diffusion_statistics = torch.cat([d_knn_mean, d_knn_median, d_knn_std], dim=1)

            # Neighborhood PCA

            flattened_neighborhoods = (neighborhood - xyz_T.unsqueeze(2)).reshape(batch_size * num_points, self.num_local, 3)
            flattened_neighborhoods = torch.nan_to_num(flattened_neighborhoods, nan=0.0, posinf=1.0, neginf=-1.0)

            eps_diag         = torch.full((3,), eps, device = device, dtype = dtype).diag().unsqueeze(0)
            neighborhood_cov = torch.bmm(flattened_neighborhoods.transpose(1, 2), flattened_neighborhoods) / (self.num_local - 1)
            neighborhood_cov = torch.nan_to_num(neighborhood_cov, nan=0.0, posinf=1.0, neginf=0.0)
            neighborhood_cov = neighborhood_cov + eps_diag
            
            # Symmetrize to ensure positive semi-definite
            neighborhood_cov = (neighborhood_cov + neighborhood_cov.transpose(-1, -2)) / 2
            
            # Use CPU fallback for eigh to avoid cusolver NaN issues
            try:
                torch.backends.cuda.preferred_linalg_library('cusolver')
                evals, evecs = torch.linalg.eigh(neighborhood_cov)
            except Exception:
                try:
                    torch.backends.cuda.preferred_linalg_library('magma')
                    evals, evecs = torch.linalg.eigh(neighborhood_cov)
                except Exception:
                    # CPU fallback
                    evals, evecs = torch.linalg.eigh(neighborhood_cov.cpu())
                    evals = evals.to(device)
                    evecs = evecs.to(device)

            evals = evals.reshape(batch_size, num_points, 3)
            
            e1, e2, e3 = evals[:, :, 0:1], evals[:, :, 1:2], evals[:, :, 2:3]

            eval_sum            = evals.sum(dim = -1, keepdim = True)
            linearity           = (e3 - e2) / torch.clip(e3, min = eps)
            planarity           = (e2 - e1) / torch.clip(e3, min = eps)
            sphericity          = e1 / torch.clip(e3, min = eps)
            anisotropy          = (e3 - e1) / torch.clip(e3, min = eps)
            omnivariance        = (e1 * e2 * e3) ** (1/3)
            eigenentropy        = - torch.mul(evals, torch.log(torch.clamp(evals, min = eps))).sum(dim = -1, keepdim = True)
            surface_variation   = e3 / torch.clip(eval_sum, min = eps)

            flip_mat = torch.tensor([[
                [ 1.,  0.,  0.],
                [ 0.,  0.,  1.],
                [ 0.,  1.,  0.]
            ]], device = device, dtype = dtype)

            left_handed = torch.linalg.det(evecs) < 0
            evecs[left_handed] = evecs[left_handed] @ flip_mat

            evecs = evecs.reshape(batch_size, num_points, 3, 3)

            features = torch.cat([
                dist_pointwise,                     # 1. dist_pointwise
                torch.log1p(dist_pointwise),        # 2. log distance
                (dist_pointwise ** 2),              # 3. squared distance
                evals.permute(0, 2, 1),             # 4 - 6. neighborhood eigenvalues
                linearity.permute(0, 2, 1),         # 7. linearity
                planarity.permute(0, 2, 1),         # 8. planarity
                sphericity.permute(0, 2, 1),        # 9. sphericity
                anisotropy.permute(0, 2, 1),        # 10. anisotropy
                omnivariance.permute(0, 2, 1),      # 11. omnivariance
                eigenentropy.permute(0, 2, 1),      # 12. eigenentropy
                surface_variation.permute(0, 2, 1), # 13. surface variation
                neighborhood_diffusion_statistics,  # 14 - 16. neighborhood diffusion statistics
            ], dim = 1)                             # [batch_size, feature_dim, num_points]

            features = features[:, self.indices, :]

            pos_enc = self.positional_encoding_features(
                xyz = xyz, 
                evecs = evecs,
                evals = evals, 
                dist_pointwise = dist_pointwise, 
                dist_pairwise = dist_pairwise, 
                attended_indices = attended_indices
            )

        return features, xyz, attn_mask, attended_indices, pos_enc



class MultiScaleInvariantAttention(nn.Module):
    """
    Invariant attention block with externally provided sparse attention mask.
    Each point attends to the points marked True in `attn_mask`.
    """
    def __init__(
            self,
            out_dim,
            model_dim = None,
            pos_enc_feature_dim = 12,
            do_rel_attention_augmentation = True,
        ):
        super().__init__()

        model_dim = model_dim if model_dim is not None else out_dim

        self.model_dim = model_dim
        self.out_dim = out_dim

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out      = nn.Linear(model_dim, out_dim)
        self.do_rel_attention_augmentation = do_rel_attention_augmentation

        if self.do_rel_attention_augmentation:
            self.pos_enc_mlp = nn.Sequential(
                nn.Linear(pos_enc_feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.pos_enc_mlp = None

    def forward(
            self,
            features: torch.Tensor,
            attended_indices: torch.Tensor = None,
            pos_enc: torch.Tensor   = None,
            attn_mask: torch.Tensor = None,
        ):

        B, C, N = features.shape
        f_t = features.transpose(1, 2)  # [B, N, C]

        Q = self.q_linear(f_t)
        K = self.k_linear(f_t)
        V = self.v_linear(f_t)

        if attended_indices.dim() != 3:
            raise ValueError(
                f"attended_indices must have shape [B, N, A], got {tuple(attended_indices.shape)}"
            )
        if attended_indices.shape[0] != B or attended_indices.shape[1] != N:
            raise ValueError(
                f"attended_indices must have leading shape {(B, N)}, got {tuple(attended_indices.shape[:2])}"
            )

        attended_indices = attended_indices.long()
        A = attended_indices.shape[-1]
        device = features.device
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, A)

        K_gathered = K[batch_idx, attended_indices]  # [B, N, A, C]
        V_gathered = V[batch_idx, attended_indices]  # [B, N, A, C]

        scores = (Q.unsqueeze(2) * K_gathered).sum(dim = -1) / (C ** 0.5)  # [B, N, A]

        if pos_enc is not None and self.do_rel_attention_augmentation:
            scores = scores + self.pos_enc_mlp(pos_enc).squeeze(-1)

        attn = F.softmax(scores, dim=-1)
        out = (attn.unsqueeze(-1) * V_gathered).sum(dim=2)  # [B, N, C]

        out = self.out(out)

        # Residual Connection
        features = features + out.transpose(1, 2) # [B, C, N] to match input format

        return features


class SE3InvariantPointTransformerEncoder(nn.Module):
    """Multi-scale invariant transformer with FPS global context."""
    def __init__(
            self,
            num_global = 16,
            num_local = 32,
            indices = list(range(GeometricInvariantExtractor.feature_dim)),
            do_rel_attention_augmentation = True,
        ):
        super().__init__()

        model_dim = 128
        attn_dim = model_dim

        self.do_rel_attention_augmentation = do_rel_attention_augmentation

        self.feature_extractor = GeometricInvariantExtractor(
            num_global  = num_global,
            num_local   = num_local,
            indices     = indices,
        )

        self.feature_dim = self.feature_extractor.feature_dim
        self.model_dim   = model_dim

        self.feature_embed = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, model_dim)
        )

        self.blocks = nn.ModuleList([
            MultiScaleInvariantAttention(
                model_dim = attn_dim,
                out_dim = model_dim,
                do_rel_attention_augmentation = do_rel_attention_augmentation,
            ),
            MultiScaleInvariantAttention(
                model_dim = attn_dim,
                out_dim = model_dim,
                do_rel_attention_augmentation = do_rel_attention_augmentation,
            )
        ])

        self.post = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc = nn.Linear(512, 512)

    def embed(self, features : torch.Tensor):
        batch_size, feature_dim, num_points = features.shape
        features = features.transpose(1, 2).reshape(batch_size * num_points, feature_dim)
        features = self.feature_embed(features)
        features = features.reshape(batch_size, num_points, -1).transpose(1, 2).contiguous()
        return features


    def forward(self, x, return_per_point_features = False):

        features, _, attn_mask, attended_indices, pos_enc = self.feature_extractor(x)

        f = self.embed(features)

        for block in self.blocks:
            f = block(f, pos_enc = pos_enc, attn_mask = attn_mask, attended_indices = attended_indices)

        f = self.post(f)
        z = self.fc(torch.max(f, dim=-1)[0])

        if return_per_point_features:
            out = z, f
        else:
            out = z

        return out

    def forward_seg(self, x):
        return self.forward(x, return_per_point_features = True)


# Backward-compatible aliases so existing callers that reference the old class
# name continue to work without any import changes.
PointTransformerInvariantEncoder = SE3InvariantPointTransformerEncoder
PointTransformerInvariant        = SE3InvariantPointTransformerEncoder


if __name__ == "__main__":

    seed = 0
    batch_size = 8
    num_points = 192
    num_global = 10

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = [0, 4, 5, 6, 10]
    model = SE3InvariantPointTransformerEncoder(num_global = num_global, indices = indices).to(device).eval()

    dummy_points = torch.randn(batch_size, 3, num_points, device=device)

    with torch.no_grad():
        global_features = model(dummy_points)
        seg_global_features, per_point_features = model.forward_seg(dummy_points)

    print(f"Input shape: {dummy_points.shape}")
    print(f"Global output shape (forward): {global_features.shape}")
    print(f"Seg global output shape (forward_seg): {seg_global_features.shape}")
    print(f"Per-point output shape (forward_seg): {per_point_features.shape}")
