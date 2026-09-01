"""Rotation-Invariant Sparse Attention (RISA) encoder.

A sparse attention mechanism that operates on rotation-invariant geometric
features (pairwise distances, angles, eigenvalue ratios) to produce
SE(3)-invariant point cloud embeddings.
"""
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

RISA_ABLATION_BASE_ARGS = {
    "num_global": 16,
    "num_local" : 32,
    "num_blocks" : 2,
    "model_dim" :  128,
    "features_out_dim" : 128,
    "encoding_out_dim" : 128,
}

RISA_ABLATION_OPTIONS = [
    ("sparse",
        [True, False]),
    ("encoding_method",
        ["mean", "max", "token"]),
    ("attn_augment",
        ["none", "add", "weight"]),
]

def make_risa_sweep_params():

    base_args = {
        "sparse": True,
        "features_out_dim": 128,
        "encoding_out_dim": 128,
        "attn_augment": "weight",
        "encoding_method": "token",
    }

    options = [
        ("num_global", [16, 32, 64]),
        ("num_local",  [8, 16, 24]),
        ("num_blocks", [2, 4, 6]),
        ("model_dim",  [128, 256]),
    ]

    return base_args, options

class GeometricInvariantExtractor(nn.Module):

    _base_feature_dim = 32       # per-scale scalar statistics (16 each × 2 scales)
    _nb_embed_dim     = 16       # output dim of the neighbor-distance MLP (per scale)
    feature_dim       = _base_feature_dim + 2 * _nb_embed_dim  # 64 total
    pos_enc_dim       = 9

    _min_neighborhood_points = 16

    """INVARIANT Feature Extractor using PCA angles + k-NN median + global context.

    In addition to the 16 per-scale scalar statistics, we embed the full sorted
    neighbour-distance sequence for each point through a small shared MLP.  This
    gives every point a fine-grained "fingerprint" of its local geometry that is
    still rotation-invariant (pairwise distances are unchanged by rotation) and
    directly addresses two weaknesses of the pure statistics approach:

      1. Information loss  — the sorted distance sequence encodes neighbourhood
         shape far more faithfully than mean / median / std alone.
      2. Feature collisions — two points need identical inter-neighbour distance
         distributions to produce the same embedding, which is rare in practice
         and essentially impossible for asymmetric shapes.

    The MLP maps [B*N, k] → [B*N, _nb_embed_dim] using a max-pool over a
    per-neighbour hidden representation (PointNet style), so the output size is
    independent of k and the operation is permutation-invariant (the sorted order
    gives determinism without breaking rotation invariance).
    """

    def __init__(self, sparse : bool = True, num_global = 128, num_local = 32, num_local_coarse = 128):
        super().__init__()

        self.sparse = sparse
        self.num_global = num_global
        self.num_local = max(self._min_neighborhood_points, num_local)
        self.num_local_coarse = num_local_coarse

        hidden = self._nb_embed_dim * 2

        # Shared MLP applied per neighbour distance, then max-pooled over k.
        # Input: 1 scalar (distance) → hidden → _nb_embed_dim.
        # Two independent copies: one for fine scale, one for coarse scale.
        self.nb_dist_mlp_fine = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self._nb_embed_dim),
        )
        self.nb_dist_mlp_coarse = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self._nb_embed_dim),
        )

    @staticmethod
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

    def positional_encoding_features(
            self,
            xyz : torch.Tensor,
            evecs : torch.Tensor,
            evals : torch.Tensor,
            dist_pointwise : torch.Tensor,
            dist_pairwise : torch.Tensor,
            indices : torch.Tensor
        ):

        eps = 1e-6
        batch_size, num_points, num_attended = indices.shape
        device = xyz.device

        batch_idx = torch.arange(batch_size, device = device).view(batch_size, 1, 1).expand(-1, num_points, num_attended)
        point_idx = torch.arange(num_points, device = device).view(1, num_points, 1).expand(batch_size, -1, num_attended)

        # Gather per-point attended j values without building dense [N, N] tensors.
        xyz_t = xyz.transpose(1, 2)  # [B, N, 3]
        xyz_j = xyz_t[batch_idx, indices]  # [B, N, A, 3]

        dist_pointwise_flat = dist_pointwise.squeeze(1)  # [B, N]

        r_i  = dist_pointwise_flat.unsqueeze(-1)
        r_j  = dist_pointwise_flat[batch_idx, indices].unsqueeze(-1)
        r_ij = dist_pairwise[batch_idx, point_idx, indices].unsqueeze(-1)

        # Eigenvalue Ratios
        evals_i = evals.unsqueeze(2)
        evals_j = torch.clamp(evals[batch_idx, indices], min = eps)
        rsig_ij = torch.sigmoid(torch.divide(evals_i, torch.clip(evals_j, min = eps)))

        # Direction Similarity
        dir_i       = torch.divide(xyz_t, torch.clamp(r_i, min = eps))
        dir_j       = torch.divide(xyz_j, torch.clamp(r_j, min = eps))
        dot_ij      = torch.multiply(dir_i.unsqueeze(2), dir_j).sum(dim = -1, keepdim = True)
        angle_ij    = torch.acos(torch.clamp(dot_ij, min = -1, max = 1))

        # PCA Relative Quaternions

        evecs_i = evecs.unsqueeze(2).expand(-1, -1, indices.shape[-1], -1, -1)
        evecs_i_flat = evecs_i.reshape(batch_size * num_points * indices.shape[-1], 3, 3)

        evecs_jT = evecs[batch_idx, indices].permute(0, 1, 2, 4, 3)
        evecs_jT_flat = evecs_jT.reshape(batch_size * num_points * indices.shape[-1], 3, 3)
        
        R_ij = torch.bmm(evecs_i_flat, evecs_jT_flat)
        dq = self._rotation_matrix_to_quaternion(R_ij)
        dq = dq.reshape(batch_size, num_points, indices.shape[-1], 4)

        pos_enc_features = torch.cat([
            r_ij,
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

    @staticmethod
    def _scale_features(xyz_T, dist_pairwise, dist_pointwise, k, batch_size, num_points, device, dtype, eps, return_evecs=False, return_nb_dists=False):
        """Compute all 16 rotation-invariant features for a given neighbourhood size k.

        Args:
            return_nb_dists: if True, also return ``dist_nb`` shaped ``[B, N, k]``
                (sorted neighbour distances).  Used by the caller to build the
                neighbour-distance embedding outside the no_grad block.
        """
        if k >= num_points - 1:
            dist_nb, nb_idx = dist_pairwise.clone(), \
                torch.arange(num_points, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)
        else:
            dist_nb, nb_idx = dist_pairwise.topk(k, dim=-1, largest=False)

        neighborhood = torch.gather(
            xyz_T.unsqueeze(1).expand(-1, num_points, -1, -1),
            dim=2,
            index=nb_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [B, N, k, 3]

        # Diffusion stats
        d_mean   = dist_nb.mean(dim=-1, keepdim=True).transpose(1, 2)
        d_median = dist_nb.median(dim=-1, keepdim=True)[0].transpose(1, 2)
        d_std    = dist_nb.std(dim=-1, keepdim=True).transpose(1, 2)
        diffusion = torch.cat([d_mean, d_median, d_std], dim=1)  # [B, 3, N]

        # PCA — use actual neighbourhood size (may differ from k if clamped)
        k_actual = nb_idx.shape[-1]
        flat = (neighborhood - xyz_T.unsqueeze(2)).reshape(batch_size * num_points, k_actual, 3)
        flat = torch.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)
        eps_diag = torch.full((3,), eps, device=device, dtype=dtype).diag().unsqueeze(0)
        cov = torch.bmm(flat.transpose(1, 2), flat) / (k_actual - 1)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=1.0, neginf=0.0)
        cov = (cov + cov.transpose(-1, -2)) / 2 + eps_diag

        try:
            torch.backends.cuda.preferred_linalg_library('cusolver')
            evals, evecs = torch.linalg.eigh(cov)
        except Exception:
            try:
                torch.backends.cuda.preferred_linalg_library('magma')
                evals, evecs = torch.linalg.eigh(cov)
            except Exception:
                evals, evecs = torch.linalg.eigh(cov.cpu())
                evals = evals.to(device)
                evecs = evecs.to(device)

        evals = evals.reshape(batch_size, num_points, 3)
        e1, e2, e3 = evals[:, :, 0:1], evals[:, :, 1:2], evals[:, :, 2:3]
        eval_sum = evals.sum(dim=-1, keepdim=True)

        feats = torch.cat([
            dist_pointwise,                                                          # 1
            evals.permute(0, 2, 1),                                                  # 2-4
            torch.log1p(dist_pointwise),                                             # 5
            dist_pointwise ** 2,                                                     # 6
            ((e3 - e2) / torch.clip(e3, min=eps)).permute(0, 2, 1),                 # 7 linearity
            ((e2 - e1) / torch.clip(e3, min=eps)).permute(0, 2, 1),                 # 8 planarity
            (e1 / torch.clip(e3, min=eps)).permute(0, 2, 1),                        # 9 sphericity
            ((e3 - e1) / torch.clip(e3, min=eps)).permute(0, 2, 1),                 # 10 anisotropy
            ((e1 * e2 * e3) ** (1/3)).permute(0, 2, 1),                             # 11 omnivariance
            (-torch.mul(evals, torch.log(torch.clamp(evals, min=eps))
).sum(dim=-1, keepdim=True)).permute(0, 2, 1),  # 12 eigenentropy
            (e3 / torch.clip(eval_sum, min=eps)).permute(0, 2, 1),                  # 13 surface_variation
            diffusion,                                                               # 14-16
        ], dim=1)  # [B, 16, N]

        if return_evecs:
            flip_mat = torch.tensor([[[1.,0.,0.],[0.,0.,1.],[0.,1.,0.]]], device=device, dtype=evecs.dtype)
            left_handed = torch.linalg.det(evecs) < 0
            evecs[left_handed] = (evecs[left_handed] @ flip_mat).to(evecs.dtype)
            evecs = evecs.reshape(batch_size, num_points, 3, 3)
            if return_nb_dists:
                return feats, evecs, evals, dist_nb
            return feats, evecs, evals
        if return_nb_dists:
            return feats, dist_nb
        return feats

    def _embed_nb_dists(self, dist_nb: torch.Tensor, mlp: nn.Sequential) -> torch.Tensor:
        """Embed sorted neighbour distances into a fixed-size per-point vector.

        Applies ``mlp`` independently to each scalar distance, then max-pools
        over the k neighbours.  This is PointNet-style aggregation: it is
        permutation-invariant over the neighbourhood, and the sorted order just
        ensures determinism.

        Args:
            dist_nb: ``[B, N, k]`` sorted neighbour distances.
            mlp: shared MLP mapping ``(1,) → (_nb_embed_dim,)``.

        Returns:
            ``[B, _nb_embed_dim, N]`` per-point neighbourhood embedding.
        """
        B, N, k = dist_nb.shape
        # [B*N, k, 1] → per-distance hidden vectors
        x = dist_nb.reshape(B * N, k, 1)
        x = mlp(x)                          # [B*N, k, _nb_embed_dim]
        x = x.max(dim=1).values             # [B*N, _nb_embed_dim]  — max-pool over k
        x = x.reshape(B, N, self._nb_embed_dim).permute(0, 2, 1)  # [B, _nb_embed_dim, N]
        return x

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

            # Local Neighborhood

            if self.num_local >= num_points - 1:
                dist_neighborhood, neighbor_indices = \
                    dist_pairwise.clone(), torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)
            else:
                dist_neighborhood, neighbor_indices = \
                    dist_pairwise.topk(self.num_local, dim = -1, largest = False)  # [B, N, self.num_local]

            # Context

            attn_mask = torch.zeros(batch_size, num_points, num_points, device = device, dtype = torch.bool)

            # Sparse attention mask, global plus local neighborhood

            if self.sparse:

                if self.num_local >= num_points - 1:

                    # Local neighborhood is all other points, attend to everything
                    attn_mask.fill_(True)
                    indices = torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)

                elif self.num_local + self.num_global < num_points:

                    fps_idx_by_seed = self._seeded_exclusive_fps(
                        xyz              = xyz_T,
                        dist_pairwise    = dist_pairwise,
                        neighbor_indices = neighbor_indices,
                        npoint           = self.num_global,
                        return_points    = False,
                    )

                    indices = torch.cat([neighbor_indices, fps_idx_by_seed], dim = -1)
                    attn_mask.scatter_(dim = 2, index = indices, value = True)
                
                else:
                    # Local neighborhood + global fps sample is all points, attend to everything
                    attn_mask.fill_(True)
                    indices = torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)

            # Full attention: all points attend to all other points
            else:
                attn_mask.fill_(True)
                indices = torch.arange(num_points, device = device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_points, -1)

            # Compute features at two scales and concatenate
            # Also get evecs/evals at fine scale for positional encoding
            features_fine, evecs, evals, dist_nb_fine = self._scale_features(
                xyz_T, dist_pairwise, dist_pointwise,
                self.num_local, batch_size, num_points, device, dtype, eps,
                return_evecs=True, return_nb_dists=True
            )
            features_coarse, dist_nb_coarse = self._scale_features(
                xyz_T, dist_pairwise, dist_pointwise,
                min(self.num_local_coarse, num_points - 1), batch_size, num_points, device, dtype, eps,
                return_nb_dists=True
            )
            features_stats = torch.cat([features_fine, features_coarse], dim=1)  # [B, 32, N]

            pos_enc = self.positional_encoding_features(
                xyz = xyz, 
                evecs = evecs,
                evals = evals, 
                dist_pointwise = dist_pointwise, 
                dist_pairwise = dist_pairwise, 
                indices = indices
            )

        # Neighbour-distance embeddings are computed outside no_grad so that the
        # MLP parameters receive gradients during training.
        nb_embed_fine   = self._embed_nb_dists(dist_nb_fine,   self.nb_dist_mlp_fine)    # [B, _nb_embed_dim, N]
        nb_embed_coarse = self._embed_nb_dists(dist_nb_coarse, self.nb_dist_mlp_coarse)  # [B, _nb_embed_dim, N]

        # Full feature vector: scalar stats (32) + nb embeddings (16 × 2) = 64
        features = torch.cat([features_stats, nb_embed_fine, nb_embed_coarse], dim=1)  # [B, 64, N]

        return features, indices, pos_enc

class SparsePointCloudAttention(nn.Module):
    """
    Attention block which attends to token indices externally provided.
    Supports dense/global attention when indices are not provided.

    Allows for different attention mechanisms:
        none: base attention
        add: additional term (pointTransformerV2 method) based on relative position encoding
        weight: elementwise weighting based on relative position encoding
    
    Allows for passing a global token whose update is computed by attending to all features.
    """
    def __init__(
            self,
            out_dim,
            model_dim = None,
            pos_enc_feature_dim = GeometricInvariantExtractor.pos_enc_dim,
            attn_augment : Literal["none", "add", "weight"] = "none",
        ):
        super().__init__()

        model_dim = model_dim if model_dim is not None else out_dim

        self.model_dim = model_dim
        self.out_dim = out_dim
        # Residual connection requires matching dimensions.  The model is always
        # instantiated with out_dim == model_dim; assert here to catch misuse early.
        assert out_dim == model_dim, (
            f"SparsePointCloudAttention: out_dim ({out_dim}) must equal model_dim ({model_dim}) "
            "for the residual connection.  Use a separate projection layer if you need dimension change."
        )

        self.pre_norm     = nn.LayerNorm(model_dim)
        self.enc_pre_norm = nn.LayerNorm(model_dim)

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.o_linear = nn.Linear(model_dim, out_dim)
        self.attn_augment = str(attn_augment).lower()

        # enc_token cross-attention: token queries all point features
        self.enc_q_linear = nn.Linear(model_dim, model_dim)
        self.enc_out      = nn.Linear(model_dim, out_dim)

        if self.attn_augment in ["add", "weight"]:

            mlp_dim = model_dim * 2
            self.pos_enc_mlp = nn.Sequential(
                nn.Linear(pos_enc_feature_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, 1 if self.attn_augment == "add" else model_dim)
            )
        else:
            self.pos_enc_mlp = None

    def forward(
            self,
            features: torch.Tensor,
            enc_token : torch.Tensor    = None,
            indices : torch.Tensor      = None,
            pos_enc : torch.Tensor      = None,
        ):

        B, C, N = features.shape

        f_norm = self.pre_norm(features.transpose(1, 2))
        Q = self.q_linear(f_norm)
        K = self.k_linear(f_norm)
        V = self.v_linear(f_norm)

        if indices.dim() != 3:
            raise ValueError(
                f"indices must have shape [B, N, A], got {tuple(indices.shape)}"
            )
        if indices.shape[0] != B or indices.shape[1] != N:
            raise ValueError(
                f"indices must have leading shape {(B, N)}, got {tuple(indices.shape[:2])}"
            )

        indices = indices.long()
        A = indices.size(-1) # [B, N, A]

        batch_idx = torch.arange(B, device = features.device).view(B, 1, 1).expand(-1, N, A)

        K_gathered = K[batch_idx, indices]  # [B, N, A, C]
        V_gathered = V[batch_idx, indices]  # [B, N, A, C]

        similarity = Q.unsqueeze(2) * K_gathered

        if pos_enc is not None and self.attn_augment == "weight":
            similarity = similarity * self.pos_enc_mlp(pos_enc)

        scores = similarity.sum(dim = -1) / (C ** 0.5)  # [B, N, A]

        if pos_enc is not None and self.attn_augment == "add":
            scores = scores + self.pos_enc_mlp(pos_enc).squeeze(-1)

        attn = F.softmax(scores, dim = -1)

        out_features = self.o_linear((attn.unsqueeze(-1) * V_gathered).sum(dim = 2))

        # Update features with residual connection
        features = features + out_features.transpose(1, 2) # [B, C, N] to match input format

        # Update global token
        if enc_token is not None:
            Q_enc           = self.enc_q_linear(self.enc_pre_norm(enc_token.transpose(1, 2)))   # [B, 1, C]
            scores_enc      = (Q_enc @ K.transpose(1, 2)) / (C ** 0.5)      # [B, 1, N]
            attn_enc        = F.softmax(scores_enc, dim=-1)                  # [B, 1, N]
            out_encoding    = self.enc_out(attn_enc @ V)                     # [B, 1, out_dim]
            # enc_out projects to out_dim == model_dim (asserted above), so
            # the residual addition is always dimension-safe.
            enc_token       = enc_token + out_encoding.transpose(1, 2)       # [B, model_dim, 1]

        return features, enc_token

class RotationInvariantSparseAttention(nn.Module):
    """Rotation invariant sparse attention transformer encoder"""

    name = "risa"

    def __init__(
            self,
            sparse : bool           = True,
            num_global : int        = 16,
            num_local : int         = 32,
            num_blocks : int        = 2,
            model_dim : int         = 128,
            features_out_dim : int  = 128,
            encoding_out_dim : int  = 128,
            attn_augment : Literal["none", "add", "weight"] = "weight",
            encoding_method : Literal["mean", "max", "token"] = "token",
            grad_checkpoint : bool  = False,
        ):
        super().__init__()

        self.grad_checkpoint = grad_checkpoint
        self.encoding_out_dim = encoding_out_dim
        self.model_dim   = model_dim

        self.attn_augment = str(attn_augment).lower()
        if self.attn_augment not in ["none", "add", "weight"]:
            raise ValueError(f"Invalid attention augmentation method: {attn_augment}")

        self.encoding_method = str(encoding_method).lower()
        if self.encoding_method not in ["mean", "max", "token"]:
            raise ValueError(f"Invalid encoding method: {encoding_method}")

        self.feature_extractor = GeometricInvariantExtractor(
            sparse      = sparse,
            num_global  = num_global,
            num_local   = num_local,
        )

        self.feature_dim = self.feature_extractor.feature_dim

        if encoding_method == "token":
            self.init_encoding_prior()

        self.feature_embed = nn.Sequential(
            nn.Linear(self.feature_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

        self.blocks = nn.ModuleList([
            SparsePointCloudAttention(
                model_dim    = model_dim,
                out_dim      = model_dim,
                attn_augment = attn_augment,
            )
            for _ in range(num_blocks)
        ])

        self.features_out_dim = features_out_dim
        self.features_post = nn.Sequential(
            nn.Conv1d(model_dim, self.features_out_dim, 1),
            nn.BatchNorm1d(self.features_out_dim),
            nn.GELU()
        )

        self.init_postprocess_object_encoding()

    def init_encoding_prior(self):
        self.encoding_token = nn.Parameter(torch.empty(1, self.model_dim, 1))
        nn.init.xavier_uniform_(self.encoding_token)
    
    def get_encoding_prior(self, batch_size : int, dtype : torch.dtype, device : torch.device):
        return self.encoding_token.expand(batch_size, -1, -1).to(device = device, dtype = dtype)
    
    def init_postprocess_object_encoding(self):
        if self.encoding_method == "token":
            self.encoding_post = nn.Sequential(
                nn.Conv1d(self.model_dim, self.encoding_out_dim, 1),
                nn.BatchNorm1d(self.encoding_out_dim),
                nn.GELU()
            )
        else:
            self.encoding_post = nn.Sequential(
                nn.Conv1d(self.features_out_dim, self.encoding_out_dim, 1),
                nn.BatchNorm1d(self.encoding_out_dim),
                nn.GELU()
            )

    def postprocess_object_encoding(self, features : Optional[torch.Tensor], encoding_token : Union[torch.Tensor, None]):
        
        if self.encoding_method == "token":
            assert self.encoding_post is not None, "Encoding postprocessor must be initialized for object encoding mode 'token'"
            assert encoding_token is not None, "Encoding token must be provided for object encoding mode 'token'"
            assert encoding_token.dim() == 3 and encoding_token.shape[2] == 1, f"Encoding token must have shape [B, C, 1], got {encoding_token.shape}"
            encoding = encoding_token

        else:
            assert features is not None, "Features must be provided for object encoding mode 'mean' or 'max'"
            assert features.dim() == 3 and features.shape[1] == self.model_dim, f"Features must have shape [B, C, N], got {features.shape}"
            
            if self.encoding_method == "mean":
                encoding = features.mean(dim = 2, keepdim = True)
            elif self.encoding_method == "max":
                encoding = features.max(dim = 2, keepdim = True).values
            else:
                raise ValueError(f"Invalid encoding method: {self.encoding_method}")

        encoding = self.encoding_post(encoding)

        return encoding

    def embed(self, features : torch.Tensor):

        batch_size, feature_dim, num_points = features.shape
        features = features.transpose(1, 2).reshape(batch_size * num_points, feature_dim)
        features = self.feature_embed(features)
        features = features.reshape(batch_size, num_points, -1).transpose(1, 2).contiguous()

        return features

    def forward(self, x):

        features, indices, pos_enc = self.feature_extractor(x)

        features = self.embed(features)

        encoding_token = self.get_encoding_prior(
            batch_size = x.size(0),
            dtype = x.dtype,
            device = x.device
        ) if self.encoding_method == "token" else None

        for block in self.blocks:
            features, encoding_token = block(
                features,
                enc_token   = encoding_token,
                pos_enc     = pos_enc,
                indices     = indices
            )

        features = self.features_post(features)
        encoding = self.postprocess_object_encoding(
            features = features,
            encoding_token = encoding_token
        )

        return encoding, features

if __name__ == "__main__":

    """
    Ablation Tests
        Attention augment:
            Elementwise weighting (default)
            Additional term (pointTransformerV2 method)
            None (base attention)

        Attention masking:
            No local neighborhood
            No global sample
            Full attention

        Geometric Invariant features:
            Switch from distance + PCA to traditional (XYZ)
                Still use masked attention and relative position encoding as in base model
    """

    import numpy as np
    from tqdm import tqdm

    seed = 0
    num_points = 192
    num_global = 10

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = RotationInvariantSparseAttention(num_global = num_global).to(device).eval()

    dummy_points = torch.from_numpy(np.random.uniform(-1, 1, (1, 3, num_points))).to(device = device, dtype = torch.float32)

    from scipy.spatial.transform import Rotation

    num_rotations = 1000
    rotations = Rotation.random(num_rotations)
    rotations = torch.from_numpy(rotations.as_matrix()).to(device = device, dtype = torch.float32)
    encodings = torch.empty((num_rotations, model.model_dim), device = device, dtype = torch.float32)

    with torch.no_grad():
        for i in tqdm(range(num_rotations)):
            rotated_points = rotations[i] @ dummy_points
            encoding, features = model(rotated_points)
            encodings[i] = encoding.squeeze()

    print(torch.linalg.det(torch.cov(encodings)))
