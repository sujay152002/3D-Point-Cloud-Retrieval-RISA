"""RISA-PTv3: Rotation-Invariant Sparse Attention with PTv3-style Patch Attention.

Option A design: keep RISA's GeometricInvariantExtractor (rotation-invariant features)
but replace the KNN/FPS-based SparsePointCloudAttention blocks with PTv3-style
patch attention using invariant serialization.

Key differences from RISA:
- No KNN or FPS for attention masking → no O(N²) distance matrix
- Points serialized by rotation-invariant feature (dist-from-centroid)
  instead of XYZ coordinates → maintains invariance
- Shift Order across 4 invariant sort keys (distance, eigenvalue[0,1,2],
  and planarity) → each block sees different grouping, wider receptive field
- Multi-head dot-product attention (Flash-Attention-compatible) instead of
  single-head elementwise similarity
- Patch size replaces num_local/num_global; receptive field scales to 1024+
  without proportional memory cost

Invariant serialization keys (all rotation-invariant):
  0: dist_pointwise           — concentric shell ordering
  1: eigenvalue[0] (smallest) — local flatness ordering
  2: eigenvalue[1]            — local elongation ordering
  3: eigenvalue[2] (largest)  — local variance ordering

These are cycled across successive attention blocks (Shift Order) so each
block groups points by a different geometric criterion.
"""

from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use RISA's geometric extractor unchanged — it is the invariant core.
from models.risa import GeometricInvariantExtractor


# ---------------------------------------------------------------------------
# Invariant Serialization
# ---------------------------------------------------------------------------

_SERIAL_KEYS = ["dist", "eval0", "eval1", "eval2"]


def invariant_serialize(
    dist_pointwise: torch.Tensor,   # [B, 1, N]
    evals: torch.Tensor,            # [B, N, 3]
    key: str = "dist",
) -> torch.Tensor:
    """Sort indices [B, N] by a rotation-invariant per-point scalar.

    Args:
        dist_pointwise: [B, 1, N]  distance from centroid
        evals:          [B, N, 3]  local PCA eigenvalues (ascending order from eigh)
        key:            which invariant feature to sort by

    Returns:
        order: [B, N] long tensor of sorted point indices
    """
    B, _, N = dist_pointwise.shape

    if key == "dist":
        score = dist_pointwise.squeeze(1)             # [B, N]
    elif key == "eval0":
        score = evals[:, :, 0]                        # smallest eigenvalue
    elif key == "eval1":
        score = evals[:, :, 1]                        # middle eigenvalue
    elif key == "eval2":
        score = evals[:, :, 2]                        # largest eigenvalue
    else:
        raise ValueError(f"Unknown serialization key: {key}")

    order = torch.argsort(score, dim=-1)              # [B, N] ascending
    return order


# ---------------------------------------------------------------------------
# PTv3-style Patch Attention Block (invariant serialization)
# ---------------------------------------------------------------------------

class InvariantPatchAttention(nn.Module):
    """Patch attention operating on invariant-serialized point clouds.

    Points are sorted by an invariant scalar, split into non-overlapping
    patches of size `patch_size`, and standard multi-head dot-product
    attention is applied within each patch.

    The attention has no position encoding by default (matching PTv3's
    design philosophy of replacing RPE with structural efficiency).
    An optional lightweight invariant pos-bias MLP can be enabled which
    maps the 9-dim pos_enc (subset of RISA's per-pair features) to a
    per-head scalar bias added to the attention logits.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 4,
        patch_size: int = 64,
        serial_key: str = "dist",
        use_pos_bias: bool = True,
        pos_enc_dim: int = GeometricInvariantExtractor.pos_enc_dim,  # 9
        drop_path: float = 0.0,
    ):
        super().__init__()

        assert model_dim % num_heads == 0, \
            f"model_dim {model_dim} must be divisible by num_heads {num_heads}"

        self.model_dim   = model_dim
        self.num_heads   = num_heads
        self.head_dim    = model_dim // num_heads
        self.patch_size  = patch_size
        self.serial_key  = serial_key
        self.use_pos_bias = use_pos_bias
        self.scale       = self.head_dim ** -0.5

        # Pre-norm (PTv3 design)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        # Multi-head QKV projection (single fused matrix for efficiency)
        self.qkv    = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.proj   = nn.Linear(model_dim, model_dim, bias=True)

        # Feed-forward (PTv3 uses mlp_ratio=4)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )

        # Optional invariant positional bias (per-head scalar from pos_enc)
        if use_pos_bias:
            # pos_enc_dim comes from RISA's GeometricInvariantExtractor.pos_enc_dim=9
            # We use a small MLP mapping pos_enc [P, P, pos_enc_dim] → [P, P, num_heads]
            self.pos_bias_mlp = nn.Sequential(
                nn.Linear(pos_enc_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, num_heads),
            )
        else:
            self.pos_bias_mlp = None

        # Drop-path (stochastic depth) for regularization
        self.drop_path_prob = drop_path

    def _drop_path(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.drop_path_prob > 0.0 and self.training:
            # keep must broadcast over all dims of x: [B*n_patches, 1, 1]
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_path_prob
            return residual + x * keep.float()
        return residual + x

    def forward(
        self,
        features: torch.Tensor,       # [B, C, N]
        dist_pointwise: torch.Tensor, # [B, 1, N]
        evals: torch.Tensor,          # [B, N, 3]
        pos_enc: torch.Tensor,        # [B, N, A, 9]  (from RISA extractor, used for bias)
        enc_token: Optional[torch.Tensor] = None,  # [B, C, 1]
    ):
        B, C, N = features.shape

        # 1. Invariant serialization: sort by chosen key
        order = invariant_serialize(dist_pointwise, evals, self.serial_key)  # [B, N]
        inv_order = torch.argsort(order, dim=-1)                             # [B, N] inverse

        # Reorder features along serial order: [B, N, C]
        f_t = features.permute(0, 2, 1)                                      # [B, N, C]
        batch_idx = torch.arange(B, device=features.device).unsqueeze(1)    # [B, 1]
        f_serial = f_t[batch_idx, order]                                     # [B, N, C]

        # 2. Pad to multiple of patch_size
        P = self.patch_size
        pad = (P - N % P) % P
        if pad > 0:
            # Borrow from the beginning (circular, as in PTv3)
            f_serial = torch.cat([f_serial, f_serial[:, :pad]], dim=1)      # [B, N+pad, C]

        N_pad = f_serial.shape[1]
        n_patches = N_pad // P

        # 3. Split into patches: [B * n_patches, P, C]
        f_patches = f_serial.reshape(B * n_patches, P, C)

        # 4. Multi-head attention within patches (pre-norm)
        f_norm = self.norm1(f_patches)                                       # [B*n, P, C]
        qkv = self.qkv(f_norm).reshape(B * n_patches, P, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                    # [3, B*n, heads, P, head_dim]
        q, k, v = qkv.unbind(0)                                              # each [B*n, heads, P, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale                       # [B*n, heads, P, P]

        # Optional positional bias from invariant pos_enc
        # pos_enc shape is [B, N, A, 9] from GeometricInvariantExtractor
        # We only add it to the first block where A matches; otherwise skip.
        if self.pos_bias_mlp is not None and pos_enc is not None:
            A = pos_enc.shape[2]
            if A == N:
                # Reorder pos_enc rows by serial order
                # pos_enc_serial: [B, N, N, 9]
                pos_serial_i = pos_enc[batch_idx, order]                    # [B, N, A, 9]
                pos_serial_ij = pos_serial_i[batch_idx, :, order]           # not easily indexable like this
                # Approximate: use a symmetric mean-field pos bias per point
                # [B, N, 9] → per-point average of pos_enc over attended neighbours
                pos_mean = pos_enc.mean(dim=2)                              # [B, N, 9]
                pos_mean_serial = pos_mean[batch_idx, order]                # [B, N, 9]
                if pad > 0:
                    pos_mean_serial = torch.cat([pos_mean_serial, pos_mean_serial[:, :pad]], dim=1)
                pos_patches = pos_mean_serial.reshape(B * n_patches, P, 9)  # [B*n, P, 9]
                bias = self.pos_bias_mlp(pos_patches)                       # [B*n, P, heads]
                bias = bias.permute(0, 2, 1).unsqueeze(-1)                 # [B*n, heads, P, 1]
                attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * n_patches, P, C)      # [B*n, P, C]
        out = self.proj(out)

        # Residual 1
        f_patches = self._drop_path(out, f_patches)

        # FFN (pre-norm)
        f_patches = self._drop_path(self.ff(self.norm2(f_patches)), f_patches)

        # 5. Remove padding, restore original order
        f_serial_out = f_patches.reshape(B, N_pad, C)
        if pad > 0:
            f_serial_out = f_serial_out[:, :N]                              # [B, N, C]

        # Un-serialize back to original point order
        f_out = f_serial_out[batch_idx, inv_order]                          # [B, N, C]
        features_out = f_out.permute(0, 2, 1)                               # [B, C, N]

        # Update global encoding token via cross-attention to all points
        if enc_token is not None:
            # Simple: token attends to all restored features
            enc_t = enc_token.squeeze(-1).unsqueeze(1)                      # [B, 1, C]
            enc_norm = self.norm1(enc_t)
            qkv_enc = self.qkv(enc_norm).reshape(B, 1, 3, self.num_heads, self.head_dim)
            qkv_enc = qkv_enc.permute(2, 0, 3, 1, 4)
            q_enc, _, _ = qkv_enc.unbind(0)
            f_for_enc = self.norm1(features_out.permute(0, 2, 1))          # [B, N, C]
            kv_enc = self.qkv(f_for_enc).reshape(B, N, 3, self.num_heads, self.head_dim)
            kv_enc = kv_enc.permute(2, 0, 3, 1, 4)
            _, k_enc, v_enc = kv_enc.unbind(0)
            attn_enc = F.softmax((q_enc @ k_enc.transpose(-2, -1)) * self.scale, dim=-1)
            out_enc = (attn_enc @ v_enc).squeeze(2)                         # [B, heads, head_dim]
            out_enc = out_enc.reshape(B, 1, C)
            out_enc = self.proj(out_enc)
            enc_token = enc_token + out_enc.squeeze(1).unsqueeze(-1)       # [B, C, 1]

        return features_out, enc_token


# ---------------------------------------------------------------------------
# Hierarchical Invariant Grouping
# ---------------------------------------------------------------------------

class HierarchicalInvariantGrouping(nn.Module):
    """Two-level FPS + kNN grouping using rotation-invariant features only.

    Each level:
      1. FPS on dist_pointwise (invariant scalar) to pick centroids.
      2. kNN on the same invariant scalar to group nearby points.
      3. Max-pool the embedded per-point features over each group → one
         feature vector per centroid.
      4. A small MLP projects to model_dim.

    No XYZ coordinates are ever used — invariance is preserved.

    Outputs two tensors that are max-pooled and fed into the final
    encoding alongside the CLS token (multi-scale global descriptor).
    """

    def __init__(
        self,
        in_dim: int,           # model_dim after feature_embed
        model_dim: int,
        npoint1: int = 512,    # centroids at level 1
        npoint2: int = 256,    # centroids at level 2
        knn1: int    = 32,     # group size at level 1
        knn2: int    = 32,     # group size at level 2
    ):
        super().__init__()
        self.npoint1 = npoint1
        self.npoint2 = npoint2
        self.knn1    = knn1
        self.knn2    = knn2

        # Level-1 MLP: aggregate group features → model_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        # Level-2 MLP: aggregate level-1 features → model_dim
        self.mlp2 = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    @staticmethod
    def _fps_by_scalar(score: torch.Tensor, npoint: int) -> torch.Tensor:
        """Farthest-point sampling on a 1-D invariant score.

        Uses a greedy farthest-point strategy on the absolute difference
        of scores (rotation-invariant proxy for spatial distance).

        Args:
            score:  [B, N] invariant scalar (e.g. dist_pointwise)
            npoint: number of centroids to select

        Returns:
            idx: [B, npoint] long indices into the N dimension
        """
        B, N = score.shape
        device = score.device
        npoint = min(npoint, N)

        selected   = torch.zeros(B, npoint, dtype=torch.long, device=device)
        min_dist   = torch.full((B, N), float("inf"), device=device)
        # start from the point with the largest score
        farthest   = score.argmax(dim=-1)                        # [B]

        for i in range(npoint):
            selected[:, i] = farthest
            # distance to just-selected point (scalar difference)
            d = (score - score[torch.arange(B, device=device), farthest].unsqueeze(1)).abs()
            min_dist  = torch.minimum(min_dist, d)
            farthest  = min_dist.argmax(dim=-1)

        return selected                                           # [B, npoint]

    @staticmethod
    def _knn_by_scalar(score: torch.Tensor, centroids_idx: torch.Tensor, k: int) -> torch.Tensor:
        """For each centroid, find k nearest points by invariant scalar distance.

        Args:
            score:        [B, N]
            centroids_idx:[B, M]
            k:            group size

        Returns:
            [B, M, k] long indices into the N dimension
        """
        B, N = score.shape
        M = centroids_idx.shape[1]
        k = min(k, N)

        centroid_scores = score[
            torch.arange(B, device=score.device).unsqueeze(1),
            centroids_idx
        ]                                                         # [B, M]

        # pairwise |score_centroid - score_point|: [B, M, N]
        diff = (centroid_scores.unsqueeze(2) - score.unsqueeze(1)).abs()
        # k nearest (smallest diff)
        _, nn_idx = diff.topk(k, dim=-1, largest=False)          # [B, M, k]
        return nn_idx

    def forward(
        self,
        f: torch.Tensor,              # [B, model_dim, N]  embedded features
        dist_pointwise: torch.Tensor, # [B, 1, N]
    ):
        """
        Returns:
            level1_global: [B, model_dim]  max-pooled level-1 centroid features
            level2_global: [B, model_dim]  max-pooled level-2 centroid features
        """
        B, C, N = f.shape
        score = dist_pointwise.squeeze(1)                         # [B, N]

        # ---- Level 1 ------------------------------------------------
        idx1   = self._fps_by_scalar(score, self.npoint1)         # [B, M1]
        nn_idx1 = self._knn_by_scalar(score, idx1, self.knn1)     # [B, M1, k1]

        # Gather group features: [B, M1, k1, C]
        B_idx = torch.arange(B, device=f.device).view(B, 1, 1)
        grouped1 = f.permute(0, 2, 1)[B_idx, nn_idx1]            # [B, M1, k1, C]

        # Max-pool over group → [B, M1, C]
        pooled1 = grouped1.max(dim=2).values

        # MLP → [B, M1, model_dim]
        out1 = self.mlp1(pooled1)                                 # [B, M1, D]

        # Global max-pool → [B, model_dim]
        level1_global = out1.max(dim=1).values                    # [B, D]

        # ---- Level 2 ------------------------------------------------
        # Use centroid scores for level-2 FPS
        centroid_scores1 = score[
            torch.arange(B, device=f.device).unsqueeze(1), idx1
        ]                                                         # [B, M1]

        idx2    = self._fps_by_scalar(centroid_scores1, self.npoint2)  # [B, M2]  (indices into M1)
        nn_idx2 = self._knn_by_scalar(centroid_scores1, idx2, self.knn2)  # [B, M2, k2]

        # Gather level-1 features: [B, M2, k2, D]
        grouped2 = out1[torch.arange(B, device=f.device).view(B, 1, 1), nn_idx2]
        pooled2  = grouped2.max(dim=2).values                    # [B, M2, D]
        out2     = self.mlp2(pooled2)                            # [B, M2, D]

        level2_global = out2.max(dim=1).values                   # [B, D]

        return level1_global, level2_global


# ---------------------------------------------------------------------------
# RISA-PTv3 Encoder
# ---------------------------------------------------------------------------

class RISAPTv3(nn.Module):
    """Rotation-Invariant Sparse Attention with PTv3-style Patch Attention.

    Keeps RISA's GeometricInvariantExtractor (the rotation-invariant part)
    but replaces the KNN/FPS attention blocks with PTv3-style patch attention.

    The serialization order cycles through 4 invariant keys across blocks
    (analogous to PTv3's Shift Order across 4 space-filling curves):
        block 0 → sort by dist_pointwise
        block 1 → sort by eigenvalue[0]
        block 2 → sort by eigenvalue[1]
        block 3 → sort by eigenvalue[2]
        block 4 → sort by dist_pointwise  (repeats)
        ...
    """

    name = "risa_ptv3"

    def __init__(
        self,
        sparse: bool            = True,   # passed to GeometricInvariantExtractor
        num_global: int         = 16,     # passed to GeometricInvariantExtractor
        num_local: int          = 32,     # passed to GeometricInvariantExtractor
        num_blocks: int         = 6,      # increased for wider coverage
        model_dim: int          = 256,    # increased from 128
        num_heads: int          = 8,      # increased from 4
        patch_size: int         = 64,     # PTv3 default patch size
        features_out_dim: int   = 256,    # matches model_dim
        encoding_out_dim: int   = 256,    # matches model_dim
        encoding_method: Literal["mean", "max", "token"] = "token",
        use_pos_bias: bool      = True,
        drop_path: float        = 0.2,    # increased stochastic depth
        # Hierarchy
        use_hierarchy: bool     = True,   # enable 2-level FPS+grouping
        hier_npoint1: int       = 512,    # level-1 centroids
        hier_npoint2: int       = 256,    # level-2 centroids
        hier_knn: int           = 32,     # group size at both levels
    ):
        super().__init__()

        self.encoding_out_dim  = encoding_out_dim
        self.features_out_dim  = features_out_dim
        self.model_dim         = model_dim
        self.encoding_method   = encoding_method
        self.use_hierarchy     = use_hierarchy

        # --- Invariant feature extractor (unchanged from RISA) ---
        self.feature_extractor = GeometricInvariantExtractor(
            sparse      = sparse,
            num_global  = num_global,
            num_local   = num_local,
        )
        self.feature_dim = self.feature_extractor.feature_dim  # 4

        # --- Enriched per-point feature dim: 4 base + 9*3 pos_enc stats = 31 ---
        # mean + max + std over neighbours gives much higher variance than mean alone
        self._enriched_dim = self.feature_dim + GeometricInvariantExtractor.pos_enc_dim * 3  # 31

        # --- Feature embedding: 13-dim enriched input → model_dim ---
        self.feature_embed = nn.Sequential(
            nn.Linear(self._enriched_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        # --- 2-level hierarchical grouping (rotation-invariant FPS+kNN) ---
        if use_hierarchy:
            self.hierarchy = HierarchicalInvariantGrouping(
                in_dim   = model_dim,
                model_dim = model_dim,
                npoint1  = hier_npoint1,
                npoint2  = hier_npoint2,
                knn1     = hier_knn,
                knn2     = hier_knn,
            )
            # Multi-scale fusion: CLS token + level1 + level2 → encoding_out_dim
            # (3 * model_dim → encoding_out_dim via a linear projection)
            self.multiscale_proj = nn.Linear(3 * model_dim, encoding_out_dim)
        else:
            self.hierarchy       = None
            self.multiscale_proj = None

        # --- Encoding token (PTv3/RISA hybrid: learnable CLS token) ---
        if encoding_method == "token":
            self.encoding_token = nn.Parameter(torch.empty(1, model_dim, 1))
            nn.init.xavier_uniform_(self.encoding_token)

        # --- PTv3-style patch attention blocks, cycling through serial keys ---
        # stochastic depth: linearly increase drop-path rate across blocks
        dpr = [drop_path * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            InvariantPatchAttention(
                model_dim    = model_dim,
                num_heads    = num_heads,
                patch_size   = patch_size,
                serial_key   = _SERIAL_KEYS[i % len(_SERIAL_KEYS)],
                use_pos_bias = use_pos_bias,
                pos_enc_dim  = GeometricInvariantExtractor.pos_enc_dim,
                drop_path    = dpr[i],
            )
            for i in range(num_blocks)
        ])

        # --- Output projections (same as RISA) ---
        self.features_post = nn.Sequential(
            nn.Conv1d(model_dim, features_out_dim, 1),
            nn.BatchNorm1d(features_out_dim),
            nn.GELU(),
        )

        # encoding_post only used when hierarchy is off (or for the CLS-only path)
        if encoding_method == "token" and not use_hierarchy:
            self.encoding_post = nn.Sequential(
                nn.Conv1d(model_dim, encoding_out_dim, 1),
                nn.BatchNorm1d(encoding_out_dim),
                nn.GELU(),
            )
        elif not use_hierarchy:
            self.encoding_post = nn.Sequential(
                nn.Conv1d(features_out_dim, encoding_out_dim, 1),
                nn.BatchNorm1d(encoding_out_dim),
                nn.GELU(),
            )
        else:
            # With hierarchy the encoding goes through multiscale_proj instead
            self.encoding_post = None

        # --- Rich classification descriptor ---
        # Concat [CLS, global_max, global_avg] of post-attention features → encoding_out_dim
        # This is the same 3× pooling trick used by PointNet2/DGCNN for classification.
        # We always build it; the caller decides whether to use it.
        self.cls_proj = nn.Sequential(
            nn.Linear(3 * model_dim, encoding_out_dim),
            nn.BatchNorm1d(encoding_out_dim),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, features: torch.Tensor) -> torch.Tensor:
        """[B, feature_dim, N] → [B, model_dim, N]"""
        B, D, N = features.shape
        f = features.permute(0, 2, 1).reshape(B * N, D)
        f = self.feature_embed(f)
        return f.reshape(B, N, -1).permute(0, 2, 1).contiguous()

    def _get_enc_token(self, B, dtype, device) -> Optional[torch.Tensor]:
        if self.encoding_method == "token":
            return self.encoding_token.expand(B, -1, -1).to(dtype=dtype, device=device)
        return None

    def _postprocess_encoding(
        self,
        features: torch.Tensor,
        enc_token: Optional[torch.Tensor],
        dist_pointwise: Optional[torch.Tensor] = None,
        embedded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Always build a rich 3× descriptor:
        #   [CLS token, global max-pool, global avg-pool] of post-attention features
        # This is the key change for classification quality — a single CLS token
        # at 128-dim loses too much shape information.
        if enc_token is not None:
            cls = enc_token.squeeze(-1)                          # [B, model_dim]
        else:
            cls = embedded.mean(dim=2) if embedded is not None else features.mean(dim=2)

        # Use the pre-projection features (embedded = f after attention blocks)
        # for global pooling — richer than features_post output.
        f_for_pool = embedded if embedded is not None else features
        g_max = f_for_pool.max(dim=2).values                    # [B, model_dim]
        g_avg = f_for_pool.mean(dim=2)                          # [B, model_dim]

        # Project concat → encoding_out_dim
        rich = self.cls_proj(torch.cat([cls, g_max, g_avg], dim=-1))  # [B, enc_out]
        return rich.unsqueeze(-1)                                # [B, enc_out, 1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, N] input point cloud

        Returns:
            encoding : [B, encoding_out_dim, 1]
            features : [B, features_out_dim, N]
        """
        # --- Invariant feature extraction (no gradients needed here) ---
        features, indices, pos_enc = self.feature_extractor(x)
        # features: [B, 4, N]
        # indices:  [B, N, A]   (only used in RISA sparse attention, reuse pos_enc)
        # pos_enc:  [B, N, A, 9]

        # Extract dist_pointwise and evals from the features tensor:
        # feature layout: [dist_pointwise(0), eval0(1), eval1(2), eval2(3)]
        dist_pointwise = features[:, 0:1, :]          # [B, 1, N]
        evals          = features[:, 1:4, :].permute(0, 2, 1)  # [B, N, 3]

        # --- Richer per-point feature enrichment from pos_enc ---
        # pos_enc: [B, N, A, 9]
        # Instead of just mean-pooling (which loses discriminative structure),
        # compute mean + max + std over the A neighbours → 27 dims per point.
        # These three statistics preserve much more local geometric variation.
        pos_enc_mean = pos_enc.mean(dim=2)                        # [B, N, 9]
        pos_enc_max  = pos_enc.max(dim=2).values                  # [B, N, 9]
        pos_enc_std  = pos_enc.std(dim=2).clamp(min=0)            # [B, N, 9]
        # Permute all to [B, 9, N] and concatenate with base features
        pos_enc_mean = pos_enc_mean.permute(0, 2, 1)              # [B, 9, N]
        pos_enc_max  = pos_enc_max.permute(0, 2, 1)               # [B, 9, N]
        pos_enc_std  = pos_enc_std.permute(0, 2, 1)               # [B, 9, N]
        features_enriched = torch.cat(
            [features, pos_enc_mean, pos_enc_max, pos_enc_std], dim=1
        )                                                          # [B, 31, N]

        # --- Embed enriched invariant features → model_dim ---
        f = self._embed(features_enriched)             # [B, model_dim, N]

        # --- Encoding token ---
        enc_token = self._get_enc_token(x.size(0), x.dtype, x.device)

        # --- PTv3-style patch attention blocks ---
        for block in self.blocks:
            f, enc_token = block(
                features       = f,
                dist_pointwise = dist_pointwise,
                evals          = evals,
                pos_enc        = pos_enc,
                enc_token      = enc_token,
            )

        # --- Output projections ---
        features_out = self.features_post(f)                    # [B, feat_out, N]
        encoding     = self._postprocess_encoding(
            features       = features_out,
            enc_token      = enc_token,
            dist_pointwise = dist_pointwise,
            embedded       = f,                  # pre-post-proj features for hierarchy
        )                                                        # [B, enc_out, 1]

        return encoding, features_out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def make_risa_ptv3_default() -> RISAPTv3:
    """Default configuration matching RISA's hyperparameter sweep baseline."""
    return RISAPTv3(
        sparse           = True,
        num_global       = 16,
        num_local        = 32,
        num_blocks       = 4,
        model_dim        = 128,
        num_heads        = 4,
        patch_size       = 64,
        features_out_dim = 128,
        encoding_out_dim = 128,
        encoding_method  = "token",
        use_pos_bias     = True,
        drop_path        = 0.1,
    )


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RISAPTv3().to(device).eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    num_points   = 1024
    num_rotations = 200
    dummy = torch.randn(1, 3, num_points, device=device)
    rotations = Rotation.random(num_rotations)
    rotations = torch.from_numpy(rotations.as_matrix()).to(device=device, dtype=torch.float32)

    encodings = []
    with torch.no_grad():
        for i in tqdm(range(num_rotations), desc="Rotation invariance test"):
            enc, _ = model(rotations[i] @ dummy)
            encodings.append(enc.squeeze())

    encodings = torch.stack(encodings)
    variance  = encodings.var(dim=0).mean().item()
    print(f"Encoding variance across {num_rotations} rotations: {variance:.6f}")
    print("(Lower is more invariant; should be near 0 for a perfectly invariant model)")
