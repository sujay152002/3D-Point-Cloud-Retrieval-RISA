"""Decoder heads for all training tasks.

All decoders expect a global shape encoding z: (B, D) produced by the encoder
wrapper in the Trainer.  For tasks that benefit from per-point features (RISA
encoder), an optional ``point_features`` tensor (B, N, D_feat) may also be
supplied to the segmentation and denoise decoders.

Input convention for spatial coordinates: channels-first (B, 3, N).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class ClassificationDecoder(nn.Module):
    """MLP classifier applied to a global shape encoding."""

    def __init__(
        self,
        num_classes: int,
        encoding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            activation(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (B, D) global encoding
        Returns:
            (B, num_classes) logits
        """
        return self.classifier(z)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

class ReconstructionDecoder(nn.Module):
    """MLP that decodes a global encoding into a point cloud."""

    def __init__(
        self,
        encoding_dim: int,
        hidden_dim: int = 512,
        num_points: int = 1024,
        dropout: float = 0.3,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        self.num_points = num_points
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_points * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (B, D) global encoding
        Returns:
            (B, num_points, 3) reconstructed point cloud
        """
        return self.decoder(z).view(z.size(0), self.num_points, 3)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

class SegmentationDecoder(nn.Module):
    """Per-point segmentation head following the standard research benchmark.

    Concatenates per-point features with the global encoding AND a one-hot
    category label (when provided), matching the PointNet/DGCNN benchmark setup.

    Args:
        encoding_dim:  Dimension of per-point feature vector.
        num_classes:   Number of part classes to predict.
        num_categories: Number of object categories for one-hot conditioning.
                        Set to 0 to disable category conditioning.
        hidden_dim:    Width of intermediate Conv1d layers.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        encoding_dim: int,
        num_classes: int,
        num_categories: int = 0,
        global_encoding_dim: Optional[int] = None,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.num_categories = num_categories
        global_dim = global_encoding_dim if global_encoding_dim is not None else encoding_dim
        # xyz + per-point features + global encoding broadcast + one-hot category
        in_dim = 3 + encoding_dim + global_dim + num_categories
        self._global_dim = global_dim

        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_classes, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        xyz: torch.Tensor,
        point_features: Optional[torch.Tensor] = None,
        category: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            z:              (B, D) global encoding
            xyz:            (B, 3, N) point positions (channels-first)
            point_features: (B, N, D_feat) per-point features (e.g. from RISA).
                            When provided, concatenated with broadcast of z.
            category:       (B,) integer category labels for one-hot conditioning.
                            When provided, one-hot is broadcast to all points.
        Returns:
            (B, num_classes, N) per-point class logits
        """
        B, _, N = xyz.shape

        # Per-point features: use RISA features if available, else broadcast global
        if point_features is not None:
            feat = point_features.permute(0, 2, 1)          # (B, D, N)
        else:
            # For models without per-point features, build local features from xyz
            # by concatenating xyz with broadcast global to give each point local context
            feat = z.unsqueeze(-1).expand(-1, -1, N)        # (B, D, N)

        # Always also broadcast global encoding for shape-level context
        global_feat = z.unsqueeze(-1).expand(-1, -1, N)     # (B, D, N)

        parts = [xyz, feat, global_feat]

        # One-hot category conditioning
        if category is not None and self.num_categories > 0:
            one_hot = F.one_hot(category, self.num_categories).float()  # (B, C)
            one_hot = one_hot.unsqueeze(-1).expand(-1, -1, N)           # (B, C, N)
            parts.append(one_hot)

        combined = torch.cat(parts, dim=1)
        return self.mlp(combined)                            # (B, num_classes, N)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

class DenoiseDecoder(nn.Module):
    """Multi-head cross-attention denoising decoder with two refinement passes.

    Args:
        encoding_dim:      Dimension of the global shape encoding z.
        point_feature_dim: Dimension of per-point query features (RISA feat_dim or 3).
        hidden_dim:        Attention hidden dimension.
        num_heads:         Number of attention heads.
        dropout:           Dropout probability.
    """

    def __init__(
        self,
        encoding_dim: int,
        point_feature_dim: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        self.q_proj = nn.Linear(3, hidden_dim)
        self.k_proj = nn.Linear(point_feature_dim, hidden_dim)
        self.v_proj = nn.Linear(point_feature_dim, hidden_dim)
        self.z_proj = nn.Linear(encoding_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Second refinement pass (self-attention over denoised queries)
        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.k2 = nn.Linear(hidden_dim, hidden_dim)
        self.v2 = nn.Linear(hidden_dim, hidden_dim)
        self.o2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def _mha(self, Q, K, V, k_lin, v_lin, o_lin):
        """Multi-head scaled dot-product attention. Q is already projected."""
        B, N, _ = Q.shape
        Nk = K.shape[1]
        Q_ = Q.view(B, N,  self.num_heads, self.head_dim).transpose(1, 2)
        K_ = k_lin(K).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V_ = v_lin(V).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** 0.5
        attn  = F.softmax((Q_ @ K_.transpose(-1, -2)) / scale, dim=-1)
        out   = (attn @ V_).transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        return o_lin(out)

    def forward(
        self,
        noisy_xyz: torch.Tensor,
        z: torch.Tensor,
        point_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            noisy_xyz:      (B, 3, N) noisy point positions (channels-first)
            z:              (B, D) global shape encoding
            point_features: (B, N, D_feat) optional per-point features from RISA
        Returns:
            (B, N, 3) denoised point positions
        """
        xyz_t = noisy_xyz.permute(0, 2, 1)   # (B, N, 3)
        src   = point_features if point_features is not None else xyz_t

        # Project queries from noisy xyz, bias with global shape encoding
        # This ensures z conditions what each noisy point attends to
        Q1 = self.q_proj(xyz_t) + self.z_proj(z).unsqueeze(1)  # (B, N, H)

        # Pass 1: cross-attention — noisy points (conditioned on z) query RISA features
        x  = self.norm1(Q1 + self._mha(Q1, src, src, self.k_proj, self.v_proj, self.o_proj))

        # Pass 2: self-attention — points refine globally consistent representation
        x  = self.norm2(x  + self._mha(x,   x,   x,  self.k2,    self.v2,    self.o2))

        delta = self.out_proj(x)       # (B, N, 3)
        return xyz_t + delta           # (B, N, 3)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_decoder(
    task: str,
    decoder_kwargs: dict = {},
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> nn.Module:
    _registry = {
        "classification": ClassificationDecoder,
        "reconstruction": ReconstructionDecoder,
        "segmentation":   SegmentationDecoder,
        "denoise":        DenoiseDecoder,
    }
    if task not in _registry:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(_registry)}")

    decoder = _registry[task](**decoder_kwargs)
    if device is not None:
        decoder.to(device)
    if dtype is not None:
        decoder.to(dtype)
    return decoder
