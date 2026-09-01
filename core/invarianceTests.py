"""Rotation-invariance testing utilities.

Takes an encoder and a point cloud, rotates the cloud in many ways, then
measures how much the encoder output changes.  A perfectly SE(3)-invariant
encoder should produce identical encodings for all rotations.
"""

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


class InvarianceTester:
    """Test rotation invariance of a point-cloud encoder.

    Usage
    -----
        tester = InvarianceTester(encoder, num_rotations=100)
        results = tester.run(xyz)   # xyz: (1, 3, N) or (B, 3, N)
        print(results['std_mean'])  # near 0 ⇒ invariant
    """

    def __init__(self, architecture: torch.nn.Module, num_rotations: int = 100):
        self.architecture  = architecture
        self.num_rotations = num_rotations
        self._rotations    = None   # lazy-init so device can be matched to input

    def _get_rotations(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate (or return cached) random rotation matrices."""
        if self._rotations is None:
            mats = Rotation.random(self.num_rotations).as_matrix()
            self._rotations = torch.from_numpy(mats).to(device=device, dtype=dtype)
        return self._rotations.to(device=device, dtype=dtype)

    @torch.no_grad()
    def run(self, xyz: torch.Tensor) -> dict:
        """Evaluate rotation invariance over many random rotations.

        Parameters
        ----------
        xyz : (B, 3, N) or (1, 3, N) point cloud tensor.

        Returns
        -------
        dict with:
            encodings   – (num_rotations, D) all encodings (first sample)
            std_mean    – mean std across encoding dims (lower ⇒ more invariant)
            cos_min     – minimum pairwise cosine similarity (higher ⇒ more invariant)
            cos_mean    – mean pairwise cosine similarity
        """
        was_training = self.architecture.training
        self.architecture.eval()

        device    = xyz.device
        dtype     = xyz.dtype
        rotations = self._get_rotations(device, dtype)  # (R, 3, 3)

        # Take single sample for rotation test
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)
        sample = xyz[:1]   # (1, 3, N)

        encodings = []
        for R in rotations:
            rotated = R @ sample          # (1, 3, N) – rotate all N columns
            out = self.architecture(rotated)
            enc = out[0].squeeze(-1) if isinstance(out, tuple) else out
            encodings.append(enc.squeeze(0))   # (D,)

        encodings = torch.stack(encodings)     # (R, D)

        # Statistics
        std_mean = encodings.std(dim=0).mean().item()

        # Pairwise cosine similarity (lower triangular)
        enc_norm  = F.normalize(encodings, dim=-1)
        cos_matrix = enc_norm @ enc_norm.T     # (R, R)
        mask       = torch.tril(torch.ones_like(cos_matrix, dtype=torch.bool), diagonal=-1)
        cos_vals   = cos_matrix[mask]

        if was_training:
            self.architecture.train()

        return {
            "encodings": encodings,
            "std_mean":  std_mean,
            "cos_min":   cos_vals.min().item()  if cos_vals.numel() > 0 else 1.0,
            "cos_mean":  cos_vals.mean().item() if cos_vals.numel() > 0 else 1.0,
        }
