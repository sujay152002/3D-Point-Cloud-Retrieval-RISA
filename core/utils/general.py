import logging
import sys
from pathlib import Path
from typing import Dict

import torch


def init_logger(save_directory: Path) -> logging.Logger:
    name = str(save_directory)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

        fh = logging.FileHandler(save_directory / "train.log", mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    return logger


def safe_batch_size(model_cls, num_points, requested_bs, device):
    """Find largest batch size that fits in GPU memory for this model."""
    if not torch.cuda.is_available():
        return requested_bs
    for bs in [requested_bs, requested_bs // 2, requested_bs // 4, max(1, requested_bs // 8)]:
        try:
            torch.cuda.empty_cache()
            m = model_cls().to(device)
            x = torch.randn(bs, 3, num_points, device=device)
            with torch.no_grad():
                m(x)
            del m, x
            torch.cuda.empty_cache()
            return bs
        except RuntimeError:
            torch.cuda.empty_cache()
    return 1


def all_model_classes() -> Dict[str, type]:
    from models import (
        DGCNNEncoder,
        PointMambaEncoder,
        PointNet2Encoder,
        RISAPTv3,
        RotationInvariantSparseAttention,
        RSCNNEncoder,
    )
    return {
        "risa":      RotationInvariantSparseAttention,
        "risa_ptv3": RISAPTv3,
        "dgcnn":     DGCNNEncoder,
        "mamba":     PointMambaEncoder,
        "pointnet2": PointNet2Encoder,
        "rscnn":     RSCNNEncoder,
    }
