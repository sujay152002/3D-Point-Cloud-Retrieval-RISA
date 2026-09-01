"""Representation similarity metrics used across all experiments.

Provides L2 distance, centered linear CKA (Kornblith et al., ICML 2019),
and cosine similarity between encoder output embeddings.
"""
import numpy as np
import torch
import torch.nn.functional as F


def compute_miou(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Global part-class mIoU: average IoU over part IDs pooled across all points."""
    ious = []
    for cls in range(num_classes):
        pred_mask  = (preds == cls)
        label_mask = (labels == cls)

        if label_mask.sum() == 0:
            continue

        intersection = (pred_mask & label_mask).sum().float()
        union        = (pred_mask | label_mask).sum().float()
        ious.append((intersection / union).item() if union > 0 else 0.0)

    return float(np.mean(ious)) if ious else 0.0


def compute_instance_miou(
    preds: torch.Tensor,
    labels: torch.Tensor,
    categories: torch.Tensor,
    seg_classes: list,
) -> float:
    """ShapeNet-standard instance mIoU: per-shape IoU over valid parts, then mean.

    Parts absent from both pred and GT count as IoU=1.0, matching the PointNet
    benchmark convention and preventing deflation for shapes with few parts.
    """
    if preds.dim() == 1:
        raise ValueError("compute_instance_miou expects per-point preds with shape (B, N)")

    shape_ious = []
    for b in range(preds.shape[0]):
        cat = int(categories[b].item())
        if cat < 0 or cat >= len(seg_classes):
            continue

        part_ious = []
        for part_id in seg_classes[cat]:
            pred_mask  = preds[b] == part_id
            label_mask = labels[b] == part_id
            if pred_mask.sum() == 0 and label_mask.sum() == 0:
                # Part absent from both: perfect score per benchmark convention
                part_ious.append(1.0)
                continue
            intersection = (pred_mask & label_mask).sum().float()
            union        = (pred_mask | label_mask).sum().float()
            part_ious.append((intersection / union).item() if union > 0 else 0.0)

        if part_ious:
            shape_ious.append(float(np.mean(part_ious)))

    return float(np.mean(shape_ious)) if shape_ious else 0.0


def segmentation_point_mask(
    labels: torch.Tensor,
    categories: torch.Tensor,
    seg_classes: list,
) -> torch.Tensor:
    """Boolean mask (B, N) for points whose part label is valid for its category."""
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for b in range(labels.shape[0]):
        cat = int(categories[b].item())
        if cat < 0 or cat >= len(seg_classes):
            continue
        for part_id in seg_classes[cat]:
            mask[b] |= labels[b] == part_id
    return mask


def mask_segmentation_logits(
    logits: torch.Tensor,
    categories: torch.Tensor,
    seg_classes: list,
) -> torch.Tensor:
    """Suppress logits for part classes invalid for each object's category.

    Uses a large negative finite value (-1e9) so softmax assigns near-zero
    probability to masked classes.  We do NOT use float('-inf') because
    label_smoothing in cross_entropy spreads a small mass to all classes;
    -inf targets produce inf loss.  Instead we disable label_smoothing when
    masked logits are present (see _compute_loss in trainer.py).
    """
    masked = logits.clone()
    num_classes = masked.shape[1]
    for b in range(masked.shape[0]):
        cat = int(categories[b].item())
        if cat < 0 or cat >= len(seg_classes):
            continue
        valid = set(seg_classes[cat])
        for part_id in range(num_classes):
            if part_id not in valid:
                masked[b, part_id, :] = -1e4
    return masked


def compute_l2_dist(z1: torch.Tensor, z2: torch.Tensor) -> float:
    return torch.norm(z1 - z2, dim=-1).mean().item()


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA (Centered Kernel Alignment).

    Automatically selects the numerically stable form based on the ratio of
    samples N to embedding dimension D:

    - When N <= D: uses the (N, N) sample Gram-matrix form.  Stable when
      there are few samples relative to dimension (e.g. a small eval batch).
    - When N > D: uses the (D, D) feature cross-covariance form, which avoids
      building an (N, N) Gram matrix that becomes near-rank-deficient when N
      is large and encodings lie on a low-dimensional manifold.  This is the
      regime that caused near-zero CKA in reconstruction eval with 512-d
      encoders accumulated over many batches.

    Both forms are mathematically equivalent for linear kernels but have very
    different numerical properties depending on the N vs D regime.

    Args:
        X: (N, D1) tensor
        Y: (N, D2) tensor — must have same N and D as X (D1 == D2 enforced
           by the feature-space branch; sample branch is dimension-agnostic).

    Returns:
        CKA scalar in [0, 1]
    """
    if X.dim() > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.shape[0], -1)

    assert X.size(0) == Y.size(0), "X and Y must have the same number of samples"
    if X.size(0) < 2:
        return 1.0

    if Y.device != X.device:
        Y = Y.to(X.device)

    N, D = X.shape

    if N > D:
        # Feature-space (D, D) form — stable when N >> D.
        # Equivalent to the sample form but avoids rank-deficient (N,N) Grams.
        def _center_features(Z: torch.Tensor) -> torch.Tensor:
            return Z - Z.mean(dim=0, keepdim=True)

        Xc = _center_features(X)   # (N, D)
        Yc = _center_features(Y)   # (N, D)
        # Cross-covariance matrices
        SXX = Xc.t() @ Xc          # (D, D)
        SYY = Yc.t() @ Yc          # (D, D)
        SXY = Xc.t() @ Yc          # (D, D)
        num = (SXY * SXY).sum()                          # ||SXY||_F^2
        den = (SXX * SXX).sum().sqrt() * (SYY * SYY).sum().sqrt()  # ||SXX||_F * ||SYY||_F
    else:
        # Sample-space (N, N) form — stable when N <= D.
        def _center_gram(K: torch.Tensor) -> torch.Tensor:
            n = K.shape[0]
            H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
            return H @ K @ H

        Kc = _center_gram(X @ X.t())
        Lc = _center_gram(Y @ Y.t())
        num = (Kc * Lc).sum()
        den = (Kc * Kc).sum().sqrt() * (Lc * Lc).sum().sqrt()

    cka = num / (den + 1e-8)
    if torch.isnan(cka) or torch.isinf(cka):
        return 0.0
    return cka.clamp(0.0, 1.0).item()


def compute_cosine_similarity(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Mean cosine similarity between corresponding row vectors."""
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)

    z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-8)
    z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True) + 1e-8)

    return (z1_norm * z2_norm).sum(dim=-1).mean().item()


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer Distance (mean over both directions, averaged over batch).

    Args:
        pred: (B, N, 3)
        gt:   (B, M, 3)
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)   # (B, N, M, 3)
    dist = diff.pow(2).sum(dim=-1)               # (B, N, M)
    min_pred_to_gt = dist.min(dim=2)[0].mean()
    min_gt_to_pred = dist.min(dim=1)[0].mean()
    return (min_pred_to_gt + min_gt_to_pred) / 2.0


def chamfer_distance_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-sample Chamfer Distance.

    Args:
        pred: (B, N, 3)
        gt:   (B, M, 3)

    Returns:
        (B,) tensor
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1)
    min_pred_to_gt = dist.min(dim=2)[0].mean(dim=1)
    min_gt_to_pred = dist.min(dim=1)[0].mean(dim=1)
    return (min_pred_to_gt + min_gt_to_pred) / 2.0


def min_rotated_chamfer_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_rotations: int = 64,
) -> float:
    """Mean per-sample minimum Chamfer distance over random rotations applied to gt."""
    from scipy.spatial.transform import Rotation as SciRotation

    B = pred.shape[0]
    rotations = torch.from_numpy(
        SciRotation.random(num_rotations).as_matrix()
    ).to(device=pred.device, dtype=pred.dtype)

    best_cds = torch.full((B,), float('inf'), device=pred.device, dtype=pred.dtype)

    with torch.inference_mode():
        for r in range(num_rotations):
            R = rotations[r]
            gt_rot = (R @ gt.transpose(-1, -2)).transpose(-1, -2)
            cd = chamfer_distance_per_sample(pred, gt_rot)
            best_cds = torch.minimum(best_cds, cd)

    return best_cds.mean().item()
