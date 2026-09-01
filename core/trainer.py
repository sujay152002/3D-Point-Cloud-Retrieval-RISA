"""Unified Trainer for 3D point-cloud encoding experiments.

Supported tasks
---------------
classification  – object-level multi-class classification
segmentation    – per-point part segmentation
reconstruction  – rotation-corrected shape reconstruction
denoise         – cross-attention denoising

Encoder API contract
--------------------
All encoders receive (B, 3, N) input (channels-first).

* Baseline encoders (DGCNN, PointNet2, Mamba, RSCNN) return a flat (B, D) global encoding.
* RISA returns a tuple:
    encoding   : (B, enc_dim, 1)  – global shape token
    features   : (B, feat_dim, N) – per-point transformer features
  Both are normalised via ``unpack_encoder_output``.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import _ALL_DATASETS, outputs_dir
from core.utils.general import init_logger, safe_batch_size
from core.memory_utils import cleanup_cuda_memory, maybe_cleanup_after_batch, memory_summary
from models.decoders import get_decoder
from core.metrics import (
    chamfer_distance,
    chamfer_distance_per_sample,
    compute_cka,
    compute_cosine_similarity,
    compute_instance_miou,
    compute_miou,
    mask_segmentation_logits,
    min_rotated_chamfer_distance,
    segmentation_point_mask,
)


# ---------------------------------------------------------------------------
# Encoder output normalisation
# ---------------------------------------------------------------------------

def unpack_encoder_output(out) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalise encoder output to ``(encoding, point_features_or_None)``."""
    if isinstance(out, tuple):
        encoding, features = out
        # RISA: encoding (B, enc_dim, 1), features (B, feat_dim, N)
        return encoding.squeeze(-1), features.permute(0, 2, 1)
    return out, None


def probe_encoder(encoder: nn.Module, device: torch.device, num_points: int = 256) -> Tuple[int, Optional[int]]:
    """Determine output dimensions via a dummy forward pass."""
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, num_points, device=device)
        out = encoder(dummy)
    if was_training:
        encoder.train()

    if isinstance(out, tuple):
        enc, feat = out
        return enc.squeeze(-1).shape[-1], feat.shape[1]
    return out.shape[-1], None


# ---------------------------------------------------------------------------
# Batch unpacking (handles 2-tuple and 3-tuple from ShapeNet)
# ---------------------------------------------------------------------------

def _unpack_batch(batch, task: str):
    """Unpack a batch that may be (pts, lbl) or (pts, obj_lbl, part_lbl).

    ShapeNet returns a 3-tuple: (pts, object_label, per_point_part_label).
    - segmentation returns both object and part labels
    - all other tasks use object_label
    """
    if len(batch) == 3:
        pts, obj_lbl, part_lbl = batch
        if task == "segmentation":
            return pts, (obj_lbl, part_lbl)
        return pts, obj_lbl
    return batch[0], batch[1]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_num_classes(loader: DataLoader) -> int:
    # Walk through Subset / wrapper layers to find the base dataset
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    for attr in ("num_classes", "categories", "class_counts"):
        val = getattr(ds, attr, None)
        if val is not None:
            return len(val) if hasattr(val, "__len__") else int(val)
    batch = next(iter(loader))
    _, labels = _unpack_batch(batch, "classification")
    return int(labels.max().item()) + 1


def get_num_points(loader: DataLoader) -> int:
    batch = next(iter(loader))
    points, _ = _unpack_batch(batch, "reconstruction")
    return points.shape[-1]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer for all point-cloud tasks."""

    TASKS     = ("classification", "segmentation", "reconstruction", "denoise")
    NOISE_STD = 0.10
    GRAD_CLIP = 1.0

    def __call__(
        self,
        encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        task: str,
        dataset_name: str,
        device: torch.device,
        epochs: int = 100,
        decoder_kwargs: dict = {},
        save_directory: Path = outputs_dir / "results",
        verbose: bool = True,
        seed: Optional[int] = None,
        eval_rotations: int = 32,
    ) -> dict:
        if task not in self.TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {self.TASKS}")

        if seed is not None:
            import random
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        logger = init_logger(save_directory)

        encoder_name = getattr(encoder, "name", encoder.__class__.__name__)
        logger.info("=" * 60)
        logger.info(f"  {encoder_name}  |  {task}  |  {dataset_name}  |  {epochs} epochs")
        logger.info(f"  Save dir: {save_directory}")
        logger.info("=" * 60)

        enc_dim, feat_dim = probe_encoder(encoder, device)
        logger.info(f"  Encoding dim: {enc_dim}  |  Per-point feature dim: {feat_dim}")

        if task == "segmentation":
            batch = next(iter(train_loader))
            _, (_, sample_labels) = _unpack_batch(batch, "segmentation")
            if sample_labels.dim() < 2:
                # Broadcast object-level label to every point so segmentation can still run
                logger.info("  Segmentation: broadcasting object-level labels to per-point labels.")

        full_dk = self._build_decoder_kwargs(task, enc_dim, feat_dim, train_loader, decoder_kwargs)
        logger.info(f"  Decoder kwargs: {full_dk}")

        self._seg_classes = full_dk.pop("seg_classes", None)
        decoder   = get_decoder(task, full_dk, device, torch.float32)
        if task == "reconstruction":
            for p in encoder.parameters():
                p.requires_grad = False
            params = list(decoder.parameters())
        else:
            params = list(encoder.parameters()) + list(decoder.parameters())
        lr = 1e-3
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
        warmup_epochs = 10
        flat_epochs   = int(epochs * 0.2)   # hold peak LR for 20% of training
        decay_start   = warmup_epochs + flat_epochs
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # linear warmup
                return (epoch + 1) / warmup_epochs
            if epoch < decay_start:
                # flat phase at peak LR
                return 1.0
            # cosine tail over the remaining epochs
            progress = (epoch - decay_start) / max(epochs - decay_start, 1)
            return max(1e-5 / lr, 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item()))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self._cls_weights: Optional[torch.Tensor] = None
        if task == "classification":
            all_labels = torch.cat([_unpack_batch(b, task)[1] for b in train_loader])
            num_cls = get_num_classes(train_loader)
            counts = torch.bincount(all_labels, minlength=num_cls).float().clamp(min=1)
            w = 1.0 / counts
            self._cls_weights = (w / w.sum() * num_cls).to(device)
        weights_dir = save_directory / "weights"
        weights_dir.mkdir(exist_ok=True)

        results: dict = {
            "model":        encoder_name,
            "dataset":      dataset_name,
            "task":         task,
            "encoding_dim": enc_dim,
            "feature_dim":  feat_dim,
            "epochs":       epochs,
            "timestamp":    datetime.now().isoformat(),
            "history":      [],
            "best":         {},
        }

        best_primary_metric = None

        for epoch in range(epochs):
            if task == "reconstruction":
                encoder.eval()
            else:
                encoder.train()
            decoder.train()

            train_loss = self._train_epoch(
                task, encoder, decoder, train_loader,
                optimizer, device, verbose, epoch, epochs,
            )
            scheduler.step()

            epoch_record: dict = {"epoch": epoch + 1, "train_loss": train_loss}

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                val_metrics = self._evaluate(
                    task, encoder, decoder, val_loader,
                    device, full_dk, eval_rotations,
                )
                epoch_record["val_metrics"] = val_metrics
                cleanup_cuda_memory(verbose=False)

                primary = self._primary_metric(task, val_metrics)
                is_best = (
                    best_primary_metric is None
                    or primary > best_primary_metric
                )
                if is_best:
                    best_primary_metric = primary
                    best_epoch          = epoch + 1
                    results["best_val"] = val_metrics
                    torch.save(encoder.state_dict(), weights_dir / "encoder_best.pt")
                    torch.save(decoder.state_dict(), weights_dir / "decoder_best.pt")

                ckpt = f"epoch_{epoch + 1:04d}"
                torch.save(encoder.state_dict(), weights_dir / f"encoder_{ckpt}.pt")
                torch.save(decoder.state_dict(), weights_dir / f"decoder_{ckpt}.pt")

                lr = scheduler.get_last_lr()[0]
                self._log_epoch(logger, task, epoch + 1, epochs, train_loss, val_metrics, lr, is_best)

            results["history"].append(epoch_record)

        # Load best checkpoint and evaluate once on held-out test set
        # Load best checkpoints using map_location to ensure correct device
        encoder.load_state_dict(torch.load(weights_dir / "encoder_best.pt", weights_only=True))
        decoder.load_state_dict(torch.load(weights_dir / "decoder_best.pt", weights_only=True))
        test_metrics = self._evaluate(task, encoder, decoder, test_loader, device, full_dk, eval_rotations)
        results["test"] = test_metrics
        cleanup_cuda_memory(verbose=False)

        results_fp = save_directory / "results.json"
        with open(results_fp, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n  Best val metrics (epoch {best_epoch}) : {results['best_val']}")
        logger.info(f"  Test metrics                          : {results['test']}")
        logger.info(f"  Results saved: {results_fp}")

        cleanup_cuda_memory(verbose=False)
        del encoder, decoder, optimizer, scheduler

        return results

    # ------------------------------------------------------------------
    # Decoder kwargs
    # ------------------------------------------------------------------

    def _build_decoder_kwargs(self, task, enc_dim, feat_dim, train_loader, extra_kwargs):
        kwargs: dict = {}

        if task == "classification":
            num_classes = extra_kwargs.get("num_classes", get_num_classes(train_loader))
            kwargs = {"num_classes": num_classes, "encoding_dim": enc_dim}

        elif task == "segmentation":
            seg_dim = feat_dim if feat_dim is not None else enc_dim
            batch = next(iter(train_loader))
            _, (_, labels) = _unpack_batch(batch, "segmentation")
            if labels.dim() >= 2:
                # Use dataset attribute if available, else scan batches to find true max
                _ds = train_loader.dataset
                _base_ds = getattr(_ds, "dataset", _ds)
                num_classes = getattr(_base_ds, "num_part_classes", None)
                if num_classes is None:
                    _all_max = 0
                    for _i, _b in enumerate(train_loader):
                        _, (_obj_lbl, _lbl) = _unpack_batch(_b, "segmentation")
                        _all_max = max(_all_max, int(_lbl.max().item()))
                        if _i >= 20: break
                    num_classes = _all_max + 1
            else:
                num_classes = extra_kwargs.get("num_classes", get_num_classes(train_loader))
            # Walk through Subset wrappers to reach the underlying dataset
            _root_ds = _base_ds
            while hasattr(_root_ds, "dataset"):
                _root_ds = _root_ds.dataset
            num_categories = 0
            for attr in ("categories", "num_classes"):
                val = getattr(_root_ds, attr, None)
                if val is not None:
                    num_categories = len(val) if hasattr(val, "__len__") else int(val)
                    break
            seg_classes = getattr(_root_ds, "part_seg_classes", None)
            kwargs = {
                "encoding_dim": seg_dim,
                "global_encoding_dim": enc_dim,
                "num_classes": num_classes,
                "num_categories": num_categories,
                "hidden_dim": extra_kwargs.get("seg_hidden_dim", 512),
                "seg_classes": seg_classes,
            }

        elif task == "reconstruction":
            num_points = extra_kwargs.get("num_points", get_num_points(train_loader))
            kwargs = {"encoding_dim": enc_dim, "num_points": num_points}

        elif task == "denoise":
            point_feature_dim = feat_dim if feat_dim is not None else 3
            kwargs = {"encoding_dim": enc_dim, "point_feature_dim": point_feature_dim}

        kwargs.update(extra_kwargs)
        return kwargs

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_epoch(self, task, encoder, decoder, loader, optimizer, device, verbose, epoch, epochs):
        total_loss  = 0.0
        num_batches = 0
        params      = [p for p in list(encoder.parameters()) + list(decoder.parameters()) if p.requires_grad]

        pbar = tqdm(
            loader,
            desc=f"  [{epoch + 1:3d}/{epochs}] {task[:6]:<6}",
            leave=False,
            disable=not verbose,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(pbar):
            if task == "segmentation":
                points, (obj_labels, labels) = _unpack_batch(batch, task)
            else:
                points, labels = _unpack_batch(batch, task)
            points = points.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = self._compute_loss(task, encoder, decoder, points, labels, device, obj_labels if task == "segmentation" else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.GRAD_CLIP)
            optimizer.step()

            total_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})
            
            # Memory cleanup every 10 batches
            maybe_cleanup_after_batch(batch_idx, cleanup_interval=10, verbose=False)

        pbar.close()
        return total_loss / max(num_batches, 1)

    def _compute_loss(self, task, encoder, decoder, points, labels, device, obj_labels=None):
        if task == "denoise":
            noise  = torch.randn_like(points) * self.NOISE_STD
            noisy  = points + noise
            noisy_enc, noisy_feat = unpack_encoder_output(encoder(noisy))
            denoised = decoder(noisy, noisy_enc, noisy_feat)
            return chamfer_distance(denoised, points.permute(0, 2, 1))

        # Apply random rotation augmentation during classification training
        # so the train distribution matches the rotated eval distribution.
        if task == "classification":
            from core.geometry import random_rotation_3d
            pts_T = points.permute(0, 2, 1)   # [B, N, 3]
            pts_T = torch.stack([random_rotation_3d(pts_T[i]) for i in range(pts_T.size(0))])
            points = pts_T.permute(0, 2, 1)   # [B, 3, N]

        encoding, point_features = unpack_encoder_output(encoder(points))

        if task == "classification":
            labels = labels.to(device, non_blocking=True)
            labels = labels.clamp(0, decoder.classifier[-1].out_features - 1)
            return F.cross_entropy(decoder(encoding), labels, weight=self._cls_weights,
                                   label_smoothing=0.2)

        elif task == "segmentation":
            labels = labels.to(device, non_blocking=True)
            if labels.dim() < 2:
                N = points.shape[-1]
                labels = labels.unsqueeze(1).expand(-1, N)
            if obj_labels is None:
                obj_labels = labels[:, 0] if labels.dim() == 2 else labels
            else:
                obj_labels = obj_labels.to(device, non_blocking=True)
            logits = decoder(encoding, points, point_features, obj_labels)

            # Always mask invalid part logits before computing loss
            if self._seg_classes is not None:
                obj_labels_clamped = obj_labels.clamp(0, len(self._seg_classes) - 1)
                logits = mask_segmentation_logits(logits, obj_labels_clamped, self._seg_classes)
                point_mask = segmentation_point_mask(labels, obj_labels_clamped, self._seg_classes)
                if point_mask.any():
                    # label_smoothing=0.0 here: masked logits contain -1e4 values
                    # which produce huge loss under label smoothing (smoothing
                    # distributes small probability mass to all classes including
                    # masked ones, making the loss ~94M instead of ~4).
                    return F.cross_entropy(
                        logits.permute(0, 2, 1)[point_mask],
                        labels[point_mask],
                        label_smoothing=0.0,
                    )
                # No valid points for any category in this batch — skip cleanly
                # rather than falling through to unmasked cross-entropy.
                return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()

            B, C, N = logits.shape
            return F.cross_entropy(
                logits.permute(0, 2, 1).reshape(-1, C),
                labels.reshape(-1),
                label_smoothing=0.1,
            )

        elif task == "reconstruction":
            recon = decoder(encoding)
            return chamfer_distance(recon, points.permute(0, 2, 1))

        raise ValueError(f"Unknown task: {task}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, task, encoder, decoder, loader, device, decoder_kwargs, eval_rotations):
        encoder.eval()
        decoder.eval()

        if task == "classification":
            return self._eval_classification(encoder, decoder, loader, device, random_rotation=True)
        elif task == "segmentation":
            return self._eval_segmentation(encoder, decoder, loader, device, decoder_kwargs["num_classes"], self._seg_classes)
        elif task == "reconstruction":
            return self._eval_reconstruction(encoder, decoder, loader, device, eval_rotations)
        elif task == "denoise":
            return self._eval_denoise(encoder, decoder, loader, device)

        raise ValueError(f"Unknown task: {task}")

    @torch.no_grad()
    def _eval_classification(self, encoder, decoder, loader, device, random_rotation: bool = True, num_votes: int = 12):
        from core.geometry import random_rotation_3d
        all_votes, all_labels = [], []

        for batch in loader:
            points, labels = _unpack_batch(batch, "classification")
            points = points.to(device, non_blocking=True)
            B = points.size(0)

            if random_rotation and num_votes > 1:
                # Accumulate softmax logits over num_votes random rotations.
                # This averages out the variance from any single unlucky rotation.
                vote_logits = None
                for _ in range(num_votes):
                    pts_T = points.permute(0, 2, 1)   # [B, N, 3]
                    pts_T = torch.stack([random_rotation_3d(pts_T[i]) for i in range(B)])
                    pts_r = pts_T.permute(0, 2, 1)
                    encoding, _ = unpack_encoder_output(encoder(pts_r))
                    logits = decoder(encoding)          # [B, num_classes]
                    probs  = F.softmax(logits, dim=-1)
                    vote_logits = probs if vote_logits is None else vote_logits + probs
                preds = vote_logits.argmax(dim=1).cpu()
            else:
                if random_rotation:
                    pts_T = points.permute(0, 2, 1)
                    pts_T = torch.stack([random_rotation_3d(pts_T[i]) for i in range(B)])
                    points = pts_T.permute(0, 2, 1)
                encoding, _ = unpack_encoder_output(encoder(points))
                preds = decoder(encoding).argmax(dim=1).cpu()

            all_votes.append(preds)
            all_labels.append(labels)

        all_preds  = torch.cat(all_votes)
        all_labels = torch.cat(all_labels)

        accuracy    = (all_preds == all_labels).float().mean().item()
        num_classes = int(all_labels.max().item()) + 1

        per_cls_correct = torch.zeros(num_classes)
        per_cls_total   = torch.zeros(num_classes)
        for c in range(num_classes):
            mask = all_labels == c
            per_cls_total[c]   = mask.sum()
            per_cls_correct[c] = (all_preds[mask] == c).sum()

        valid          = per_cls_total > 0
        mean_class_acc = (per_cls_correct[valid] / per_cls_total[valid]).mean().item() if valid.any() else 0.0

        # precision, recall, f1 (macro)
        per_cls_tp = torch.zeros(num_classes)
        per_cls_fp = torch.zeros(num_classes)
        per_cls_fn = torch.zeros(num_classes)
        for c in range(num_classes):
            pred_c  = (all_preds  == c)
            label_c = (all_labels == c)
            per_cls_tp[c] = (pred_c & label_c).sum().float()
            per_cls_fp[c] = (pred_c & ~label_c).sum().float()
            per_cls_fn[c] = (~pred_c & label_c).sum().float()

        valid     = per_cls_total > 0
        precision = (per_cls_tp / (per_cls_tp + per_cls_fp).clamp(min=1e-8))[valid].mean().item()
        recall    = (per_cls_tp / (per_cls_tp + per_cls_fn).clamp(min=1e-8))[valid].mean().item()
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "accuracy":           accuracy,
            "mean_class_accuracy": mean_class_acc,
            "precision":          precision,
            "recall":             recall,
            "f1":                 f1,
        }

    @torch.no_grad()
    def _eval_segmentation(self, encoder, decoder, loader, device, num_classes, seg_classes=None):
        all_preds, all_labels, all_categories = [], [], []

        for batch in loader:
            points, (obj_labels, labels) = _unpack_batch(batch, "segmentation")
            points = points.to(device, non_blocking=True)
            obj_labels = obj_labels.to(device, non_blocking=True)
            if labels.dim() < 2:
                labels = labels.unsqueeze(1).expand(-1, points.shape[-1])
            encoding, point_features = unpack_encoder_output(encoder(points))
            logits = decoder(encoding, points, point_features, obj_labels)
            if seg_classes is not None:
                obj_labels_clamped = obj_labels.clamp(0, len(seg_classes) - 1)
                logits = mask_segmentation_logits(logits, obj_labels_clamped, seg_classes)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())
            all_categories.append(obj_labels.cpu())

        all_preds      = torch.cat(all_preds)
        all_labels     = torch.cat(all_labels)
        all_categories = torch.cat(all_categories)

        preds_flat  = all_preds.reshape(-1)
        labels_flat = all_labels.reshape(-1)

        miou_class  = compute_miou(preds_flat, labels_flat, num_classes)
        accuracy    = (preds_flat == labels_flat).float().mean().item()
        miou        = (
            compute_instance_miou(all_preds, all_labels, all_categories, seg_classes)
            if seg_classes is not None
            else miou_class
        )

        per_cls_iou = {}
        for c in range(num_classes):
            pred_c  = (preds_flat == c)
            label_c = (labels_flat == c)
            inter   = (pred_c & label_c).sum().float()
            union   = (pred_c | label_c).sum().float()
            if union > 0:
                per_cls_iou[c] = (inter / union).item()

        return {
            "miou":          miou,
            "miou_class":    miou_class,
            "accuracy":      accuracy,
            "per_class_iou": per_cls_iou,
        }

    @torch.no_grad()
    def _eval_reconstruction(self, encoder, decoder, loader, device, eval_rotations):
        total_cd, n_batches = 0.0, 0
        all_z_orig, all_z_recon, cos_sims, per_sample_cds = [], [], [], []

        for batch in loader:
            points, _ = _unpack_batch(batch, "reconstruction")
            points = points.to(device, non_blocking=True)
            gt     = points.permute(0, 2, 1)

            encoding, _ = unpack_encoder_output(encoder(points))
            recon       = decoder(encoding)

            total_cd  += chamfer_distance(recon, gt).item()
            n_batches += 1

            recon_enc, _ = unpack_encoder_output(encoder(recon.permute(0, 2, 1)))
            cos_sims.append(F.cosine_similarity(encoding, recon_enc, dim=1).mean().item())
            all_z_orig.append(encoding.cpu())
            all_z_recon.append(recon_enc.cpu())

            per_sample_cds.append(chamfer_distance_per_sample(recon, gt).cpu())

        mean_cd  = total_cd / n_batches
        mean_cos = float(np.mean(cos_sims))
        all_cd   = torch.cat(per_sample_cds)

        try:
            z_orig  = torch.cat(all_z_orig)
            z_recon = torch.cat(all_z_recon)
            orig_var  = z_orig.var(dim=0).mean().item()
            recon_var = z_recon.var(dim=0).mean().item()
            if orig_var > 1e-6 and recon_var > 1e-6:
                cka = compute_cka(z_orig, z_recon)
            else:
                cka = 0.0
        except Exception:
            cka = 0.0

        min_cd = self._eval_min_rotation_cd(encoder, decoder, loader, device, eval_rotations)

        return {
            "chamfer_distance":           mean_cd,
            "chamfer_std":                all_cd.std().item(),
            "rotation_corrected_chamfer": min_cd,
            "cosine_similarity":          mean_cos,
            "cka":                        cka,
        }

    @torch.no_grad()
    def _eval_min_rotation_cd(self, encoder, decoder, loader, device, eval_rotations):
        try:
            points, _ = _unpack_batch(next(iter(loader)), "reconstruction")
            points = points.to(device)
            encoding, _ = unpack_encoder_output(encoder(points))
            recon = decoder(encoding)
            return min_rotated_chamfer_distance(recon, points.permute(0, 2, 1), num_rotations=eval_rotations)
        except Exception:
            return 0.0

    @torch.no_grad()
    def _eval_denoise(self, encoder, decoder, loader, device):
        total_cd, noisy_cd, n_batches = 0.0, 0.0, 0

        rng = torch.Generator(device=device)
        rng.manual_seed(0)
        for batch in loader:
            points, _ = _unpack_batch(batch, "denoise")
            points = points.to(device, non_blocking=True)
            gt     = points.permute(0, 2, 1)

            noise   = torch.randn(points.shape, generator=rng, device=device, dtype=points.dtype) * self.NOISE_STD
            noisy   = points + noise

            noisy_enc, noisy_feat = unpack_encoder_output(encoder(noisy))
            denoised = decoder(noisy, noisy_enc, noisy_feat)

            total_cd += chamfer_distance(denoised, gt).item()
            noisy_cd += chamfer_distance(noisy.permute(0, 2, 1), gt).item()  # noisy vs GT
            n_batches += 1

        mean_cd         = total_cd / n_batches
        noisy_ref       = noisy_cd / n_batches
        improvement     = noisy_ref - mean_cd          # positive = denoised closer to GT
        improvement_pct = improvement / max(noisy_ref, 1e-8) * 100

        return {
            "chamfer_distance":  mean_cd,
            "noisy_chamfer":     noisy_ref,
            "improvement":       improvement,
            "improvement_pct":   improvement_pct,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _primary_metric(task: str, metrics: dict) -> float:
        return {
            "classification": metrics.get("accuracy",         0.0),
            "segmentation":   metrics.get("miou",              0.0),
            "reconstruction": -metrics.get("chamfer_distance", float("inf")),
            "denoise":        metrics.get("improvement_pct",   float("-inf")),
        }[task]

    @staticmethod
    def _log_epoch(logger, task, epoch, epochs, train_loss, metrics, lr, is_best):
        mark = "  ★ best" if is_best else ""
        # Exclude per_class_iou (too verbose) and miou_class (secondary metric;
        # primary segmentation metric is instance mIoU logged as 'miou').
        loggable   = {k: v for k, v in metrics.items() if k != "per_class_iou"}
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in loggable.items())
        logger.info(
            f"  [{epoch:3d}/{epochs}] loss={train_loss:.4f}  {metric_str}"
            f"  lr={lr:.2e}{mark}"
        )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _all_model_classes() -> Dict[str, type]:
    from models import (
        DGCNNEncoder,
        DiPVNetEncoder,
        PointMambaEncoder,
        PointNet2Encoder,
        RINet,
        RISAPTv3,
        RotationInvariantSparseAttention,
        RSCNNEncoder,
    )
    return {
        "risa":      RotationInvariantSparseAttention,
        "risa_ptv3": RISAPTv3,
        "dipvnet":   DiPVNetEncoder,
        "dgcnn":     DGCNNEncoder,
        "mamba":     PointMambaEncoder,
        "pointnet2": PointNet2Encoder,
        "rinet":     RINet,
        "rscnn":     RSCNNEncoder,
    }

def run_experiments(
    models:         Union[str, List]  = "all",
    datasets:       Union[str, List]  = "shapenet",
    tasks:          Union[str, List]  = "all",
    data_root:      str               = "data",
    device:         Union[str, torch.device, None] = None,
    epochs:         int               = 100,
    batch_size:     int               = 32,
    num_points:     int               = 128,
    encoding_dim:   int               = 512,
    save_directory: Optional[Path]    = None,
    verbose:        bool              = True,
    seed:           Optional[int]     = None,
    eval_rotations: int               = 32,
    num_workers:    int               = 4,
    risa_kwargs:    dict              = None,
) -> dict:
    """Run a grid of experiments over all combinations of models × datasets × tasks.
    
    Args:
        encoding_dim: Global shape encoding dimension (default 128). Larger values
                     improve reconstruction (higher CKA) but increase GPU memory usage.
                     Recommend: 128 (baseline), 256 (better reconstruction), 512 (best).
    """
    from core.datasets import get_dataloader, get_dataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    all_models = _all_model_classes()

    def _resolve(arg, all_vals):
        if arg == "all":
            return list(all_vals)
        return [arg] if isinstance(arg, str) else list(arg)

    model_names   = _resolve(models,   all_models.keys())
    dataset_names = _resolve(datasets, _ALL_DATASETS)
    task_names    = _resolve(tasks,    Trainer.TASKS)

    for m in model_names:
        if m not in all_models:
            raise ValueError(f"Unknown model '{m}'. Available: {list(all_models)}")
    for t in task_names:
        if t not in Trainer.TASKS:
            raise ValueError(f"Unknown task '{t}'. Available: {list(Trainer.TASKS)}")

    if save_directory is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = outputs_dir / "experiments" / ts
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    trainer      = Trainer()
    all_results  : dict = {}
    summary_rows : list = []

    total_runs = len(model_names) * len(dataset_names) * len(task_names)
    run_idx    = 0

    for model_name in model_names:
        model_cls = all_models[model_name]
        all_results[model_name] = {}

        safe_bs = safe_batch_size(model_cls, num_points, batch_size, device)
        if safe_bs != batch_size:
            print(f"[{model_name}] Reduced batch size {batch_size} -> {safe_bs} to fit GPU memory")

        for dataset_name in dataset_names:
            all_results[model_name][dataset_name] = {}

            try:
                train_ds = get_dataset(dataset_name, split="train", root=data_root, num_points=num_points)
                val_ds   = get_dataset(dataset_name, split="val",   root=data_root, num_points=num_points)
                test_ds  = get_dataset(dataset_name, split="test",  root=data_root, num_points=num_points)
            except Exception as exc:
                print(f"[SKIP] Dataset '{dataset_name}' unavailable: {exc}")
                continue

            train_loader = get_dataloader(train_ds, batch_size=safe_bs, shuffle=True,  num_workers=num_workers, drop_last=True)
            val_loader   = get_dataloader(val_ds,   batch_size=safe_bs, shuffle=False, num_workers=num_workers)
            test_loader  = get_dataloader(test_ds,  batch_size=safe_bs, shuffle=False, num_workers=num_workers)

            for task in task_names:
                run_idx += 1
                run_dir  = save_directory / model_name / dataset_name / task
                run_dir.mkdir(parents=True, exist_ok=True)

                # Skip segmentation if the underlying dataset does not provide per-point part labels
                if task == "segmentation" and getattr(train_ds, "part_labels", None) is None:
                    # Check the underlying dataset when wrapped in a Subset
                    _base = getattr(train_ds, "dataset", train_ds)
                    if getattr(_base, "part_labels", None) is None:
                        print(f"[SKIP] {model_name}/{dataset_name}/segmentation — no per-point part labels")
                        all_results[model_name][dataset_name][task] = {"skipped": "no part labels"}
                        summary_rows.append({"model": model_name, "dataset": dataset_name, "task": task, "skipped": True})
                        continue

                print(f"\n{'='*60}\n  Run {run_idx}/{total_runs}: {model_name} | {dataset_name} | {task}\n{'='*60}")

                cleanup_cuda_memory(verbose=False)
                
                # Instantiate model with encoding_dim for RISA and RISA-PTv3
                if model_name in ("risa", "risa_ptv3"):
                    _mdim = (risa_kwargs.get("model_dim", encoding_dim) if risa_kwargs else encoding_dim)
                    kwargs = dict(
                        encoding_out_dim = encoding_dim,
                        features_out_dim = _mdim,
                        model_dim        = _mdim,
                    )
                    if risa_kwargs:
                        # Keys only valid for the base RISA model, not risa_ptv3
                        _risa_only = {"attn_augment", "grad_checkpoint"}
                        _skip = {"encoding_out_dim", "features_out_dim", "model_dim"}
                        if model_name == "risa_ptv3":
                            _skip |= _risa_only
                        kwargs.update({k: v for k, v in risa_kwargs.items() if k not in _skip})
                    encoder = model_cls(**kwargs).to(device)
                else:
                    encoder = model_cls().to(device)

                try:
                    result = trainer(
                        encoder        = encoder,
                        train_loader   = train_loader,
                        val_loader     = val_loader,
                        test_loader    = test_loader,
                        task           = task,
                        dataset_name   = dataset_name,
                        device         = device,
                        epochs         = epochs,
                        save_directory = run_dir,
                        verbose        = verbose,
                        seed           = seed,
                        eval_rotations = eval_rotations,
                    )
                except Exception as exc:
                    print(f"[ERROR] {model_name}/{dataset_name}/{task}: {exc}")
                    import traceback; traceback.print_exc()
                    result = {"error": str(exc)}

                all_results[model_name][dataset_name][task] = result
                best = result.get("test", result.get("best_val", {}))
                summary_rows.append({"model": model_name, "dataset": dataset_name, "task": task, **best})

    master_fp = save_directory / "all_results.json"
    with open(master_fp, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {master_fp}")

    if summary_rows:
        all_keys = list(dict.fromkeys(k for row in summary_rows for k in row))
        csv_fp   = save_directory / "summary.csv"
        with open(csv_fp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV saved to {csv_fp}")

    return all_results
