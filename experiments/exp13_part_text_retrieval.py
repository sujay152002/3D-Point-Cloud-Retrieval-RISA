"""Experiment 13 — Part-Composition Text Retrieval

Motivation
----------
Exp11/11b align shapes to ImageBind's class-level text embeddings.
This works for "a 3D model of a chair" but fails for structural queries like
"something with 4 legs" that should return BOTH chairs AND tables.

The core idea: train a projection MLP so that shape embeddings align to
*part-composition descriptions* of each shape, not class names.
At query time, structural text queries match across class boundaries.

Design
------
    shape → RISA encoder (frozen, exp10 checkpoint)
          → z_risa  [512]
          → ProjectionMLP  (512 → 768)
          → z_proj  [768]   ←→   sentence-transformer("a chair with a back, seat, and 4 legs")

Text encoder: all-mpnet-base-v2 (sentence-transformers)
  - 768-dim embeddings
  - Strong at semantic/structural similarity
  - Understands part words: "leg", "wing", "handle", "blade" etc.

Per-shape description generation (from part_labels):
  - Count dominant points per part ID
  - Map global part ID → part name via SHAPE_NET_PART_SEG_CLASSES + SHAPENET_PART_NAMES
  - Count-aware phrasing: "3 legs" vs "a leg", "2 wings" vs "a wing"
  - Example outputs:
      chair (parts 12,13,14)    → "a chair with a back, a seat, and 4 legs"
      chair (parts 12,13,14,15) → "a chair with a back, a seat, 4 legs, and 2 arms"
      table (parts 47,48)       → "a table with a top and 4 legs"
      airplane (parts 0,1,2,3)  → "an airplane with a body, 2 wings, a tail, and 2 engines"

Evaluation
----------
  1. Part-query retrieval: queries like "something with 4 legs", "a shape with wings"
     → checks both correct class AND cross-class matches
  2. Standard class-level R@1, R@5, mAP@5 (regression check vs exp11)
  3. Cross-class retrieval: for leg/wheel/handle queries, measures fraction of
     results that contain the queried part regardless of class

Usage
-----
    python experiments/exp13_part_text_retrieval.py

    # Custom checkpoints:
    python experiments/exp13_part_text_retrieval.py \\
        --risa_ckpt outputs/checkpoints/exp10_aggregate_K.pt \\
        --prior aggregate_K --K 256 --epochs 50
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.datasets import get_dataset, SHAPE_NET_PART_SEG_CLASSES
from core.trainer import unpack_encoder_output
from experiments.exp10_prior_selection import RISAWithPrior, PRIOR_NAMES, set_seed
from experiments.exp1_retrieval_eval import recall_at_k, mean_ap_at_k

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = str(ROOT / "data")
N_POINTS  = 1024
ST_DIM    = 768    # sentence-transformers all-mpnet-base-v2 output dim

# ── Part name mapping (from exp9, mirrors SHAPE_NET_PART_SEG_CLASSES order) ──

SHAPENET_CLASS_NAMES = [
    "airplane", "bag", "cap", "car", "chair",
    "earphone", "guitar", "knife", "lamp", "laptop",
    "motorbike", "mug", "pistol", "rocket", "skateboard", "table",
]

SHAPENET_PART_NAMES = {
    "airplane":   ["body", "wing", "tail", "engine"],
    "bag":        ["body", "handle"],
    "cap":        ["brim", "crown"],
    "car":        ["body", "wheel", "hood", "roof"],
    "chair":      ["back", "seat", "leg", "arm"],
    "earphone":   ["earcup", "headband", "data cord"],
    "guitar":     ["head", "neck", "body"],
    "knife":      ["blade", "handle"],
    "lamp":       ["base", "shade", "tube", "bulb"],
    "laptop":     ["lid", "base"],
    "motorbike":  ["gas tank", "seat", "wheel", "handle", "light", "engine"],
    "mug":        ["body", "handle"],
    "pistol":     ["barrel", "handle", "trigger guard"],
    "rocket":     ["nose", "body", "fin"],
    "skateboard": ["deck", "wheel", "truck"],
    "table":      ["top", "leg", "support"],
}

# Build global part ID → (class_name, part_name) lookup table
PART_ID_TO_INFO: dict[int, tuple[str, str]] = {}
for cls_idx, part_ids in enumerate(SHAPE_NET_PART_SEG_CLASSES):
    cls_name   = SHAPENET_CLASS_NAMES[cls_idx]
    part_names = SHAPENET_PART_NAMES[cls_name]
    for local_idx, global_id in enumerate(part_ids):
        pname = part_names[local_idx] if local_idx < len(part_names) else f"part{local_idx}"
        PART_ID_TO_INFO[global_id] = (cls_name, pname)

# ── PartNet part name mapping ─────────────────────────────────────────────────
# PartNet uses per-category 0-based part IDs (remapped during preprocessing).
# These names come from the after_merging_label_ids level-3 taxonomy.
# Each entry maps local_part_id → part_name for that category.
PARTNET_PART_NAMES: dict[str, dict[int, str]] = {
    "Bag":              {0: "body", 1: "handle", 2: "strap", 3: "pocket"},
    "Bed":              {0: "bed frame", 1: "headboard", 2: "footboard",
                         3: "leg", 4: "sleep area", 5: "side rail",
                         6: "ladder", 7: "support", 8: "surface",
                         9: "drawer", 10: "slat", 11: "mattress",
                         12: "pillow", 13: "bar", 14: "column"},
    "Bottle":           {0: "body", 1: "cap", 2: "neck",
                         3: "mouth", 4: "base", 5: "label",
                         6: "handle", 7: "nozzle", 8: "pump"},
    "Bowl":             {0: "body", 1: "base", 2: "lip", 3: "handle"},
    "Chair":            {0: "back", 1: "seat", 2: "leg", 3: "arm",
                         4: "armrest", 5: "back support", 6: "seat support",
                         7: "footrest", 8: "wheel", 9: "star base",
                         10: "seat cushion", 11: "back cushion",
                         12: "back frame", 13: "seat frame",
                         14: "headrest", 15: "connector",
                         16: "rocker", 17: "stretcher",
                         18: "frame", 19: "base",
                         20: "central support", 21: "runner",
                         22: "arm sofa", 23: "back sofa",
                         24: "ottoman", 25: "ottoman frame",
                         26: "ottoman base", 27: "ottoman cushion",
                         28: "ottoman leg", 29: "ottoman wheel",
                         30: "ottoman arm", 31: "side panel",
                         32: "lounge chair back", 33: "lounge chair base",
                         34: "lounge chair leg", 35: "lounge chair arm",
                         36: "lounge chair frame", 37: "lounge bar",
                         38: "rocking bar"},
    "Clock":            {0: "body", 1: "face", 2: "hand",
                         3: "frame", 4: "base", 5: "pendulum",
                         6: "knob", 7: "battery", 8: "button",
                         9: "leg", 10: "hook"},
    "Dishwasher":       {0: "body", 1: "door", 2: "rack",
                         3: "handle", 4: "control panel", 5: "base",
                         6: "foot"},
    "Display":          {0: "screen", 1: "base", 2: "support", 3: "frame"},
    "Door":             {0: "door", 1: "frame", 2: "handle",
                         3: "hinge", 4: "knob"},
    "Earphone":         {0: "earcup", 1: "headband", 2: "cable",
                         3: "driver", 4: "cushion", 5: "microphone",
                         6: "jack", 7: "adjuster", 8: "clip",
                         9: "cover"},
    "Faucet":           {0: "body", 1: "handle", 2: "spout",
                         3: "base", 4: "knob", 5: "valve",
                         6: "aerator", 7: "pipe", 8: "escutcheon",
                         9: "cartridge", 10: "supply line", 11: "drain"},
    "Hat":              {0: "crown", 1: "brim", 2: "band",
                         3: "button", 4: "visor", 5: "ear flap"},
    "Keyboard":         {0: "key", 1: "body", 2: "cable"},
    "Knife":            {0: "blade", 1: "handle", 2: "guard",
                         3: "bolster", 4: "tip", 5: "spine",
                         6: "edge", 7: "rivet", 8: "tang", 9: "pommel"},
    "Lamp":             {0: "base", 1: "shade", 2: "bulb",
                         3: "tube", 4: "arm", 5: "canopy",
                         6: "socket", 7: "switch", 8: "cord",
                         9: "diffuser", 10: "head", 11: "clamp",
                         12: "harp", 13: "finial", 14: "stem",
                         15: "support", 16: "column", 17: "foot",
                         18: "reflector", 19: "cover", 20: "chain",
                         21: "ring", 22: "bar", 23: "mounting",
                         24: "ceiling plate", 25: "globe",
                         26: "chimney", 27: "font", 28: "bracket",
                         29: "base plate", 30: "neck", 31: "connector",
                         32: "rail", 33: "spotlight",
                         34: "string light", 35: "rope",
                         36: "pendant", 37: "wire", 38: "cage",
                         39: "frame", 40: "panel"},
    "Laptop":           {0: "screen", 1: "keyboard", 2: "hinge"},
    "Microwave":        {0: "body", 1: "door", 2: "handle",
                         3: "control panel", 4: "foot", 5: "window"},
    "Mug":              {0: "body", 1: "handle", 2: "base", 3: "lip"},
    "Refrigerator":     {0: "body", 1: "door", 2: "handle",
                         3: "hinge", 4: "shelf", 5: "foot", 6: "drawer"},
    "Scissors":         {0: "blade", 1: "handle", 2: "pivot"},
    "StorageFurniture": {0: "body", 1: "door", 2: "drawer",
                         3: "leg", 4: "shelf", 5: "back panel",
                         6: "top panel", 7: "handle", 8: "side panel",
                         9: "bottom panel", 10: "foot", 11: "rail",
                         12: "bar", 13: "frame", 14: "support",
                         15: "knob", 16: "lock", 17: "hinge",
                         18: "column", 19: "base", 20: "pull",
                         21: "runner", 22: "divider", 23: "countertop"},
    "Table":            {0: "top", 1: "leg", 2: "support",
                         3: "shelf", 4: "drawer", 5: "bar stretcher",
                         6: "pedestal", 7: "frame", 8: "apron",
                         9: "base", 10: "wheel", 11: "leaf",
                         12: "cross stretcher", 13: "tray",
                         14: "handle", 15: "connector",
                         16: "side panel", 17: "back panel",
                         18: "door", 19: "bottom panel",
                         20: "runner", 21: "pull", 22: "knob",
                         23: "lock", 24: "foot", 25: "column",
                         26: "glass top", 27: "rim", 28: "center support",
                         29: "lower shelf", 30: "back foot",
                         31: "front foot", 32: "desktop",
                         33: "partition", 34: "keyboard tray",
                         35: "monitor stand", 36: "cup holder",
                         37: "side rail", 38: "central support",
                         39: "round top", 40: "bench top",
                         41: "bench leg", 42: "bench support",
                         43: "bench foot", 44: "bench frame",
                         45: "bench apron", 46: "bench stretcher",
                         47: "bench connector", 48: "bench drawer",
                         49: "bench shelf", 50: "bench rail"},
    "TrashCan":         {0: "body", 1: "lid", 2: "base",
                         3: "handle", 4: "wheel", 5: "pedal",
                         6: "inner", 7: "bag holder",
                         8: "foot", 9: "frame", 10: "opening"},
    "Vase":             {0: "body", 1: "base", 2: "lip",
                         3: "handle", 4: "neck", 5: "foot"},
}


def get_partnet_part_name(category: str, local_part_id: int) -> str:
    """Look up part name for a PartNet (category, local_part_id) pair.

    Falls back to a generic name if the category or id is unknown.
    Category matching is case-insensitive.
    """
    # Case-insensitive category lookup
    cat_key = next(
        (k for k in PARTNET_PART_NAMES if k.lower() == category.lower()),
        None
    )
    if cat_key is None:
        return f"part{local_part_id}"
    return PARTNET_PART_NAMES[cat_key].get(local_part_id, f"part{local_part_id}")


# Parts where multiple instances genuinely exist per shape.
# Maps part_name → typical max count (used as upper bound for connected-component estimate).
COUNTABLE_PARTS: dict[str, int] = {
    "leg":    6,   # chairs/tables: 3-4, stools: 3
    "wing":   2,   # airplanes: always 2
    "wheel":  6,   # cars: 4, motorbikes: 2, skateboards: 4
    "engine": 4,   # airplanes: 1-4
    "fin":    4,   # rockets: 3-4
    "arm":    2,   # chairs: 0-2
    "handle": 2,   # bags/mugs: 1, motorbikes: 2
}

# Articles: parts that sound better with "an" than "a"
AN_PARTS = {"engine", "arm"}


# ── Per-shape description generation ─────────────────────────────────────────

def _article(part_name: str) -> str:
    return "an" if part_name in AN_PARTS else "a"


def _pluralise(part_name: str, count: int) -> str:
    """Return '4 legs', '2 wings', 'a back', 'a seat' etc."""
    if part_name in COUNTABLE_PARTS and count > 1:
        if part_name.endswith("ch") or part_name.endswith("sh"):
            plural = part_name + "es"
        else:
            plural = part_name + "s"
        return f"{count} {plural}"
    return f"{_article(part_name)} {part_name}"


def _count_instances_cc(part_points: np.ndarray, max_count: int) -> int:
    """Estimate number of instances of a part via 3D connected components.

    part_points : [M, 3] XYZ of points labelled with this part ID.
    max_count   : upper bound (from COUNTABLE_PARTS) — clips outliers.

    Algorithm: union-find on a kNN graph (k=5, radius threshold = 0.15
    in unit-sphere normalised coords). Each connected component = one instance.
    This is O(M log M) and runs on CPU in <1ms for M<=1024.
    """
    M = len(part_points)
    if M == 0:
        return 0
    if M < 5:
        return 1

    # Normalise to unit sphere so threshold is scale-invariant
    pts = part_points - part_points.mean(0)
    scale = np.linalg.norm(pts, axis=1).max() + 1e-8
    pts = pts / scale

    # Build kNN graph
    k   = min(5, M - 1)
    # pairwise distances — M is small (<<1024 per part) so O(M^2) is fine
    diff = pts[:, None, :] - pts[None, :, :]          # [M, M, 3]
    dist = np.linalg.norm(diff, axis=-1)              # [M, M]
    np.fill_diagonal(dist, np.inf)

    RADIUS = 0.15
    # Union-Find
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(M):
        nn_idx = np.argpartition(dist[i], k)[:k]
        for j in nn_idx:
            if dist[i, j] < RADIUS:
                union(i, j)

    n_components = len({find(i) for i in range(M)})
    return min(n_components, max_count)


def generate_description(class_name: str, part_ids_present: list[int],
                         part_point_counts: dict[int, int],
                         part_point_xyz: dict[int, np.ndarray] = None) -> str:
    """Generate a natural-language part-composition description for one shape.

    Parameters
    ----------
    class_name        : e.g. "chair"
    part_ids_present  : sorted list of global part IDs present in this shape
    part_point_counts : global_part_id → number of points with that label
    part_point_xyz    : global_part_id → [M, 3] XYZ coords of those points
                        If provided, uses connected components to count instances.
                        If None, falls back to a single-instance assumption.

    Returns
    -------
    e.g. "a chair with a back, a seat, 4 legs, and 2 arms"
    """
    if not part_ids_present:
        return f"a {class_name}"

    part_phrases = _build_part_phrases(part_ids_present, part_point_counts, part_point_xyz)
    parts_str    = _join_part_phrases(part_phrases)

    article = "an" if class_name[0] in "aeiou" else "a"
    return f"{article} {class_name} with {parts_str}"


# Prefixes for class-agnostic descriptions — mirror the eval query style and
# add structural rephrasings for text augmentation diversity. Each prefix
# becomes a separate training view so the projection learns invariance to
# phrasing while keeping part composition as the discriminative signal.
_AGOSTIC_PREFIXES = [
    "something with",
    "a shape with",
    "an object with",
    "a 3D model with",
    "a 3D object with",
    "a model with",
    "something that has",
    "a shape that has",
    "an object that has",
]


def _build_part_phrases(
    part_ids_present: list[int],
    part_point_counts: dict[int, int],
    part_point_xyz: dict[int, np.ndarray] = None,
) -> list[str]:
    """Shared helper: build count-aware part phrases for a shape.

    Returns a list like ["a back", "a seat", "4 legs", "2 arms"].
    Used by both generate_description() and generate_agnostic_description().
    """
    part_phrases = []
    seen_names   = set()

    for pid in part_ids_present:
        _, pname = PART_ID_TO_INFO.get(pid, ("", f"part{pid}"))
        if pname in seen_names:
            continue
        seen_names.add(pname)

        if pname in COUNTABLE_PARTS:
            max_count = COUNTABLE_PARTS[pname]
            if part_point_xyz is not None and pid in part_point_xyz:
                count = _count_instances_cc(part_point_xyz[pid], max_count)
            else:
                count = 1
        else:
            count = 1

        part_phrases.append(_pluralise(pname, count))

    return part_phrases


def _join_part_phrases(part_phrases: list[str]) -> str:
    """Join part phrases with Oxford comma: 'a back, a seat, and 4 legs'."""
    if len(part_phrases) == 1:
        return part_phrases[0]
    elif len(part_phrases) == 2:
        return f"{part_phrases[0]} and {part_phrases[1]}"
    else:
        return ", ".join(part_phrases[:-1]) + f", and {part_phrases[-1]}"


def generate_agnostic_description(
    class_name: str,
    part_ids_present: list[int],
    part_point_counts: dict[int, int],
    part_point_xyz: dict[int, np.ndarray] = None,
    rng: random.Random = None,
) -> str:
    """Generate a class-agnostic part-composition description for one shape.

    Like generate_description() but replaces the class name with a generic
    prefix ("something with", "a shape with", "an object with") to match
    the phrasing used by PART_QUERIES at evaluation time.

    e.g. "a chair with a back, a seat, and 4 legs"
      → "something with a back, a seat, and 4 legs"
    """
    if not part_ids_present:
        prefix = rng.choice(_AGOSTIC_PREFIXES) if rng else _AGOSTIC_PREFIXES[0]
        return f"{prefix} parts"

    part_phrases = _build_part_phrases(part_ids_present, part_point_counts, part_point_xyz)
    parts_str    = _join_part_phrases(part_phrases)

    prefix = rng.choice(_AGOSTIC_PREFIXES) if rng else _AGOSTIC_PREFIXES[0]
    return f"{prefix} {parts_str}"


def generate_agnostic_descriptions_all(
    class_name: str,
    part_ids_present: list[int],
    part_point_counts: dict[int, int],
    part_point_xyz: dict[int, np.ndarray] = None,
) -> list[str]:
    """Generate ALL class-agnostic variants for one shape (one per prefix).

    Returns a list of strings, one per prefix in _AGOSTIC_PREFIXES.
    Each string uses the same part composition but a different prefix.

    e.g. ["something with a back, a seat, and 4 legs",
          "a shape with a back, a seat, and 4 legs",
          "an object with a back, a seat, and 4 legs",
          ...]
    """
    if not part_ids_present:
        return [f"{prefix} parts" for prefix in _AGOSTIC_PREFIXES]

    part_phrases = _build_part_phrases(part_ids_present, part_point_counts, part_point_xyz)
    parts_str    = _join_part_phrases(part_phrases)

    return [f"{prefix} {parts_str}" for prefix in _AGOSTIC_PREFIXES]


def build_shape_descriptions(dataset, categories: dict) -> list[str]:
    """Generate one description per shape in dataset using part_labels.

    Handles both plain datasets and torch.utils.data.Subset correctly:
    iterates only the indices present in the Subset so the returned list
    is aligned with dataset[0..len(dataset)-1].

    Uses connected-component instance counting for COUNTABLE_PARTS.
    Returns list of strings, one per shape, aligned with dataset indices.
    """
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        raw_ds  = dataset.dataset
        indices = dataset.indices
    else:
        raw_ds  = dataset
        indices = range(len(raw_ds))

    part_labels_all = raw_ds.part_labels   # [N_raw, 1024] global part IDs
    labels_all      = raw_ds.labels        # [N_raw] class indices
    points_all      = getattr(raw_ds, "points", None)  # [N_raw, 1024, 3]

    descriptions = []
    for raw_i in indices:
        cls_idx   = int(labels_all[raw_i])
        cls_name  = categories.get(cls_idx, f"class{cls_idx}")
        per_point = part_labels_all[raw_i]   # [1024] global part IDs

        cnt       = Counter(per_point.tolist())
        valid_ids = set(SHAPE_NET_PART_SEG_CLASSES[cls_idx])
        present   = sorted(pid for pid in cnt if pid in valid_ids)

        part_xyz = None
        if points_all is not None:
            xyz = points_all[raw_i]                    # [1024, 3]
            if hasattr(xyz, 'numpy'):
                xyz = xyz.numpy()
            if xyz.shape[0] == 3 and xyz.shape[1] != 3:
                xyz = xyz.T
            part_xyz = {pid: xyz[per_point == pid] for pid in present}

        descriptions.append(generate_description(cls_name, present, cnt, part_xyz))

    return descriptions


def build_shape_descriptions_dual(
    dataset, categories: dict,
    dataset_name: str = "shapenet",
) -> tuple[list[str], list[list[str]]]:
    """Generate class-bearing and ALL class-agnostic descriptions per shape.

    Works with both ShapeNet (global 50 part IDs, 16 classes) and PartNet
    (per-category 0-based part IDs, 24 classes).  Pass dataset_name="partnet"
    to activate PartNet part-name lookups.

    Returns (class_descs, agnostic_descs_all) — two lists aligned with
    dataset indices.

    Class-bearing:      "a chair with a back, a seat, and 4 legs"
    Class-agnostic (all): ["something with a back, a seat, and 4 legs",
                           "a shape with a back, a seat, and 4 legs",
                           ...]  — one per prefix in _AGOSTIC_PREFIXES
    """
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        raw_ds  = dataset.dataset
        indices = dataset.indices
    else:
        raw_ds  = dataset
        indices = range(len(raw_ds))

    part_labels_all = raw_ds.part_labels
    labels_all      = raw_ds.labels
    points_all      = getattr(raw_ds, "points", None)
    is_partnet      = dataset_name.lower() == "partnet"

    class_descs        = []
    agnostic_descs_all = []
    for raw_i in indices:
        cls_idx   = int(labels_all[raw_i])
        cls_name  = categories.get(cls_idx, f"class{cls_idx}")
        per_point = part_labels_all[raw_i]

        cnt = Counter(per_point.tolist())

        if is_partnet:
            # PartNet: part IDs are already local (0-based per category).
            # All non-negative IDs present in this shape are valid.
            present = sorted(pid for pid in cnt if pid >= 0)

            # Build a per-shape PART_ID_TO_INFO equivalent using PartNet names
            partnet_id_info = {
                pid: (cls_name, get_partnet_part_name(cls_name, pid))
                for pid in present
            }

            part_xyz = None
            if points_all is not None:
                xyz = points_all[raw_i]
                if hasattr(xyz, "numpy"):
                    xyz = xyz.numpy()
                if xyz.shape[0] == 3 and xyz.ndim == 2:
                    xyz = xyz.T
                part_xyz = {pid: xyz[per_point == pid] for pid in present}

            # Temporarily override PART_ID_TO_INFO for this shape
            _orig = dict(PART_ID_TO_INFO)
            PART_ID_TO_INFO.update(partnet_id_info)
            class_descs.append(
                generate_description(cls_name, present, cnt, part_xyz)
            )
            agnostic_descs_all.append(
                generate_agnostic_descriptions_all(cls_name, present, cnt, part_xyz)
            )
            # Restore
            PART_ID_TO_INFO.clear()
            PART_ID_TO_INFO.update(_orig)

        else:
            # ShapeNet: part IDs are global (0–49); validate against class's valid set
            valid_ids = set(SHAPE_NET_PART_SEG_CLASSES[cls_idx])
            present   = sorted(pid for pid in cnt if pid in valid_ids)

            part_xyz = None
            if points_all is not None:
                xyz = points_all[raw_i]
                if hasattr(xyz, "numpy"):
                    xyz = xyz.numpy()
                if xyz.shape[0] == 3 and xyz.ndim == 2:
                    xyz = xyz.T
                part_xyz = {pid: xyz[per_point == pid] for pid in present}

            class_descs.append(
                generate_description(cls_name, present, cnt, part_xyz)
            )
            agnostic_descs_all.append(
                generate_agnostic_descriptions_all(cls_name, present, cnt, part_xyz)
            )

    return class_descs, agnostic_descs_all


# ── Sentence-transformer encoder ──────────────────────────────────────────────

def load_sentence_transformer(model_name: str = "all-mpnet-base-v2"):
    """Load sentence-transformers model. Installs if missing."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not found — installing...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "sentence-transformers==3.4.1", "--user", "-q"
        ])
        from sentence_transformers import SentenceTransformer

    print(f"  Loading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name)
    model.eval()
    return model


@torch.inference_mode()
def encode_texts(texts: list[str], st_model, batch_size: int = 256) -> torch.Tensor:
    """Encode list of strings → [N, ST_DIM] L2-normalised tensor."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        embs  = st_model.encode(chunk, convert_to_tensor=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)   # [N, ST_DIM]


# ── Projection MLP (512 → 768) ────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """Two-layer MLP projecting RISA 512-dim → sentence-transformer 768-dim."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 1024, out_dim: int = ST_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        # log_inv_temp = log(1/T); clamp keeps T in [0.01, 100] — CLIP convention
        self.log_inv_temp = nn.Parameter(torch.tensor(np.log(1.0 / init_temp)))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        inv_temp = self.log_inv_temp.clamp(min=np.log(0.01), max=np.log(100.0)).exp()
        logits   = (z_a @ z_b.T) * inv_temp
        labels   = torch.arange(logits.size(0), device=logits.device)
        loss_ab  = F.cross_entropy(logits,   labels)
        loss_ba  = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2.0


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed); np.random.seed(seed)


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the position index i alongside each item.

    The returned index i is the position within this dataset (0..len-1),
    which aligns with text_embs built by build_shape_descriptions(dataset)
    since both iterate the dataset in the same order.
    """
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return (*item, i) if isinstance(item, tuple) else (item, i)


# ── Encode gallery ────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_dataset(encoder, proj, dataset, batch_size: int = 32):
    """Encode all shapes → [N, ST_DIM] L2-normalised, plus labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_z, all_labels = [], []
    encoder.eval(); proj.eval()

    for batch in loader:
        pts = batch[0].to(DEVICE)
        lbl = batch[1]
        if pts.shape[1] != 3:
            pts = pts.permute(0, 2, 1)
        z, _ = unpack_encoder_output(encoder(pts))
        all_z.append(F.normalize(proj(z), dim=-1).cpu())
        all_labels.append(lbl)

    return torch.cat(all_z), torch.cat(all_labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train_projection(
    encoder,
    proj,
    criterion,
    dataset,
    val_dataset,
    text_embs:     torch.Tensor,   # [N_train, n_views, ST_DIM] — multiple views per shape
    val_text_embs: torch.Tensor,   # [N_val,   n_views, ST_DIM]
    epochs:        int,
    batch_size:    int,
    lr:            float,
    ckpt_dir:      Path,
    prior:         str,
) -> tuple[list, list]:
    """Train ProjectionMLP so RISA embeddings align to per-shape part descriptions.

    text_embs can be 2-D [N, D] (single view) or 3-D [N, n_views, D] (multi-view).
    When multi-view, a random view is sampled per shape each batch — text augmentation
    that teaches the projection to be invariant to description phrasing.

    Tracks val loss each epoch and saves the best checkpoint.
    Returns (train_history, val_history).
    """
    text_embs     = text_embs.to(DEVICE)
    val_text_embs = val_text_embs.to(DEVICE)

    # Detect multi-view: 3-D [N, n_views, D]
    multi_view = text_embs.dim() == 3
    n_views    = text_embs.size(1) if multi_view else 1

    # _IndexedDataset returns (pts, label, part_label, idx) for ShapeNet
    # batch[0]=pts, batch[-1]=idx; label and part_label unused during training
    loader = DataLoader(
        _IndexedDataset(dataset), batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        _IndexedDataset(val_dataset), batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=2, pin_memory=True,
    )

    opt = optim.AdamW(
        list(proj.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_val_loss  = float("inf")
    best_ckpt_path = ckpt_dir / f"exp13_{prior}_proj_best.pt"
    train_history, val_history = [], []
    encoder.eval()

    for epoch in range(epochs):
        proj.train(); criterion.train()
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            pts = batch[0].to(DEVICE)
            idx = batch[-1]              # dataset indices [B]
            if pts.shape[1] != 3:
                pts = pts.permute(0, 2, 1)

            if multi_view:
                # text_embs[idx]: [B, n_views, D] → sample one random view per shape
                views = text_embs[idx]                           # [B, n_views, D]
                view_idx = torch.randint(0, n_views, (views.size(0),),
                                          device=DEVICE)          # [B]
                z_text = views[torch.arange(views.size(0), device=DEVICE),
                                view_idx]                         # [B, D]
            else:
                z_text = text_embs[idx]                           # [B, D]

            with torch.no_grad():
                z_risa, _ = unpack_encoder_output(encoder(pts))
            z_proj = proj(z_risa)
            loss   = criterion(z_proj, z_text)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_avg = total_loss / max(n_batches, 1)
        train_history.append(train_avg)

        # Validation loss — average over all views for stability
        proj.eval(); criterion.eval()
        val_loss, val_n = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                pts = batch[0].to(DEVICE)
                idx = batch[-1]
                if pts.shape[1] != 3:
                    pts = pts.permute(0, 2, 1)
                z_risa, _ = unpack_encoder_output(encoder(pts))
                z_proj = proj(z_risa)

                if multi_view:
                    # Average loss across all views
                    views = val_text_embs[idx]      # [B, n_views, D]
                    v_loss = 0.0
                    for v in range(n_views):
                        v_loss += criterion(z_proj, views[:, v, :]).item()
                    val_loss += v_loss / n_views
                else:
                    val_loss += criterion(z_proj, val_text_embs[idx]).item()

                val_n += 1
        val_avg = val_loss / max(val_n, 1)
        val_history.append(val_avg)

        is_best = val_avg < best_val_loss
        if is_best:
            best_val_loss = val_avg
            torch.save(proj.state_dict(), best_ckpt_path)

        temp = 1.0 / criterion.log_inv_temp.clamp(
            min=np.log(0.01), max=np.log(100.0)).exp().item()
        print(f"  epoch {epoch+1:3d}/{epochs}  "
              f"loss={train_avg:.4f}  val={val_avg:.4f}  "
              f"temp={temp:.3f}  lr={scheduler.get_last_lr()[0]:.2e}"
              f"{'  ★' if is_best else ''}")

    # Restore best checkpoint
    if best_ckpt_path.exists():
        proj.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE,
                                         weights_only=True))
        print(f"  Restored best checkpoint (val={best_val_loss:.4f}) ← {best_ckpt_path}")
    proj.eval()

    return train_history, val_history


# ── Part-composition evaluation ───────────────────────────────────────────────

# ShapeNet part-composition queries — designed to span multiple classes.
# Tuple: (query_text, {part_names_to_match}, {expected_class_names})
PART_QUERIES = [
    # Single-part structural queries (cross-class)
    ("something with legs",            {"leg"},         {"chair", "table"}),
    ("a shape with wings",             {"wing"},        {"airplane"}),
    ("an object with wheels",          {"wheel"},       {"car", "motorbike", "skateboard"}),
    ("something with a handle",        {"handle"},      {"bag", "knife", "mug", "motorbike", "pistol"}),
    ("a shape with an engine",         {"engine"},      {"airplane", "motorbike"}),
    ("something with a blade",         {"blade"},       {"knife"}),
    ("an object with fins",            {"fin"},         {"rocket"}),
    # Multi-part queries (more specific)
    ("a shape with a back and a seat", {"back", "seat"},{"chair"}),
    ("something with a top and legs",  {"top", "leg"},  {"table"}),
    ("an object with a barrel",        {"barrel"},      {"pistol"}),
    ("a shape with a neck",            {"neck"},        {"guitar"}),
    ("something with a deck",          {"deck"},        {"skateboard"}),
]

# PartNet part-composition queries — use PartNet category names (title-case)
# and PartNet part vocabulary from PARTNET_PART_NAMES.
PARTNET_PART_QUERIES = [
    # Single-part structural queries (cross-class)
    ("something with legs",            {"leg"},         {"Chair", "Table", "Bed", "StorageFurniture"}),
    ("a shape with a handle",          {"handle"},      {"Bag", "Knife", "Mug", "Faucet", "Door",
                                                          "Dishwasher", "Refrigerator", "StorageFurniture"}),
    ("an object with a blade",         {"blade"},       {"Knife", "Scissors"}),
    ("something with a drawer",        {"drawer"},      {"StorageFurniture", "Table", "Refrigerator"}),
    ("a shape with a door",            {"door"},        {"StorageFurniture", "Dishwasher",
                                                          "Refrigerator", "Microwave"}),
    ("an object with a lid",           {"lid"},         {"TrashCan", "Bottle"}),
    ("something with a shade",         {"shade"},       {"Lamp"}),
    ("a shape with a wheel",           {"wheel"},       {"Chair", "TrashCan"}),
    ("an object with a screen",        {"screen"},      {"Display", "Laptop"}),
    # Multi-part queries (more specific)
    ("a shape with a back and a seat", {"back", "seat"},{"Chair"}),
    ("something with a top and legs",  {"top", "leg"},  {"Table"}),
    ("an object with a body and a cap",{"body", "cap"}, {"Bottle"}),
    ("a shape with a blade and handle",{"blade", "handle"}, {"Knife", "Scissors"}),
]


def evaluate_part_queries(
    queries:         list,
    gallery_z:       torch.Tensor,   # [N, ST_DIM] L2-normalised
    gallery_labels:  torch.Tensor,   # [N] class indices
    gallery_indices: torch.Tensor,   # [N] raw dataset indices for part lookup
    categories:      dict,
    st_model,
    raw_ds,
    topk:            int = 10,
    dataset_name:    str = "shapenet",
) -> list:
    """For each part query, retrieve top-k shapes and measure:
    - part_hit_rate: fraction of top-k that contain the queried part
    - class_hit_rate: fraction of top-k whose class is in expected_classes

    Works with both ShapeNet (global part IDs via PART_ID_TO_INFO) and
    PartNet (local per-category 0-based part IDs via PARTNET_PART_NAMES).
    Pass dataset_name="partnet" to activate PartNet part-name matching.

    gallery_indices maps gallery positions back to raw_ds indices so part
    lookup is always aligned regardless of dataset wrapping/ordering.
    """
    part_labels_all = raw_ds.part_labels   # [N_raw, P]
    is_partnet      = dataset_name.lower() == "partnet"

    if is_partnet:
        # For PartNet: build per-shape {part_name → set_of_local_ids} lookup
        # using the shape's category name and PARTNET_PART_NAMES.
        labels_all = raw_ds.labels
        shape_part_name_sets: list[set] = []
        for i in range(len(part_labels_all)):
            cls_idx  = int(labels_all[i])
            cls_name = categories.get(cls_idx, "")
            local_ids = set(np.unique(part_labels_all[i]).tolist())
            names = {
                get_partnet_part_name(cls_name, pid)
                for pid in local_ids if pid >= 0
            }
            shape_part_name_sets.append(names)
    else:
        # ShapeNet: match via global part IDs → names from PART_ID_TO_INFO
        shape_part_sets = [
            set(np.unique(part_labels_all[i]).tolist())
            for i in range(len(part_labels_all))
        ]
        part_name_to_ids: dict[str, set] = {}
        for gid, (_, pname) in PART_ID_TO_INFO.items():
            part_name_to_ids.setdefault(pname, set()).add(gid)

    # Batch-encode all query texts at once
    query_texts = [q[0] for q in queries]
    q_embs = encode_texts(query_texts, st_model)   # [Q, ST_DIM]

    results = []
    for qi, (query_text, part_names, expected_classes) in enumerate(queries):
        sims     = (q_embs[qi:qi+1] @ gallery_z.T).squeeze(0)   # [N]
        topk_idx = torch.topk(sims, k=min(topk, len(sims))).indices.tolist()

        part_hits, class_hits = 0, 0
        retrieved_classes = []
        for gal_pos in topk_idx:
            cls_name = categories.get(int(gallery_labels[gal_pos]), "")
            retrieved_classes.append(cls_name)

            if cls_name in expected_classes:
                class_hits += 1

            raw_idx = int(gallery_indices[gal_pos])
            if is_partnet:
                shape_names = shape_part_name_sets[raw_idx] \
                    if raw_idx < len(shape_part_name_sets) else set()
                hit = all(pn in shape_names for pn in part_names)
            else:
                shape_parts = shape_part_sets[raw_idx] \
                    if raw_idx < len(shape_part_sets) else set()
                hit = all(
                    any(gid in shape_parts for gid in part_name_to_ids.get(pn, set()))
                    for pn in part_names
                )
            if hit:
                part_hits += 1

        n = len(topk_idx)
        results.append({
            "query":            query_text,
            "part_names":       list(part_names),
            "expected_classes": list(expected_classes),
            "top_classes":      retrieved_classes,
            "part_hit_rate":    part_hits / n,
            "class_hit_rate":   class_hits / n,
        })
        print(f"  '{query_text}'")
        print(f"    part_hit@{topk}: {part_hits}/{n} ({100*part_hits/n:.0f}%)  "
              f"class_hit@{topk}: {class_hits}/{n} ({100*class_hits/n:.0f}%)")
        print(f"    top classes: {retrieved_classes[:5]}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior",      default="aggregate_K", choices=PRIOR_NAMES)
    parser.add_argument("--K",          type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--risa_ckpt",  type=str,   default=None)
    parser.add_argument("--proj_ckpt",  type=str,   default=None,
                        help="If set, skip training and evaluate only.")
    parser.add_argument("--st_model",   default="all-mpnet-base-v2",
                        help="sentence-transformers model name")
    parser.add_argument("--dataset",    default="shapenet")
    parser.add_argument("--topk_eval",  type=int,   default=10,
                        help="Top-k for part-query evaluation")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"  seed={args.seed}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"exp13_{args.prior}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to log file (background job)
    log_path = ROOT / "outputs" / "exp13_run.log"
    sys.stdout = open(log_path, "w", buffering=1)
    print(f"  log → {log_path}")


    print(f"\n{'='*60}")
    print(f"  Exp 13 — Part-Composition Text Retrieval")
    print(f"  prior={args.prior}  K={args.K}  dataset={args.dataset}")
    print(f"  text encoder: {args.st_model}  (dim={ST_DIM})")
    print(f"  output: {out_dir}")
    print(f"{'='*60}\n")

    # ── RISA encoder (frozen) ─────────────────────────────────────────────────
    encoder = RISAWithPrior(
        prior            = args.prior,
        K                = args.K,
        num_local        = 32,
        num_global       = 32,
        num_blocks       = 4,
        model_dim        = 256,
        features_out_dim = 256,
        encoding_out_dim = 512,
    ).to(DEVICE)

    ckpt_dir  = ROOT / "outputs" / "checkpoints"
    risa_ckpt = args.risa_ckpt or str(ckpt_dir / f"exp10_{args.prior}.pt")
    if not Path(risa_ckpt).exists():
        print(f"  Error: RISA checkpoint not found at {risa_ckpt}")
        sys.exit(1)

    state = torch.load(risa_ckpt, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(state, strict=False)
    print(f"  Loaded RISA checkpoint: {risa_ckpt}")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ── Projection MLP ────────────────────────────────────────────────────────
    proj      = ProjectionMLP(in_dim=512, hidden_dim=1024, out_dim=ST_DIM).to(DEVICE)
    criterion = InfoNCELoss(init_temp=0.07).to(DEVICE)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = get_dataset(args.dataset, split="train", root=DATA_ROOT, num_points=N_POINTS)
    test_ds  = get_dataset(args.dataset, split="test",  root=DATA_ROOT, num_points=N_POINTS)

    base_train = getattr(train_ds, "dataset", train_ds)
    base_test  = getattr(test_ds,  "dataset", test_ds)
    categories = getattr(base_train, "categories", {})

    if base_train.part_labels is None:
        print("  Error: dataset has no part_labels. Use --dataset shapenet.")
        sys.exit(1)

    # ── Sentence-transformer ──────────────────────────────────────────────────
    st_model = load_sentence_transformer(args.st_model)

    # ── Train or load projection ──────────────────────────────────────────────
    auto_proj_ckpt = ckpt_dir / f"exp13_{args.prior}_proj.pt"

    if args.proj_ckpt and Path(args.proj_ckpt).exists():
        proj.load_state_dict(
            torch.load(args.proj_ckpt, map_location=DEVICE, weights_only=True)
        )
        print(f"  Loaded projection checkpoint: {args.proj_ckpt}")
        history = []
    elif auto_proj_ckpt.exists():
        proj.load_state_dict(
            torch.load(auto_proj_ckpt, map_location=DEVICE, weights_only=True)
        )
        print(f"  Loaded cached projection checkpoint: {auto_proj_ckpt}")
        history = []
    else:
        # Generate per-shape dual descriptions (class-bearing + all class-agnostic)
        print("  Generating per-shape part descriptions for training set…")
        train_class_descs, train_agnostic_descs_all = build_shape_descriptions_dual(
            train_ds, categories
        )
        n_agnostic = len(train_agnostic_descs_all[0]) if train_agnostic_descs_all else 0
        print(f"  {n_agnostic} class-agnostic variants per shape")
        print(f"  Sample descriptions (class-bearing / class-agnostic):")
        shown = set()
        raw_ds_train = getattr(train_ds, "dataset", train_ds)
        train_indices = getattr(train_ds, "indices", range(len(train_ds)))
        for pos, (raw_i, cd, ads) in enumerate(zip(train_indices, train_class_descs, train_agnostic_descs_all)):
            cls = int(raw_ds_train.labels[raw_i])
            if cls not in shown:
                print(f"    [{categories[cls]}] {cd}")
                for ad in ads:
                    print(f"            → {ad}")
                shown.add(cls)
            if len(shown) >= 4:
                break

        # Encode all view sets and stack into [N_train, 1 + n_agnostic, ST_DIM]
        # View 0: class-bearing, Views 1..n: class-agnostic (one per prefix)
        print(f"\n  Encoding {len(train_class_descs)} class-bearing + {len(train_class_descs) * n_agnostic} class-agnostic descriptions…")
        train_class_embs = encode_texts(train_class_descs, st_model)     # [N, D]

        # Flatten all agnostic variants: [N * n_agnostic] → encode → reshape
        flat_agnostic = [d for shape_ads in train_agnostic_descs_all for d in shape_ads]
        train_agnostic_embs = encode_texts(flat_agnostic, st_model)      # [N * n_agnostic, D]
        train_agnostic_embs = train_agnostic_embs.view(len(train_class_descs), n_agnostic, -1)  # [N, n_agnostic, D]

        # Stack: [N, 1 + n_agnostic, D]
        train_text_embs = torch.cat([
            train_class_embs.unsqueeze(1),      # [N, 1, D]
            train_agnostic_embs,                 # [N, n_agnostic, D]
        ], dim=1)
        print(f"  Text embeddings: {train_text_embs.shape}  (1 class + {n_agnostic} agnostic views)")

        # Save descriptions for inspection
        desc_path = out_dir / "train_descriptions.json"
        with open(desc_path, "w") as f:
            json.dump({
                "class": train_class_descs[:500],
                "agnostic": [ads[:3] for ads in train_agnostic_descs_all[:500]],
            }, f, indent=2)
        print(f"  Sample descriptions saved → {desc_path}")

        # Generate val dual descriptions for best-checkpoint tracking
        print("  Generating val shape descriptions…")
        val_ds_train = get_dataset(args.dataset, split="val", root=DATA_ROOT, num_points=N_POINTS)
        val_class_descs, val_agnostic_descs_all = build_shape_descriptions_dual(
            val_ds_train, categories
        )
        val_class_embs = encode_texts(val_class_descs, st_model)
        flat_val_agnostic = [d for shape_ads in val_agnostic_descs_all for d in shape_ads]
        val_agnostic_embs = encode_texts(flat_val_agnostic, st_model)
        val_agnostic_embs = val_agnostic_embs.view(len(val_class_descs), n_agnostic, -1)
        val_text_embs_train = torch.cat([
            val_class_embs.unsqueeze(1),
            val_agnostic_embs,
        ], dim=1)  # [N_val, 1 + n_agnostic, D]
        print(f"  Val text embeddings: {val_text_embs_train.shape}")

        print(f"\n  Training projection MLP (multi-view text augmentation)…")
        train_history, val_history = train_projection(
            encoder       = encoder,
            proj          = proj,
            criterion     = criterion,
            dataset       = train_ds,
            val_dataset   = val_ds_train,
            text_embs     = train_text_embs,
            val_text_embs = val_text_embs_train,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            lr            = args.lr,
            ckpt_dir      = ckpt_dir,
            prior         = args.prior,
        )
        history = {"train": train_history, "val": val_history}
        torch.save(proj.state_dict(), out_dir / "proj_best.pt")
        torch.save(proj.state_dict(), auto_proj_ckpt)
        print(f"  Projection checkpoint saved → {auto_proj_ckpt}")

    # ── Build test gallery (database) and val set (queries) ──────────────────
    # Proper query/gallery split: val → test (disjoint), matching exp11 protocol.
    val_ds   = get_dataset(args.dataset, split="val", root=DATA_ROOT, num_points=N_POINTS)
    base_val = getattr(val_ds, "dataset", val_ds)

    print("\n  Building test gallery…")
    gallery_z, gallery_labels = encode_dataset(encoder, proj, test_ds)
    # encode_dataset already L2-normalises; re-normalise for safety
    gallery_z = F.normalize(gallery_z, dim=-1)
    print(f"  Gallery: {gallery_z.shape}")

    # gallery_indices: DataLoader(shuffle=False) preserves order → position i = raw index i
    gallery_indices = torch.arange(len(base_test))

    # ── Rotation-stratified retrieval (val→test, 9 angles) ───────────────────
    print("\n  Rotation-stratified retrieval (val→test, 9 angles)…")
    from experiments.exp1_retrieval_eval import rotation_by_angle

    z_q0, q_labels = encode_dataset(encoder, proj, val_ds)
    db_n = gallery_z

    rotation_results = []
    for angle in [0, 15, 30, 45, 60, 90, 120, 150, 180]:
        if angle == 0:
            z_q = z_q0
        else:
            R_mat  = rotation_by_angle(angle, axis="y", device=DEVICE)
            loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                num_workers=2, pin_memory=True)
            all_z  = []
            encoder.eval(); proj.eval()
            with torch.inference_mode():
                for batch in loader:
                    pts = batch[0].to(DEVICE)
                    if pts.shape[1] != 3:
                        pts = pts.permute(0, 2, 1)
                    pts = R_mat @ pts
                    z_risa, _ = unpack_encoder_output(encoder(pts))
                    all_z.append(F.normalize(proj(z_risa), dim=-1).cpu())
            z_q = torch.cat(all_z)

        sim  = F.normalize(z_q, dim=-1) @ db_n.T
        r1   = recall_at_k(sim,  q_labels, gallery_labels, 1)
        r5   = recall_at_k(sim,  q_labels, gallery_labels, 5)
        map5 = mean_ap_at_k(sim, q_labels, gallery_labels, 5)
        rotation_results.append({"angle": angle, "R@1": r1, "R@5": r5, "mAP@5": map5})
        print(f"    θ={angle:5.1f}°  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")

    # ── Standard class-level retrieval (val text desc → test gallery) ─────────
    # Queries: val set descriptions encoded with ST
    # Gallery: test set shape embeddings
    # val and test are disjoint — no diagonal exclusion needed
    print("\n  Standard class-level retrieval (val text desc → test gallery)…")
    print("  Generating val shape descriptions…")
    val_descs  = build_shape_descriptions(val_ds, categories)
    val_embs   = encode_texts(val_descs, st_model)   # [N_val, ST_DIM]
    val_embs_n = F.normalize(val_embs, dim=-1)

    sim  = val_embs_n @ gallery_z.T   # [N_val, N_test] — no self-retrieval possible
    r1   = recall_at_k(sim,  q_labels, gallery_labels, 1)
    r5   = recall_at_k(sim,  q_labels, gallery_labels, 5)
    map5 = mean_ap_at_k(sim, q_labels, gallery_labels, 5)
    print(f"  text(desc)→shape  R@1={r1:.3f}  R@5={r5:.3f}  mAP@5={map5:.3f}")

    # ── Part-composition evaluation ───────────────────────────────────────────
    print(f"\n  Part-composition query evaluation (top-{args.topk_eval})…")
    part_results = evaluate_part_queries(
        queries          = PART_QUERIES,
        gallery_z        = gallery_z,
        gallery_labels   = gallery_labels,
        gallery_indices  = gallery_indices,
        categories       = categories,
        st_model         = st_model,
        raw_ds           = base_test,
        topk             = args.topk_eval,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    mean_part_hit  = float(np.mean([r["part_hit_rate"]  for r in part_results]))
    mean_class_hit = float(np.mean([r["class_hit_rate"] for r in part_results]))
    print(f"\n  Aggregate cross-class retrieval:  "
          f"mean part_hit@{args.topk_eval}={mean_part_hit:.3f}  "
          f"mean class_hit@{args.topk_eval}={mean_class_hit:.3f}")

    results = {
        "config":             vars(args),
        "rotation_retrieval": rotation_results,
        "standard_retrieval": {"R@1": r1, "R@5": r5, "mAP@5": map5},
        "part_query_results": part_results,
        "cross_class_summary": {
            f"mean_part_hit@{args.topk_eval}":  mean_part_hit,
            f"mean_class_hit@{args.topk_eval}": mean_class_hit,
        },
        "train_loss_history": history,
        "timestamp":          ts,
    }
    out_path = out_dir / "exp13_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    site_path = ROOT / "outputs" / "exp13_part_text_retrieval.json"
    with open(site_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Site results: {site_path}")


if __name__ == "__main__":
    main()
