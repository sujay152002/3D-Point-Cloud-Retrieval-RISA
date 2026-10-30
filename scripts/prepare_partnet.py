"""prepare_partnet.py — Convert PartNet sem_seg_h5 → project npz format.

Input (sem_seg_h5/ from HuggingFace ShapeNet/PartNet-archive):
    sem_seg_h5/
        Chair-3/          # <Category>-<level>  (level 3 = finest granularity)
            train-00.h5   # keys: data [N,10000,3], data_num [N], label_seg [N,10000]
            train-01.h5
            ...
            val-00.h5
            test-00.h5
        Table-3/
        ...

Output (data/partnet/):
    partnet_train.npz  — points [N,num_points,3]  float32
                          labels [N]               int64   (category index)
                          part_labels [N,num_points] int32 (per-point part id, 0-based per category)
                          categories               dict    {int: str}
    partnet_val.npz
    partnet_test.npz

Usage
-----
    # Default: level-3 (finest), 10,000 pts kept as-is (all categories)
    python scripts/prepare_partnet.py

    # Subsample to 2048 points per shape
    python scripts/prepare_partnet.py --num_points 2048

    # Custom input/output paths
    python scripts/prepare_partnet.py \\
        --h5_root data/partnet/sem_seg_h5 \\
        --out_dir data/partnet \\
        --num_points 2048 --level 3

    # Only a few categories
    python scripts/prepare_partnet.py --categories Chair Table Lamp --num_points 2048
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# Categories present in PartNet sem_seg_h5 at each level.
# Not every category has all three levels — the script auto-detects what's available.
ALL_CATEGORIES = [
    "Bag", "Bed", "Bottle", "Bowl", "Chair",
    "Clock", "Dishwasher", "Display", "Door", "Earphone",
    "Faucet", "Hat", "Keyboard", "Knife", "Lamp",
    "Laptop", "Microwave", "Mug", "Refrigerator", "Scissors",
    "StorageFurniture", "Table", "TrashCan", "Vase",
]


def load_h5_files(h5_dir: Path, split: str, num_points: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Load all H5 files for one category/split directory.

    Returns:
        pts        : float32 [N, num_points, 3]
        part_labels: int32   [N, num_points]   (0-based, remapped per-category)
    """
    # Find all h5 files for this split: train-00.h5, train-01.h5, …
    files = sorted(h5_dir.glob(f"{split}-*.h5"))
    if not files:
        return np.empty((0, num_points, 3), dtype=np.float32), \
               np.empty((0, num_points),    dtype=np.int32)

    all_pts, all_seg = [], []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            data     = f["data"][:]       # [B, 10000, 3]  float64
            data_num = f["data_num"][:]   # [B]            int32 — actual pts per shape
            label    = f["label_seg"][:]  # [B, 10000]     int32

        for i in range(len(data)):
            n     = int(data_num[i])
            pts   = data[i, :n, :].astype(np.float32)   # [n, 3]
            segs  = label[i, :n].astype(np.int32)        # [n]

            if n < num_points:
                idx = rng.choice(n, num_points, replace=True)
            else:
                idx = rng.choice(n, num_points, replace=False)

            all_pts.append(pts[idx])
            all_seg.append(segs[idx])

    if not all_pts:
        return np.empty((0, num_points, 3), dtype=np.float32), \
               np.empty((0, num_points),    dtype=np.int32)

    pts_arr = np.stack(all_pts, axis=0).astype(np.float32)  # [N, P, 3]
    seg_arr = np.stack(all_seg, axis=0).astype(np.int32)    # [N, P]

    # Keep original PartNet part IDs — do NOT remap.
    # Remapping would break the correspondence with PARTNET_PART_NAMES
    # which uses the original taxonomy IDs from the PartNet benchmark.
    return pts_arr, seg_arr


def main():
    parser = argparse.ArgumentParser(
        description="Convert PartNet sem_seg_h5 → partnet_train/val/test.npz"
    )
    parser.add_argument(
        "--h5_root", default=None,
        help="Path to sem_seg_h5/ directory "
             "(default: <project>/data/partnet/sem_seg_h5/)"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: <project>/data/partnet/)"
    )
    parser.add_argument(
        "--level", type=int, default=3, choices=[1, 2, 3],
        help="Part granularity level: 3=finest (default), 1=coarsest"
    )
    parser.add_argument(
        "--num_points", type=int, default=10000,
        help="Points to sample per shape (default: 10000 = keep all). "
             "Use 2048 or 1024 to match ShapeNet."
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Categories to include (default: all available)"
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    h5_root = Path(args.h5_root) if args.h5_root \
              else project_root / "data" / "partnet" / "sem_seg_h5"
    out_dir = Path(args.out_dir) if args.out_dir \
              else project_root / "data" / "partnet"

    if not h5_root.exists():
        print(f"Error: h5_root not found: {h5_root}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Discover which categories are available at the requested level
    requested = args.categories or ALL_CATEGORIES
    available = []
    for cat in requested:
        d = h5_root / f"{cat}-{args.level}"
        if d.exists():
            available.append(cat)
        else:
            # Try falling back to a lower level
            for fallback in [2, 1]:
                d2 = h5_root / f"{cat}-{fallback}"
                if d2.exists():
                    print(f"  [{cat}] level-{args.level} not found, using level-{fallback}")
                    available.append(cat)
                    break
            else:
                print(f"  [{cat}] not found at any level — skipping")

    if not available:
        print("No categories found. Check --h5_root path.")
        sys.exit(1)

    cat_to_label = {cat: i for i, cat in enumerate(available)}
    label_to_cat = {i: cat for i, cat in enumerate(available)}

    print(f"\nPartNet sem_seg_h5 : {h5_root}")
    print(f"Output             : {out_dir}")
    print(f"Level              : {args.level} (finest=3)")
    print(f"Points/shape       : {args.num_points}")
    print(f"Categories ({len(available)}): {available}\n")

    for split in ["train", "val", "test"]:
        print(f"{'='*55}")
        print(f"  Split: {split}")
        print(f"{'='*55}")

        all_pts, all_labels, all_part_labels = [], [], []

        for cat in available:
            # Use the highest available level for this category
            cat_dir = None
            for lvl in [args.level, 2, 1]:
                d = h5_root / f"{cat}-{lvl}"
                if d.exists():
                    cat_dir = d
                    break

            pts_arr, seg_arr = load_h5_files(cat_dir, split, args.num_points, rng)

            if len(pts_arr) == 0:
                print(f"  [{cat}] 0 shapes in {split} — skipping")
                continue

            n_parts = int(seg_arr.max()) + 1
            print(f"  [{cat}] {len(pts_arr):5d} shapes  {n_parts} part ids  ← {cat_dir.name}")

            lab = np.full(len(pts_arr), cat_to_label[cat], dtype=np.int64)
            all_pts.append(pts_arr)
            all_labels.append(lab)
            all_part_labels.append(seg_arr)

        if not all_pts:
            print(f"  No shapes for split '{split}' — skipping.\n")
            continue

        points      = np.concatenate(all_pts,         axis=0)  # [N, P, 3]
        labels      = np.concatenate(all_labels,      axis=0)  # [N]
        part_labels = np.concatenate(all_part_labels, axis=0)  # [N, P]

        # Shuffle
        perm        = rng.permutation(len(points))
        points      = points[perm]
        labels      = labels[perm]
        part_labels = part_labels[perm]

        out_path = out_dir / f"partnet_{split}.npz"
        np.savez_compressed(
            out_path,
            points      = points,
            labels      = labels,
            part_labels = part_labels,
            categories  = np.array(label_to_cat, dtype=object),
        )

        n_parts_global = int(part_labels.max()) + 1
        print(f"\n  → Saved {len(points):,} shapes to {out_path}")
        print(f"     points      : {points.shape}  {points.dtype}")
        print(f"     labels      : {labels.shape}  ({len(np.unique(labels))} classes)")
        print(f"     part_labels : {part_labels.shape}  "
              f"(max part id per-cat remapped; global range 0–{n_parts_global-1})\n")

    print("Done!")
    print()
    print("Verify with:")
    print("  python -c \"")
    print("  import sys; sys.path.insert(0,'.')") 
    print("  from core.datasets import get_dataset")
    print("  for s in ['train','val','test']:")
    print("      ds = get_dataset('partnet', split=s, root='data')")
    print("      print(s, len(ds))\"")


if __name__ == "__main__":
    main()
