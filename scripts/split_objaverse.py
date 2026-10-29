#!/usr/bin/env python3
"""Split the single objaverse_train.npz into train/val/test npz files.

Uses the same deterministic permutation (seed 42) as get_dataset() so the
train portion is identical to the current carved split — the existing train
text cache (39637 samples) remains valid.

Split ratio: 80 / 10 / 10  (train / val / test)
"""
import shutil
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "objaverse"
SRC      = DATA_DIR / "objaverse_train.npz"
BACKUP   = DATA_DIR / "objaverse_train_full.npz"   # safety backup of original

TRAIN_END = int(49547 * 0.80)   # 39637
VAL_END   = int(49547 * 0.90)   # 44592  (train + 10%)

def main():
    assert SRC.exists(), f"Source not found: {SRC}"

    # Load everything into memory (not memory-mapped) so overwriting the
    # source file during the train split doesn't corrupt subsequent reads.
    npz  = np.load(SRC, allow_pickle=True)
    keys = list(npz.keys())
    data = {k: npz[k] for k in keys}          # eager load into RAM
    npz.close()
    n = len(data["points"])
    print(f"Loaded {SRC.name}: {n} objects, keys={keys}")

    # Same permutation as core/datasets.py get_dataset()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = perm[:TRAIN_END]
    val_idx   = perm[TRAIN_END:VAL_END]
    test_idx  = perm[VAL_END:]

    print(f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # Backup original if not already backed up
    if not BACKUP.exists():
        shutil.copy2(SRC, BACKUP)
        print(f"Backed up original → {BACKUP.name}")

    def save_split(indices, split_name):
        out = DATA_DIR / f"objaverse_{split_name}.npz"
        kwargs = {}
        for k in keys:
            kwargs[k] = data[k][indices]
        np.savez(out, **kwargs)
        print(f"  Saved {out.name}: {len(indices)} objects")

    # Write the three splits
    save_split(train_idx, "train")
    save_split(val_idx,   "val")
    save_split(test_idx,  "test")

    print("\nDone. The original file has been overwritten with the 80% train split.")
    print(f"Full dataset preserved at: {BACKUP}")

if __name__ == "__main__":
    main()
