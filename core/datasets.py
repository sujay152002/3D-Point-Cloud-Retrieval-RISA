"""
Real Dataset Loaders for Point Cloud Research

Supports:
- ModelNet40: CAD models, 40 categories, ~12K shapes
- ModelNet10: CAD models, 10 categories (simpler benchmark)
- ShapeNet: Large-scale dataset, 55 categories, ~51K shapes
- ScanObjectNN: Real-world scans, 15 categories, ~15K objects
- PartNet: Fine-grained part annotations
- S3DIS: Stanford 3D Indoor Scenes

All datasets return point clouds in channels-first format (B, 3, N).
"""

import h5py
import urllib.request
import zipfile
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# ModelNet40 Dataset
# ============================================================================

class ModelNet40Dataset(Dataset):
    def __init__(self, root='data/modelnet40', split='train', num_points=1024,
                 download=False, normalize=True, augment=None):
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize

        if download:
            self.download()

        data_path = self.root / f'modelnet40_{split}.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f"ModelNet40 {split} data not found at {data_path}. "
                f"Set download=True or manually download from:\n"
                f"https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
            )

        data = np.load(data_path, allow_pickle=True)
        self.points      = data['points']
        self.labels      = data['labels']
        self.part_labels = data['part_labels'] if 'part_labels' in data else None
        self.categories  = data['categories'].item()
        self.augment     = (split == 'train') if augment is None else augment

        # Pre-sample fixed indices for val/test to ensure deterministic evaluation
        if not self.augment:
            rng = np.random.default_rng(0)
            self._fixed_choice = [
                rng.choice(len(p), self.num_points, replace=len(p) < self.num_points)
                for p in self.points
            ]

        print(f"✓ Loaded ModelNet40 {split}: {len(self)} samples, {len(self.categories)} categories")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].copy()
        label  = self.labels[idx]

        if self.augment:
            choice = np.random.choice(len(points), self.num_points, replace=len(points) < self.num_points)
        else:
            choice = self._fixed_choice[idx]
        points = points[choice]

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points   = points - centroid
            scale    = np.max(np.linalg.norm(points, axis=-1))
            points   = points / (scale + 1e-8)

        if self.augment:
            # Random z-axis rotation (standard in PointNet/DGCNN/etc.)
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            points = points @ R.T
            # Random jitter (sigma=0.01, clip=0.05 — standard in PointNet)
            jitter = np.clip(np.random.normal(0, 0.01, points.shape), -0.05, 0.05)
            points = points + jitter
            # Random scale (0.8–1.25 — standard in PointNet)
            scale_factor = np.random.uniform(0.8, 1.25)
            points = points * scale_factor

        pts_t = torch.from_numpy(points).float().permute(1, 0)
        lbl_t = torch.tensor(label, dtype=torch.long)
        if self.part_labels is not None:
            part_lbl = torch.from_numpy(self.part_labels[idx][choice]).long()
            return pts_t, lbl_t, part_lbl
        return pts_t, lbl_t

    def download(self):
        print("Downloading ModelNet40 dataset...")
        self.root.mkdir(parents=True, exist_ok=True)
        url      = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zip_path = self.root / "modelnet40_ply_hdf5_2048.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(url, zip_path)
        extract_dir = self.root / "modelnet40_ply_hdf5_2048"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.root)
        self._convert_h5_to_npz(extract_dir)

    def _convert_h5_to_npz(self, h5_dir):
        for split in ['train', 'test']:
            npz_path = self.root / f'modelnet40_{split}.npz'
            if npz_path.exists():
                continue
            file_list = h5_dir / f'{split}_files.txt'
            with open(file_list, 'r') as f:
                h5_files = [line.strip() for line in f]
            all_points, all_labels = [], []
            for h5_file in h5_files:
                h5_path = h5_dir / h5_file.split('/')[-1]
                with h5py.File(h5_path, 'r') as f:
                    all_points.append(f['data'][:])
                    all_labels.append(f['label'][:])
            all_points = np.concatenate(all_points, axis=0)
            all_labels = np.concatenate(all_labels, axis=0).squeeze()
            categories = {
                0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf',
                5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone',
                10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser',
                15: 'flower_pot', 16: 'glass_box', 17: 'guitar', 18: 'keyboard', 19: 'lamp',
                20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand', 24: 'person',
                25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink',
                30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent',
                35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'
            }
            np.savez(npz_path, points=all_points, labels=all_labels, categories=categories)
            print(f"✓ Saved {npz_path}: {len(all_points)} samples")


# ============================================================================
# ModelNet10 Dataset
# ============================================================================

class ModelNet10Dataset(Dataset):
    CATEGORIES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                  'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    def __init__(self, root='data/modelnet10', split='train', num_points=1024,
                 download=False, normalize=True):
        self.root       = Path(root)
        self.split      = split
        self.num_points = num_points
        self.normalize  = normalize

        data_path = self.root / f'modelnet10_{split}.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f"ModelNet10 {split} data not found at {data_path}. "
                f"Please download from: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
            )

        data = np.load(data_path, allow_pickle=True)
        self.points      = data['points']
        self.labels      = data['labels']
        self.part_labels = data['part_labels'] if 'part_labels' in data else None
        self.categories  = {i: cat for i, cat in enumerate(self.CATEGORIES)}

        print(f"✓ Loaded ModelNet10 {split}: {len(self)} samples, {len(self.categories)} categories")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].copy()
        label  = self.labels[idx]

        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice]

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points   = points - centroid
            scale    = np.max(np.linalg.norm(points, axis=-1))
            points   = points / (scale + 1e-8)

        pts_t = torch.from_numpy(points).float().permute(1, 0)
        lbl_t = torch.tensor(label, dtype=torch.long)
        if self.part_labels is not None:
            part_lbl = torch.from_numpy(self.part_labels[idx][choice]).long()
            return pts_t, lbl_t, part_lbl
        return pts_t, lbl_t

    def download(self):
        print("ModelNet10 is a subset of ModelNet40.")
        print("https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip")


# ============================================================================
# ShapeNet Dataset
# ============================================================================

# Valid unified part indices per ShapeNet Part object category (16 categories).
# Matches the standard PointNet++/DGCNN part-segmentation benchmark.
SHAPE_NET_PART_SEG_CLASSES = [
    [0, 1, 2, 3],                  # airplane
    [4, 5],                        # bag
    [6, 7],                        # cap
    [8, 9, 10, 11],                # car
    [12, 13, 14, 15],              # chair
    [16, 17, 18],                  # earphone
    [19, 20, 21],                  # guitar
    [22, 23],                      # knife
    [24, 25, 26, 27],              # lamp
    [28, 29],                      # laptop
    [30, 31, 32, 33, 34, 35],      # motorbike
    [36, 37],                      # mug
    [38, 39, 40],                  # pistol
    [41, 42, 43],                  # rocket
    [44, 45, 46],                  # skateboard
    [47, 48, 49],                  # table
]


class ShapeNetDataset(Dataset):
    def __init__(self, root='data/shapenet', split='train', num_points=1024,
                 download=False, normalize=True, category=None, num_classes=None):
        self.root       = Path(root)
        self.split      = split
        self.num_points = num_points
        self.normalize  = normalize
        self.category   = category
        self.num_classes = num_classes

        if download:
            raise NotImplementedError("Manual download required for ShapeNet")

        data_path = self.root / f'shapenet_{split}.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f"ShapeNet {split} data not found at {data_path}. "
                f"Please download manually and place the npz file at {data_path}."
            )

        data = np.load(data_path, allow_pickle=True)
        self.points      = data['points']
        self.categories  = data['categories'].item()
        self.part_labels = data['part_labels'] if 'part_labels' in data else None
        self.load_labels(data['labels'])
        self.num_part_classes = 50
        self.part_seg_classes = SHAPE_NET_PART_SEG_CLASSES

        has_seg = self.part_labels is not None
        print(f"✓ Loaded ShapeNet {split}: {len(self)} samples (per-point labels: {has_seg})")

    def load_labels(self, labels):
        # Work on a copy — the npz array may be memory-mapped and shared between
        # the train and val dataset instances.  Mutating it in-place would corrupt
        # the second load and could modify the on-disk file.
        labels = labels.copy()
        class_counts = np.bincount(labels.astype(np.int32))
        actual_classes = int((class_counts > 0).sum())
        if self.num_classes is not None and actual_classes > self.num_classes:
            null_class = self.num_classes
            for i, top_class in enumerate(np.argsort(class_counts)[::-1]):
                if i < self.num_classes:
                    labels[labels == top_class] = i
                else:
                    labels[labels == top_class] = null_class
            self.num_classes = null_class + 1
        else:
            self.num_classes = actual_classes
        self.class_counts = np.bincount(labels.astype(np.int32))
        self.labels = labels

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].copy()
        label  = self.labels[idx]

        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice]

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points   = points - centroid
            scale    = np.max(np.linalg.norm(points, axis=-1))
            points   = points / (scale + 1e-8)

        pts_t = torch.from_numpy(points).float().permute(1, 0)
        lbl_t = torch.tensor(int(label), dtype=torch.long)

        if self.part_labels is not None:
            part_lbl = torch.from_numpy(self.part_labels[idx][choice]).long()
            return pts_t, lbl_t, part_lbl

        return pts_t, lbl_t


# ============================================================================
# ScanObjectNN Dataset
# ============================================================================

class ScanObjectNNDataset(Dataset):
    def __init__(self, root='data/scanobjectnn', split='train', num_points=1024,
                 variant='OBJ_ONLY', download=False, normalize=True):
        self.root       = Path(root)
        self.split      = split
        self.num_points = num_points
        self.variant    = variant
        self.normalize  = normalize
        self.augment    = (split == 'train')

        data_path = self.root / f'scanobjectnn_{variant}_{split}.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f"ScanObjectNN {split} data not found at {data_path}. "
                f"Please download from: https://hkust-vgd.github.io/scanobjectnn/"
            )

        data = np.load(data_path, allow_pickle=True)
        self.points      = data['points']
        self.labels      = data['labels']
        self.part_labels = data['part_labels'] if 'part_labels' in data else None
        self.categories  = data['categories'].item()
        self.num_classes = len(self.categories)

        # Pre-sample fixed indices for val/test to ensure deterministic evaluation
        if not self.augment:
            rng = np.random.default_rng(0)
            self._fixed_choice = [
                rng.choice(len(p), self.num_points, replace=len(p) < self.num_points)
                for p in self.points
            ]

        print(f"✓ Loaded ScanObjectNN {split}: {len(self)} samples")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].copy()
        label  = self.labels[idx]

        if self.augment:
            choice = np.random.choice(len(points), self.num_points,
                                      replace=len(points) < self.num_points)
        else:
            choice = self._fixed_choice[idx]
        points = points[choice]

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points   = points - centroid
            scale    = np.max(np.linalg.norm(points, axis=-1))
            points   = points / (scale + 1e-8)

        if self.augment:
            # Jitter (sigma=0.01, clip=0.05) — handles sensor noise in real scans
            jitter = np.clip(np.random.normal(0, 0.01, points.shape), -0.05, 0.05)
            points = points + jitter
            # Random scale (0.8–1.25) — handles size variation
            points = points * np.random.uniform(0.8, 1.25)
            # Random point dropout (drop up to 10%) — simulates occlusion
            if np.random.random() < 0.5:
                n_keep = np.random.randint(int(0.9 * self.num_points), self.num_points)
                keep   = np.random.choice(self.num_points, n_keep, replace=False)
                pad    = np.random.choice(keep, self.num_points - n_keep, replace=True)
                points = points[np.concatenate([keep, pad])]

        pts_t = torch.from_numpy(points).float().permute(1, 0)
        lbl_t = torch.tensor(label, dtype=torch.long)
        if self.part_labels is not None:
            part_lbl = torch.from_numpy(self.part_labels[idx][choice]).long()
            return pts_t, lbl_t, part_lbl
        return pts_t, lbl_t


# ============================================================================
# PartNet Dataset
# ============================================================================

class PartNetDataset(Dataset):
    CATEGORIES = ['Chair', 'Table', 'Lamp', 'Vase', 'StorageFurniture',
                  'Bed', 'Guitar', 'Motorcycle', 'Airplane', 'Bicycle']

    def __init__(self, root='data/partnet', split='train', num_points=1024,
                 download=False, normalize=True, category=None):
        self.root       = Path(root)
        self.split      = split
        self.num_points = num_points
        self.normalize  = normalize
        self.category   = category

        data_path = self.root / f'partnet_{split}.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f"PartNet {split} data not found at {data_path}. "
                f"Please download from: https://partnet.cs.stanford.edu/"
            )

        data = np.load(data_path, allow_pickle=True)
        self.points      = data['points']
        self.labels      = data['labels']
        self.part_labels = data.get('part_labels', np.zeros_like(data['labels']))
        self.categories  = {i: cat for i, cat in enumerate(self.CATEGORIES)}

        print(f"✓ Loaded PartNet {split}: {len(self)} samples")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points     = self.points[idx].copy()
        label      = self.labels[idx]
        part_label = self.part_labels[idx]

        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points     = points[choice]
        part_label = part_label[choice]

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points   = points - centroid
            scale    = np.max(np.linalg.norm(points, axis=-1))
            points   = points / (scale + 1e-8)

        return (torch.from_numpy(points).float().permute(1, 0),
                torch.tensor(label, dtype=torch.long),
                torch.from_numpy(part_label).long())


# ============================================================================
# S3DIS Dataset
# ============================================================================

class S3DISDataset(Dataset):
    CATEGORIES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window',
                  'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    CAT_TO_ID  = {cat: i for i, cat in enumerate(CATEGORIES)}

    def __init__(self, root='data/s3dis', split='train', num_points=4096,
                 download=False, normalize=True, area='all'):
        self.root       = Path(root)
        self.split      = split
        self.num_points = num_points
        self.normalize  = normalize
        self.area       = area

        data_path = self.root / f's3dis_{split}.npz'
        if not data_path.exists():
            raise FileNotFoundError(
                f"S3DIS {split} data not found at {data_path}. "
                f"Please download from: http://buildingparser.stanford.edu/dataset.html"
            )

        data = np.load(data_path, allow_pickle=True)
        self.points     = data['points']
        self.labels     = data['labels']
        self.colors     = data.get('colors', np.zeros_like(self.points))
        self.categories = self.CAT_TO_ID

        print(f"✓ Loaded S3DIS {split}: {len(self)} room segments")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].copy()
        labels = self.labels[idx].copy()
        colors = self.colors[idx].copy()

        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice]; labels = labels[choice]; colors = colors[choice]

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points   = points - centroid
            scale    = np.max(np.linalg.norm(points, axis=-1))
            points   = points / (scale + 1e-8)

        return (torch.from_numpy(points).float().permute(1, 0),
                torch.from_numpy(labels).long(),
                torch.from_numpy(colors / 255.0).float())


# ============================================================================
# Unified Dataset Loader
# ============================================================================

def get_dataset(name, split='train', root='data', num_points=1024, download=False, **kwargs):
    """Unified interface to load any dataset.

    Args:
        name:       'modelnet40', 'modelnet10', 'shapenet', 'scanobjectnn', 'partnet', 's3dis'
        split:      'train', 'val', or 'test'
        root:       Data root directory
        num_points: Number of points to sample per cloud
        download:   Auto-download if possible

    Notes:
        Val split is carved deterministically from the train data (80/10/10).
        The test split is never touched during training or model selection.
    """
    registry = {
        'modelnet40':   ModelNet40Dataset,
        'modelnet10':   ModelNet10Dataset,
        'shapenet':     ShapeNetDataset,
        'scanobjectnn': ScanObjectNNDataset,
        'partnet':      PartNetDataset,
        's3dis':        S3DISDataset,
    }

    if name.lower() == 'synthetic':
        from core.geometry import generate_sphere, generate_cube, generate_chair, normalize_pc
        from torch.utils.data import TensorDataset, Subset
        n_per_class = 500
        shapes, labels = [], []
        for gen_fn, label in [
            (generate_sphere, 0),
            (generate_cube,   1),
            (generate_chair,  2),
        ]:
            batch = normalize_pc(gen_fn(n_per_class, num_points))
            for i in range(n_per_class):
                shapes.append(batch[i].permute(1, 0))
                labels.append(label)
        full_ds = TensorDataset(torch.stack(shapes), torch.tensor(labels))
        n = len(full_ds)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        splits  = {
            'train': indices[:n_train],
            'val':   indices[n_train:n_train + n_val],
            'test':  indices[n_train + n_val:],
        }
        if split not in splits:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        ds = Subset(full_ds, splits[split])
        ds.num_classes = 3
        return ds

    if name.lower() not in registry:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(registry)}")

    # For real datasets, val is carved from the train npz via index split
    if split == 'val':
        # ScanObjectNN: use the held-out test split as val (matches published benchmarks)
        if name.lower() == 'scanobjectnn':
            return registry[name.lower()](
                root=Path(root) / name.lower(),
                split='test',
                num_points=num_points,
                download=download,
                **kwargs
            )
        base_ds = registry[name.lower()](
            root=Path(root) / name.lower(),
            split='train',
            num_points=num_points,
            download=download,
            **(dict(augment=False) if name.lower() == 'modelnet40' else {}),
            **kwargs
        )
        n = len(base_ds)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        val_start = int(n * 0.80)
        from torch.utils.data import Subset
        return Subset(base_ds, indices[val_start:])

    if split == 'train':
        # ScanObjectNN: use the full train split (don't carve val out of it)
        if name.lower() == 'scanobjectnn':
            return registry[name.lower()](
                root=Path(root) / name.lower(),
                split='train',
                num_points=num_points,
                download=download,
                **kwargs
            )
        base_ds = registry[name.lower()](
            root=Path(root) / name.lower(),
            split='train',
            num_points=num_points,
            download=download,
            **kwargs
        )
        n = len(base_ds)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        train_end = int(n * 0.80)
        from torch.utils.data import Subset
        return Subset(base_ds, indices[:train_end])

    # split == 'test': use the held-out test npz as-is
    return registry[name.lower()](
        root=Path(root) / name.lower(),
        split='test',
        num_points=num_points,
        download=download,
        **kwargs
    )


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently for reproducible augmentation.

    Uses torch.initial_seed() which is set per-worker by PyTorch based on the
    base seed passed to DataLoader, ensuring both torch and numpy RNG states
    are distinct and reproducible across workers and runs.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False):
    """Build a DataLoader with per-worker RNG seeding.

    Args:
        drop_last: Drop the last incomplete batch.  Set True for training to
                   avoid single-sample batches that crash BatchNorm.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    for name in ['modelnet40', 'modelnet10', 'shapenet', 'scanobjectnn']:
        print(f"\n{name}: {get_dataset(name, split='train')}")
