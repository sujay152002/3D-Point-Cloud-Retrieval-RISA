"""Core package: geometry, metrics, datasets, and trainer utilities."""

from pathlib import Path

outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)

_ALL_DATASETS = ["shapenet", "modelnet40", "scanobjectnn", "modelnet10", "synthetic"]
