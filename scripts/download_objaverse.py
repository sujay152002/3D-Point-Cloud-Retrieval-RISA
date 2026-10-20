"""Download and preprocess Objaverse → objaverse_train.npz

Steps:
  1. Load all annotations, filter to downloadable objects with a usable name
  2. Sample N_OBJECTS UIDs (seeded for reproducibility)
  3. Download GLBs in parallel via objaverse.load_objects()
  4. Sample NUM_POINTS points from each mesh surface via trimesh
  5. Build text string: name + tags + description
  6. Save data/objaverse/objaverse_train.npz and delete GLB cache

Usage:
    python scripts/download_objaverse.py
    python scripts/download_objaverse.py --n_objects 50000 --num_points 2048
    python scripts/download_objaverse.py --resume   # skip already-processed UIDs
"""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "objaverse"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Text construction ─────────────────────────────────────────────────────────

def build_text(ann: dict) -> str:
    """Build a text description from Objaverse annotation fields."""
    parts = []
    name = (ann.get("name") or "").strip()
    if name:
        parts.append(name)

    tags = [t["name"] for t in (ann.get("tags") or []) if t.get("name")]
    if tags:
        parts.append(", ".join(tags[:10]))   # cap at 10 tags

    desc = (ann.get("description") or "").strip()
    if desc:
        parts.append(desc[:200])             # cap description length

    return ". ".join(parts) if parts else "3D object"


# ── Point cloud sampling ──────────────────────────────────────────────────────

def mesh_to_pointcloud(glb_path: str, num_points: int) -> np.ndarray | None:
    """Sample num_points from mesh surface. Returns [num_points, 3] float32 or None."""
    try:
        import trimesh
        scene = trimesh.load(glb_path, force="scene", process=False)

        # Merge all meshes in the scene
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values()
                      if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
            if not meshes:
                return None
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            return None

        if len(mesh.faces) == 0:
            return None

        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        return pts.astype(np.float32)

    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_objects",  type=int, default=50_000)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--processes",  type=int, default=8,
                        help="Parallel download workers")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Download and process in batches of this size")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip UIDs already in partial save")
    parser.add_argument("--delete_glb", action="store_true", default=True,
                        help="Delete GLB files after processing to save disk")
    parser.add_argument("--no_delete_glb", dest="delete_glb", action="store_false")
    args = parser.parse_args()

    import objaverse

    # ── Load annotations ──────────────────────────────────────────────────────
    print("Loading Objaverse annotations…")
    anns = objaverse.load_annotations()
    print(f"  Total objects: {len(anns)}")

    # Filter: downloadable + has a name
    valid_uids = [
        uid for uid, ann in anns.items()
        if ann.get("isDownloadable") and (ann.get("name") or "").strip()
    ]
    print(f"  Downloadable with name: {len(valid_uids)}")

    # Sample
    rng = random.Random(args.seed)
    rng.shuffle(valid_uids)
    selected_uids = valid_uids[:args.n_objects]
    print(f"  Selected: {len(selected_uids)}")

    # ── Resume: load partial results ──────────────────────────────────────────
    partial_path = OUT_DIR / "objaverse_partial.npz"
    done_uids    = set()
    all_points, all_texts, all_uids = [], [], []

    if args.resume and partial_path.exists():
        partial = np.load(partial_path, allow_pickle=True)
        all_points = list(partial["points"])
        all_texts  = list(partial["texts"])
        all_uids   = list(partial["uids"])
        done_uids  = set(all_uids)
        print(f"  Resuming: {len(done_uids)} already processed")

    remaining = [u for u in selected_uids if u not in done_uids]
    print(f"  Remaining: {len(remaining)}")

    # ── Process in batches ────────────────────────────────────────────────────
    n_failed = 0
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_uids = remaining[batch_start : batch_start + args.batch_size]
        batch_num  = batch_start // args.batch_size + 1
        n_batches  = (len(remaining) + args.batch_size - 1) // args.batch_size
        print(f"\n  Batch {batch_num}/{n_batches}  ({len(all_points)} processed so far)")

        # Download GLBs
        print(f"    Downloading {len(batch_uids)} objects…")
        glb_paths = objaverse.load_objects(
            uids       = batch_uids,
            download_processes = args.processes,
        )

        # Process each downloaded object
        for uid in batch_uids:
            glb_path = glb_paths.get(uid)
            if not glb_path or not Path(glb_path).exists():
                n_failed += 1
                continue

            pts = mesh_to_pointcloud(glb_path, args.num_points)

            if args.delete_glb:
                try:
                    os.remove(glb_path)
                except OSError:
                    pass

            if pts is None:
                n_failed += 1
                continue

            text = build_text(anns[uid])
            all_points.append(pts)
            all_texts.append(text)
            all_uids.append(uid)

        print(f"    Processed: {len(all_points)}  Failed: {n_failed}")

        # Save partial checkpoint every batch
        np.savez(
            partial_path,
            points = np.array(all_points, dtype=np.float32),
            texts  = np.array(all_texts,  dtype=object),
            uids   = np.array(all_uids,   dtype=object),
        )
        print(f"    Partial save → {partial_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    print(f"\n  Final: {len(all_points)} objects  ({n_failed} failed)")

    out_path = OUT_DIR / "objaverse_train.npz"
    np.savez(
        out_path,
        points = np.array(all_points, dtype=np.float32),
        texts  = np.array(all_texts,  dtype=object),
        uids   = np.array(all_uids,   dtype=object),
    )
    print(f"  Saved → {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1e9:.2f} GB")

    if partial_path.exists():
        partial_path.unlink()
        print(f"  Removed partial save")

    # Save UID list for reference
    uid_list_path = OUT_DIR / "objaverse_train_uids.json"
    with open(uid_list_path, "w") as f:
        json.dump(all_uids, f)
    print(f"  UID list → {uid_list_path}")


if __name__ == "__main__":
    main()
