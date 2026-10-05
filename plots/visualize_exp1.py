"""Visualization for Exp 1 — Rotation-Stratified Retrieval results."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "outputs/exp1_retrieval.json"
OUT_DIR = ROOT / "plots/exp1_all_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    raw = json.load(f)

MODELS = list(raw.keys())
DATASETS = list(next(iter(raw.values())).keys())

MODEL_COLORS = {
    "PointNet++": "#e63946",
    "DGCNN":      "#457b9d",
    "DiPVNet":    "#f4a261",
    "RINet":      "#8ecae6",
    "RISA":       "#2a9d8f",
}
METRIC_STYLES = {
    "r1":   {"lw": 2.5, "ls": "-",  "marker": "o", "ms": 4},
    "r5":   {"lw": 1.5, "ls": "--", "marker": "s", "ms": 3},
    "map5": {"lw": 1.5, "ls": ":",  "marker": "^", "ms": 3},
}

# ── Figure 1: R@1 all models on one plot per dataset ─────────────────────────
for ds in DATASETS:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODELS:
        if ds not in raw[model]:
            continue
        rows   = raw[model][ds]
        angles = [r["angle"] for r in rows]
        r1     = [r["r1"]    for r in rows]
        mean   = np.mean(r1)
        drop   = max(r1) - min(r1)
        color  = MODEL_COLORS.get(model, "#888888")
        label  = f"{model}  (mean={mean:.3f}, drop={drop:.4f})"
        ax.plot(angles, r1, color=color, label=label, **METRIC_STYLES["r1"])

    ax.set_title(f"R@1 vs Rotation Angle — {ds}\nFlat curves = rotation invariant",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Rotation angle (°)", fontsize=11)
    ax.set_ylabel("Recall@1", fontsize=11)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, 181, 10))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / f"exp1_r1_{ds}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Figure 2: per-model subplots (R@1, R@5, mAP@5) for each dataset ──────────
for ds in DATASETS:
    n = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Exp 1 — All Models on {ds}\nR@1 / R@5 / mAP@5 vs Rotation Angle",
                 fontsize=12, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        if ds not in raw[model]:
            ax.set_visible(False)
            continue
        rows   = raw[model][ds]
        angles = [r["angle"] for r in rows]
        r1     = [r["r1"]    for r in rows]
        r5     = [r["r5"]    for r in rows]
        map5   = [r["map5"]  for r in rows]
        color  = MODEL_COLORS.get(model, "#888888")

        ax.plot(angles, r1,   color=color,    label="R@1",   **METRIC_STYLES["r1"])
        ax.plot(angles, r5,   color="#aaaaaa", label="R@5",   **METRIC_STYLES["r5"])
        ax.plot(angles, map5, color="#555555", label="mAP@5", **METRIC_STYLES["map5"])
        ax.fill_between(angles, [min(r1)] * len(angles), [max(r1)] * len(angles),
                        color=color, alpha=0.08)
        drop = max(r1) - min(r1)
        ax.set_title(f"{model}\nR@1 drop={drop:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Rotation angle (°)", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xlim(0, 180)
        ax.set_ylim(max(0, min(r1) - 0.05), min(1.05, max(r5) + 0.05))
        ax.set_xticks(range(0, 181, 60))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    out = OUT_DIR / f"exp1_per_model_{ds}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Figure 3: summary bar chart — mean R@1 / R@5 / mAP@5 per model ───────────
for ds in DATASETS:
    models_with_ds = [m for m in MODELS if ds in raw[m]]
    x     = np.arange(len(models_with_ds))
    width = 0.25

    means_r1   = [np.mean([r["r1"]   for r in raw[m][ds]]) for m in models_with_ds]
    means_r5   = [np.mean([r["r5"]   for r in raw[m][ds]]) for m in models_with_ds]
    means_map5 = [np.mean([r["map5"] for r in raw[m][ds]]) for m in models_with_ds]
    drops      = [max(r["r1"] for r in raw[m][ds]) - min(r["r1"] for r in raw[m][ds])
                  for m in models_with_ds]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(models_with_ds)), 5))
    bars_r1   = ax.bar(x - width, means_r1,   width, label="R@1",   color="#e63946", alpha=0.85)
    bars_r5   = ax.bar(x,         means_r5,   width, label="R@5",   color="#457b9d", alpha=0.85)
    bars_map5 = ax.bar(x + width, means_map5, width, label="mAP@5", color="#2a9d8f", alpha=0.85)

    for bar, drop in zip(bars_r1, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"Δ={drop:.4f}", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(models_with_ds, fontsize=10)
    ax.set_ylabel("Score (mean over all angles)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Mean Retrieval Scores — {ds}\nΔ = R@1 drop (lower = more rotation invariant)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    out = OUT_DIR / f"exp1_summary_bars_{ds}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Print summary table ───────────────────────────────────────────────────────
for ds in DATASETS:
    print(f"\n{'='*70}")
    print(f"Dataset: {ds}")
    print(f"{'Model':<15} {'Mean R@1':>9} {'Mean R@5':>9} {'Mean mAP@5':>11} {'R@1 Drop':>10}")
    print("-" * 70)
    for model in MODELS:
        if ds not in raw[model]:
            continue
        rows  = raw[model][ds]
        mr1   = np.mean([r["r1"]   for r in rows])
        mr5   = np.mean([r["r5"]   for r in rows])
        mmap5 = np.mean([r["map5"] for r in rows])
        drop  = max(r["r1"] for r in rows) - min(r["r1"] for r in rows)
        print(f"{model:<15} {mr1:>9.4f} {mr5:>9.4f} {mmap5:>11.4f} {drop:>10.4f}")
    print("=" * 70)
