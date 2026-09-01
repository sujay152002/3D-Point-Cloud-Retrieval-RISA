"""Plot all paper figures from saved experiment outputs.

Generates:
  Fig 1 — Recall@1 vs rotation angle (Exp 1)  [main result]
  Fig 2 — t-SNE semantic collapse (Exp 2)      [already saved by exp2]
  Fig 3 — Ablation bar chart (Exp 3)
  Fig 4 — Cross-dataset transfer curves (Exp 4)
  Fig 5 — Noise robustness heatmap (Exp 5)
  Fig 6 — μCKA vs worst-case Recall@1 (Exp 6) [already saved by exp6]

Usage:
    cd /users/grad/smenon/retrieval_paper
    python experiments/plot_all.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT     = Path(__file__).parent.parent
PLOTS    = ROOT / "plots"
OUTPUTS  = ROOT / "outputs"
PLOTS.mkdir(exist_ok=True)

MODEL_COLORS = {
    "PointNet++"      : "#e6194b",
    "PointMamba"      : "#3cb44b",
    "RS-CNN"          : "#4363d8",
    "PointTransformer": "#f58231",
    "RISA"            : "#000000",   # black — stands out as the proposed method
    "DGCNN"           : "#911eb4",
    "RINet"           : "#42d4f4",
}

MODEL_STYLES = {
    "RISA": {"linewidth": 2.5, "linestyle": "-",  "zorder": 10},
}
DEFAULT_STYLE = {"linewidth": 1.2, "linestyle": "--", "zorder": 5}


# ── Fig 1: Recall@1 vs rotation angle ────────────────────────────────────────

def plot_retrieval_curves(json_path: Path):
    if not json_path.exists():
        print(f"[skip] {json_path} not found"); return

    with open(json_path) as f:
        data = json.load(f)

    datasets = list(next(iter(data.values())).keys())
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), sharey=True)
    if n_ds == 1:
        axes = [axes]

    fig.suptitle("Recall@1 vs Rotation Angle", fontsize=13, fontweight="bold")

    for ax, ds_name in zip(axes, datasets):
        for model_name, ds_results in data.items():
            if ds_name not in ds_results:
                continue
            angle_res = ds_results[ds_name]
            angles = [r["angle"] for r in angle_res]
            r1     = [r["r1"]    for r in angle_res]
            color  = MODEL_COLORS.get(model_name, "#888888")
            style  = MODEL_STYLES.get(model_name, DEFAULT_STYLE)
            ax.plot(angles, r1, color=color, label=model_name, **style)

        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel("Rotation angle (°)")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Recall@1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(data),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    out = PLOTS / "fig1_retrieval_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Fig 3: Ablation bar chart ─────────────────────────────────────────────────

def plot_ablation(json_path: Path):
    if not json_path.exists():
        print(f"[skip] {json_path} not found"); return

    with open(json_path) as f:
        data = json.load(f)

    key_angles = [0, 45, 90, 180]
    conditions = list(data.keys())
    x          = np.arange(len(key_angles))
    width      = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, cond in enumerate(conditions):
        row  = {r["angle"]: r["r1"] for r in data[cond]}
        vals = [row.get(a, 0) for a in key_angles]
        ax.bar(x + i * width, vals, width, label=cond)

    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels([f"{a}°" for a in key_angles])
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: What Drives RISA's Retrieval Gain?", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "fig3_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Fig 4: Cross-dataset transfer curves ─────────────────────────────────────

def plot_cross_dataset(json_path: Path):
    if not json_path.exists():
        print(f"[skip] {json_path} not found"); return

    with open(json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, angle_res in data.items():
        angles = [r["angle"] for r in angle_res]
        r1     = [r["r1"]    for r in angle_res]
        color  = MODEL_COLORS.get(model_name, "#888888")
        style  = MODEL_STYLES.get(model_name, DEFAULT_STYLE)
        ax.plot(angles, r1, color=color, label=model_name, **style)

    ax.set_xlabel("Rotation angle (°)")
    ax.set_ylabel("Recall@1")
    ax.set_title("Cross-Dataset Transfer: MN40 → ScanObjectNN", fontweight="bold")
    ax.set_xlim(0, 180); ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "fig4_cross_dataset.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Fig 5: Noise robustness heatmap ──────────────────────────────────────────

def plot_noise_heatmap(json_path: Path):
    if not json_path.exists():
        print(f"[skip] {json_path} not found"); return

    with open(json_path) as f:
        data = json.load(f)

    models      = list(data.keys())
    noise_levels = sorted({r["sigma"] for r in next(iter(data.values()))})
    matrix       = np.array([
        [next(r["r1_mean"] for r in data[m] if r["sigma"] == s)
         for s in noise_levels]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f"σ={s:.2f}" for s in noise_levels])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Recall@1 under Rotation + Noise", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Recall@1")
    for i in range(len(models)):
        for j in range(len(noise_levels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    out = PLOTS / "fig5_noise_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating all paper figures…\n")
    plot_retrieval_curves(OUTPUTS / "exp1_retrieval.json")
    plot_ablation(        OUTPUTS / "exp3_ablation.json")
    plot_cross_dataset(   OUTPUTS / "exp4_cross_dataset.json")
    plot_noise_heatmap(   OUTPUTS / "exp5_noise_robustness.json")
    print("\nDone. Check plots/")
