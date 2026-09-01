"""Visualization for Exp 7 — Proper Training Comparison."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "outputs/exp7_proper_training.json"
OUT_DIR = ROOT / "plots/risa_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA) as f:
    results = json.load(f)

models  = list(results.keys())
angles  = [0, 90, 180]

# Colour: invariant models green, baselines red/grey
INVARIANT = {"RISA", "VNN", "DiPVNet"}
def color(m):
    if m == "RISA":    return "#e63946"
    if m == "VNN":     return "#2a9d8f"
    if m == "DiPVNet": return "#457b9d"
    return "#adb5bd"

r1 = {m: {r["angle"]: r["r1"] for r in results[m]} for m in models}

# ── Figure 1: grouped bar chart R@1 at 0° / 90° / 180° ──────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(models))
width = 0.25
alpha_vals = [1.0, 0.65, 0.35]
labels_ang = ["0° (canonical)", "90° (rotated)", "180° (rotated)"]

for i, (theta, alpha, lbl) in enumerate(zip(angles, alpha_vals, labels_ang)):
    vals = [r1[m].get(theta, 0) for m in models]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  color=[color(m) for m in models], alpha=alpha, label=lbl,
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, rotation=15, ha="right")
ax.set_ylabel("Recall@1", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title("Exp 7 — ProxyAnchor Training: R@1 at 0° / 90° / 180°\n"
             "Invariant models (RISA, VNN, DiPVNet) should show no drop across angles",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

# Annotate drop for each model
for i, m in enumerate(models):
    drop = r1[m].get(0, 0) - r1[m].get(180, 0)
    clr  = "#2a9d8f" if drop < 0.05 else "#e63946"
    ax.text(x[i], -0.08, f"Δ={drop:.3f}", ha="center", fontsize=8,
            color=clr, fontweight="bold")

ax.text(0.5, -0.13, "Δ = R@1(0°) − R@1(180°)   |   green = invariant (Δ<0.05)   |   red = not invariant",
        ha="center", transform=ax.transAxes, fontsize=8, color="#555")

plt.tight_layout()
out1 = OUT_DIR / "exp7_grouped_bars.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out1}")


# ── Figure 2: drop chart — how much each model degrades ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
drops = [r1[m].get(0, 0) - r1[m].get(180, 0) for m in models]
bars  = ax.barh(models, drops, color=[color(m) for m in models], alpha=0.85, edgecolor="white")

ax.axvline(0.05, color="black", lw=1.5, ls="--", alpha=0.5, label="Invariance threshold (Δ=0.05)")
for bar, d in zip(bars, drops):
    ax.text(d + 0.002, bar.get_y() + bar.get_height()/2,
            f"{d:.3f}", va="center", fontsize=9)

ax.set_xlabel("R@1 Drop  (0° → 180°)", fontsize=11)
ax.set_title("Exp 7 — Rotation Sensitivity per Model\n"
             "Lower = more invariant  |  dashed line = invariance threshold",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="x", alpha=0.3)
ax.set_xlim(0, max(drops) * 1.2)
plt.tight_layout()
out2 = OUT_DIR / "exp7_drop_chart.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out2}")


# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"{'Model':<18} {'R@1 (0°)':>9} {'R@1 (90°)':>10} {'R@1 (180°)':>11} {'Drop':>8}  {'Invariant?':>10}")
print("-"*72)
for m in models:
    r0   = r1[m].get(0,   float("nan"))
    r90  = r1[m].get(90,  float("nan"))
    r180 = r1[m].get(180, float("nan"))
    drop = r0 - r180
    inv  = "✓" if drop < 0.05 else "✗"
    print(f"{m:<18} {r0:>9.3f} {r90:>10.3f} {r180:>11.3f} {drop:>8.3f}  {inv:>10}")
print("="*72)
