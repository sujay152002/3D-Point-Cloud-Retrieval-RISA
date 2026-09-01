# Rotation-Robust 3D Shape Retrieval

**Paper:** *SE(3)-Invariant Encodings are Necessary and Sufficient for Rotation-Robust 3D Shape Retrieval*

## Overview

This repository contains all experiments for the paper. The core claim is:

> Existing 3D encoders fail at shape retrieval under arbitrary rotation because their
> representations are pose-dependent. We show this failure is directly predicted by
> μCKA, provide the first rotation-stratified retrieval benchmark, and demonstrate
> that RISA's architectural SE(3)-invariance closes the gap completely.

## Experiments

| # | Script | What it shows | Key figure |
|---|--------|---------------|------------|
| 1 | `exp1_retrieval_eval.py`        | Recall@1 vs rotation angle for all models × datasets | Fig 1 — main result |
| 2 | `exp2_semantic_collapse.py`     | t-SNE: rotation scatters baseline encodings across class boundaries | Fig 2 |
| 3 | `exp3_ablation.py`              | Architectural vs learned vs TTA invariance | Fig 3 |
| 4 | `exp4_cross_dataset_transfer.py`| MN40 (CAD) → ScanObjectNN (real scans) retrieval | Fig 4 |
| 5 | `exp5_noise_robustness.py`      | Retrieval under rotation + Gaussian noise | Fig 5 |
| 6 | `exp6_cka_retrieval_bound.py`   | μCKA vs worst-case Recall@1 correlation | Fig 6 |

## Quick Start

```bash
# Install dependencies (same as 3D_Encoding_Gap)
pip install -r requirements.txt

# Run all experiments
bash run_all.sh

# Or run individually
python experiments/exp1_retrieval_eval.py --datasets modelnet40 --epochs 20
python experiments/exp2_semantic_collapse.py --dataset modelnet40
python experiments/exp3_ablation.py --dataset modelnet40
python experiments/exp4_cross_dataset_transfer.py
python experiments/exp5_noise_robustness.py --dataset modelnet40
python experiments/exp6_cka_retrieval_bound.py --dataset modelnet40

# Generate all figures from saved results
python experiments/plot_all.py
```

## Paper Story

```
§1 Introduction
   — Rotation sensitivity is an unsolved problem in 3D retrieval

§2 Background
   — μCKA, Vg metrics (from 3D_Encoding_Gap)
   — RISA architecture

§3 The Encoding Gap Causes Retrieval Failure  [Exp 1 + Exp 2]
   — Rotation-stratified benchmark
   — Semantic collapse visualization

§4 Why Architectural Invariance Beats Learned Invariance  [Exp 3]
   — Ablation: RISA vs SO3-aug vs TTA vs RISA-XYZ

§5 Generalisation to Real-World Scans  [Exp 4]
   — Cross-dataset transfer MN40 → ScanObjectNN

§6 Robustness to Noise  [Exp 5]
   — Joint rotation + noise benchmark

§7 μCKA as a Retrieval Bound  [Exp 6]
   — Empirical validation of the theoretical connection
```

## Project Structure

```
retrieval_paper/
├── experiments/
│   ├── exp1_retrieval_eval.py        # Core retrieval benchmark
│   ├── exp2_semantic_collapse.py     # t-SNE visualization
│   ├── exp3_ablation.py              # Ablation study
│   ├── exp4_cross_dataset_transfer.py# Domain transfer
│   ├── exp5_noise_robustness.py      # Noise robustness
│   ├── exp6_cka_retrieval_bound.py   # μCKA → Recall@1 bound
│   └── plot_all.py                   # Generate all figures
├── core/                             # Copied from 3D_Encoding_Gap
├── models/                           # Copied from 3D_Encoding_Gap
├── data/                             # Symlinked/copied from 3D_Encoding_Gap
├── outputs/                          # JSON results
├── plots/                            # Generated figures
└── run_all.sh                        # Run everything
```
