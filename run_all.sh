#!/usr/bin/env bash
# run_all.sh — Run all paper experiments in sequence
# Usage: bash run_all.sh [--epochs 20] [--datasets modelnet40 shapenet]
#
# Experiments:
#   Exp 1 — Rotation-stratified retrieval benchmark   (~2h per model)
#   Exp 2 — Semantic collapse t-SNE visualization     (~30min)
#   Exp 3 — Ablation: what drives RISA's gain         (~1h)
#   Exp 4 — Cross-dataset transfer (MN40 → ScanObjNN) (~1h)
#   Exp 5 — Noise robustness                          (~1h)
#   Exp 6 — μCKA vs worst-case Recall@1 bound         (~1h)

set -e
cd "$(dirname "$0")"

EPOCHS=20
DATASETS="modelnet40 shapenet scanobjectnn synthetic"

echo "============================================================"
echo " Exp 1: Rotation-Stratified Retrieval"
echo "============================================================"
python experiments/exp1_retrieval_eval.py \
    --epochs $EPOCHS \
    --datasets $DATASETS \
    --n_angles 19 \
    --n_random 3

echo "============================================================"
echo " Exp 2: Semantic Collapse Visualization"
echo "============================================================"
python experiments/exp2_semantic_collapse.py \
    --model_a RISA \
    --model_b "PointNet++" \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --n_classes 8 \
    --n_shapes 5 \
    --n_rotations 32

echo "============================================================"
echo " Exp 3: Ablation"
echo "============================================================"
python experiments/exp3_ablation.py \
    --dataset modelnet40 \
    --epochs $EPOCHS

echo "============================================================"
echo " Exp 4: Cross-Dataset Transfer"
echo "============================================================"
python experiments/exp4_cross_dataset_transfer.py \
    --epochs $EPOCHS

echo "============================================================"
echo " Exp 5: Noise Robustness"
echo "============================================================"
python experiments/exp5_noise_robustness.py \
    --dataset modelnet40 \
    --epochs $EPOCHS

echo "============================================================"
echo " Exp 6: μCKA vs Recall@1 Bound"
echo "============================================================"
python experiments/exp6_cka_retrieval_bound.py \
    --dataset modelnet40 \
    --epochs $EPOCHS

echo "============================================================"
echo " Plotting all figures"
echo "============================================================"
python experiments/plot_all.py

echo ""
echo "All experiments complete. Results in outputs/, figures in plots/"
