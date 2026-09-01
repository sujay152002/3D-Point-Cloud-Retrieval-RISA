#!/usr/bin/env bash
# run_risa_only.sh — Run all experiments with RISA only
# Usage: bash run_risa_only.sh [--epochs 30]
#
# Outputs land in outputs/risa_only/ and plots/risa_only/

set -e
cd "$(dirname "$0")"

EPOCHS=${1:-30}
LOG=outputs/risa_only_run.log
mkdir -p outputs/risa_only plots/risa_only

echo "RISA-only experiment run  (epochs=$EPOCHS)" | tee $LOG
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# ── Exp 1: Rotation-stratified retrieval on all 4 datasets ───────────────────
echo "============================================================" | tee -a $LOG
echo " Exp 1: Rotation-Stratified Retrieval (RISA)" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 experiments/exp1_retrieval_eval.py \
    --models RISA \
    --datasets synthetic modelnet40 shapenet scanobjectnn \
    --epochs $EPOCHS \
    --n_angles 19 \
    --n_random 5 \
    --out outputs/risa_only/exp1_retrieval.json \
    2>&1 | tee -a $LOG

# ── Exp 2: Semantic collapse t-SNE (RISA vs DGCNN) ───────────────────────────
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo " Exp 2: Semantic Collapse t-SNE" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 experiments/exp2_semantic_collapse.py \
    --model_a RISA \
    --model_b DGCNN \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --n_classes 8 \
    --n_shapes 5 \
    --n_rotations 32 \
    2>&1 | tee -a $LOG

# ── Exp 3: Ablation (RISA vs DGCNN+SO3 vs DGCNN+TTA vs RISA-XYZ) ────────────
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo " Exp 3: Ablation" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 experiments/exp3_ablation.py \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --out outputs/risa_only/exp3_ablation.json \
    2>&1 | tee -a $LOG

# ── Exp 4: Cross-dataset transfer MN40 → ScanObjectNN ────────────────────────
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo " Exp 4: Cross-Dataset Transfer" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 experiments/exp4_cross_dataset_transfer.py \
    --models RISA \
    --epochs $EPOCHS \
    --out outputs/risa_only/exp4_cross_dataset.json \
    2>&1 | tee -a $LOG

# ── Exp 5: Noise robustness ───────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo " Exp 5: Noise Robustness" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 experiments/exp5_noise_robustness.py \
    --models RISA \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --out outputs/risa_only/exp5_noise.json \
    2>&1 | tee -a $LOG

# ── Exp 6: μCKA vs Recall@1 bound ────────────────────────────────────────────
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo " Exp 6: μCKA vs Recall@1 Bound" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 experiments/exp6_cka_retrieval_bound.py \
    --models RISA \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --out outputs/risa_only/exp6_cka_bound.json \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo " All RISA experiments complete." | tee -a $LOG
echo " Results : outputs/risa_only/" | tee -a $LOG
echo " Plots   : plots/" | tee -a $LOG
echo " Log     : $LOG" | tee -a $LOG
echo " Finished: $(date)" | tee -a $LOG
