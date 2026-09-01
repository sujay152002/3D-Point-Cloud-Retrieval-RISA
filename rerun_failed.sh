#!/usr/bin/env bash
# rerun_failed.sh — re-run exp2 (tsne fix) and exp3 (full, condition D OOM fix)
# Run this AFTER the current background job finishes (exp4/5/6)

set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
EPOCHS=30
LOG=outputs/risa_only_run2.log

echo "" | tee -a $LOG
echo "=== RE-RUN: Exp 2 (TSNE fix) ===" | tee -a $LOG
stdbuf -oL python3 experiments/exp2_semantic_collapse.py \
    --model_a RISA --model_b DGCNN \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --n_classes 8 --n_shapes 5 --n_rotations 32 \
    2>&1 | stdbuf -oL tee -a $LOG

echo "" | tee -a $LOG
echo "=== RE-RUN: Exp 3 (condition D OOM fix) ===" | tee -a $LOG
stdbuf -oL python3 experiments/exp3_ablation.py \
    --dataset modelnet40 \
    --epochs $EPOCHS \
    --out outputs/risa_only/exp3_ablation.json \
    2>&1 | stdbuf -oL tee -a $LOG

echo "RE-RUNS DONE $(date)" | tee -a $LOG
