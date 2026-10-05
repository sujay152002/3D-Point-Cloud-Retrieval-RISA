#!/bin/bash
# Waits for exp3 to finish, then runs exp2 → exp9 sequentially.

EXP3_PID=2961303
ROOT="/home/grad/smenon/retrieval_paper"
LOG_DIR="$ROOT/outputs"

echo "Waiting for exp3 (PID $EXP3_PID) to finish..."
while kill -0 $EXP3_PID 2>/dev/null; do
    sleep 60
done
echo "exp3 finished. Starting queue..."

cd "$ROOT"

echo "=== Running exp2 ===" 
python3 experiments/exp2_semantic_collapse.py \
    --models PointNet++ DGCNN DiPVNet RINet RISA \
    --checkpoint_dir outputs/checkpoints \
    --dataset shapenet \
    > "$LOG_DIR/exp2_run_$(date +%Y%m%d_%H%M%S).log" 2>&1
echo "exp2 done."

echo "=== Running exp9 ==="
python3 experiments/exp9_risa_sift.py \
    --checkpoint_dir outputs/checkpoints \
    > "$LOG_DIR/exp9_run_$(date +%Y%m%d_%H%M%S).log" 2>&1
echo "exp9 done."

echo "All experiments finished."
