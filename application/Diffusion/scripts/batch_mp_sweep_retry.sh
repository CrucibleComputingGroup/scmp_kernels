#!/bin/bash
# Retry script for 29 failed mixed-precision configurations
# Only runs the configs that didn't produce sample_sc.png

set -e
cd "$(dirname "$0")/.."

NUM_GPUS=8
MAX_PARALLEL=$NUM_GPUS

BASE_CMD="python scripts/quant_sc_main.py \
  --wbits 8 --abits 8 --w_sym --a_sym \
  --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
  --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
  --sc_prec 8 --image-size 256 --num-sampling-steps 100 \
  --cfg-scale 4 --batch-size 16 --sc_enable \
  --adaptive_mp --mp_levels 256,128,64,32,16 \
  --mp_alpha_av 0.3 --mp_beta_av 0.1 \
  --mp_alpha_qk 0.3 --mp_beta_qk 0.1"

# 29 failed configs: proj input_proj fc1 fc2
CONFIGS=(
  "0.3 0.1  0.5 0.25  0.5 0.25  0.3 0.1"
  "0.3 0.1  0.5 0.25  0.7 0.4   0.3 0.1"
  "0.3 0.1  0.7 0.4   0.3 0.1   0.3 0.1"
  "0.3 0.1  0.7 0.4   0.5 0.25  0.3 0.1"
  "0.3 0.1  0.7 0.4   0.5 0.25  0.5 0.25"
  "0.3 0.1  0.7 0.4   0.7 0.4   0.5 0.25"
  "0.5 0.25 0.3 0.1   0.3 0.1   0.5 0.25"
  "0.5 0.25 0.3 0.1   0.5 0.25  0.5 0.25"
  "0.5 0.25 0.5 0.25  0.3 0.1   0.3 0.1"
  "0.5 0.25 0.5 0.25  0.3 0.1   0.5 0.25"
  "0.5 0.25 0.5 0.25  0.5 0.25  0.3 0.1"
  "0.5 0.25 0.5 0.25  0.7 0.4   0.3 0.1"
  "0.5 0.25 0.5 0.25  0.7 0.4   0.5 0.25"
  "0.5 0.25 0.7 0.4   0.3 0.1   0.3 0.1"
  "0.5 0.25 0.7 0.4   0.3 0.1   0.5 0.25"
  "0.5 0.25 0.7 0.4   0.5 0.25  0.3 0.1"
  "0.5 0.25 0.7 0.4   0.7 0.4   0.3 0.1"
  "0.5 0.25 0.7 0.4   0.7 0.4   0.5 0.25"
  "0.7 0.4  0.3 0.1   0.3 0.1   0.3 0.1"
  "0.7 0.4  0.3 0.1   0.5 0.25  0.3 0.1"
  "0.7 0.4  0.3 0.1   0.7 0.4   0.3 0.1"
  "0.7 0.4  0.3 0.1   0.7 0.4   0.5 0.25"
  "0.7 0.4  0.5 0.25  0.3 0.1   0.5 0.25"
  "0.7 0.4  0.5 0.25  0.5 0.25  0.5 0.25"
  "0.7 0.4  0.5 0.25  0.7 0.4   0.5 0.25"
  "0.7 0.4  0.7 0.4   0.3 0.1   0.5 0.25"
  "0.7 0.4  0.7 0.4   0.5 0.25  0.3 0.1"
  "0.7 0.4  0.7 0.4   0.7 0.4   0.3 0.1"
  "0.7 0.4  0.7 0.4   0.7 0.4   0.5 0.25"
)

# Track PID -> job name mapping
declare -A PID_TO_NAME
RUNNING_PIDS=()
gpu_idx=0
finished=0

LOG_DIR="logs_mp_sweep"
mkdir -p "$LOG_DIR"

TOTAL=${#CONFIGS[@]}
echo "=== Retry Sweep: $TOTAL failed jobs, $NUM_GPUS GPUs ==="

for ((i=0; i<TOTAL; i++)); do
  read -r proj_a proj_b inproj_a inproj_b fc1_a fc1_b fc2_a fc2_b <<< "${CONFIGS[$i]}"

  tag="proj_a${proj_a}_b${proj_b}_inproj_a${inproj_a}_b${inproj_b}_fc1_a${fc1_a}_b${fc1_b}_fc2_a${fc2_a}_b${fc2_b}"

  cmd="$BASE_CMD \
    --mp_alpha_proj $proj_a --mp_beta_proj $proj_b \
    --mp_alpha_input_proj $inproj_a --mp_beta_input_proj $inproj_b \
    --mp_alpha_mlp_fc1 $fc1_a --mp_beta_mlp_fc1 $fc1_b \
    --mp_alpha_mlp_fc2 $fc2_a --mp_beta_mlp_fc2 $fc2_b"

  # If we've reached max parallel, wait for one to finish
  while [ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]; do
    wait -n 2>/dev/null || true
    NEW_PIDS=()
    for pid in "${RUNNING_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      else
        finished=$((finished + 1))
        echo "[FINISHED $finished/$TOTAL] ${PID_TO_NAME[$pid]}"
        unset PID_TO_NAME[$pid]
      fi
    done
    RUNNING_PIDS=("${NEW_PIDS[@]}")
  done

  gpu=$((gpu_idx % NUM_GPUS))
  gpu_idx=$((gpu_idx + 1))

  echo "[LAUNCH $((i+1))/$TOTAL] GPU $gpu: $tag"

  eval "CUDA_VISIBLE_DEVICES=$gpu $cmd" \
    > "${LOG_DIR}/${tag}.retry.log" 2>&1 &

  pid=$!
  RUNNING_PIDS+=($pid)
  PID_TO_NAME[$pid]="$tag"

  sleep 10
done

# Wait for all remaining jobs
while [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
  wait -n 2>/dev/null || true
  NEW_PIDS=()
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      NEW_PIDS+=("$pid")
    else
      finished=$((finished + 1))
      echo "[FINISHED $finished/$TOTAL] ${PID_TO_NAME[$pid]}"
      unset PID_TO_NAME[$pid]
    fi
  done
  RUNNING_PIDS=("${NEW_PIDS[@]}")
done

echo "=== All $TOTAL retry jobs completed ==="
