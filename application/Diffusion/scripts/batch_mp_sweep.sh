#!/bin/bash
# Batch sweep over mixed-precision alpha/beta configurations
# Uses GPUs 0-7, queuing jobs round-robin.
#
# Configurations (alpha, beta):
#   conservative: 0.3, 0.1
#   balanced:     0.5, 0.25
#   aggressive:   0.7, 0.4
#
# av:         conservative only
# qk:         conservative only
# proj:       conservative / balanced / aggressive
# input_proj: conservative / balanced / aggressive
# mlp_fc1:    conservative / balanced / aggressive
# mlp_fc2:    conservative / balanced
#
# Total: 1 x 1 x 3 x 3 x 3 x 2 = 54 jobs

set -e
cd "$(dirname "$0")/.."

NUM_GPUS=8
MAX_PARALLEL=$NUM_GPUS

# (alpha, beta) for each config level
CONS_A=0.3;  CONS_B=0.1
BAL_A=0.5;   BAL_B=0.25
AGG_A=0.7;   AGG_B=0.4

# Fixed: av and qk are always conservative
AV_A=$CONS_A;  AV_B=$CONS_B
QK_A=$CONS_A;  QK_B=$CONS_B

# Sweep ranges: "alpha,beta" pairs
PROJ_CFGS=("$CONS_A,$CONS_B" "$BAL_A,$BAL_B" "$AGG_A,$AGG_B")
INPROJ_CFGS=("$CONS_A,$CONS_B" "$BAL_A,$BAL_B" "$AGG_A,$AGG_B")
FC1_CFGS=("$CONS_A,$CONS_B" "$BAL_A,$BAL_B" "$AGG_A,$AGG_B")
FC2_CFGS=("$CONS_A,$CONS_B" "$BAL_A,$BAL_B")

# Base command (shared arguments)
BASE_CMD="python scripts/quant_sc_main.py \
  --wbits 8 --abits 8 --w_sym --a_sym \
  --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
  --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
  --sc_prec 8 --image-size 256 --num-sampling-steps 100 \
  --cfg-scale 4 --batch-size 16 --sc_enable \
  --adaptive_mp --mp_levels 256,128,64,32,16"

# Collect all jobs
JOBS=()
JOB_NAMES=()

for proj_ab in "${PROJ_CFGS[@]}"; do
  IFS=',' read -r proj_a proj_b <<< "$proj_ab"
  for inproj_ab in "${INPROJ_CFGS[@]}"; do
    IFS=',' read -r inproj_a inproj_b <<< "$inproj_ab"
    for fc1_ab in "${FC1_CFGS[@]}"; do
      IFS=',' read -r fc1_a fc1_b <<< "$fc1_ab"
      for fc2_ab in "${FC2_CFGS[@]}"; do
        IFS=',' read -r fc2_a fc2_b <<< "$fc2_ab"

        tag="proj_a${proj_a}_b${proj_b}_inproj_a${inproj_a}_b${inproj_b}_fc1_a${fc1_a}_b${fc1_b}_fc2_a${fc2_a}_b${fc2_b}"

        cmd="$BASE_CMD \
          --mp_alpha_av $AV_A --mp_beta_av $AV_B \
          --mp_alpha_qk $QK_A --mp_beta_qk $QK_B \
          --mp_alpha_proj $proj_a --mp_beta_proj $proj_b \
          --mp_alpha_input_proj $inproj_a --mp_beta_input_proj $inproj_b \
          --mp_alpha_mlp_fc1 $fc1_a --mp_beta_mlp_fc1 $fc1_b \
          --mp_alpha_mlp_fc2 $fc2_a --mp_beta_mlp_fc2 $fc2_b"

        JOBS+=("$cmd")
        JOB_NAMES+=("$tag")
      done
    done
  done
done

TOTAL=${#JOBS[@]}
echo "=== Mixed-Precision Sweep: $TOTAL jobs, $NUM_GPUS GPUs ==="

# Track PID -> job name mapping
declare -A PID_TO_NAME
declare -A PID_TO_IDX
RUNNING_PIDS=()
gpu_idx=0
finished=0

LOG_DIR="logs_mp_sweep"
mkdir -p "$LOG_DIR"

for ((i=0; i<TOTAL; i++)); do
  # If we've reached max parallel, wait for one to finish
  while [ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]; do
    wait -n 2>/dev/null || true
    # Clean up finished PIDs and report
    NEW_PIDS=()
    for pid in "${RUNNING_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      else
        finished=$((finished + 1))
        echo "[FINISHED $finished/$TOTAL] ${PID_TO_NAME[$pid]}"
        unset PID_TO_NAME[$pid]
        unset PID_TO_IDX[$pid]
      fi
    done
    RUNNING_PIDS=("${NEW_PIDS[@]}")
  done

  gpu=$((gpu_idx % NUM_GPUS))
  gpu_idx=$((gpu_idx + 1))

  echo "[LAUNCH $((i+1))/$TOTAL] GPU $gpu: ${JOB_NAMES[$i]}"

  eval "CUDA_VISIBLE_DEVICES=$gpu ${JOBS[$i]}" \
    > "${LOG_DIR}/${JOB_NAMES[$i]}.log" 2>&1 &

  pid=$!
  RUNNING_PIDS+=($pid)
  PID_TO_NAME[$pid]="${JOB_NAMES[$i]}"
  PID_TO_IDX[$pid]=$i

  # Wait 10s between launches so each job creates its output directory
  # before the next one counts existing directories
  sleep 10
done

# Wait for all remaining jobs and report
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
      unset PID_TO_IDX[$pid]
    fi
  done
  RUNNING_PIDS=("${NEW_PIDS[@]}")
done

echo "=== All $TOTAL jobs completed ==="
