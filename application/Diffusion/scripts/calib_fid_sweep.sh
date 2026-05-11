#!/bin/bash
# FID sweep: adaptive vs uniform at 5 budgets, with class-balanced sampling.
#
# - 5 target avg_sl values (128 / 96 / 64 / 48 / 32)
# - 2 modes per target: adaptive (calibration table) vs uniform (sc_cfg_uniform*)
# - Default 10000 samples (10 per class × 1000 ImageNet classes)
# - Default Owen mode: bitrev
#
# Single-GPU sequential runtime ≈ 14 hr per config × 10 configs = ~140 hr.
# Override with NUM_FID=2000 for a fast sanity sweep (~30 hr).
#
# Output: /scratch/.../fid_sweep_${OWEN_MODE}/{adaptive,uniform}_avg{SL}/samples/*.png

set -euo pipefail

# --- knobs ---
NUM_FID="${NUM_FID:-10000}"            # samples per config
BATCH="${BATCH:-40}"                    # batch size; lower if OOM
OWEN_MODE="${OWEN_MODE:-bitrev}"        # counter | bitrev | random
SEED="${SEED:-0}"                       # base seed
NUM_STEPS="${NUM_STEPS:-50}"            # diffusion sampling steps
CFG_SCALE="${CFG_SCALE:-4}"             # CFG scale

# --- paths ---
REPO=/gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm
CALIB_DIR=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256_${OWEN_MODE}
UNIFORM_CFG_DIR=$REPO/Q-DiT/results
OUT_BASE=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/fid_sweep_${OWEN_MODE}

mkdir -p "$OUT_BASE"
export SC_OWEN_MODE="$OWEN_MODE"
export PYTHONUNBUFFERED=1

LOG="$OUT_BASE/sweep.log"
echo "FID sweep started $(date)  NUM_FID=$NUM_FID  BATCH=$BATCH  OWEN_MODE=$OWEN_MODE" | tee "$LOG"

cd "$REPO/Q-DiT"

run_one() {
  local TAG="$1"        # e.g. "adaptive_avg64" or "uniform_avg64"
  local DIR="$OUT_BASE/$TAG"
  shift                  # remaining args are passed through
  if [[ -d "$DIR/samples" && $(ls "$DIR/samples"/*.png 2>/dev/null | wc -l) -ge "$NUM_FID" ]]; then
    echo "[skip] $TAG already has $NUM_FID samples in $DIR/samples" | tee -a "$LOG"
    return
  fi
  mkdir -p "$DIR"
  echo "[run]  $TAG  start $(date +%H:%M:%S)" | tee -a "$LOG"
  python -u scripts/quant_sc_main.py \
    --wbits 8 --abits 8 --w_sym --a_sym \
    --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
    --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
    --sc_prec 8 --sc_fixed_level_prec --sc_enable \
    --image-size 256 --num-sampling-steps "$NUM_STEPS" --cfg-scale "$CFG_SCALE" \
    --batch-size "$BATCH" --seed "$SEED" \
    --generate-fid-samples --num-fid-samples "$NUM_FID" \
    --results-dir "$DIR" \
    "$@" \
    > "$DIR/run.log" 2>&1
  # Surface the auto-numbered samples dir up one level
  GENERATED=$(find "$DIR" -type d -name samples | head -1)
  if [[ -n "$GENERATED" && "$GENERATED" != "$DIR/samples" ]]; then
    mv "$GENERATED" "$DIR/samples"
  fi
  COUNT=$(ls "$DIR/samples"/*.png 2>/dev/null | wc -l)
  echo "[ok]   $TAG  end   $(date +%H:%M:%S)  samples=$COUNT" | tee -a "$LOG"
}

for SL in 128 96 64 48 32; do
  CALIB_JSON="$CALIB_DIR/calib_fix_avg${SL}_l256_ref192.json"
  UNIFORM_JSON="$UNIFORM_CFG_DIR/sc_cfg_uniform${SL}_all.json"

  if [[ -f "$CALIB_JSON" ]]; then
    run_one "adaptive_avg${SL}" \
      --adaptive_mp \
      --adaptive_mp_table "$CALIB_JSON" \
      --mp_levels 256,192,128,96,64,48,32,16
  else
    echo "[skip] adaptive_avg${SL}: $CALIB_JSON missing" | tee -a "$LOG"
  fi

  if [[ -f "$UNIFORM_JSON" ]]; then
    run_one "uniform_avg${SL}" \
      --sc_config "$UNIFORM_JSON"
  else
    echo "[skip] uniform_avg${SL}: $UNIFORM_JSON missing" | tee -a "$LOG"
  fi
done

echo "Sample generation finished $(date)" | tee -a "$LOG"

# --- FID computation ---
# Requires `torch-fidelity` or `clean-fid` installed in the conda env.
# Reference: ImageNet validation set (256x256) — set IMAGENET_REF env var to its path.
#
# Example (uncomment and adjust IMAGENET_REF):
# IMAGENET_REF="${IMAGENET_REF:-/path/to/imagenet/val}"
# for D in "$OUT_BASE"/{adaptive,uniform}_avg*/samples; do
#   TAG=$(basename $(dirname "$D"))
#   echo "FID $TAG..."
#   python -m pytorch_fid "$IMAGENET_REF" "$D" --device cuda > "$OUT_BASE/${TAG}.fid"
# done
# echo "FID done $(date). Results:"
# grep -H FID "$OUT_BASE"/*.fid

echo "DONE $(date)"
