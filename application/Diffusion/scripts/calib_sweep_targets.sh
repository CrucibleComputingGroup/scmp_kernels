#!/bin/bash
# Sequential calibration sweep at 5 target avg_stoc_len values.
# Reuses --metric cosine --teacher fp + --sc_fixed_level_prec on Blackwell.
set -euo pipefail

OUT_BASE=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256
mkdir -p "$OUT_BASE"

# (target_avg_sl, budget_ratio against ref=256)
declare -a TARGETS=(
  "128:0.5"
  "96:0.375"
  "64:0.25"
  "48:0.1875"
  "32:0.125"
)

for entry in "${TARGETS[@]}"; do
  TARGET=${entry%:*}
  BR=${entry#*:}
  OUT_JSON="$OUT_BASE/calib_fix_avg${TARGET}_l256_ref192.json"
  if [[ -f "$OUT_JSON" ]]; then
    echo "[skip] avg_sl=$TARGET already exists at $OUT_JSON"
    continue
  fi
  echo "[run]  avg_sl=$TARGET (b=$BR) start $(date +%H:%M:%S)"
  python -u scripts/calibrate_mp_thresholds.py \
    --mp_levels 256,192,128,96,64,48,32,16 \
    --budget_ratio "$BR" \
    --budget_ref_stoc_len 256 \
    --metric cosine --teacher fp \
    --sc_prec 8 --sc_fixed_level_prec --sc_enable \
    --wbits 8 --abits 8 --w_sym --a_sym \
    --image-size 256 --num-sampling-steps 50 \
    --num_calib_batches 1 --num_calib_timesteps 6 \
    --timestep_buckets 4 --layer_buckets 4 \
    --teacher_cfg_scale 0.0 \
    --calib_output_json "$OUT_JSON" \
    > "$OUT_BASE/calib_fix_avg${TARGET}.log" 2>&1
  cp "$(dirname "$OUT_JSON")/../Q-DiT/threshold_mp_calibration_summary.csv" \
     "$OUT_BASE/calib_fix_avg${TARGET}_summary.csv" 2>/dev/null || true
  cp /gpfs/accounts/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/Q-DiT/threshold_mp_calibration_summary.csv \
     "$OUT_BASE/calib_fix_avg${TARGET}_summary.csv"
  echo "[ok]   avg_sl=$TARGET end   $(date +%H:%M:%S) -> $OUT_JSON"
done

echo "DONE $(date)"
