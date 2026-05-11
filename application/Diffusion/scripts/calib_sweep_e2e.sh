#!/bin/bash
# After calib_sweep_targets.sh finishes (5 JSONs in calib_sweep_ref256/),
# run quant_sc_main on each at the standard 8-image visual sample (--seed 0).
set -euo pipefail

CALIB_DIR=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256
E2E_DIR=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/calib_sweep_ref256_e2e
mkdir -p "$E2E_DIR"

# Wait for all 5 calibrations to finish
echo "[wait] until 5 calibration JSONs exist in $CALIB_DIR ..."
until [[ $(ls "$CALIB_DIR"/calib_fix_avg*.json 2>/dev/null | wc -l) -ge 5 ]]; do
  sleep 30
done
echo "[ok]   all 5 calibrations done $(date)"

for TARGET in 128 96 64 48 32; do
  CALIB_JSON="$CALIB_DIR/calib_fix_avg${TARGET}_l256_ref192.json"
  OUT_DIR="$E2E_DIR/avg${TARGET}"
  if [[ -f "$OUT_DIR/sample_sc.png" ]]; then
    echo "[skip] avg_sl=$TARGET already has sample_sc.png"
    continue
  fi
  mkdir -p "$OUT_DIR"
  echo "[run]  e2e avg_sl=$TARGET start $(date +%H:%M:%S)"
  python -u scripts/quant_sc_main.py \
    --wbits 8 --abits 8 --w_sym --a_sym \
    --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
    --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
    --sc_prec 8 --sc_fixed_level_prec --sc_enable \
    --adaptive_mp \
    --adaptive_mp_table "$CALIB_JSON" \
    --mp_levels 256,192,128,96,64,48,32,16 \
    --image-size 256 --num-sampling-steps 50 --cfg-scale 4 --batch-size 8 \
    --seed 0 \
    --results-dir "$OUT_DIR" \
    > "$OUT_DIR/run.log" 2>&1
  GENERATED=$(find "$OUT_DIR" -name sample_sc.png -print -quit)
  if [[ -n "$GENERATED" ]]; then
    cp "$GENERATED" "$OUT_DIR/sample_sc.png"
    rm -rf "$OUT_DIR"/000-*
    echo "[ok]   e2e avg_sl=$TARGET end   $(date +%H:%M:%S) -> $OUT_DIR/sample_sc.png"
  else
    echo "[FAIL] avg_sl=$TARGET no sample_sc.png in $OUT_DIR"
  fi
done

echo "DONE $(date)"
