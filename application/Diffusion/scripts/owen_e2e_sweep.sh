#!/bin/bash
# E2E DiT sweep: SC_OWEN_MODE ∈ {counter, random} × stoc_len ∈ {16,32,48,64,96,128}
# Run from Q-DiT/ on a node with 1 GPU. Each run: 50 steps, 8 images.
set -euo pipefail

LEVELS=(16 32 48 64 96 128)
MODES=(counter random bitrev off)

OUT_BASE=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/owen_e2e
mkdir -p "$OUT_BASE"

SUMMARY="$OUT_BASE/sweep_summary.txt"
echo "owen_e2e sweep started $(date)" > "$SUMMARY"

for MODE in "${MODES[@]}"; do
  for L in "${LEVELS[@]}"; do
    DIR="$OUT_BASE/${MODE}_uniform${L}"
    if [[ -f "$DIR/sample_sc.png" ]] || ls "$DIR"/*/sample_sc.png > /dev/null 2>&1; then
      echo "[skip] $MODE / sl=$L already has sample_sc.png" | tee -a "$SUMMARY"
      continue
    fi
    mkdir -p "$DIR"
    echo "[run]  $MODE / sl=$L  start $(date +%H:%M:%S)" | tee -a "$SUMMARY"
    SC_OWEN_MODE=$MODE python -u scripts/quant_sc_main.py \
      --wbits 8 --abits 8 --w_sym --a_sym \
      --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
      --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
      --sc_prec 8 --sc_fixed_level_prec --sc_enable \
      --sc_config "results/sc_cfg_uniform${L}_all.json" \
      --image-size 256 --num-sampling-steps 50 --cfg-scale 4 --batch-size 8 \
      --results-dir "$DIR" \
      > "$DIR/run.log" 2>&1
    # Find the auto-numbered subdir and surface the sample_sc.png
    GENERATED=$(find "$DIR" -name sample_sc.png -print -quit)
    if [[ -n "$GENERATED" ]]; then
      cp "$GENERATED" "$DIR/sample_sc.png"
      echo "[ok]   $MODE / sl=$L  end   $(date +%H:%M:%S)  -> $DIR/sample_sc.png" | tee -a "$SUMMARY"
    else
      echo "[FAIL] $MODE / sl=$L  no sample_sc.png in $DIR" | tee -a "$SUMMARY"
    fi
  done
done

echo "owen_e2e sweep finished $(date)" >> "$SUMMARY"
