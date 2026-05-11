#!/bin/bash
# Same as owen_e2e_sweep.sh but with --seed $1 and output dir suffix _seed${1}.
# Usage: bash owen_e2e_sweep_seed.sh <seed_int>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
LEVELS=(16 32 48 64 96 128)
MODES=(counter random bitrev off)

OUT_BASE=/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm/owen_e2e_seed${SEED}
mkdir -p "$OUT_BASE"

SUMMARY="$OUT_BASE/sweep_summary.txt"
echo "owen_e2e seed=${SEED} sweep started $(date)" > "$SUMMARY"

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
      --seed "${SEED}" \
      --results-dir "$DIR" \
      > "$DIR/run.log" 2>&1
    GENERATED=$(find "$DIR" -name sample_sc.png -print -quit)
    if [[ -n "$GENERATED" ]]; then
      cp "$GENERATED" "$DIR/sample_sc.png"
      rm -rf "$DIR"/000-*
      echo "[ok]   $MODE / sl=$L  end   $(date +%H:%M:%S)  -> $DIR/sample_sc.png" | tee -a "$SUMMARY"
    else
      echo "[FAIL] $MODE / sl=$L  no sample_sc.png in $DIR" | tee -a "$SUMMARY"
    fi
  done
done

echo "owen_e2e seed=${SEED} sweep finished $(date)" >> "$SUMMARY"
