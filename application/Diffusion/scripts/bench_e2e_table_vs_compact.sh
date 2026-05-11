#!/bin/bash
# End-to-end benchmark: table-based vs compact SC kernel for full DiT inference.
#
# Usage:
#   cd Q-DiT && bash scripts/bench_e2e_table_vs_compact.sh
#   STEPS=50 BATCH=16 bash scripts/bench_e2e_table_vs_compact.sh
#
# Env vars:
#   STEPS  — diffusion sampling steps (default: 20)
#   BATCH  — batch size (default: 8)
#   SC_PREC — SC precision (default: 8)
#   QK_LW  — qklayerwise fraction (default: 1.0)
#   TW     — timewise fraction (default: 1.0)

set -euo pipefail

STEPS=${STEPS:-20}
BATCH=${BATCH:-8}
SC_PREC=${SC_PREC:-8}
QK_LW=${QK_LW:-1.0}
TW=${TW:-1.0}
RESULTS_DIR="../results_bench_e2e"

COMMON_ARGS=(
    --wbits 8 --abits 8
    --w_sym --a_sym
    --timewise "$TW"
    --qklayerwise "$QK_LW"
    --sc_prec "$SC_PREC"
    --sc_enable
    --image-size 256
    --num-sampling-steps "$STEPS"
    --batch-size "$BATCH"
    --cfg-scale 1.5
    --results-dir "$RESULTS_DIR"
)

echo "============================================================"
echo "E2E Benchmark: Table vs Compact SC Kernel"
echo "  sc_prec=$SC_PREC  steps=$STEPS  batch=$BATCH"
echo "  timewise=$TW  qklayerwise=$QK_LW"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

mkdir -p "$RESULTS_DIR"

# --- Table-based path ---
echo ""
echo "[1/2] Running TABLE-BASED path..."
unset SC_FORCE_COMPACT 2>/dev/null || true
export SC_FORCE_TABLE=1

START_TABLE=$(date +%s%N)
python scripts/quant_sc_main.py "${COMMON_ARGS[@]}" 2>&1 | tail -5
END_TABLE=$(date +%s%N)
TABLE_MS=$(( (END_TABLE - START_TABLE) / 1000000 ))
TABLE_S=$(echo "scale=1; $TABLE_MS / 1000" | bc)
echo "  Table path wall time: ${TABLE_S}s"

unset SC_FORCE_TABLE

# --- Compact path ---
echo ""
echo "[2/2] Running COMPACT (on-the-fly) path..."
unset SC_FORCE_TABLE 2>/dev/null || true
export SC_FORCE_COMPACT=1

START_COMPACT=$(date +%s%N)
python scripts/quant_sc_main.py "${COMMON_ARGS[@]}" 2>&1 | tail -5
END_COMPACT=$(date +%s%N)
COMPACT_MS=$(( (END_COMPACT - START_COMPACT) / 1000000 ))
COMPACT_S=$(echo "scale=1; $COMPACT_MS / 1000" | bc)
echo "  Compact path wall time: ${COMPACT_S}s"

unset SC_FORCE_COMPACT

# --- Summary ---
echo ""
echo "============================================================"
echo "E2E Summary:"
echo "  Table path:   ${TABLE_S}s"
echo "  Compact path: ${COMPACT_S}s"
if [ "$COMPACT_MS" -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $TABLE_MS / $COMPACT_MS" | bc)
    echo "  Speedup (compact/table): ${SPEEDUP}x  (>1 = compact faster)"
fi
echo "============================================================"
