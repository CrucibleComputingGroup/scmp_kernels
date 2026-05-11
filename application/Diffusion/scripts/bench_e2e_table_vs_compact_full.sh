#!/bin/bash
# Full end-to-end benchmark: table-based vs compact SC kernel.
# Runs the user's exact command twice with different force flags.
#
# Usage:
#   cd Q-DiT && bash scripts/bench_e2e_table_vs_compact_full.sh
#
# Output: results_bench_e2e/e2e_benchmark_report.txt

set -euo pipefail

RESULTS_DIR="../results_bench_e2e"
REPORT="$RESULTS_DIR/e2e_benchmark_report.txt"
mkdir -p "$RESULTS_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

COMMON_ARGS=(
    --wbits 8 --abits 8
    --w_sym --a_sym
    --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0
    --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0
    --sc_prec 8 --sc_enable
    --adaptive_mp --mp_levels 256,128,64,32,16
    --mp_alpha 0.3 --mp_beta 0.1
    --image-size 256 --num-sampling-steps 100 --batch-size 16
    --results-dir "$RESULTS_DIR"
)

cat << EOF | tee "$REPORT"
============================================================
E2E Benchmark: Table vs Compact SC Kernel
============================================================
Date: $(date)
GPU: $GPU_NAME ($GPU_MEM)
sc_prec=8  steps=100  batch=16  image=256px
All SC operators active (qk/av/proj/mlp/input_proj = 1.0)
Adaptive MP: levels=256,128,64,32,16 alpha=0.3 beta=0.1
============================================================

EOF

# --- Run 1: Table-based path (force table for attention, where it normally applies) ---
echo "[1/2] Running TABLE-BASED path..." | tee -a "$REPORT"
echo "Start time: $(date)" | tee -a "$REPORT"

unset SC_FORCE_COMPACT 2>/dev/null || true
export SC_FORCE_TABLE=1

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$RESULTS_DIR/gpu_mem_before_table.txt"

START_TABLE=$(date +%s%N)
python scripts/quant_sc_main.py "${COMMON_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/log_table.txt"
END_TABLE=$(date +%s%N)

TABLE_MS=$(( (END_TABLE - START_TABLE) / 1000000 ))
TABLE_S=$(echo "scale=2; $TABLE_MS / 1000" | bc)
echo "  Table path wall time: ${TABLE_S}s" | tee -a "$REPORT"
echo "  End time: $(date)" | tee -a "$REPORT"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$RESULTS_DIR/gpu_mem_after_table.txt"
unset SC_FORCE_TABLE

# Clear GPU cache between runs
python -c "import torch; torch.cuda.empty_cache()"

echo "" | tee -a "$REPORT"

# --- Run 2: Compact path (force compact for everything) ---
echo "[2/2] Running COMPACT (on-the-fly) path..." | tee -a "$REPORT"
echo "Start time: $(date)" | tee -a "$REPORT"

unset SC_FORCE_TABLE 2>/dev/null || true
export SC_FORCE_COMPACT=1

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$RESULTS_DIR/gpu_mem_before_compact.txt"

START_COMPACT=$(date +%s%N)
python scripts/quant_sc_main.py "${COMMON_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/log_compact.txt"
END_COMPACT=$(date +%s%N)

COMPACT_MS=$(( (END_COMPACT - START_COMPACT) / 1000000 ))
COMPACT_S=$(echo "scale=2; $COMPACT_MS / 1000" | bc)
echo "  Compact path wall time: ${COMPACT_S}s" | tee -a "$REPORT"
echo "  End time: $(date)" | tee -a "$REPORT"

nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$RESULTS_DIR/gpu_mem_after_compact.txt"
unset SC_FORCE_COMPACT

echo "" | tee -a "$REPORT"

# --- Summary ---
cat << EOF | tee -a "$REPORT"
============================================================
E2E Summary:
  Table path:   ${TABLE_S}s
  Compact path: ${COMPACT_S}s
EOF

if [ "$COMPACT_MS" -gt 0 ]; then
    SPEEDUP=$(echo "scale=3; $TABLE_MS / $COMPACT_MS" | bc)
    echo "  Ratio (table/compact): ${SPEEDUP}x  (<1 = table faster, >1 = compact faster)" | tee -a "$REPORT"
fi

cat << EOF | tee -a "$REPORT"
============================================================

GPU Memory (before/after each run):
  Table:   $(cat "$RESULTS_DIR/gpu_mem_before_table.txt") -> $(cat "$RESULTS_DIR/gpu_mem_after_table.txt")
  Compact: $(cat "$RESULTS_DIR/gpu_mem_before_compact.txt") -> $(cat "$RESULTS_DIR/gpu_mem_after_compact.txt")
============================================================
EOF

echo ""
echo "Full logs saved to $RESULTS_DIR/log_table.txt and $RESULTS_DIR/log_compact.txt"
echo "Report saved to $REPORT"
