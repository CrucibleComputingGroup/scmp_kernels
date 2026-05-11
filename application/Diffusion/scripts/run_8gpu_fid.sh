#!/bin/bash
# ============================================================
# Generate 5000 FID samples across 8 NVIDIA RTX 6000 GPUs
# Each GPU generates ceil(5000/8) = 625 images in parallel,
# then all images are merged into a single directory.
# ============================================================

set -euo pipefail

NUM_GPUS=7
TOTAL_SAMPLES=5000
SAMPLES_PER_GPU=$(( (TOTAL_SAMPLES + NUM_GPUS - 1) / NUM_GPUS ))  # 625

RESULTS_BASE="../results_fid_8gpu"
MERGED_DIR="${RESULTS_BASE}/merged_samples"
mkdir -p "${MERGED_DIR}"

# Common flags (excluding per-GPU overrides)
COMMON_FLAGS="\
  --wbits 8 --abits 8 --w_sym --a_sym \
  --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
  --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
  --sc_prec 8 --image-size 256 --num-sampling-steps 100 \
  --cfg-scale 4 --batch-size 40 --sc_enable \
  --adaptive_mp --mp_levels 256,224,192,160,128,96,64,32,16 \
  --mp_alpha_av 0.7 --mp_beta_av 0.4 \
  --mp_alpha_qk 0.7 --mp_beta_qk 0.4 \
  --mp_alpha_proj 0.7 --mp_beta_proj 0.4 \
  --mp_alpha_input_proj 0.5 --mp_beta_input_proj 0.25 \
  --mp_alpha_mlp_fc1 0.7 --mp_beta_mlp_fc1 0.4 \
  --mp_alpha_mlp_fc2 0.5 --mp_beta_mlp_fc2 0.25 \
  --range_mp --range_mp_levels 256,224,192,160,128,96,64,32,16 --range_mp_threshold 0.2 \
  --generate-fid-samples"

export PYTHONUNBUFFERED=1

echo "=== Launching ${NUM_GPUS} GPU processes, ${SAMPLES_PER_GPU} samples each ==="
echo "=== Total target: ${TOTAL_SAMPLES} samples ==="

PIDS=()

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_RESULTS="${RESULTS_BASE}/gpu_${GPU_ID}"

    echo "[GPU ${GPU_ID}] Starting: seed=${GPU_ID}, samples=${SAMPLES_PER_GPU}, results=${GPU_RESULTS}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u scripts/quant_sc_main.py \
        ${COMMON_FLAGS} \
        --num-fid-samples ${SAMPLES_PER_GPU} \
        --seed ${GPU_ID} \
        --results-dir "${GPU_RESULTS}" \
        > "${RESULTS_BASE}/gpu_${GPU_ID}.log" 2>&1 &

    PIDS+=($!)

    # Stagger launches to avoid CUDA device contention
    sleep 3
done

echo "=== All ${NUM_GPUS} processes launched. PIDs: ${PIDS[*]} ==="
echo "=== Waiting for completion... ==="

# Wait for all processes and track failures
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[GPU ${i}] Finished successfully."
    else
        echo "[GPU ${i}] FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done

if [ ${FAILED} -gt 0 ]; then
    echo "ERROR: ${FAILED} GPU process(es) failed. Check logs in ${RESULTS_BASE}/gpu_*.log"
    exit 1
fi

# Merge all per-GPU sample directories into one with unique filenames
echo "=== Merging samples into ${MERGED_DIR} ==="

GLOBAL_IDX=0
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    # Find the experiment subdirectory (numbered like 000-xxx)
    GPU_SAMPLE_DIR=$(find "${RESULTS_BASE}/gpu_${GPU_ID}" -type d -name "samples" | head -1)

    if [ -z "${GPU_SAMPLE_DIR}" ]; then
        echo "WARNING: No samples directory found for GPU ${GPU_ID}"
        continue
    fi

    COUNT=0
    for IMG in "${GPU_SAMPLE_DIR}"/*.png; do
        [ -f "${IMG}" ] || continue
        cp "${IMG}" "${MERGED_DIR}/$(printf '%06d' ${GLOBAL_IDX}).png"
        GLOBAL_IDX=$((GLOBAL_IDX + 1))
        COUNT=$((COUNT + 1))
    done
    echo "[GPU ${GPU_ID}] Merged ${COUNT} images."
done

echo "=== Done! Total merged samples: ${GLOBAL_IDX} ==="
echo "=== Samples directory: ${MERGED_DIR} ==="
