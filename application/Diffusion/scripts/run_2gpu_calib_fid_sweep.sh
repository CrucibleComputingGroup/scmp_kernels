#!/bin/bash
# ============================================================
# 2-GPU FID-10k sweep: adaptive (calibration table) vs uniform
# at avg_sl ∈ {32, 48, 64, 96, 128}, with bitrev Owen mode.
#
# 10 configs × 10000 samples each, fan-out across 2 GPUs.
# Each GPU gets ceil(NUM_FID/2)=5000 samples with a unique --seed.
# Per-config samples are merged into a single dir, then FID is
# computed via pytorch-fid against an ImageNet val pool.
#
# Usage (run from gl1807 or any 2-GPU box):
#   ssh gl1807
#   conda activate qdit
#   cd /gpfs/.../scmp_llm/Q-DiT
#   bash scripts/run_2gpu_calib_fid_sweep.sh
#
# Override knobs (env vars):
#   NUM_FID=10000                 # samples per config
#   BATCH=32                      # per-GPU batch (lower than 8-GPU script: 2 GPUs share host RAM, OOM at 40)
#   OWEN_MODE=bitrev              # counter | bitrev | random
#   IMAGENET_REF=/path/to/val     # ImageNet val pool for FID; if unset, FID is skipped
#   CALIB_DIR=...                 # default: /scratch/.../calib_sweep_ref256_${OWEN_MODE}
#   OUT_BASE=...                  # default: /scratch/.../fid_sweep_${OWEN_MODE}
#
# Estimated runtime on 2× RTX-PRO-6000 Blackwell:
#   adaptive_mp configs: ~7 hr per config × 5 = 35 hr
#   uniform configs:     ~3 hr per config × 5 = 15 hr
#   Total:               ~50 hr (~2 days) for full FID-10k sweep.
#   Cut NUM_FID to 5000 → ~25 hr.  NUM_FID to 2000 → ~10 hr.
# ============================================================

set -euo pipefail

# --- knobs ---
NUM_GPUS=2
NUM_FID="${NUM_FID:-10000}"
BATCH="${BATCH:-32}"
OWEN_MODE="${OWEN_MODE:-bitrev}"
NUM_STEPS="${NUM_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-4}"
# --- paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/nbleier_owned_root/nbleier_owned1/zhkangqi/scmp_llm}"
CALIB_DIR="${CALIB_DIR:-${SCRATCH_ROOT}/calib_sweep_ref256_${OWEN_MODE}}"
UNIFORM_CFG_DIR="${REPO_ROOT}/results"
OUT_BASE="${OUT_BASE:-${SCRATCH_ROOT}/fid_sweep_${OWEN_MODE}}"

mkdir -p "${OUT_BASE}"
SWEEP_LOG="${OUT_BASE}/sweep.log"

NUM_CLASSES="${NUM_CLASSES:-1000}"
if (( NUM_FID % NUM_CLASSES != 0 )); then
    echo "ERROR: NUM_FID (${NUM_FID}) must be divisible by NUM_CLASSES (${NUM_CLASSES})" >&2
    echo "       so the global label array has the same count per class." >&2
    exit 1
fi

export SC_OWEN_MODE="${OWEN_MODE}"
export PYTHONUNBUFFERED=1

echo "============================================================" | tee -a "${SWEEP_LOG}"
echo "FID sweep started $(date)  (work-steal layout)" | tee -a "${SWEEP_LOG}"
echo "  NUM_GPUS=${NUM_GPUS}  NUM_FID=${NUM_FID}  BATCH=${BATCH}  NUM_CLASSES=${NUM_CLASSES}" | tee -a "${SWEEP_LOG}"
echo "  OWEN_MODE=${OWEN_MODE}  NUM_STEPS=${NUM_STEPS}  CFG_SCALE=${CFG_SCALE}" | tee -a "${SWEEP_LOG}"
echo "  CALIB_DIR=${CALIB_DIR}" | tee -a "${SWEEP_LOG}"
echo "  OUT_BASE=${OUT_BASE}" | tee -a "${SWEEP_LOG}"
echo "============================================================" | tee -a "${SWEEP_LOG}"

cd "${REPO_ROOT}"

run_config() {
    local TAG="$1"          # adaptive_avg64 | uniform_avg64
    local CFG_DIR="${OUT_BASE}/${TAG}"
    shift                    # remaining args: --adaptive_mp ... | --sc_config ...
    local SAMPLES="${CFG_DIR}/samples"
    local IDX_DIR="${CFG_DIR}/_indices"
    mkdir -p "${SAMPLES}" "${IDX_DIR}"

    local DONE
    DONE=$(find "${SAMPLES}" -maxdepth 1 -type f -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' 2>/dev/null | wc -l)
    if [[ ${DONE} -ge ${NUM_FID} ]]; then
        echo "[skip] ${TAG} already complete (${DONE}/${NUM_FID})" | tee -a "${SWEEP_LOG}"
        return
    fi

    echo "[plan] ${TAG} ${DONE}/${NUM_FID} done; planning across ${NUM_GPUS} GPUs at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
    python -u "${REPO_ROOT}/scripts/_plan_missing_indices.py" \
        "${SAMPLES}" "${NUM_FID}" "${NUM_GPUS}" "${IDX_DIR}" \
        | tee -a "${SWEEP_LOG}"

    PIDS=()
    for GPU_ID in 0 1; do
        local INDICES_FILE="${IDX_DIR}/gpu_${GPU_ID}.txt"
        if [[ ! -s "${INDICES_FILE}" ]]; then
            echo "  [GPU ${GPU_ID}] no work assigned" | tee -a "${SWEEP_LOG}"
            continue
        fi
        local GPU_LOG_DIR="${CFG_DIR}/_logs/gpu_${GPU_ID}"
        mkdir -p "${GPU_LOG_DIR}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        SC_OWEN_MODE="${OWEN_MODE}" \
        python -u scripts/quant_sc_main.py \
            --wbits 8 --abits 8 --w_sym --a_sym \
            --timewise 1 --qklayerwise 1.0 --avlayerwise 1.0 \
            --projlayerwise 1.0 --mlplayerwise 1.0 --inputprojlayerwise 1.0 \
            --sc_prec 8 --sc_fixed_level_prec --sc_enable \
            --image-size 256 --num-sampling-steps "${NUM_STEPS}" --cfg-scale "${CFG_SCALE}" \
            --batch-size "${BATCH}" \
            --generate-fid-samples \
            --balanced_classes \
            --num-classes "${NUM_CLASSES}" \
            --balanced_total_samples "${NUM_FID}" \
            --num-fid-samples "${NUM_FID}" \
            --target_indices_path "${INDICES_FILE}" \
            --samples_dir_override "${SAMPLES}" \
            --seed ${GPU_ID} \
            --results-dir "${GPU_LOG_DIR}" \
            "$@" \
            > "${GPU_LOG_DIR}/run.log" 2>&1 &
        PIDS+=($!)
        sleep 5
    done

    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  [worker $i] done at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
        else
            echo "  [worker $i] FAILED (rc=$?); see ${CFG_DIR}/_logs/" | tee -a "${SWEEP_LOG}"
            FAILED=$((FAILED + 1))
        fi
    done

    local FINAL
    FINAL=$(find "${SAMPLES}" -maxdepth 1 -type f -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' 2>/dev/null | wc -l)
    if [[ ${FINAL} -ge ${NUM_FID} ]]; then
        echo "[ok]   ${TAG} ${FINAL}/${NUM_FID} samples in ${SAMPLES} at $(date +%H:%M:%S)" | tee -a "${SWEEP_LOG}"
        return 0
    fi
    if [[ ${FAILED} -gt 0 ]]; then
        echo "[partial] ${TAG} ${FINAL}/${NUM_FID} after ${FAILED} worker failure(s); rerun this sweep to resume" | tee -a "${SWEEP_LOG}"
    else
        echo "[partial] ${TAG} ${FINAL}/${NUM_FID} (workers OK but short of target); rerun to resume" | tee -a "${SWEEP_LOG}"
    fi
    return 1
}

# --- 10 configs: 5 budgets × {adaptive, uniform} ---
# Process uniform first within each budget — they're ~2.5× faster than adaptive,
# so the user gets early signal on whether the pipeline is wired up correctly.
for SL in 128 96 64 48 32; do
    UNIFORM_JSON="${UNIFORM_CFG_DIR}/sc_cfg_uniform${SL}_all.json"
    CALIB_JSON="${CALIB_DIR}/calib_fix_avg${SL}_l256_ref192.json"

    if [[ -f "${UNIFORM_JSON}" ]]; then
        run_config "uniform_avg${SL}" \
            --sc_config "${UNIFORM_JSON}"
    else
        echo "[skip] uniform_avg${SL}: ${UNIFORM_JSON} not found" | tee -a "${SWEEP_LOG}"
    fi

    if [[ -f "${CALIB_JSON}" ]]; then
        run_config "adaptive_avg${SL}" \
            --adaptive_mp \
            --adaptive_mp_table "${CALIB_JSON}" \
            --mp_levels 256,192,128,96,64,48,32,16
    else
        echo "[skip] adaptive_avg${SL}: ${CALIB_JSON} not found" | tee -a "${SWEEP_LOG}"
    fi
done

echo "Sample generation finished $(date)" | tee -a "${SWEEP_LOG}"

# --- FID computation (optional) ---
if [[ -n "${IMAGENET_REF:-}" && -d "${IMAGENET_REF}" ]]; then
    echo "Computing FID against ${IMAGENET_REF} ..." | tee -a "${SWEEP_LOG}"
    if ! python -c "import pytorch_fid" 2>/dev/null; then
        echo "  pytorch_fid not installed; pip install pytorch-fid" | tee -a "${SWEEP_LOG}"
    else
        for D in "${OUT_BASE}"/{adaptive,uniform}_avg*/samples; do
            [[ -d "${D}" ]] || continue
            TAG=$(basename "$(dirname "${D}")")
            FID_OUT="${OUT_BASE}/${TAG}.fid"
            echo "  FID ${TAG} ..." | tee -a "${SWEEP_LOG}"
            python -m pytorch_fid "${IMAGENET_REF}" "${D}" --device cuda \
                > "${FID_OUT}" 2>&1
        done
        echo "FID results:" | tee -a "${SWEEP_LOG}"
        grep -H FID "${OUT_BASE}"/*.fid | tee -a "${SWEEP_LOG}"
    fi
else
    echo "FID skipped (set IMAGENET_REF to enable)." | tee -a "${SWEEP_LOG}"
fi

echo "DONE $(date)" | tee -a "${SWEEP_LOG}"
