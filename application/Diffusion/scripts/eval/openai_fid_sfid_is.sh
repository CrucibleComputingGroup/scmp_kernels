#!/bin/bash
# Run OpenAI guided-diffusion evaluator (FID/sFID/IS/Precision/Recall) on a
# samples npz against the VIRTUAL_imagenet256_labeled reference.
#
# Why a wrapper: OpenAI's evaluator.py needs TF 2.15 with bundled CUDA libs,
# and TF must be able to dlopen those libs at runtime. Plain `pip install
# tensorflow[and-cuda]` puts them under site-packages/nvidia/*/lib but doesn't
# add them to LD_LIBRARY_PATH. We do that here so TF finds the GPU.
#
# Set TFEVAL_ENV (default 'tfeval'), REF (path to VIRTUAL npz), and EVAL_PY
# (path to evaluator.py) before invoking; or pass three positional args.
#
# Usage:
#   ./openai_fid_sfid_is.sh <samples_npz> <out_txt>
#   ./openai_fid_sfid_is.sh <samples_npz> <out_txt> <ref_npz>
#
# One-time env setup:
#   conda create -n tfeval python=3.11 -y
#   conda activate tfeval
#   pip install "tensorflow[and-cuda]==2.15.*" numpy scipy tqdm requests

set -euo pipefail

SAMPLES_NPZ=${1:?samples npz required}
OUT_TXT=${2:?output txt path required}
REF=${3:-${REF:-}}
EVAL_PY=${EVAL_PY:-$(dirname "$0")/../../models/evaluations/evaluator.py}
TFEVAL_ENV=${TFEVAL_ENV:-tfeval}

if [[ -z "$REF" || ! -f "$REF" ]]; then
    echo "error: pass <ref_npz> as 3rd arg or set \$REF to VIRTUAL_imagenet256_labeled.npz" >&2
    exit 2
fi
if [[ ! -f "$EVAL_PY" ]]; then
    echo "error: evaluator.py not found at $EVAL_PY (set \$EVAL_PY)" >&2
    exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TFEVAL_ENV"

NV=$(python -c "import os, nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH=${NV}/cudnn/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cuda_nvrtc/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${LD_LIBRARY_PATH:-}
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== node $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

echo "=== eval  ref=$REF  samples=$SAMPLES_NPZ ==="
python -u "$EVAL_PY" "$REF" "$SAMPLES_NPZ" 2>&1 | tee "$OUT_TXT"
echo "=== done $(date) ==="
