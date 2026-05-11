#!/bin/bash
# Evaluate generated samples using FID, sFID, IS, Precision, Recall
# Usage: ./evaluate.sh <sample_dir_or_image> [ref_batch]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_BATCH="${SCRIPT_DIR}/reference_batch/VIRTUAL_imagenet256_labeled.npz"

# Parse arguments
INPUT=$1
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <sample_dir_or_image> [ref_batch]"
    echo "  sample_dir_or_image: Path to grid image (*.png) or folder of individual images"
    echo "  ref_batch: Path to reference batch npz (default: $REF_BATCH)"
    exit 1
fi

if [ -n "$2" ]; then
    REF_BATCH=$2
fi

# Check reference batch exists
if [ ! -f "$REF_BATCH" ]; then
    echo "Error: Reference batch not found at $REF_BATCH"
    echo "Download it from: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
    exit 1
fi

# Determine output npz path
if [ -f "$INPUT" ]; then
    # Single file (grid image)
    SAMPLE_NPZ="${INPUT%.png}.npz"
elif [ -d "$INPUT" ]; then
    # Directory
    SAMPLE_NPZ="${INPUT%/}.npz"
else
    echo "Error: Input path does not exist: $INPUT"
    exit 1
fi

# Convert to npz if not already exists
if [ ! -f "$SAMPLE_NPZ" ]; then
    echo "Converting samples to npz format..."
    python "${SCRIPT_DIR}/convert_npz.py" "$INPUT" -o "$SAMPLE_NPZ"
else
    echo "Using existing npz: $SAMPLE_NPZ"
fi

# Run evaluation
echo ""
echo "Running evaluation..."
echo "  Reference: $REF_BATCH"
echo "  Samples:   $SAMPLE_NPZ"
echo ""

python "${SCRIPT_DIR}/evaluator.py" "$REF_BATCH" "$SAMPLE_NPZ"
