#!/bin/bash

# SC Parameter Sweep Script
# Runs quant_sc_main.py with different timewise and qklayerwise combinations

# Configuration - just specify the number of steps
NUM_TIMEWISE=25      # Number of timewise values (0.0 to 1.0); 50 layers
NUM_QKLAYERWISE=14   # Number of qklayerwise values (0.0 to 1.0); 28 blocks
SC_PREC=8
IMAGE_SIZE=256

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="../results/sc_sweep_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo "========================================"
echo "SC Parameter Sweep"
echo "========================================"
echo "Results directory: ${RESULTS_DIR}"
echo "Timewise steps: ${NUM_TIMEWISE}"
echo "Qklayerwise steps: ${NUM_QKLAYERWISE}"
echo "========================================"

# Run sweep
for ((i=0; i<NUM_TIMEWISE; i++)); do
    if [ ${NUM_TIMEWISE} -eq 1 ]; then
        timewise="0.0"
    else
        timewise=$(echo "scale=2; $i / (${NUM_TIMEWISE} - 1)" | bc)
    fi

    for ((j=0; j<NUM_QKLAYERWISE; j++)); do
        if [ ${NUM_QKLAYERWISE} -eq 1 ]; then
            qklayerwise="0.0"
        else
            qklayerwise=$(echo "scale=2; $j / (${NUM_QKLAYERWISE} - 1)" | bc)
        fi

        # Skip if only one of them is 0 (no effect), but run if both are 0 (baseline)
        if [ $(echo "${timewise} == 0" | bc) -eq 1 ] && [ $(echo "${qklayerwise} != 0" | bc) -eq 1 ]; then
            echo "Skipping: timewise=${timewise}, qklayerwise=${qklayerwise} (timewise=0 has no effect)"
            continue
        fi
        if [ $(echo "${timewise} != 0" | bc) -eq 1 ] && [ $(echo "${qklayerwise} == 0" | bc) -eq 1 ]; then
            echo "Skipping: timewise=${timewise}, qklayerwise=${qklayerwise} (qklayerwise=0 has no effect)"
            continue
        fi

        echo ""
        echo "Running: timewise=${timewise}, qklayerwise=${qklayerwise}"
        echo "----------------------------------------"

        python scripts/quant_sc_main.py \
            --wbits 8 --abits 8 \
            --w_sym --a_sym \
            --timewise ${timewise} \
            --qklayerwise ${qklayerwise} \
            --sc_prec ${SC_PREC} \
            --image-size ${IMAGE_SIZE} \
            --results-dir ${RESULTS_DIR}

        echo "Done: timewise=${timewise}, qklayerwise=${qklayerwise}"
    done
done

echo ""
echo "========================================"
echo "Sweep complete! Results in: ${RESULTS_DIR}"
echo "========================================"
