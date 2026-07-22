#!/bin/bash
#SBATCH --job-name=sweep_rows
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/sweep_rows_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TMPDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/tmp
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
cd /home/allenjin/Projects/scmp_llm/kernels
echo "=== row-throughput sweep ==="
python -u tests/sweep_rows_throughput.py
