#!/bin/bash
#SBATCH --job-name=verify_cum_cache
#SBATCH --account=nbleier_owned1
#SBATCH --reservation=rtx6000_arph_nodes
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/verify_cum_cache_%j.out
set +u
source ~/.bashrc
conda activate annstention
set -u
export HF_HOME=/nfs/turbo/coe-nbleier/allenjin/hf_cache
export TMPDIR=/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/tmp
export SC_OWEN_MODE=bitrev
export SC_SCRAMBLE_MASKS=64
cd /home/allenjin/Projects/scmp_llm/kernels
echo "=== B1 bit-identity + speedup ==="
python -u tests/verify_cum_cache.py
echo "=== per-group ladder runtime tests (need torch) ==="
python -u -m unittest tests.test_group_ladders -v 2>&1 | tail -20
