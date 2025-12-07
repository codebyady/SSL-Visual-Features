#!/bin/bash -l
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1     # A100 GPU node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --job-name=moco_r50_full
#SBATCH --output=logs/moco_r50_full_%j.out


echo "[$(date)] Starting job on host: $(hostname)"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-not set}"

# Go to your project directory (where .venv already exists)
cd /scratch/at6370/mocomotion_hpc
echo "Working directory: $(pwd)"

# Ensure logs directory exists for the SBATCH --output path
mkdir -p logs

echo "Listing working directory contents:"
ls -alh

# Activate your existing virtual environment
echo "Activating virtual environment at .venv..."
source .venv/bin/activate

echo "Python executable: $(which python)"
python --version

echo "Starting MoCo pretraining on FULL dataset..."
python -m scripts.pretrain_moco \
  --config configs/moco_r50_96.yaml \
  --max-samples -1

echo "[$(date)] Training finished."
