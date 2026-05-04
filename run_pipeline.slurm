#!/bin/bash
#SBATCH --job-name=ppri_pipeline
#SBATCH --output=slurm-%j.out
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:t4:1

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME   GPUs: $CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="/scratch/user/saher02/PPRI 3"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }

export HF_HOME=/scratch/user/saher02/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


# ---------------------------------------------------------------------------
# 1. Load modules
# ---------------------------------------------------------------------------
module purge
module load Anaconda3/2024.02-1

# ---------------------------------------------------------------------------
# 2. Force the env's Python — bypass conda activate entirely
# ---------------------------------------------------------------------------
export PATH=/scratch/user/saher02/ppri_env/bin:$PATH
export CONDA_PREFIX=/scratch/user/saher02/ppri_env

# Confirm correct Python
echo "Python path: $(which python)"
python --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ---------------------------------------------------------------------------
# 3. Run pipeline
# ---------------------------------------------------------------------------
echo "Starting PPRI pipeline..."
python main.py

echo "Pipeline finished: $(date)"
echo "Exit code: $?"
