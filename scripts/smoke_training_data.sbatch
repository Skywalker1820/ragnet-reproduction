#!/usr/bin/env bash
#SBATCH --job-name=ragnet-data-smoke
#SBATCH --account=mairl
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/shared/rc/mairl/results/ragnet-reproduction/logs/%x-%j.out

set -euo pipefail

source /home/dg5804/miniconda3/etc/profile.d/conda.sh
conda activate ragnet

export HF_HOME=/scratch/dg5804/ragnet/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /home/dg5804/projects/ragnet-reproduction
python -u -m scripts.smoke_training_data
