#!/usr/bin/env bash
#SBATCH --job-name=ragnet-data
#SBATCH --account=mairl
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/shared/rc/mairl/results/ragnet-reproduction/logs/%x-%j.out

set -euo pipefail

source /home/dg5804/miniconda3/etc/profile.d/conda.sh
conda activate ragnet

export HF_HOME=/scratch/dg5804/ragnet/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

cd /home/dg5804/projects/ragnet-reproduction
bash scripts/prepare_training_assets.sh "${1:-public}"
