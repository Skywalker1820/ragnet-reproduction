#!/usr/bin/env bash
#SBATCH --job-name=ragnet-train
#SBATCH --account=mairl
#SBATCH --partition=tigris
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/rc/mairl/results/ragnet-reproduction/logs/%x-%j.out

set -euo pipefail

source /home/dg5804/miniconda3/etc/profile.d/conda.sh
conda activate ragnet

default_cuda_home=/.autofs/tools/spack/opt/spack/linux-rhel9-neoverse_v2/gcc-12.3.1/cuda-12.4.0-5ncc6rpt2kvuwx5iomvpkdadb5oxvb6s
default_gcc_home=/.autofs/tools/spack/opt/spack/linux-rhel9-neoverse_v2/gcc-12.3.1/gcc-12.3.1-bjxl5z3yzmlwhp6so6xyj6c4rftezdxt
export CUDA_HOME="${RAGNET_CUDA_HOME:-${default_cuda_home}}"
RAGNET_GCC_HOME="${RAGNET_GCC_HOME:-${default_gcc_home}}"

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA 12.4 toolkit not found; set RAGNET_CUDA_HOME." >&2
    exit 3
fi
if [[ -z "${RAGNET_GCC_HOME:-}" ]]; then
    echo "GCC 12.3.1 not found; set RAGNET_GCC_HOME." >&2
    exit 3
fi

export PATH="${CUDA_HOME}/bin:${RAGNET_GCC_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${RAGNET_GCC_HOME}/bin/gcc"
export CXX="${RAGNET_GCC_HOME}/bin/g++"
export HF_HOME=/scratch/dg5804/ragnet/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=/scratch/dg5804/ragnet/cache/torch_extensions
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export WORLD_SIZE="${SLURM_NTASKS}"

# Paper target: global batch size 40. The released train.sh passes 40 as a
# per-device batch, but train_aff.py defines the option per GPU.
if [[ -z "${RAGNET_BATCH_SIZE:-}" ]]; then
    if (( 40 % WORLD_SIZE != 0 )); then
        echo "WORLD_SIZE=${WORLD_SIZE} does not divide global batch 40." >&2
        exit 1
    fi
    export RAGNET_BATCH_SIZE="$((40 / WORLD_SIZE))"
fi
export RAGNET_GRAD_ACCUM="${RAGNET_GRAD_ACCUM:-1}"
export RAGNET_EXP_NAME="${RAGNET_EXP_NAME:-ragnet-paper-${WORLD_SIZE}gh200}"

echo "nodes=${SLURM_NNODES} world_size=${WORLD_SIZE} batch_per_gpu=${RAGNET_BATCH_SIZE}"
echo "global_batch=$((WORLD_SIZE * RAGNET_BATCH_SIZE * RAGNET_GRAD_ACCUM))"
echo "master=${MASTER_ADDR}:${MASTER_PORT} exp=${RAGNET_EXP_NAME}"
echo "cuda_home=${CUDA_HOME}"
nvcc_release="$("${CUDA_HOME}/bin/nvcc" --version | awk '/release/{gsub(",", "", $5); print $5}')"
echo "nvcc=${nvcc_release} gcc=$("${CC}" -dumpfullversion)"

cd /home/dg5804/projects/ragnet-reproduction
srun \
    --kill-on-bad-exit=1 \
    --label \
    bash scripts/train_tigris_worker.sh
