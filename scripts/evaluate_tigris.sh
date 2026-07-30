#!/usr/bin/env bash

set -euo pipefail

dataset="${1:-handal_mini}"
shift || true

case "$dataset" in
    handal_all|handal_mini|graspnet_test_seen|graspnet_test_novel|3doi)
        evaluation_mode="--eval_affordance"
        ;;
    handal_hard_reasoning|handal_easy_reasoning|3doi_easy_reasoning)
        evaluation_mode="--eval_reason_aff"
        ;;
    *)
        echo "Unsupported evaluation dataset: $dataset" >&2
        exit 2
        ;;
esac

ragnet_project="${RAGNET_PROJECT:-$HOME/projects/ragnet-reproduction}"
ragnet_results="${RAGNET_RESULTS:-/shared/rc/mairl/results/ragnet-reproduction}"
ragnet_model="${RAGNET_MODEL:-$ragnet_results/models/AffordanceVLM-7B}"
ragnet_sam="${RAGNET_SAM:-$ragnet_results/models/sam_vit_h_4b8939.pth}"
ragnet_scratch="${RAGNET_SCRATCH:-/scratch/$USER/ragnet}"
ragnet_data="${RAGNET_DATA:-$ragnet_scratch/data}"

master_port="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"

if [[ -n "${RAGNET_CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$RAGNET_CUDA_HOME"
elif [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    for candidate in \
        /.autofs/tools/spack/opt/spack/linux-rhel9-neoverse_v2/gcc-12.3.1/cuda-12.4.0-*; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            export CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "CUDA 12.4 toolkit not found; set RAGNET_CUDA_HOME." >&2
    exit 3
fi

if [[ -z "${RAGNET_GCC_HOME:-}" ]]; then
    for candidate in \
        /.autofs/tools/spack/opt/spack/linux-rhel9-neoverse_v2/gcc-12.3.1/gcc-12.3.1-*; do
        if [[ -x "$candidate/bin/gcc" && -x "$candidate/bin/g++" ]]; then
            RAGNET_GCC_HOME="$candidate"
            break
        fi
    done
fi

if [[ -z "${RAGNET_GCC_HOME:-}" ]]; then
    echo "GCC 12.3.1 not found; set RAGNET_GCC_HOME." >&2
    exit 3
fi

export PATH="$CUDA_HOME/bin:$RAGNET_GCC_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$RAGNET_GCC_HOME/bin/gcc"
export CXX="$RAGNET_GCC_HOME/bin/g++"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ragnet_scratch"

exec deepspeed \
    --num_gpus 1 \
    --master_port "$master_port" \
    "$ragnet_project/train_aff.py" \
    --version "$ragnet_model" \
    --dataset_dir "$ragnet_data" \
    --dataset "reason_seg" \
    --sample_rates "1" \
    --vision_pretrained "$ragnet_sam" \
    --log_base_dir "$ragnet_results/evaluations/tensorboard" \
    --exp_name "$dataset" \
    --precision bf16 \
    --lora_r 0 \
    --workers "${RAGNET_WORKERS:-4}" \
    --eval_only \
    "$evaluation_mode" \
    --val_dataset "$dataset" \
    "$@"
