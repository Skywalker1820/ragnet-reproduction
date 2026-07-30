#!/usr/bin/env bash
set -euo pipefail

RAGNET_REPO="${RAGNET_REPO:-/home/dg5804/projects/ragnet-reproduction}"
RAGNET_DATA_DIR="${RAGNET_DATA_DIR:-${RAGNET_REPO}/data}"
RAGNET_MODEL="${RAGNET_MODEL:-/shared/rc/mairl/results/ragnet-reproduction/models/LLaVA-Lightning-7B-v1-1}"
RAGNET_SAM="${RAGNET_SAM:-/shared/rc/mairl/results/ragnet-reproduction/models/sam_vit_h_4b8939.pth}"
RAGNET_RUN_ROOT="${RAGNET_RUN_ROOT:-/shared/rc/mairl/results/ragnet-reproduction/runs}"
RAGNET_EXP_NAME="${RAGNET_EXP_NAME:-ragnet-paper}"

RANK="${SLURM_PROCID:-${RANK:-0}}"
LOCAL_RANK="${SLURM_LOCALID:-${LOCAL_RANK:-0}}"
WORLD_SIZE="${SLURM_NTASKS:-${WORLD_SIZE:-1}}"
export RANK LOCAL_RANK WORLD_SIZE

if [[ ! -f "${RAGNET_MODEL}/config.json" ]]; then
    echo "Missing base model: ${RAGNET_MODEL}" >&2
    exit 1
fi
if [[ ! -f "${RAGNET_SAM}" ]]; then
    echo "Missing SAM checkpoint: ${RAGNET_SAM}" >&2
    exit 1
fi
if [[ ! -d "${RAGNET_DATA_DIR}" ]]; then
    echo "Missing dataset directory: ${RAGNET_DATA_DIR}" >&2
    exit 1
fi

extra_args=()
if [[ "${RAGNET_NO_EVAL:-0}" == "1" ]]; then
    extra_args+=(--no_eval)
fi

cd "${RAGNET_REPO}"

python -u train_aff.py \
    --local_rank="${LOCAL_RANK}" \
    --version="${RAGNET_MODEL}" \
    --vision_pretrained="${RAGNET_SAM}" \
    --dataset_dir="${RAGNET_DATA_DIR}" \
    --dataset="${RAGNET_DATASETS:-sem_seg||refer_seg||vqa||reason_seg||aff_seg||reason_aff}" \
    --sample_rates="${RAGNET_SAMPLE_RATES:-3,1,1,1,9,3}" \
    --sem_seg_data="${RAGNET_SEM_SEG_DATA:-ade20k||cocostuff||pascal_part||paco_lvis||mapillary}" \
    --refer_seg_data="${RAGNET_REFER_SEG_DATA:-refclef||refcoco||refcoco+||refcocog}" \
    --vqa_data="${RAGNET_VQA_DATA:-llava_instruct_150k}" \
    --reason_seg_data="${RAGNET_REASON_SEG_DATA:-ReasonSeg|train}" \
    --aff_seg_data="${RAGNET_AFF_SEG_DATA:-handal||openx||egoobjects||graspnet||rlbench}" \
    --aff_sample_rates="${RAGNET_AFF_SAMPLE_RATES:-2,2,4,2,1}" \
    --reason_aff_data="${RAGNET_REASON_AFF_DATA:-handal_hard_reasoning||egoobjects_easy_reasoning||egoobjects_hard_reasoning}" \
    --reason_aff_sample_rates="${RAGNET_REASON_AFF_SAMPLE_RATES:-1,1,1}" \
    --val_dataset="${RAGNET_VAL_DATASET:-ReasonSeg|val}" \
    --log_base_dir="${RAGNET_RUN_ROOT}" \
    --exp_name="${RAGNET_EXP_NAME}" \
    --batch_size="${RAGNET_BATCH_SIZE:-5}" \
    --grad_accumulation_steps="${RAGNET_GRAD_ACCUM:-1}" \
    --steps_per_epoch="${RAGNET_STEPS_PER_EPOCH:-500}" \
    --epochs="${RAGNET_EPOCHS:-10}" \
    --lr="${RAGNET_LR:-2e-5}" \
    --workers="${RAGNET_WORKERS:-4}" \
    --precision=bf16 \
    "${extra_args[@]}"
