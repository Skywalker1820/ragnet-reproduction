#!/usr/bin/env python3
"""Initialize the released training mixture and collate one sample per task."""

from __future__ import annotations

import argparse

import deepspeed  # noqa: F401 - import first to match train_aff.py's compatible order
from transformers import AutoTokenizer

from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/shared/rc/mairl/datasets/ragnet/processed/data",
    )
    parser.add_argument(
        "--model",
        default=(
            "/shared/rc/mairl/results/ragnet-reproduction/models/"
            "LLaVA-Lightning-7B-v1-1"
        ),
    )
    parser.add_argument(
        "--sem-seg-data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens("[AFF]")
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
        special_tokens=True,
    )
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    task_names = [
        "sem_seg",
        "refer_seg",
        "vqa",
        "reason_seg",
        "aff_seg",
        "reason_aff",
    ]
    dataset = HybridDataset(
        args.data_root,
        tokenizer,
        "openai/clip-vit-large-patch14",
        samples_per_epoch=len(task_names),
        precision="bf16",
        image_size=1024,
        dataset="||".join(task_names),
        sample_rate=[1] * len(task_names),
        sem_seg_data=args.sem_seg_data,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        aff_seg_data="handal||openx||egoobjects||graspnet||rlbench",
        aff_sample_rate=[2, 2, 4, 2, 1],
        reason_aff_data=(
            "handal_hard_reasoning||"
            "egoobjects_easy_reasoning||"
            "egoobjects_hard_reasoning"
        ),
        reason_aff_sample_rate=[1, 1, 1],
    )

    for task_name, task_dataset in zip(task_names, dataset.all_datasets):
        sample = (*task_dataset[0], False)
        batch = collate_fn(
            [sample],
            tokenizer=tokenizer,
            conv_type="llava_v1",
            use_mm_start_end=True,
            local_rank=-1,
        )
        print(
            f"OK {task_name}: image={tuple(batch['images'].shape)} "
            f"clip={tuple(batch['images_clip'].shape)} "
            f"tokens={tuple(batch['input_ids'].shape)} "
            f"masks={tuple(batch['masks_list'][0].shape)}"
        )


if __name__ == "__main__":
    main()
