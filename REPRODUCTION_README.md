# RAGNet Reproduction on RIT TIGRIS

本文件只记录当前复现使用的代码入口、模型、数据、实验配置、运行命令和
结果位置。官方项目说明见 [`README.md`](README.md)。

最后更新：2026-07-30

## 1. 路径总览

| 内容 | 位置 |
|---|---|
| Git 仓库 | `/home/dg5804/projects/ragnet-reproduction` |
| Conda 环境 | `/home/dg5804/miniconda3/envs/ragnet` |
| 原始数据 | `/shared/rc/mairl/datasets/ragnet/raw` |
| 解压数据 | `/shared/rc/mairl/datasets/ragnet/processed/data` |
| 模型 | `/shared/rc/mairl/results/ragnet-reproduction/models` |
| Slurm 日志 | `/shared/rc/mairl/results/ragnet-reproduction/logs` |
| 训练输出 | `/shared/rc/mairl/results/ragnet-reproduction/runs` |
| 评估输出 | `/shared/rc/mairl/results/ragnet-reproduction/evaluations` |
| Scratch 数据/缓存 | `/scratch/dg5804/ragnet` |

仓库内的数据入口是软链接：

```text
/home/dg5804/projects/ragnet-reproduction/data
  -> /shared/rc/mairl/datasets/ragnet/processed/data
```

## 2. 关键代码与脚本

| 文件 | 用途 |
|---|---|
| [`model/AffordanceVLM.py`](model/AffordanceVLM.py) | AffordanceVLM 主模型；连接 LLaVA、`[AFF]` embedding 与 SAM，计算 CE/BCE/Dice loss |
| [`model/llava/`](model/llava/) | LLaVA 语言模型、CLIP vision tower 和多模态融合 |
| [`model/segment_anything/`](model/segment_anything/) | SAM image encoder、prompt encoder 和 mask decoder |
| [`train_aff.py`](train_aff.py) | 训练和评估总入口；模型、DeepSpeed、训练循环、验证指标与 checkpoint |
| [`utils/dataset.py`](utils/dataset.py) | 六类训练任务的混合采样、tokenization 和 batch collate |
| [`utils/aff_seg_dataset.py`](utils/aff_seg_dataset.py) | HANDAL、GraspNet、3DOI 等普通 affordance 数据 |
| [`utils/reason_aff_dataset.py`](utils/reason_aff_dataset.py) | Reasoning affordance 数据 |
| [`scripts/evaluate_tigris.sh`](scripts/evaluate_tigris.sh) | TIGRIS 单卡官方 checkpoint 评估 setting |
| [`scripts/train_tigris.sbatch`](scripts/train_tigris.sbatch) | 正式多节点 Slurm 资源、rank、CUDA 和 global batch |
| [`scripts/train_tigris_worker.sh`](scripts/train_tigris_worker.sh) | 训练数据组合、采样比例和超参数 |
| [`scripts/prepare_training_assets.sh`](scripts/prepare_training_assets.sh) | 下载和解压训练数据 |
| [`scripts/check_training_assets.py`](scripts/check_training_assets.py) | 检查模型与训练数据是否完整 |
| [`scripts/smoke_training_data.sbatch`](scripts/smoke_training_data.sbatch) | CPU 数据加载 smoke test |
| [`chat.py`](chat.py) | 单图推理并保存 mask/overlay |
| [`merge_lora_weights_and_save_hf_model.py`](merge_lora_weights_and_save_hf_model.py) | 导出训练后的 Hugging Face checkpoint |
| `/home/dg5804/slurm/ragnet-gpu-job.sbatch` | 通用单 GH200 评估 wrapper |

实验 setting 的读取顺序：

1. 训练资源：`scripts/train_tigris.sbatch`
2. 训练数据与超参数：`scripts/train_tigris_worker.sh`
3. DeepSpeed、optimizer、scheduler 和 loss：`train_aff.py`
4. 评估参数：`scripts/evaluate_tigris.sh`

## 3. 模型

| 用途 | 模型 | 位置 |
|---|---|---|
| 官方 checkpoint 评估 | AffordanceVLM-7B | `/shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B` |
| 从头复现训练的基座 | LLaVA-Lightning-7B-v1-1 | `/shared/rc/mairl/results/ragnet-reproduction/models/LLaVA-Lightning-7B-v1-1` |
| 分割模块 | SAM ViT-H | `/shared/rc/mairl/results/ragnet-reproduction/models/sam_vit_h_4b8939.pth` |
| 视觉编码器 | CLIP ViT-L/14 | Hugging Face cache：`/scratch/dg5804/ragnet/cache/huggingface` |

评估使用已合并的 AffordanceVLM checkpoint，并设置 `--lora_r 0`。训练从
LLaVA 基座初始化，不从 AffordanceVLM checkpoint 继续训练。

## 4. 数据集

| 数据 | 位置 | 用途 |
|---|---|---|
| HANDAL without-depth | `data/HANDAL/without_depth` | HANDAL train/mini/full/easy/hard |
| RAGNet affordance 数据 | `data/{openx,egoobjects,graspnet,rlbench,3doi}` | affordance train/eval |
| RAGNet pickle 索引 | `data/*.pkl` | train/validation split 与 reasoning 标注 |
| Semantic segmentation | `data/lisa_data/{ade20k,cocostuff,vlpart,mapillary}` | `sem_seg` |
| Referring segmentation | `data/lisa_data/{refer_seg,refer_seg/images}` | `refer_seg` |
| VQA | `data/lisa_data/llava_dataset` | `vqa` |
| ReasonSeg | `data/lisa_data/reason_seg` | `reason_seg` |

Mapillary 位于：

```text
data/lisa_data/mapillary/
├── config_v2.0.json
└── training/
    ├── images/          # 18,000
    └── v2.0/labels/     # 18,000
```

全部训练资产检查：

```bash
cd /home/dg5804/projects/ragnet-reproduction
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ragnet
python scripts/check_training_assets.py
```

## 5. 官方 checkpoint 评估

### Setting

| 项目 | 设置 |
|---|---|
| 模型 | AffordanceVLM-7B + SAM ViT-H |
| 资源 | 1×GH200 |
| Precision | BF16 |
| DeepSpeed | Stage 0，eval only |
| LoRA | `lora_r=0` |
| Batch | 1 |
| 配置脚本 | `scripts/evaluate_tigris.sh` |

支持的数据集名称：

```text
handal_all  handal_mini  graspnet_test_seen  graspnet_test_novel  3doi
handal_easy_reasoning  handal_hard_reasoning  3doi_easy_reasoning
```

小型验证集使用 Scratch 数据：

```bash
sbatch --parsable \
  --time=01:00:00 \
  --job-name=ragnet-eval \
  /home/dg5804/slurm/ragnet-gpu-job.sbatch \
  bash /home/dg5804/projects/ragnet-reproduction/scripts/evaluate_tigris.sh \
  handal_mini
```

完整 HANDAL 直接读取 MAIRL：

```bash
sbatch --parsable \
  --time=04:00:00 \
  --job-name=ragnet-handall \
  /home/dg5804/slurm/ragnet-gpu-job.sbatch \
  env RAGNET_DATA=/shared/rc/mairl/datasets/ragnet/processed/data \
  bash /home/dg5804/projects/ragnet-reproduction/scripts/evaluate_tigris.sh \
  handal_all
```

### 结果

| 数据集 | 样本 | GIoU | cIoU | 论文 GIoU/cIoU |
|---|---:|---:|---:|---:|
| HANDAL mini | 1,003 | 60.63 | 60.04 | 60.5 / 60.3 |
| HANDAL full | 65,827 | 60.83 | 60.53 | 60.3 / 60.8 |
| GraspNet seen | 1,008 | 63.34 | 64.11 | 63.3 / 64.0 |
| GraspNet novel | 1,018 | 47.26 | 35.60 | 45.6 / 33.2 |
| 3DOI | 1,012 | 37.50 | 37.51 | 37.4 / 37.4 |
| HANDAL easy reasoning | 1,003 | 59.27 | 58.83 | 58.3 / 58.1 |
| HANDAL hard reasoning | 1,003 | 58.98 | 58.72 | 58.2 / 57.8 |
| 3DOI easy reasoning | 1,012 | 37.87 | 38.94 | 38.1 / 39.4 |

结果文件：

```text
/shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B/eval_result.txt
```

完整 HANDAL：Job `23247`，日志 `logs/ragnet-handall-23247.out`。

## 6. 训练复现

### 正式 setting

| 项目 | 设置 |
|---|---|
| 初始化模型 | LLaVA-Lightning-7B-v1-1 + SAM ViT-H |
| 资源 | 8 节点 × 1 GH200 |
| 并行 | DeepSpeed ZeRO-2 data parallel |
| Precision | BF16 |
| Epochs / steps | 10 × 500 |
| Batch | 40/GPU，global batch 320 |
| Gradient accumulation | 1 |
| Optimizer / LR | AdamW / `2e-5` |
| Warmup | 100 steps |
| LoRA | rank 8，`q_proj,v_proj` |

论文写 global batch 40；官方发布脚本的 `--batch_size=40` 实际是每卡
40。本次按发布脚本语义运行，并将实验命名为 B320。

训练任务比例：

```text
sem_seg : refer_seg : vqa : reason_seg : aff_seg : reason_aff
   3    :     1     :  1  :     1      :    9    :      3
```

正式训练：

```bash
cd /home/dg5804/projects/ragnet-reproduction
sbatch --parsable scripts/train_tigris.sbatch
```

当前任务：

| Job | 状态 | 日志 | 输出 |
|---:|---|---|---|
| `23269` | RUNNING | `logs/ragnet-train-23269.out` | `runs/ragnet-b320-8gh200` |

2×GH200 smoke test：

```bash
RAGNET_BATCH_SIZE=1 \
RAGNET_STEPS_PER_EPOCH=2 \
RAGNET_EPOCHS=1 \
RAGNET_NO_EVAL=1 \
RAGNET_EXP_NAME=smoke-2gh200-20260730-v3 \
sbatch --parsable \
  --job-name=ragnet-smoke \
  --nodes=2 \
  --ntasks=2 \
  --time=00:30:00 \
  scripts/train_tigris.sbatch
```

Smoke Job `23258` 已完成 2 次 forward/backward/step，checkpoint：

```text
/shared/rc/mairl/results/ragnet-reproduction/runs/
└── smoke-2gh200-20260730-v3/ckpt_model/global_step2
```

## 7. 日志、结果与监控

```text
/shared/rc/mairl/results/ragnet-reproduction/
├── logs/          # Slurm stdout/stderr
├── models/        # baseline、训练基座、SAM、eval_result.txt
├── runs/          # TensorBoard 与 DeepSpeed checkpoints
├── evaluations/   # 单图 mask/overlay 与评估 TensorBoard
└── manifests/     # 实验记录
```

查看正式训练：

```bash
squeue -j 23269
tail -f \
  /shared/rc/mairl/results/ragnet-reproduction/logs/ragnet-train-23269.out
sacct -j 23269 --format=JobID,JobName,State,Elapsed,ExitCode
```

DeepSpeed checkpoint 转 FP32：

```bash
cd <run_dir>/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
```
