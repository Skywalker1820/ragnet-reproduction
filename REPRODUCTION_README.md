# RAGNet Reproduction on RIT TIGRIS

> 这是本仓库的本地复现说明，记录我们在 RIT TIGRIS 上实际使用的环境、数据、模型、脚本和实验结果。
> 官方项目介绍仍见 [`README.md`](README.md)；不要用本文替代论文或官方文档。

最后更新：2026-07-30

## 1. 当前状态

已经完成：

- 在 `ragnet` Conda 环境中跑通 AffordanceVLM-7B 单图推理。
- 下载并校验 RAGNet 官方数据包。
- 从 HANDAL 官方来源补齐 17 类 without-depth 数据。
- 跑完 7 个约 1,000 样本的官方验证集，以及完整 HANDAL（65,827 张）
  评估；完整 HANDAL 为 GIoU 60.83 / cIoU 60.53。
- 下载 LLaVA-Lightning-7B-v1-1 训练基座。
- 下载并校验全部训练资产，包括 Mapillary Vistas v2.0 的 18,000 张
  训练图像与标签。
- 完成 TIGRIS 一节点一 GH200 的多节点 DeepSpeed 训练脚本。
- 完成 2×GH200、2 step 的分布式训练 smoke test 和 ZeRO-2 checkpoint。
- 提交 8×GH200、每卡 batch 40、global batch 320 的正式训练任务
  `23269`。

尚未完成：

- 正式训练任务 `23269` 正在等待 8 张 GH200 同时可用；尚未产生完整
  训练 checkpoint 和训练后评估结果。

当前结论：**官方 checkpoint 的推理与评估链路已经复现；完整训练数据
和多节点训练链路均已验证，正式 B320 训练已进入 Slurm 队列。**

## 2. 项目在做什么

RAGNet 的核心任务是：给定图像和自然语言任务或对象名称，预测目标物体上适合抓取/操作的 affordance 区域。

模型的数据流如下：

```mermaid
flowchart LR
    A["RGB image"] --> B["CLIP ViT-L/14"]
    Q["Object query or task instruction"] --> C["LLaVA / AffordanceVLM"]
    B --> C
    C --> D["Hidden state at [AFF] token"]
    D --> E["MLP projection: 4096 -> 256"]

    A --> F["SAM ViT-H image encoder"]
    E --> G["SAM prompt encoder"]
    F --> H["SAM mask decoder"]
    G --> H
    H --> I["Predicted affordance mask"]
    I --> J["GIoU / cIoU"]
```

关键连接点是特殊 token `[AFF]`：

1. CLIP 和 LLaVA 共同编码图像与文本。
2. 取 `[AFF]` 位置的语言隐藏状态。
3. 用 `text_hidden_fcs` 将 4096 维隐藏状态映射到 256 维。
4. 将它作为文本 prompt embedding 送入 SAM。
5. SAM mask decoder 输出 affordance 二值掩码。

## 3. 关键代码

| 文件 | 作用 | 当前状态 |
|---|---|---|
| [`model/AffordanceVLM.py`](model/AffordanceVLM.py) | AffordanceVLM 主模型；连接 LLaVA、`[AFF]` embedding 和 SAM；定义 CE/BCE/Dice loss | 官方核心代码 |
| [`model/llava/model/llava_arch.py`](model/llava/model/llava_arch.py) | LLaVA 图文融合、vision tower 与 multimodal projector | 官方基础代码 |
| [`model/segment_anything/modeling/`](model/segment_anything/modeling/) | SAM image encoder、prompt encoder 和 mask decoder | 官方基础代码 |
| [`train_aff.py`](train_aff.py) | 训练/评测总入口；模型与数据、DeepSpeed、训练循环、checkpoint 和验证指标 | 已增加 TIGRIS 适配 |
| [`utils/dataset.py`](utils/dataset.py) | 混合训练数据集、batch collate、conversation tokenization | 官方核心代码 |
| [`utils/aff_seg_dataset.py`](utils/aff_seg_dataset.py) | 普通 affordance 训练/验证数据；HANDAL、GraspNet、3DOI 等 | 官方核心代码 |
| [`utils/reason_aff_dataset.py`](utils/reason_aff_dataset.py) | reasoning-based affordance 训练/验证数据 | 官方核心代码 |
| [`utils/utils.py`](utils/utils.py) | IoU 统计、CUDA 搬运、meter 等公共函数 | 官方核心代码 |
| [`chat.py`](chat.py) | 单图自由文本推理，保存预测 mask 和叠加图 | 已增加非交互 `--prompt/--image` 模式 |
| [`app.py`](app.py) | Gradio demo | 尚未作为复现主入口 |
| [`scripts/evaluate_tigris.sh`](scripts/evaluate_tigris.sh) | 我们在 TIGRIS 上使用的统一评估入口 | 已验证 |
| [`scripts/evaluate.sh`](scripts/evaluate.sh) | 官方顺序评估脚本，路径和 CUDA 配置不适合直接在 TIGRIS 运行 | 仅供参考 |
| [`scripts/train.sh`](scripts/train.sh) | 官方单机 8 GPU 训练命令 | 仅供参考 |
| [`scripts/train_tigris.sbatch`](scripts/train_tigris.sbatch) | TIGRIS 多节点资源、CUDA、缓存和全局 batch 配置 | 正式训练入口 |
| [`scripts/train_tigris_worker.sh`](scripts/train_tigris_worker.sh) | 每个 Slurm task 启动一个训练进程 | 正式训练使用 |
| [`scripts/prepare_training_assets.sh`](scripts/prepare_training_assets.sh) | 登录节点下载、计算节点离线校验/解压训练资产 | 使用中 |
| [`scripts/check_training_assets.py`](scripts/check_training_assets.py) | 检查完整训练混合所需的目录、文件数和模型 shard | 已实现 |
| [`merge_lora_weights_and_save_hf_model.py`](merge_lora_weights_and_save_hf_model.py) | 将训练后的 LoRA/DeepSpeed 权重导出为 Hugging Face checkpoint | 尚未在本次复现中使用 |
| [`data_curation/check_dataset.py`](data_curation/check_dataset.py) | 校验官方 pickle 引用的数据是否存在 | 已验证通过 |

### 我们对训练代码的最小修改

`--eval_only` 时：

- 不再构建 `HybridDataset` 训练集。
- 不再开启 input gradients 和 gradient checkpointing。
- 不再构建训练 optimizer/dataloader。
- 使用 DeepSpeed ZeRO stage 0 做单 GPU 评估。

这些修改只绕开评估时不需要的训练初始化，不改变模型预测、mask 后处理或 GIoU/cIoU 计算。

多节点训练时：

- 从 Slurm/torch distributed 环境读取全局 `RANK`、`WORLD_SIZE` 和节点内
  `LOCAL_RANK`。
- 只允许全局 rank 0 写 TensorBoard、meta log 和公共 checkpoint。
- 用全局 world size 计算训练 epoch 的样本数。
- 修正 `utils/dataset.py` 中只适用于字面路径 `./data` 的字符串替换，
  使 MAIRL 绝对路径稳定解析到 `lisa_data`。
- 在 BF16 训练中，将 SAM image embedding、dense positional embedding 和
  prompt embedding 显式对齐到 mask decoder dtype；解决新版
  PyTorch/DeepSpeed 下的 `float != bfloat16` 矩阵乘错误。

## 4. 模型与 checkpoint

### AffordanceVLM-7B

- Hugging Face：[`Dongming97/AffordanceVLM`](https://huggingface.co/Dongming97/AffordanceVLM)
- 本地目录：

  ```text
  /shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B
  ```

- 大小：约 16 GB。
- 模型类：`AffordanceVLMForCausalLM`。
- 基座配置：LLaVA-Lightning-7B-v1-1。
- hidden size：4096。
- checkpoint 已经是合并后的完整模型；评估时使用 `--lora_r 0`，避免再次注入随机 LoRA 层。

### LLaVA 训练基座

- 正式训练从 LLaVA-Lightning-7B-v1-1 初始化，而不是从已训练好的
  AffordanceVLM baseline 继续训练。
- 本地目录：

  ```text
  /shared/rc/mairl/results/ragnet-reproduction/models/LLaVA-Lightning-7B-v1-1
  ```

- 大小：约 13 GB；由 `scripts/train_tigris_worker.sh` 的 `--version`
  传给 `train_aff.py`。

### CLIP vision tower

- 模型：`openai/clip-vit-large-patch14`
- 本地 Hugging Face cache：

  ```text
  /scratch/dg5804/ragnet/cache/huggingface
  ```

### SAM

- 模型：SAM ViT-H。
- checkpoint：

  ```text
  /shared/rc/mairl/results/ragnet-reproduction/models/sam_vit_h_4b8939.pth
  ```

- 大小：约 2.4 GB。
- SHA-256：

  ```text
  a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e
  ```

### 训练时哪些参数会更新

官方训练逻辑冻结：

- CLIP vision tower。
- LLaVA multimodal projector。
- SAM image encoder 和 prompt encoder。

官方训练逻辑更新：

- LLaVA `q_proj` / `v_proj` 的 LoRA 参数。
- token embedding 与 `lm_head`。
- `text_hidden_fcs` 投影层。
- SAM mask decoder。

## 5. Conda 环境

环境位置：

```text
/home/dg5804/miniconda3/envs/ragnet
```

激活：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ragnet
```

已经验证的关键版本：

| 包 | 版本 |
|---|---:|
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu124 |
| torchvision | 0.20.1 |
| transformers | 4.31.0 |
| PEFT | 0.4.0 |
| Accelerate | 0.21.0 |
| DeepSpeed | 0.19.3 |
| SciPy | 1.11.2 |
| scikit-image | 0.21.0 |
| OpenCV | 4.8.0 |

注意：

- 官方 [`requirements.txt`](requirements.txt) 仍固定到 PyTorch 1.13.1 + CUDA 11.7，不应在当前环境中重新无脑执行，否则可能降级已经验证的 PyTorch。
- TIGRIS 是 Linux AArch64；PyTorch/torchvision wheel 可用性与常见 x86_64 机器不同。
- 当前没有安装 FlashAttention；RAGNet 的 `train_aff.py` 不导入 LLaVA 的
  `train_mem.py` monkey patch，因此本次推理、评估和训练入口都不依赖它。
- 检查当前环境是否存在依赖冲突：

  ```bash
  python -m pip check
  ```

## 6. 数据集与存储位置

### 存储分层

| 内容 | 位置 | 说明 |
|---|---|---|
| Git 仓库和 Conda 环境 | `/home/dg5804` | 小文件、代码、环境 |
| 原始数据 | `/shared/rc/mairl/datasets/ragnet/raw` | MAIRL 持久化存储，约 186 GB |
| 解压数据 | `/shared/rc/mairl/datasets/ragnet/processed/data` | MAIRL 持久化存储，约 199 GB |
| 模型和实验结果 | `/shared/rc/mairl/results/ragnet-reproduction` | MAIRL 持久化存储 |
| 高频小型验证子集 | `/scratch/dg5804/ragnet/data` | Scratch，约 4.9 GB，可能按集群策略清理 |
| HF/编译缓存 | `/scratch/dg5804/ragnet/cache` | 可重新生成 |

训练和评估统一从仓库内的 `data` 读取；它不是数据副本，而是指向 MAIRL
持久化目录的软链接：

```text
/home/dg5804/projects/ragnet-reproduction/data
  -> /shared/rc/mairl/datasets/ragnet/processed/data
```

不要把唯一 checkpoint 或唯一实验结果只放在 Scratch。

### 下载的数据

#### RAGNet 官方数据包

- 来源：[`Dongming97/RAGNet`](https://huggingface.co/datasets/Dongming97/RAGNet)
- 文件：

  ```text
  /shared/rc/mairl/datasets/ragnet/raw/data.zip
  ```

- 大小：37,687,969,525 bytes。
- SHA-256：

  ```text
  220616e3df60ca741f24b31ad60af51ca2bd6d5729a0b7577d3a23e821e72784
  ```

- `unzip -t` 完整性检查已通过。

它提供以下本地数据和索引：

- Open-X：约 398 MB。
- EgoObjects：约 4.4 GB。
- GraspNet：约 31 GB。
- RLBench：约 1.6 GB。
- 3DOI：约 376 MB。
- 训练/验证 pickle 索引。

#### HANDAL

- 来源：[`NVlabs/HANDAL`](https://github.com/NVlabs/HANDAL)
- 下载的是 17 个类别的 `without_depth` 压缩包。
- 解压目录：

  ```text
  /shared/rc/mairl/datasets/ragnet/processed/data/HANDAL/without_depth
  ```

- 解压后约 90 GB。
- 17 个 zip 均通过 CRC 检查。
- 完整 test split 共 65,827 张 RGB 图像，每张预期的 handle mask 均存在。

### 当前目录结构

```text
/shared/rc/mairl/datasets/ragnet/processed/data/
├── HANDAL/without_depth/
├── openx/{images,masks}/
├── egoobjects/{images,masks}/
├── graspnet/
│   ├── images/
│   ├── masks/
│   ├── test_seen/
│   └── test_novel/
├── rlbench/{images,masks}/
├── 3doi/{images,masks}/
├── openx_train.pkl
├── egoobjects_train.pkl
├── graspnet_train.pkl
├── rlbench_train.pkl
├── handal_hard_reasoning_train.pkl
├── egoobjects_easy_reasoning_train.pkl
├── egoobjects_hard_reasoning_train.pkl
├── handal_mini_val.pkl
├── graspnet_test_seen_val.pkl
├── graspnet_test_novel_val.pkl
├── 3doi_val.pkl
├── handal_easy_reasoning_val.pkl
├── handal_hard_reasoning_val.pkl
└── 3doi_easy_reasoning_val.pkl
```

### LISA 通用训练资产

原始压缩包：

```text
/shared/rc/mairl/datasets/ragnet/raw/lisa
```

解压目标：

```text
/shared/rc/mairl/datasets/ragnet/processed/data/lisa_data
```

已下载、解压并通过 `scripts/check_training_assets.py` 检查：

- ADE20K、COCO 2017、COCO-Stuff。
- Pascal-Part、PASCAL VOC 2010、PACO-LVIS。
- RefCOCO、RefCOCO+、RefCOCOg、RefCLEF metadata。
- COCO 2014 图像和 RefCLEF `saiapr_tc-12` 图像。
- LLaVA-Instruct-150K annotations。
- ReasonSeg train/val/test 和 explanatory annotations。
- Mapillary Vistas v2.0。

RefCLEF 的原 UNC 下载主机已经失效，因此图像 zip 来自
`cxz0416/saiapr_tc` 镜像；已核对官方 REFER metadata 引用的
19,997/19,997 张图像。

Mapillary 的关键目录：

```text
/shared/rc/mairl/datasets/ragnet/processed/data/lisa_data/mapillary/
├── config_v2.0.json
└── training/
    ├── images/          # 18,000 JPG
    └── v2.0/labels/     # 18,000 PNG
```

其 gated 原始包保存在：

```text
/shared/rc/mairl/datasets/ragnet/raw/lisa/
└── mapillary-vistas-dataset_public_v2.0.zip
```

## 7. 评估实现到底输入和输出什么

### 普通 affordance segmentation

每个样本包含：

- 一张 RGB 图像。
- 一个对象类别名。
- 一个用于评分的 GT mask 或 object ID。
- 固定问题：

  ```text
  You are an embodied robot.
  What is the affordance map of {class_name} in this image?
  ```

数据集会构建包含 `[AFF]` 的完整 conversation。验证代码提取 `[AFF]`
位置的隐藏状态，再由 SAM 预测 mask。

### Reasoning-based affordance segmentation

每个样本包含：

- 一张 RGB 图像。
- 自然语言任务问题。
- 官方参考回答。
- 目标对象类别和 GT affordance mask。

easy 问题通常直接给出对象线索，例如寻找锤子；hard 问题可能只描述
“把钉子钉进墙里”，模型需要从任务推断应使用锤子并定位手柄。

### 一个重要的实现细节

官方 benchmark 的 `train_aff.py::validate()` 并不是自由生成文本：

- `AffValDataset` 会将 assistant 内容设为 `[AFF].`。
- `ReasonAffValDataset` 会把官方参考回答和 `[AFF]` 一起放入 conversation。
- `validate()` 对完整 conversation 做 forward，提取 `[AFF]` hidden state 并解码 mask。

而 [`chat.py`](chat.py) 使用 `model.evaluate()` 自回归生成回答和 `[AFF]`，
更接近真实交互推理。研究改进时应明确区分这两条路径。

### 评估输出

模型内部输出：

- 连续值 affordance mask logits。
- 用 `logit > 0` 得到二值预测 mask。

官方批量评估只持久化：

- GIoU：先计算每个样本的前景 IoU，再取平均。
- cIoU：累计所有样本的前景 intersection 和 union 后计算 IoU。

结果追加到：

```text
/shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B/eval_result.txt
```

当前批量评估不会保存逐样本预测 mask；单图 `chat.py` 会保存 mask 和
叠加可视化。

## 8. 已完成的官方评估

所有数字均为百分数。

| 验证集 | 有效样本 | 本次 GIoU | 本次 cIoU | 论文 GIoU | 论文 cIoU |
|---|---:|---:|---:|---:|---:|
| HANDAL mini（HANDAL†） | 1,003 | 60.63 | 60.04 | 60.5 | 60.3 |
| GraspNet test seen | 1,008 | 63.34 | 64.11 | 63.3 | 64.0 |
| GraspNet test novel | 1,018 | 47.26 | 35.60 | 45.6 | 33.2 |
| 3DOI affordance | 1,012 | 37.50 | 37.51 | 37.4 | 37.4 |
| HANDAL easy reasoning | 1,003 | 59.27 | 58.83 | 58.3 | 58.1 |
| HANDAL hard reasoning | 1,003 | 58.98 | 58.72 | 58.2 | 57.8 |
| 3DOI easy reasoning | 1,012 | 37.87 | 38.94 | 38.1 | 39.4 |
| HANDAL full | 65,827 | 60.83 | 60.53 | 60.3 | 60.8 |

3DOI 的两个原始 pickle 各有 1,013 条记录；官方代码跳过损坏的
`EK_frame_0000040462.jpg`，实际各评估 1,012 条。

GraspNet novel 本次结果比论文高约 1.7/2.4 个百分点。其余结果与论文
非常接近，说明 checkpoint、数据、输入模板、SAM 和指标链路已经连通。

下载的 checkpoint 目录中原本就带有一条 `handal_all` 结果；本次完整
重跑是作业 `23247` 最后追加的 60.83 / 60.53，耗时 2:15:19，exit 0。

## 9. 如何运行评估

### 支持的验证集名称

普通 affordance：

```text
handal_all
handal_mini
graspnet_test_seen
graspnet_test_novel
3doi
```

Reasoning affordance：

```text
handal_hard_reasoning
handal_easy_reasoning
3doi_easy_reasoning
```

### 在已有 GPU 交互会话中运行小型验证集

只在 `sinteractive` 已经分配 GPU 后执行：

```bash
cd /home/dg5804/projects/ragnet-reproduction
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ragnet

bash scripts/evaluate_tigris.sh handal_mini
```

不要在登录节点直接运行模型。

### 使用 `sbatch` 运行小型验证集

```bash
sbatch --parsable \
  --time=01:00:00 \
  --job-name=ragnet-seen \
  /home/dg5804/slurm/ragnet-gpu-job.sbatch \
  bash /home/dg5804/projects/ragnet-reproduction/scripts/evaluate_tigris.sh \
  graspnet_test_seen
```

通用 wrapper 默认从 `/scratch/dg5804/ragnet/data` 读取数据。

### 使用 `sbatch` 运行完整 HANDAL

完整 HANDAL 不复制到 Scratch，直接读取 MAIRL 持久化目录：

```bash
sbatch --parsable \
  --time=04:00:00 \
  --job-name=ragnet-handall \
  /home/dg5804/slurm/ragnet-gpu-job.sbatch \
  env RAGNET_DATA=/shared/rc/mairl/datasets/ragnet/processed/data \
  bash /home/dg5804/projects/ragnet-reproduction/scripts/evaluate_tigris.sh \
  handal_all
```

### 查看 Slurm 状态和日志

```bash
squeue -j 23247
```
```bash
tail -f \
  /shared/rc/mairl/results/ragnet-reproduction/logs/ragnet-handall-23247.out
```

```bash
sacct -j 23247 \
  --format=JobID,JobName,State,Elapsed,ExitCode
```

`sbatch` 作业与 VS Code/SSH 会话解耦；关闭 VS Code 不会终止作业。

## 10. 单图 smoke test

输入：

- 图像：`vis_output/my_workspace.JPG`
- Prompt：`Please segment the affordance map of mug in this image.`

运行命令：

```bash
python -u chat.py \
  --version /shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B \
  --precision bf16 \
  --prompt "Please segment the affordance map of mug in this image." \
  --image /home/dg5804/projects/ragnet-reproduction/vis_output/my_workspace.JPG \
  --vis_save_path \
    /shared/rc/mairl/results/ragnet-reproduction/evaluations/smoke-test
```

模型文本输出：

```text
Sure, [AFF] .
```

生成文件：

```text
/shared/rc/mairl/results/ragnet-reproduction/evaluations/smoke-test/
├── my_workspace_mask_0.jpg
└── my_workspace_masked_img_0.jpg
```

预测区域覆盖了杯子把手。

## 11. 训练配置与 TIGRIS 启动方式

官方入口是 [`scripts/train.sh`](scripts/train.sh)，但发布脚本、论文配置
和 TIGRIS 拓扑之间存在几处重要差异：

| 配置 | 论文 | 发布脚本的实际含义 | 本次 TIGRIS 配置 |
|---|---:|---:|---:|
| GPU | 8×A100 80 GB | 单机 GPU 0–7 | 8 节点×1 GH200 |
| Precision | BF16 | BF16 | BF16 |
| DeepSpeed | 未单列 | ZeRO stage 2 | ZeRO stage 2 |
| Epochs | 10 | 10 | 10 |
| Steps per epoch | 500 | 500 | 500 |
| Global batch | 40 | `40 × 8 = 320` | 320 |
| Micro-batch / GPU | 5 | 40 | 40 |
| Gradient accumulation | 1 | 1 | 1 |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 2e-5 | 未传参，落到默认 3e-4 | 2e-5 |
| Warmup | 100 steps | 100 steps | 100 steps |
| LoRA | rank 8 | `q_proj,v_proj` | `q_proj,v_proj` |

`train_aff.py` 明确将 `--batch_size` 定义为每卡每 step 的
micro-batch。论文写的是 global batch 40，但发布脚本实际形成
`8 × 40 × 1 = 320`。本次按发布代码的语义使用每卡 40；这更接近
官方脚本，但与论文表述存在差异，因此实验记录中明确标记为 B320。

训练混合数据及顶层采样权重：

```text
sem_seg : refer_seg : vqa : reason_seg : aff_seg : reason_aff
   3    :     1     :  1  :     1      :    9    :      3
```

Affordance 数据内部权重：

```text
HANDAL : Open-X : EgoObjects : GraspNet : RLBench
   2   :   2    :     4      :    2     :    1
```

Reasoning affordance 数据内部权重：

```text
HANDAL hard : EgoObjects easy : EgoObjects hard
      1     :        1        :        1
```

### 为什么不能直接运行官方 `train.sh`

1. 脚本硬编码 `/data/cuda/cuda-11.7`，而当前环境是 CUDA 12.4。
2. 脚本假设一台机器上有 8 张 GPU。
3. TIGRIS 每个 GH 节点只有一张 GH200；8 GPU 训练会跨 8 个节点。
4. `--batch_size=40` 会在 8 卡上形成全局 batch 320；本次保留该发布
   脚本行为，并在实验名中标记 B320。
5. 脚本没有显式传入论文所述的 `2e-5` learning rate。
6. 多节点 rank 不能用单机的 `torch.cuda.device_count()` 推断。

因此不要直接提交：

```bash
bash scripts/train.sh
```

### 当前训练作业

| Job ID | 用途 | 资源 | 配置 | 日志 |
|---:|---|---|---|---|
| `23252` | 公开训练数据幂等复核 | 1 CPU 节点 | 完成，exit 0 | `logs/ragnet-data-check-23252.out` |
| `23253` | RefCLEF 图像校验和解压 | 1 CPU 节点 | 完成，exit 0 | `logs/ragnet-refclef-23253.out` |
| `23265` | Mapillary v2.0 校验和解压 | 1 CPU 节点 | 完成，exit 0 | `logs/ragnet-mapillary-23265.out` |
| `23267` | Mapillary 专项真实数据加载 | 1 CPU 节点 | 完成，exit 0 | `logs/ragnet-mapillary-smoke-23267.out` |
| `23269` | 正式 B320 训练 | 8×GH200 | 已提交，等待资源 | `logs/ragnet-train-23269.out` |
| `23256` | 六类训练 data loader smoke | 1 CPU 节点 | 完成，exit 0 | `logs/ragnet-data-smoke-23256.out` |
| `23258` | 分布式训练 smoke | 2 节点×1 GH200 | 完成，2 次 forward/backward/step，exit 0 | `logs/ragnet-smoke-23258.out` |

首次 GPU smoke `23249` 已正确取得两个节点并启动两个 rank，但因 CUDA
toolkit 路径没有稳定解析，在 DeepSpeed 导入阶段失败。`23257` 继续定位出
BF16 SAM embedding 与 decoder dtype 不一致；修正后的 `23258` 完整成功。

`23258` 的两步 rank-0 训练日志：

```text
Epoch: [0][1/2] Loss 11.6266  CeLoss 9.5000  MaskLoss 2.1266
Epoch: [0][2/2] Loss  9.7699  CeLoss 8.3750  MaskLoss 1.3949
```

smoke checkpoint：

```text
/shared/rc/mairl/results/ragnet-reproduction/runs/
└── smoke-2gh200-20260730-v3/
    └── ckpt_model/global_step2/
```

该目录约 32 GB，包含两个 ZeRO optimizer shard 和约 30 GB 的 model state。

日志根目录：

```text
/shared/rc/mairl/results/ragnet-reproduction/logs
```

### 正式 8×GH200 训练

正式任务提交前满足了以下条件：

1. `23258` 完成一次真实 forward/backward/checkpoint，并以 exit code 0
   结束（已满足）。
2. `scripts/check_training_assets.py` 全部通过（已满足）。
3. Mapillary Vistas v2.0 已经通过授权入口获得，18,000 张训练标签通过
   资产检查，并完成一次真实图像/标签/token/mask collate（已满足）。

本次提交命令为：

```bash
cd /home/dg5804/projects/ragnet-reproduction
sbatch --parsable scripts/train_tigris.sbatch
```

默认训练输出：

```text
/shared/rc/mairl/results/ragnet-reproduction/runs/ragnet-b320-8gh200
```

如未补 Mapillary 或人为删掉 RefCLEF，只能称为 reduced-mixture
ablation，不能称为论文训练复现。

## 12. 训练产物与合并

训练时 TensorBoard 和 checkpoint 默认位于：

```text
<log_base_dir>/<exp_name>/
├── events.out.tfevents...
├── meta_log_giou*_ciou*.pth
└── ckpt_model/
```

DeepSpeed checkpoint 转 FP32：

```bash
cd <run_dir>/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin
```

将 LoRA/训练权重合并为 Hugging Face 模型：

```bash
CUDA_VISIBLE_DEVICES="" \
python merge_lora_weights_and_save_hf_model.py \
  --version <base_llava_path> \
  --weight <run_dir>/pytorch_model.bin \
  --save_path <output_model_dir>
```

不要覆盖官方 baseline：

```text
/shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B
```

建议每个实验使用独立输出目录，例如：

```text
/shared/rc/mairl/results/ragnet-reproduction/experiments/<experiment_name>/
```

## 13. 实验结果与日志位置

```text
/shared/rc/mairl/results/ragnet-reproduction/
├── evaluations/
│   ├── smoke-test/
│   └── tensorboard/
├── logs/
│   ├── ragnet-handall-23247.out
│   └── ragnet-train-23269.out
├── manifests/
│   └── smoke-test-2026-07-30.md
├── models/
│   ├── AffordanceVLM-7B/
│   │   └── eval_result.txt
│   ├── LLaVA-Lightning-7B-v1-1/
│   └── sam_vit_h_4b8939.pth
└── runs/
    ├── smoke-2gh200-20260730-v3/
    └── ragnet-b320-8gh200/
```

## 14. 接下来改进时最值得关注的位置

模型方法：

- [`model/AffordanceVLM.py`](model/AffordanceVLM.py)：`[AFF]` embedding、
  text projection、SAM prompt/mask decoder 的连接方式。
- LoRA target modules、rank 和训练范围。
- easy/hard reasoning 的文本监督形式。
- 是否让语言模型真正生成 reasoning，再用生成的 `[AFF]` 做分割。

数据与采样：

- [`utils/dataset.py`](utils/dataset.py)：六类训练任务的混合比例。
- [`utils/aff_seg_dataset.py`](utils/aff_seg_dataset.py)：普通 affordance prompt。
- [`utils/reason_aff_dataset.py`](utils/reason_aff_dataset.py)：reasoning question/answer 模板。
- seen/novel 类别划分与跨数据集泛化。

评估：

- 当前 `val_batch_size` 强制为 1。
- 每个样本都会调用 `torch.cuda.empty_cache()`，GH200 利用率约 57%。
- 当前不保存逐样本预测，后续可增加 mask、overlay、per-sample IoU 和失败案例导出。
- 修改评估速度时必须保持预测与指标一致，不能混入方法改动。

## 15. 最短操作清单

进入环境：

```bash
cd /home/dg5804/projects/ragnet-reproduction
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ragnet
```

确认环境：

```bash
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

运行一个小验证集（必须已经获得 GPU）：

```bash
bash scripts/evaluate_tigris.sh handal_mini
```

查看已有指标：

```bash
tail \
  /shared/rc/mairl/results/ragnet-reproduction/models/AffordanceVLM-7B/eval_result.txt
```

查看正式训练状态和日志：

```bash
squeue -j 23269
tail -f \
  /shared/rc/mairl/results/ragnet-reproduction/logs/ragnet-train-23269.out
```
