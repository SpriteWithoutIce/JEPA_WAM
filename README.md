# JEPA-WAM

一个面向机器人操作的 Vision-Language-Action 算法框架。这个仓库以 `Prismatic/OpenVLA` 风格训练流水线为底座，加入了基于 **V-JEPA 2.1** 的视觉表征、**Qwen2.5-0.5B + LoRA** 的语言条件建模，以及 **Flow Matching / JEPA 辅助监督** 的动作学习模块。

当前仓库里同时保留了两层实现：

- `jepa_wam/`：更清晰、模块化的 JEPA-VLA 算法原型实现。
- `prismatic/ + vla-scripts/`：实际用于训练、恢复、评估和部署的工程化流水线。

## 1. 项目目标

这个仓库主要解决的是多任务机器人操作中的 VLA 建模问题，核心思路是：

- 用冻结的 `V-JEPA 2.1` 提供时空视觉表征，而不是传统 DINO/SigLIP 视觉 backbone。
- 用 `Qwen2.5-0.5B` 承接语言条件和跨模态融合，通过 `LoRA` 控制可训练参数量。
- 用 `GR00T` 风格的 `Flow Matching Action Head` 预测连续动作序列。
- 用未来帧 JEPA 表征预测或视觉 token 对齐损失，增强动作前的隐变量表征质量。

这套框架目前围绕 `LIBERO` 任务做了较完整的训练和评估支持，同时保留了 ALOHA、CALVIN、部署脚本等实验资产。

## 2. 算法结构

训练时的数据流可以概括为：

```text
多视角当前帧 + 文本指令 + proprio + 未来帧
        │
        ├─ 冻结 V-JEPA 2.1 编码当前图像
        │      -> patch tokens
        │
        ├─ Vision Projector
        │      -> 映射到 LLM hidden space
        │
        ├─ Qwen2.5-0.5B
        │      -> 视觉 token 在前，文本 token 在后
        │      -> LLM hidden states
        │
        ├─ Action Head
        │      -> Flow Matching / L1 / joint JEPA variants
        │      -> 连续动作序列
        │
        └─ Aux Head / Visual Token Cosine Head
               -> 监督未来视觉表征或配对视觉 token
```

核心设计对应代码：

- `jepa_wam/models/jepa_vla.py`：总模型组装，包含视觉、LLM、动作头、aux head。
- `jepa_wam/models/vision_projector.py`：V-JEPA token 到 LLM 空间的两层 MLP 投影。
- `jepa_wam/models/llm_wrapper.py`：Qwen2.5 + LoRA 封装。
- `jepa_wam/models/action_head.py`：GR00T 风格 Flow Matching 动作头。
- `jepa_wam/models/aux_head.py`：未来帧 JEPA embedding 预测头。
- `architecture_spec.md`：更完整的算法规格说明和张量形状约定。

## 3. 仓库结构

```text
JEPA-WAM/
├── jepa_wam/                     # 轻量、模块化 JEPA-VLA 实现
│   ├── conf/                     # dataclass 配置
│   ├── datasets/                 # RLDS 数据包装、batch transform、collator
│   ├── models/                   # V-JEPA + Qwen + action/aux heads
│   └── training/                 # 简化 trainer / metrics
├── prismatic/                    # 底层 VLM / VLA 主体实现
├── vla-scripts/                  # 实际训练、恢复、转换、评估脚本
├── experiments/robot/            # LIBERO / ALOHA / 部署相关实验代码
├── pretrained_models/            # 本地模型配置或导出资产
├── meta/                         # 数据集元信息
└── architecture_spec.md          # 算法说明文档
```

如果你是第一次读这个仓库，建议按这个顺序：

1. 读 `architecture_spec.md`
2. 看 `jepa_wam/models/jepa_vla.py`
3. 看 `vla-scripts/train.py`
4. 再看具体实验脚本，例如 `vla-scripts/run_jepa_only.sh`

## 4. 依赖与环境

`pyproject.toml` 已经列出了主要 Python 依赖，但这个仓库还有几项仓库外前置条件需要手动满足。

### 基础依赖

- Python `>= 3.8`
- PyTorch `2.2.0`
- `torchvision==0.17.0`
- `transformers`
- `peft`
- `tensorflow==2.15.0`
- `tensorflow_datasets==4.9.3`
- `dlimp`
- `timm`

安装方式：

```bash
pip install -e .
```

如需和当前训练环境尽量保持一致，可参考根目录的 `our_envs.txt`。

### 建议额外安装

某些训练配置建议安装：

```bash
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 必须自行准备的外部资产

下面这些通常不在仓库内：

- `Qwen2.5-0.5B` 本地权重目录
- `V-JEPA 2.1` checkpoint
- 基础 `base VLM` 运行目录
- RLDS 格式的数据集目录
- `vjepa21` Python 模块或本地源码环境
- 评估时的 `LIBERO` 环境目录

其中 `jepa_wam/models/jepa_vla.py` 直接依赖：

```python
from vjepa21.extractor import VJEPA21Encoder
```

如果这个模块不在当前 Python 环境里，模型会在导入阶段报错。

## 5. 数据与权重准备

完整训练通常至少需要四类路径：

- `data_root_dir`：RLDS 数据根目录，例如 `modified_libero_rlds`
- `llm_checkpoint_path`：Qwen 本地目录
- `vla.base_vlm`：基础 VLM 运行目录，里面至少要有 `config.json` 和 `checkpoints/latest-checkpoint.pt`
- `vla.vjepa_checkpoint_path`：V-JEPA 2.1 `.pt` 权重

`vla-scripts/train.py` 在 JEPA-VLA 训练时会先读取 `base_vlm` 的 projector 和 LLM 权重，再挂接新的 action head / aux head。

## 6. 训练入口

### 推荐入口

实际训练建议优先使用：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py ...
```

这个脚本支持：

- 从已有 `base_vlm` 运行目录重建 JEPA-VLA
- LoRA 训练
- full finetune / frozen vision / last-layer unfreeze 等不同 stage
- `flow_gr00t`、`flow_gr00t_jepa`、`l1` 三类动作头
- `aux head` 与 `visual token cosine` 等辅助监督
- RLDS 数据混合采样与 future frame supervision

### 一个常用训练示例

下面是按当前仓库逻辑整理过的通用模板：

```bash
export PYTHONPATH=$PWD:${PYTHONPATH:-}

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/train.py \
  --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90 \
  --vla.base_vlm /path/to/base_vlm_run \
  --vla.data_mix libero_4_task_suites_no_noops \
  --vla.vjepa_checkpoint_path /path/to/vjepa2_1_vitl.pt \
  --llm_checkpoint_path /path/to/Qwen2.5-0.5B \
  --data_root_dir /path/to/modified_libero_rlds \
  --run_root_dir ./runs \
  --vla.expected_world_size 4 \
  --vla.global_batch_size 128 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-4 \
  --vla.min_learning_rate 1e-5 \
  --vla.lr_scheduler_type linear-warmup+cosine-decay \
  --vla.warmup_ratio 0.03 \
  --vla.max_steps 60000 \
  --vla.shuffle_buffer_size 20000 \
  --vla.use_lora True \
  --vla.freeze_vision_backbone True \
  --vla.lora_rank 32 \
  --vla.lora_alpha 64 \
  --vla.lora_dropout 0.1 \
  --vla.action_head_type flow_gr00t \
  --vla.use_aux_head False \
  --use_wrist_image True \
  --save_interval 5000 \
  --seed 7
```

### 仓库里现成的训练脚本

你可以直接从这些脚本改路径和超参：

- `vla-scripts/full_run.sh`：带 `aux head` 的完整训练。
- `vla-scripts/run.sh`：JEPA-SigLIP 路线的训练脚本。
- `vla-scripts/run_jepa_only.sh`：JEPA + Flow Matching + visual token cosine。
- `vla-scripts/run_jepa_joint_head.sh`：`flow_gr00t_jepa` 联合动作头。
- `vla-scripts/run_visual_cosine.sh`：以视觉 token cosine 监督为主的变体。
- `vla-scripts/run_jepa_only_fullft.sh`：更偏 full finetune 的变体。

## 7. 评估入口

### LIBERO 仿真评估

最直接的入口是：

```bash
bash vla-scripts/run_libero_eval.sh /path/to/checkpoint.pt libero_goal 10
```

这个包装脚本最终会调用：

```bash
python experiments/robot/libero/run_libero_eval.py
```

评估前需要保证：

- `LIBERO_PATH` 指向本地 LIBERO 环境
- `llm_checkpoint_path` 可访问
- checkpoint 与训练配置兼容

支持的 task suite：

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`
- `libero_90`

### 其他实验代码

- `experiments/robot/aloha/`：ALOHA 训练、假客户端验证、真实机器人部署相关代码。
- `experiments/robot/server_deploy/`：服务式部署脚本。
- `vla-scripts/evaluate_calvin.py`：CALVIN 评估入口。

## 8. `jepa_wam` 轻量实现说明

`jepa_wam/` 这一层更像算法骨架，适合做结构验证、论文实现整理或小规模二次开发。

它包含：

- `conf/config.py`：训练、数据、视觉、LLM、head 配置 dataclass。
- `datasets/dataset.py`：RLDS 数据入口。
- `datasets/batch_transform.py`：把 RLDS 样本转成 JEPA-VLA 输入格式。
- `datasets/collator.py`：文本 padding、多视角图像堆叠、future frame 组批。
- `training/trainer.py`：简单版 trainer，负责 AMP、优化器、LR schedule、checkpoint。

需要注意的是，真正长期维护、实验脚本更丰富的仍然是 `prismatic + vla-scripts` 这一套。

## 9. 关键实验开关

训练时最值得关注的参数包括：

- `--vla.action_head_type`
  可选 `l1`、`flow_gr00t`、`flow_gr00t_jepa`
- `--vla.use_aux_head`
  是否启用未来 JEPA 表征预测辅助头
- `--vla.use_visual_token_cosine_head`
  是否启用视觉 token 对齐监督
- `--vla.future_obs_window_size`
  需要提取多少未来帧
- `--use_wrist_image`
  是否加入 wrist view
- `--vla.use_lora`
  是否只训练 LoRA / head
- `--pretrained_checkpoint`
  是否从已有训练 checkpoint 恢复

## 10. 常见问题

### 1. 为什么 `README` 里强调 `base_vlm` 是运行目录而不是单个权重文件？

因为 `vla-scripts/train.py` 在 JEPA-VLA 模式下会读取：

- `config.json`
- `checkpoints/latest-checkpoint.pt` 或 `step-*.pt`

单独给一个 `.pt` 文件通常不够。

### 2. 为什么 `jepa_wam` 和 `prismatic` 里都有类似功能？

因为这两个层次的目标不同：

- `jepa_wam` 强调结构清晰、便于单独理解 JEPA-VLA。
- `prismatic` 强调完整训练链路、历史兼容和多实验变体复用。

### 3. 训练入口应该优先用哪个？

如果你要复现实验，优先用 `vla-scripts/train.py`。  
如果你要改算法结构、整理论文实现或者快速定位模块逻辑，优先看 `jepa_wam/`。

## 11. 相关文件

- 算法说明：`architecture_spec.md`
- 包配置：`pyproject.toml`
- 环境参考：`our_envs.txt`
- ALOHA 补充文档：`experiments/robot/aloha/README.md`

## 12. License

本仓库根目录采用 `MIT License`，但训练所依赖的数据集、基础模型和外部环境各自可能有独立许可约束。使用前需要分别确认。
