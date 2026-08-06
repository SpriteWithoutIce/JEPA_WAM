<div align="center">
  <img src="docs/assets/logo-ai.webp" alt="JEPA-WAM logo" width="140" />
  <h1>JEPA-WAM</h1>
  <h3>Learning Vision-Language-Action Policies with Joint-Embedding World Modeling</h3>

  <p>
    <a href="https://spritewithoutice.github.io/JEPA_WAM/">
      <img src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge" alt="Project page" />
    </a>
    <a href="https://huggingface.co/CokeAnd1ce/JEPA_WAM">
      <img src="https://img.shields.io/badge/Models-Hugging_Face-ffd21e?style=for-the-badge" alt="Hugging Face models" />
    </a>
    <a href="docs/assets/demo.mp4">
      <img src="https://img.shields.io/badge/Demo-Video-16a34a?style=for-the-badge" alt="Demo video" />
    </a>
    <img src="https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b?style=for-the-badge" alt="arXiv coming soon" />
  </p>
</div>

<p align="center">
  <img src="docs/assets/teaser.webp" alt="JEPA-WAM overview" width="760" />
</p>

JEPA-WAM is a vision-language-action policy that adds joint-embedding world-model supervision to robot policy
learning. Instead of reconstructing future RGB observations, it aligns policy visual states with future features from a
frozen V-JEPA 2.1 encoder. This provides transition-aware supervision during training without adding an image decoder
or extra perception pass at deployment time.

This repository releases the fixed JEPA-WAM recipe used for LIBERO training and LIBERO-Plus evaluation. It contains
one training launcher, one evaluation launcher, the model implementation, and focused regression tests.

## Release Status

- [x] Training and LIBERO-Plus evaluation code
- [x] Pretrained base VLM and LIBERO policy checkpoint
- [x] Project page and demo
- [ ] arXiv paper and BibTeX

## Highlights

- Frozen V-JEPA 2.1 ViT-L encoder for primary and wrist observations.
- Qwen2.5-0.5B policy backbone adapted with LoRA.
- GR00T-style flow-matching head for continuous action chunks.
- Dense cosine alignment between policy visual tokens and paired future V-JEPA targets.
- Reproducible launchers with full and end-to-end smoke-test modes.

## Contents

- [Method](#method)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Models](#pretrained-models)
- [Path Configuration](#path-configuration)
- [Training](#training)
- [LIBERO-Plus Evaluation](#libero-plus-evaluation)
- [Results](#results)
- [Verification](#verification)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Method

<p align="center">
  <img src="docs/assets/method-balanced.webp" alt="JEPA-WAM method" width="900" />
</p>

For each training sample, the current primary and wrist observations are encoded by a frozen V-JEPA 2.1 encoder. The
resulting visual tokens are projected into Qwen2.5 and concatenated with the language instruction and learned action
placeholder tokens. Qwen keeps its native causal attention mask.

The final action-placeholder states condition a flow-matching action head. In parallel, a two-layer MLP projects the
final Qwen visual states back to the V-JEPA embedding dimension and aligns them with detached paired-frame targets:

```text
loss = action_flow_matching_loss + 0.5 * visual_token_cosine_loss
```

The auxiliary predictor is used only for training. See [architecture_spec.md](architecture_spec.md) for the tensor
shapes and complete model contract.

## Installation

The released environment was tested with Python 3.10.16, PyTorch 2.2.0, and CUDA 12.1. The full training recipe uses
eight GPUs; the smoke-test mode runs the same model and data path on one GPU for one optimizer step.

### 1. Create the environment

```bash
git clone https://github.com/SpriteWithoutIce/JEPA_WAM.git
cd JEPA_WAM

conda create -n jepa-wam python=3.10.16 -y
conda activate jepa-wam
```

### 2. Install PyTorch and dependencies

Install PyTorch first so FlashAttention can build against the active PyTorch installation:

```bash
pip install torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt --no-build-isolation
pip install -e .
pip check
```

The checked-in [requirements.txt](requirements.txt) is the tested Python 3.10/CUDA 12.1 runtime lock. It pins the
OpenVLA `dlimp` fork to the exact commit used during validation.

<details>
<summary>Tested core versions</summary>

| Component | Version |
|---|---|
| Python | 3.10.16 |
| PyTorch / torchvision | 2.2.0 / 0.17.0 |
| CUDA | 12.1 |
| Transformers | 4.57.0 |
| PEFT | 0.13.2 |
| FlashAttention | 2.7.4.post1 |
| TensorFlow / TFDS | 2.15.0 / 4.9.3 |

</details>

Model weights, datasets, and simulator assets are not stored in this Git repository.

## Data Preparation

JEPA-WAM uses the no-op-filtered LIBERO datasets in RLDS format for training and the official LIBERO-Plus simulator
and assets for robustness evaluation.

### LIBERO RLDS training data

Download [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds):

```bash
DATA_ROOT=/path/to/datasets

hf download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}/modified_libero_rlds"
```

The directory passed as `LIBERO_DATA` must contain these four TFDS datasets:

```text
modified_libero_rlds/
├── libero_spatial_no_noops/
├── libero_object_no_noops/
├── libero_goal_no_noops/
└── libero_10_no_noops/
```

### LIBERO-Plus benchmark

Clone the official [LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus) and install it as an editable
package:

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git /path/to/LIBERO-plus
pip install -e /path/to/LIBERO-plus
pip install -r /path/to/LIBERO-plus/extra_requirements.txt
```

LIBERO-Plus also requires its extended simulator assets. Download `assets.zip` from the official
[Sylvest/LIBERO-plus dataset repository](https://huggingface.co/datasets/Sylvest/LIBERO-plus):

```bash
hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset \
  --local-dir /path/to/libero_plus_assets

unzip /path/to/libero_plus_assets/assets.zip \
  -d /path/to/LIBERO-plus/libero/libero
```

After extraction, `/path/to/LIBERO-plus/libero/libero/assets` should contain the additional objects, scenes, and
textures. Refer to the LIBERO-Plus installation guide for its system packages if MuJoCo or ImageMagick dependencies
are missing on your machine.

## Pretrained Models

JEPA-WAM needs three groups of weights: the official Qwen2.5 language model, the official V-JEPA 2.1 visual encoder,
and the JEPA-WAM base VLM/policy checkpoints.

Set a common download root first:

```bash
ASSET_ROOT=/path/to/jepa_wam_assets
mkdir -p "${ASSET_ROOT}"
```

### Qwen2.5-0.5B

Download from the official [Qwen model repository](https://huggingface.co/Qwen/Qwen2.5-0.5B):

```bash
hf download Qwen/Qwen2.5-0.5B \
  --local-dir "${ASSET_ROOT}/Qwen2.5-0.5B"
```

### V-JEPA 2.1 ViT-L/16

Download the 384px ViT-L checkpoint listed by the official
[V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints):

```bash
mkdir -p "${ASSET_ROOT}/vjepa2"
wget \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -O "${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
```

### JEPA-WAM base VLM and policy

The released checkpoints are hosted at [CokeAnd1ce/JEPA_WAM](https://huggingface.co/CokeAnd1ce/JEPA_WAM). The command
below downloads the files required by the public training and evaluation launchers:

```bash
hf download CokeAnd1ce/JEPA_WAM \
  "checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7/config.json" \
  "checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7/checkpoints/latest-checkpoint.pt" \
  "checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/config.json" \
  "checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/dataset_statistics.json" \
  "checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt" \
  --local-dir "${ASSET_ROOT}/JEPA_WAM"
```

The base VLM directory must retain this structure:

```text
prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7/
├── config.json
└── checkpoints/
    └── latest-checkpoint.pt
```

## Path Configuration

The launchers do not contain machine-specific paths. Export the following variables after downloading the data and
weights:

```bash
export ASSET_ROOT=/path/to/jepa_wam_assets
export LIBERO_DATA=/path/to/datasets/modified_libero_rlds
export LIBERO_PATH=/path/to/LIBERO-plus

export QWEN_PATH="${ASSET_ROOT}/Qwen2.5-0.5B"
export VJEPA_CKPT="${ASSET_ROOT}/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
export BASE_VLM_RUN="${ASSET_ROOT}/JEPA_WAM/checkpoints/pretrained_vlm/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7"
export CHECKPOINT="${ASSET_ROOT}/JEPA_WAM/checkpoints/libero/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--visual-cosine-projector-allviews--20260723_232305/checkpoints/step-040000-epoch-37-loss=0.0262.pt"
```

| Variable | Used by | Description |
|---|---|---|
| `LIBERO_DATA` | Training | Root containing the four modified LIBERO RLDS datasets |
| `QWEN_PATH` | Both | Local Qwen2.5-0.5B directory |
| `VJEPA_CKPT` | Both | V-JEPA 2.1 ViT-L checkpoint file |
| `BASE_VLM_RUN` | Both | Pretrained JEPA-WAM base VLM run directory |
| `LIBERO_PATH` | Evaluation | LIBERO-Plus repository checkout |
| `CHECKPOINT` | Evaluation | Trained or released JEPA-WAM policy checkpoint |

## Training

### Full training recipe

The public launcher implements the fixed eight-GPU configuration used by this release:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash vla-scripts/run_visual_cosine_primary.sh
```

| Setting | Value |
|---|---:|
| GPUs | 8 |
| Global / per-device batch size | 256 / 32 |
| Training steps | 40,000 |
| Learning rate / minimum learning rate | 2e-4 / 1e-5 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.1 |
| Action horizon | 8 |
| Paired-frame offset | 31 |
| Visual cosine weight | 0.5 |
| Random seed | 7 |

Checkpoints and run metadata are written to `RUNS_DIR` (default: `./runs`). Console logs are written to `LOG_DIR`
(default: `./logs`). Both locations can be overridden with environment variables.

### One-step training smoke test

Use smoke mode to validate model loading, all four datasets, forward/backward, the optimizer step, and checkpoint
writing on one GPU:

```bash
SMOKE_TEST=1 \
CUDA_VISIBLE_DEVICES=0 \
RUNS_DIR=/tmp/jepa_wam_smoke \
LOG_DIR=/tmp/jepa_wam_smoke_logs \
bash vla-scripts/run_visual_cosine_primary.sh
```

Smoke mode changes only world size, batch size, shuffle-buffer size, and maximum steps. It uses the same model, loss,
data mixture, pretrained weights, and checkpoint code as the full run.

To inspect the resolved command without starting training, set `DRY_RUN=1`.

## LIBERO-Plus Evaluation

### Evaluate a checkpoint

The following command evaluates the released policy on the LIBERO-Plus spatial suite across all perturbation
categories:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash vla-scripts/libero_plus.sh \
  "${CHECKPOINT}" \
  libero_spatial \
  all \
  1
```

The positional arguments are:

```text
bash vla-scripts/libero_plus.sh CHECKPOINT TASK_SUITE CATEGORIES TRIALS
```

Supported task suites are `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, and `libero_90`.
`CATEGORIES` accepts `all` or a comma-separated list of `camera`, `robot`, `language`, `light`, `background`, `sensor`,
and `layout`.

When `CHECKPOINT` is omitted, the launcher selects the newest
`RUNS_DIR/*/checkpoints/latest-checkpoint.pt`. Evaluation logs are saved under `experiments/logs`, and rollout videos
are saved under `rollout`.

### Environment-to-action smoke test

This diagnostic run constructs one LIBERO-Plus environment and executes one predicted action:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_TASKS=1 \
MAX_EPISODE_STEPS=1 \
SAVE_ROLLOUTS=False \
bash vla-scripts/libero_plus.sh "${CHECKPOINT}" libero_spatial all 1
```

`MAX_TASKS` and `MAX_EPISODE_STEPS` are disabled by default and should not be set for full benchmark evaluation.
Set `DRY_RUN=1` to print the evaluator command without launching MuJoCo.

## Results

JEPA-WAM reaches an average success rate of **79.2%** across the seven LIBERO-Plus distribution-shift categories.

| Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 79.2 | 59.2 | 68.2 | 93.3 | 94.6 | 83.6 | 76.1 | **79.2** |

RoboTwin 2.0, ablation, and real-world experiment tables are available on the
[project page](https://spritewithoutice.github.io/JEPA_WAM/).

## Verification

The complete workflow was exercised in the reference environment on an H100 GPU:

- `requirements.txt` resolved successfully and `pip check` reported no dependency conflicts.
- A one-step run loaded Qwen, V-JEPA, the base VLM, and all four LIBERO RLDS datasets.
- Forward, backward, optimizer update, and checkpoint save completed with a smoke-test loss of `1.4728`.
- The evaluator reconstructed a compatible checkpoint and produced an action in a LIBERO-Plus environment.

Run the focused repository checks with:

```bash
python -m pytest tests/test_visual_token_cosine.py
bash -n vla-scripts/run_visual_cosine_primary.sh
bash -n vla-scripts/libero_plus.sh
```

## Repository Structure

```text
JEPA_WAM/
├── prismatic/                         # model, data, training, and checkpoint code
│   └── training/train.py              # distributed JEPA-WAM training entry point
├── experiments/robot/libero/          # LIBERO-Plus evaluator and environment helpers
├── vla-scripts/
│   ├── run_visual_cosine_primary.sh   # fixed training launcher
│   └── libero_plus.sh                 # LIBERO-Plus evaluation launcher
├── tests/                              # architecture regression tests
├── docs/                               # GitHub Pages project site
├── requirements.txt                    # tested Python 3.10/CUDA 12.1 runtime lock
└── architecture_spec.md               # released architecture specification
```

## Citation

The paper and BibTeX entry will be added when the arXiv version is available.

## Acknowledgements

JEPA-WAM builds on [V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms), [OpenVLA](https://github.com/openvla/openvla), and
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus). We thank the authors for releasing their code, models,
datasets, and benchmarks.

## License

The code is released under the [MIT License](LICENSE). Third-party models, datasets, and simulators remain subject to
their respective licenses.
