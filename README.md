<div align="center">
  <img src="docs/assets/logo-ai.webp" alt="JEPA-WAM logo" width="140" />
  <h1>JEPA-WAM</h1>
  <h3>Learning Vision-Language-Action Policies with Joint-Embedding World Modeling</h3>

  <p>
    <a href="https://spritewithoutice.github.io/JEPA_WAM/"><strong>Project Page</strong></a>
    &nbsp;|&nbsp;
    <strong>arXiv (Coming Soon)</strong>
    &nbsp;|&nbsp;
    <a href="https://github.com/SpriteWithoutIce/JEPA_WAM"><strong>Code</strong></a>
  </p>
</div>

<p align="center">
  <img src="docs/assets/teaser.webp" alt="JEPA-WAM overview" width="760" />
</p>

JEPA-WAM is a vision-language-action policy that learns robot control with joint-embedding world modeling. Instead of
generating future RGB observations, the released recipe supervises the policy in the latent space of a frozen V-JEPA
2.1 encoder. The same Qwen representation used for action prediction is aligned with paired future visual tokens,
providing transition-aware supervision without an image decoder at deployment time.

This repository contains one fixed public recipe for LIBERO training and LIBERO-Plus evaluation. Experimental
backbones, alternative action heads, and unrelated robot platforms are intentionally excluded.

## Highlights

- Frozen V-JEPA 2.1 ViT-L visual encoder with primary and wrist camera inputs.
- Qwen2.5-0.5B multimodal policy adapted with LoRA.
- GR00T-style flow-matching head for continuous action chunks.
- Dense visual-token cosine supervision from paired V-JEPA targets.
- One reproducible training launcher and one LIBERO-Plus evaluation launcher.

## Method

<p align="center">
  <img src="docs/assets/method-balanced.webp" alt="JEPA-WAM method" width="900" />
</p>

For every training sample, JEPA-WAM processes the current primary and wrist observations with a frozen V-JEPA 2.1
encoder. The visual tokens are projected into Qwen2.5, followed by the language instruction and learned action
placeholder tokens. Qwen keeps its native causal attention mask.

The final action-placeholder states condition a flow-matching action head. In parallel, a two-layer MLP maps the final
Qwen visual states back to the V-JEPA embedding dimension and aligns them with detached paired-frame targets:

```text
loss = action_flow_matching_loss + 0.5 * visual_token_cosine_loss
```

See [architecture_spec.md](architecture_spec.md) for the tensor shapes and the complete model contract.

## Installation

The checked-in [requirements.txt](requirements.txt) is a focused runtime lock derived from the working
`/ssd_node5/jepa_copy` environment:

| Component | Tested version |
|---|---|
| Python | 3.10.16 |
| PyTorch | 2.2.0 + CUDA 12.1 |
| torchvision | 0.17.0 |
| Transformers | 4.57.0 |
| PEFT | 0.13.2 |
| FlashAttention | 2.7.4.post1 |
| TensorFlow / TFDS | 2.15.0 / 4.9.3 |
| LIBERO-Plus | 0.1.0 |

Install PyTorch first so that FlashAttention can build against it, then install the locked dependencies without build
isolation:

```bash
git clone https://github.com/SpriteWithoutIce/JEPA_WAM.git
cd JEPA_WAM

conda create -n jepa-wam python=3.10.16 -y
conda activate jepa-wam

pip install torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
pip install -e /path/to/LIBERO-Plus
pip install -e .
pip check
```

The requirements file pins the OpenVLA `dlimp` fork to the exact commit tested by this repository. LIBERO-Plus remains
an editable external checkout because its benchmark tasks, BDDL files, initial states, and assets are required at
runtime.

Model weights, datasets, and simulator assets are not included in this repository.

## Required Assets

The launchers contain the tested server paths below. Every value can be overridden with an environment variable.

| Variable | Tested default |
|---|---|
| `JEPA_ENV` | `/ssd_node5/jepa_copy` |
| `LIBERO_DATA` | `/ssd/linyihan/datasets/modified_libero_rlds` |
| `QWEN_PATH` | `/ssd/linyihan/ckpt/Qwen2.5-0.5B` |
| `VJEPA_CKPT` | `/ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt` |
| `BASE_VLM_RUN` | `/ssd/linyihan/ckpt/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7` |
| `LIBERO_PATH` | `/root/linyihan/LIBERO-plus` |

`BASE_VLM_RUN` must contain the pretrained language-model and vision-projector weights:

```text
base_vlm_run/
├── config.json
└── checkpoints/
    └── latest-checkpoint.pt
```

The training data root must expose the four no-op-filtered RLDS datasets used by the fixed mixture:

```text
libero_spatial_no_noops
libero_object_no_noops
libero_goal_no_noops
libero_10_no_noops
```

## Training

On the tested server, the fixed eight-GPU recipe can be launched directly because all asset paths are already defined
in the script:

```bash
bash vla-scripts/run_visual_cosine_primary.sh
```

On another machine, override the same paths before launching:

```bash
export LIBERO_DATA=/path/to/modified_libero_rlds
export QWEN_PATH=/path/to/Qwen2.5-0.5B
export VJEPA_CKPT=/path/to/vjepa2_1_vitl_checkpoint.pt
export BASE_VLM_RUN=/path/to/pretrained_qwen25_vjepa_vlm_run

bash vla-scripts/run_visual_cosine_primary.sh
```

Run a complete one-step training smoke test on one GPU with:

```bash
SMOKE_TEST=1 \
CUDA_VISIBLE_DEVICES=0 \
RUNS_DIR=/tmp/jepa_wam_smoke \
LOG_DIR=/tmp/jepa_wam_smoke_logs \
bash vla-scripts/run_visual_cosine_primary.sh
```

The smoke mode overrides only world size, batch size, shuffle buffer, and max steps. The model, loss, LIBERO mixture,
V-JEPA/Qwen/base-VLM loading path, forward pass, backward pass, optimizer step, and checkpoint writer are unchanged.

The fixed configuration uses:

| Setting | Value |
|---|---:|
| GPUs | 8 |
| Global batch size | 256 |
| Per-device batch size | 32 |
| Training steps | 40,000 |
| Learning rate | 2e-4 |
| Minimum learning rate | 1e-5 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.1 |
| Action horizon | 8 |
| Paired-frame offset | 31 |
| Visual cosine weight | 0.5 |
| Random seed | 7 |

Checkpoints and run metadata are written below `RUNS_DIR`, which defaults to `./runs`. Console logs are written below
`LOG_DIR`, which defaults to `./logs`.

## LIBERO-Plus Evaluation

The evaluation launcher also contains tested defaults for the environment, reconstruction assets, and a local
JEPA-WAM checkpoint. On the tested server:

```bash
CUDA_VISIBLE_DEVICES=7 bash vla-scripts/libero_plus.sh
```

To evaluate another checkpoint or run on another machine:

```bash
export BASE_VLM_RUN=/path/to/pretrained_qwen25_vjepa_vlm_run
export QWEN_PATH=/path/to/Qwen2.5-0.5B
export VJEPA_CKPT=/path/to/vjepa2_1_vitl_checkpoint.pt
export LIBERO_PATH=/path/to/LIBERO-Plus
export CUDA_VISIBLE_DEVICES=0

bash vla-scripts/libero_plus.sh \
  ./runs/<run-name>/checkpoints/latest-checkpoint.pt \
  libero_spatial \
  all \
  1
```

The positional arguments are `CHECKPOINT`, `TASK_SUITE`, `CATEGORIES`, and `TRIALS`. Supported suites are
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, and `libero_90`. `CATEGORIES` can be `all` or a
comma-separated list using the aliases `camera`, `robot`, `language`, `light`, `background`, `sensor`, and `layout`.
When `CHECKPOINT` is omitted, the launcher selects the newest `RUNS_DIR/*/checkpoints/latest-checkpoint.pt`, then falls
back to the tested server checkpoint.

For a fast end-to-end check that constructs one environment and executes one predicted action:

```bash
CHECKPOINT=/path/to/checkpoints/latest-checkpoint.pt \
CUDA_VISIBLE_DEVICES=0 \
MAX_TASKS=1 \
MAX_EPISODE_STEPS=1 \
SAVE_ROLLOUTS=False \
bash vla-scripts/libero_plus.sh
```

`MAX_TASKS` and `MAX_EPISODE_STEPS` are diagnostic limits only. They default to disabled for the full benchmark.

Evaluation logs are saved under `experiments/logs`, and rollout videos are saved under `rollout`.

## End-to-End Validation

The complete path was exercised in the reference environment on an H100 GPU:

- `requirements.txt` resolved successfully, including the pinned `dlimp` commit, and `pip check` reported no conflicts.
- A one-step training run loaded V-JEPA, Qwen, the base VLM, and all four LIBERO RLDS datasets.
- The training smoke test completed forward, backward, optimizer update, and checkpoint save with loss `1.4728`.
- The saved checkpoint was reconstructed by the evaluator and produced an action in a LIBERO-Plus environment.
- The launcher's built-in JEPA-WAM checkpoint also completed the same environment-to-action smoke test.

## Results

JEPA-WAM reaches an average success rate of **79.2%** across the seven LIBERO-Plus distribution-shift categories.

| Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 79.2 | 59.2 | 68.2 | 93.3 | 94.6 | 83.6 | 76.1 | **79.2** |

Additional RoboTwin 2.0, ablation, and real-world results are available on the
[project page](https://spritewithoutice.github.io/JEPA_WAM/).

## Repository Structure

```text
JEPA_WAM/
├── prismatic/                         # model, data, training, and checkpoint code
│   └── training/train.py              # fixed distributed training implementation
├── experiments/robot/libero/          # LIBERO-Plus evaluator and environment helpers
├── vla-scripts/
│   ├── run_visual_cosine_primary.sh   # training launcher
│   └── libero_plus.sh                 # evaluation launcher
├── tests/                              # focused architecture regression tests
├── docs/                               # GitHub Pages project site
├── requirements.txt                    # tested Python 3.10 / CUDA 12.1 runtime lock
└── architecture_spec.md               # fixed public architecture specification
```

## Verification

```bash
pip install --dry-run --no-build-isolation -r requirements.txt
pip check
python -m pytest tests/test_visual_token_cosine.py
bash -n vla-scripts/run_visual_cosine_primary.sh
bash -n vla-scripts/libero_plus.sh
```

## Checkpoints

Pretrained JEPA-WAM checkpoints and the base VLM checkpoint will be released soon.

## Citation

The paper and BibTeX entry will be added when the arXiv version is available.

## Acknowledgements

This codebase builds on ideas and components from V-JEPA, Qwen, Prismatic/OpenVLA, GR00T-style flow matching, and
LIBERO. We thank the authors of these projects for making their work available to the research community.

## License

The code is released under the [MIT License](LICENSE). Third-party models, datasets, and simulators are distributed
under their respective licenses.
