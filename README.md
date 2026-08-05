# JEPA-WAM

JEPA-WAM is a Vision-Language-Action model for LIBERO manipulation. The public training recipe in this repository uses one fixed architecture:

- frozen V-JEPA 2.1 ViT-L visual encoder;
- Qwen2.5-0.5B with native causal attention and LoRA;
- primary and wrist camera views;
- a GR00T-style flow-matching action head;
- visual-token cosine supervision from paired V-JEPA targets.

The supported entry point is `vla-scripts/run_visual_cosine_primary.sh`.

## Architecture

```text
current primary + wrist images
            |
            v
  frozen V-JEPA 2.1 encoder
            |
            v
      vision projector
            |
            v
Qwen2.5 visual tokens + text + action placeholders
            |
            | native Qwen causal attention
            |
            +--------------------------+
            |                          |
            v                          v
 action placeholder states     final-layer visual states
            |                          |
            v                          v
  Flow GR00T action head       Linear -> GELU -> Linear
            |                          |
            v                          v
       action loss          cosine loss against detached
                            paired V-JEPA embeddings
```

The model does not replace Qwen's causal mask with bidirectional vision-text attention.

For visual-token supervision, the final Qwen visual hidden states are mapped from the Qwen hidden dimension to the V-JEPA dimension by a two-layer MLP. The target is the raw, detached V-JEPA embedding from the paired frame. It is not passed through the VLM vision projector.

The training objective is:

```text
loss = action_flow_matching_loss + 0.5 * visual_token_cosine_loss
```

See `architecture_spec.md` for the tensor-level definition.

## Repository Layout

```text
JEPA-WAM/
├── prismatic/                         # model, data, training, and checkpoint code
├── vla-scripts/
│   ├── train.py                       # distributed training entry point
│   ├── run_visual_cosine_primary.sh   # supported public recipe
│   └── run_libero_eval.sh             # standard LIBERO evaluation launcher
├── experiments/robot/libero/          # LIBERO evaluation code
├── tests/                              # focused architecture regression tests
└── architecture_spec.md               # fixed public architecture specification
```

## Environment

The code targets Python 3.10 and PyTorch 2.2+. Install the Python dependencies and this package with:

```bash
pip install -e .
```

RLDS loading additionally requires the OpenVLA `dlimp` package, installed from its source checkout. LIBERO is only
required for evaluation and should be installed from the official LIBERO repository. Model and environment assets are
not vendored here.

## Required Assets

Training requires four external paths:

- `LIBERO_DATA`: modified LIBERO RLDS dataset root;
- `QWEN_PATH`: local Qwen2.5-0.5B checkpoint directory;
- `VJEPA_CKPT`: V-JEPA 2.1 ViT-L checkpoint;
- `BASE_VLM_RUN`: pretrained Qwen2.5 + V-JEPA VLM run directory.

`BASE_VLM_RUN` must contain its model configuration and a base checkpoint with the pretrained Qwen and vision-projector weights.

## Training

Set the required paths and launch the fixed eight-GPU recipe:

```bash
export LIBERO_DATA=/path/to/modified_libero_rlds
export QWEN_PATH=/path/to/Qwen2.5-0.5B
export VJEPA_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export BASE_VLM_RUN=/path/to/pretrained_qwen25_vjepa_vlm_run

bash vla-scripts/run_visual_cosine_primary.sh
```

The script uses:

| Setting | Value |
|---|---:|
| GPUs | 8 |
| Global batch size | 256 |
| Per-device batch size | 32 |
| Max steps | 40,000 |
| Learning rate | 2e-4 |
| Minimum learning rate | 1e-5 |
| LoRA rank / alpha / dropout | 32 / 64 / 0.1 |
| Paired-frame offset | 31 |
| Visual cosine weight | 0.5 |
| Seed | 7 |

Run outputs are written below `RUNS_DIR`, which defaults to `./runs`.

SwanLab logging is enabled by the launch script. Remove `--use_swanlab True` to keep only local JSONL metrics.

## Evaluation

Set the local model and LIBERO paths, then evaluate a native JEPA-WAM checkpoint:

```bash
export BASE_VLM_RUN=/path/to/pretrained_qwen25_vjepa_vlm_run
export LLM_PATH=/path/to/Qwen2.5-0.5B
export VJEPA_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export LIBERO_PATH=/path/to/LIBERO

bash vla-scripts/run_libero_eval.sh \
  ./runs/<run>/checkpoints/latest-checkpoint.pt libero_goal 10
```

## Fixed Model Contract

The registered `jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90` configuration fixes the architecture used by the public recipe:

- V-JEPA is frozen;
- Qwen is adapted with LoRA;
- Qwen uses its original causal attention;
- action head type is `flow_gr00t`;
- the action head consumes action-placeholder hidden states, not the full Qwen sequence;
- there is no auxiliary decoder or future-observation prediction branch;
- primary and wrist views are enabled;
- visual-token cosine supervision is enabled;
- the cosine head is always the two-layer MLP;
- the V-JEPA target is raw and detached;
- rotation representation is axis-angle.

## Verification

The focused tests can be run with:

```bash
python -m pytest tests/test_visual_token_cosine.py
bash -n vla-scripts/run_visual_cosine_primary.sh
bash -n vla-scripts/run_libero_eval.sh
```

## Public Scope

This release intentionally excludes unrelated experimental backbones, alternate action heads, ALOHA/RoboTwin/CALVIN
training code, LeRobot integrations, LIBERO-PRO utilities, deployment servers, local tokenizer copies, and pretrained
weights. The `docs/` directory is retained solely for the project page.

## License

The repository is released under the MIT License. Dataset, environment, Qwen, and V-JEPA assets may have separate licenses.
