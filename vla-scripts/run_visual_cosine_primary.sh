#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_jepa_visual_cosine_primary_$(date +%Y%m%d_%H%M%S).log"
: "${LIBERO_DATA:?Set LIBERO_DATA to the modified LIBERO RLDS dataset root}"
: "${QWEN_PATH:?Set QWEN_PATH to the Qwen2.5-0.5B checkpoint directory}"
: "${VJEPA_CKPT:?Set VJEPA_CKPT to the V-JEPA 2.1 ViT-L checkpoint}"
: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the pretrained Qwen2.5 + V-JEPA VLM run directory}"
RUNS_DIR="${RUNS_DIR:-./runs}"

CMD=(
    torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py
    --vla.base_vlm "${BASE_VLM_RUN}"
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --llm_checkpoint_path "${QWEN_PATH}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir "${RUNS_DIR}"
    --run_id_note visual-cosine-projector-allviews
    --save_interval 10000
    --seed 7
    --use_swanlab True
    --debug_memory_stats False
    --debug_batch_shapes False
)

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
