#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="/root/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b32+x7--llava-lvis4v-lrv-vjepa--20260528_222015/checkpoints/latest-checkpoint.pt"

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
TRIALS="${2:-${TRIALS:-10}}"
CUDA_ID="${3:-${CUDA_ID:-7}}"

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

LIBERO_PATH="${LIBERO_PATH:-/root/linyihan/LIBERO}"
QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-flow_gr00t}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
LOAD_VISUAL_TOKEN_COSINE_HEAD="${LOAD_VISUAL_TOKEN_COSINE_HEAD:-False}"

LIBERO_PATH_VALUE="${LIBERO_PATH:-/root/linyihan/LIBERO}"
export LIBERO_PATH="${LIBERO_PATH_VALUE}"

mkdir -p eval_log

LOG_FILE="eval_log/${TASK_SUITE}_trials${TRIALS}_gpu${CUDA_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluating checkpoint with action_head_type=${ACTION_HEAD_TYPE}, num_images_in_input=${NUM_IMAGES_IN_INPUT}"
echo "Logging to ${LOG_FILE}"

CMD=(
    "${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py
    --model_family openvla
    --pretrained_checkpoint "${CHECKPOINT}"
    --llm_checkpoint_path "${QWEN_PATH}"
    --vjepa_checkpoint_path "${VJEPA_CKPT}"
    --task_suite_name "${TASK_SUITE}"
    --num_trials_per_task "${TRIALS}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --use_proprio True
    --action_head_type "${ACTION_HEAD_TYPE}"
    --load_visual_token_cosine_head "${LOAD_VISUAL_TOKEN_COSINE_HEAD}"
    --use_minivlm True
    --center_crop False
    --use_aux_head False
    --use_wandb True
    --wandb_project vla_libero
    --save_version vla-adapter
    --log_per_task_metrics True
)

[[ -n "${SIGLIP_LOCAL_PATH}" ]] && CMD+=(--siglip_local_path "${SIGLIP_LOCAL_PATH}")

"${CMD[@]}" > "${LOG_FILE}" 2>&1 &
echo "Running in background. PID: $!"
echo "Log: ${LOG_FILE}"