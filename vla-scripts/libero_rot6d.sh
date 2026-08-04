#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIBERO_PATH_VALUE="${LIBERO_PATH:-/root/linyihan/LIBERO}"

export PYTHONPATH="${REPO_ROOT}:${LIBERO_PATH_VALUE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-visual-cosine-qwen25-primary-rot6d}"
CHECKPOINT="${CHECKPOINT:-/root/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b16+x7--visual-cosine-qwen25-primary-rot6d--20260602_221728/checkpoints/latest-checkpoint.pt}"

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
TRIALS="${2:-${TRIALS:-10}}"
CUDA_ID="${3:-${CUDA_ID:-7}}"

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_ID}}"

QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt}"
BASE_VLM_RUN="${BASE_VLM_RUN:-/ssd/linyihan/ckpt/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-flow_gr00t}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
LOAD_VISUAL_TOKEN_COSINE_HEAD="${LOAD_VISUAL_TOKEN_COSINE_HEAD:-False}"
STITCH_PRIMARY_AND_WRIST_IMAGES="${STITCH_PRIMARY_AND_WRIST_IMAGES:-False}"
ROTATION_REPRESENTATION="${ROTATION_REPRESENTATION:-rot6d}"
USE_WANDB="${USE_WANDB:-False}"

export LIBERO_PATH="${LIBERO_PATH_VALUE}"
NAME="${NAME:-qwen25_visual_cosine_primary_rot6d}"
mkdir -p eval_log

LOG_FILE="eval_log/${TASK_SUITE}_${NAME}_trials${TRIALS}_gpu${CUDA_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluating LIBERO rot6d checkpoint with action_head_type=${ACTION_HEAD_TYPE}, num_images_in_input=${NUM_IMAGES_IN_INPUT}, stitch_primary_and_wrist_images=${STITCH_PRIMARY_AND_WRIST_IMAGES}, trials=${TRIALS}"
echo "Using CHECKPOINT=${CHECKPOINT}"
echo "Using LIBERO_PATH=${LIBERO_PATH}"
echo "Using QWEN_PATH=${QWEN_PATH}"
echo "Using VJEPA_CKPT=${VJEPA_CKPT}"
echo "Using BASE_VLM_RUN=${BASE_VLM_RUN}"
echo "Using ROTATION_REPRESENTATION=${ROTATION_REPRESENTATION}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
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
    --stitch_primary_and_wrist_images "${STITCH_PRIMARY_AND_WRIST_IMAGES}"
    --rotation_representation "${ROTATION_REPRESENTATION}"
    --use_proprio True
    --action_head_type "${ACTION_HEAD_TYPE}"
    --load_visual_token_cosine_head "${LOAD_VISUAL_TOKEN_COSINE_HEAD}"
    --use_minivlm True
    --center_crop False
    --use_aux_head False
    --use_wandb "${USE_WANDB}"
    --wandb_project vla_libero
    --save_version vla-adapter
    --log_per_task_metrics True
)

[[ -n "${BASE_VLM_RUN}" ]] && CMD+=(--base_vlm "${BASE_VLM_RUN}")
[[ -n "${SIGLIP_LOCAL_PATH}" ]] && CMD+=(--siglip_local_path "${SIGLIP_LOCAL_PATH}")

"${CMD[@]}" > "${LOG_FILE}" 2>&1 &
echo "Running in background. PID: $!"
echo "Log: ${LOG_FILE}"
