#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LIBERO_PATH_VALUE="${LIBERO_PATH:-/root/linyihan/LIBERO-plus}"

export PYTHONPATH="${REPO_ROOT}:${LIBERO_PATH_VALUE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-visual-cosine-qwen25-primary}"
if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(
        find "${RUNS_DIR}" -path "*--${RUN_ID_NOTE}--*/checkpoints/latest-checkpoint.pt" -type f -printf '%T@ %p\n' \
            | sort -n \
            | tail -n 1 \
            | cut -d' ' -f2-
    )"
fi
if [[ -z "${CHECKPOINT}" || ! -e "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found. Set CHECKPOINT=/path/to/checkpoint.pt or run vla-scripts/run_visual_cosine_primary.sh first." >&2
    exit 1
fi

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
CUDA_ID="${2:-${CUDA_ID:-7}}"
CATEGORY_FILTER="${3:-${CATEGORY_FILTER:-all}}"

TRIALS=1
export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_ID}}"

QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt}"
BASE_VLM_RUN="${BASE_VLM_RUN:-/ssd/linyihan/ckpt/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_plus_config}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-flow_gr00t}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
LOAD_VISUAL_TOKEN_COSINE_HEAD="${LOAD_VISUAL_TOKEN_COSINE_HEAD:-False}"
STITCH_PRIMARY_AND_WRIST_IMAGES="${STITCH_PRIMARY_AND_WRIST_IMAGES:-False}"
ROTATION_REPRESENTATION="${ROTATION_REPRESENTATION:-axis_angle}"
EVAL_ACTION_DIM="${EVAL_ACTION_DIM:-7}"
EVAL_PROPRIO_DIM="${EVAL_PROPRIO_DIM:-14}"
USE_WANDB="${USE_WANDB:-False}"

export LIBERO_PATH="${LIBERO_PATH_VALUE}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"

mkdir -p "${LIBERO_CONFIG_DIR}"
mkdir -p eval_log

cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_PATH_VALUE}/libero/libero
bddl_files: ${LIBERO_PATH_VALUE}/libero/libero/bddl_files
init_states: ${LIBERO_PATH_VALUE}/libero/libero/init_files
datasets: ${LIBERO_PATH_VALUE}/libero/datasets
assets: ${LIBERO_PATH_VALUE}/libero/libero/assets
EOF

NAME="${NAME:-qwen25_visual_cosine_primary_14d}"
LOG_FILE="eval_log/libero_plus_${TASK_SUITE}_${NAME}_cat${CATEGORY_FILTER}_trials${TRIALS}_gpu${CUDA_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "Evaluating LIBERO-plus with action_head_type=${ACTION_HEAD_TYPE}, num_images_in_input=${NUM_IMAGES_IN_INPUT}, stitch_primary_and_wrist_images=${STITCH_PRIMARY_AND_WRIST_IMAGES}, trials=${TRIALS}, categories=${CATEGORY_FILTER}"
echo "Using CHECKPOINT=${CHECKPOINT}"
echo "Using RUN_ID_NOTE=${RUN_ID_NOTE}"
echo "Using BASE_VLM_RUN=${BASE_VLM_RUN}"
echo "Using ROTATION_REPRESENTATION=${ROTATION_REPRESENTATION}"
echo "Using EVAL_ACTION_DIM=${EVAL_ACTION_DIM}"
echo "Using EVAL_PROPRIO_DIM=${EVAL_PROPRIO_DIM}"
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
    --libero_plus_categories "${CATEGORY_FILTER}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --stitch_primary_and_wrist_images "${STITCH_PRIMARY_AND_WRIST_IMAGES}"
    --rotation_representation "${ROTATION_REPRESENTATION}"
    --eval_action_dim "${EVAL_ACTION_DIM}"
    --eval_proprio_dim "${EVAL_PROPRIO_DIM}"
    --use_proprio True
    --action_head_type "${ACTION_HEAD_TYPE}"
    --load_visual_token_cosine_head "${LOAD_VISUAL_TOKEN_COSINE_HEAD}"
    --use_minivlm True
    --center_crop False
    --use_aux_head False
    --use_wandb "${USE_WANDB}"
    --wandb_project vla_libero_plus
    --save_version vla-adapter
    --log_per_task_metrics True
)

[[ -n "${BASE_VLM_RUN}" ]] && CMD+=(--base_vlm "${BASE_VLM_RUN}")
[[ -n "${SIGLIP_LOCAL_PATH}" ]] && CMD+=(--siglip_local_path "${SIGLIP_LOCAL_PATH}")

# "${CMD[@]}"
"${CMD[@]}" > "${LOG_FILE}" 2>&1 &
echo "Running in background. PID: $!"
echo "Log: ${LOG_FILE}"
