#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLA_ROBOT_PLATFORM="ALOHA"

LOG_DIR="${LOG_DIR:-./logs}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-aloha-robotwin-15-full-llm-projector-action-query}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_${RUN_ID_NOTE}_$(date +%Y%m%d_%H%M%S).log"

ALOHA_DATA="${ALOHA_DATA:-/home/linyihan/datasets/aloha_dataset}"
DATA_MIX="${DATA_MIX:-aloha_robotwin_all}"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
ROBOTWIN_ALOHA_MOSAIC="${ROBOTWIN_ALOHA_MOSAIC:-True}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/linyihan/jepa_copy/bin/torchrun}"
VLM_LEARNING_RATE="${VLM_LEARNING_RATE:-2e-5}"
VLM_MIN_LEARNING_RATE="${VLM_MIN_LEARNING_RATE:-1e-6}"
ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-2e-4}"
ACTION_MIN_LEARNING_RATE="${ACTION_MIN_LEARNING_RATE:-1e-5}"

VISUAL_TOKEN_PAIR_OFFSET="${VISUAL_TOKEN_PAIR_OFFSET:-31}"
VISUAL_TOKEN_COSINE_LAYER_IDX="${VISUAL_TOKEN_COSINE_LAYER_IDX:--1}"
LAMBDA_VISUAL_TOKEN_COSINE="${LAMBDA_VISUAL_TOKEN_COSINE:-0.5}"
USE_WANDB="${USE_WANDB:-True}"

QWEN_PATH="${QWEN_PATH:-/home/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/home/linyihan/datasets/vjepa2_1_vitl_dist_vitG_384.pt}"
BASE_VLM_RUN="${BASE_VLM_RUN:-/home/linyihan/datasets/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"

CMD=(
  "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS}" vla-scripts/train.py
  --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90 \
  --vla.data_mix "${DATA_MIX}" \
  --vla.base_vlm "${BASE_VLM_RUN}" \
  --vla.vjepa_checkpoint_path "${VJEPA_CKPT}" \
  --llm_checkpoint_path "${QWEN_PATH}" \
  --data_root_dir "${ALOHA_DATA}" \
  --run_root_dir "${RUNS_DIR}" \
  --vla.expected_world_size "${NUM_GPUS}" \
  --vla.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --vla.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --vla.max_steps "${MAX_STEPS}" \
  --vla.learning_rate "${VLM_LEARNING_RATE}" \
  --vla.min_learning_rate "${VLM_MIN_LEARNING_RATE}" \
  --vla.vlm_learning_rate "${VLM_LEARNING_RATE}" \
  --vla.vlm_min_learning_rate "${VLM_MIN_LEARNING_RATE}" \
  --vla.action_head_learning_rate "${ACTION_LEARNING_RATE}" \
  --vla.action_head_min_learning_rate "${ACTION_MIN_LEARNING_RATE}" \
  --vla.lr_scheduler_type linear-warmup+cosine-decay \
  --vla.warmup_ratio 0.03 \
  --vla.shuffle_buffer_size 20000 \
  --vla.use_lora False \
  --vla.freeze_vision_backbone True \
  --vla.freeze_llm_backbone False \
  --vla.freeze_projector False \
  --vla.unfreeze_last_llm_layer False \
  --vla.use_vlm_peft False \
  --vla.use_action_queries True \
  --vla.fm_state_dropout 0.5 \
  --vla.flow_gr00t_use_full_llm_hidden False \
  --vla.use_aux_head False \
  --vla.use_visual_token_cosine_head True \
  --vla.visual_token_cosine_use_projector_target False \
  --vla.visual_token_cosine_layer_idx "${VISUAL_TOKEN_COSINE_LAYER_IDX}" \
  --vla.visual_token_cosine_projection_type mlp \
  --vla.lambda_visual_token_cosine "${LAMBDA_VISUAL_TOKEN_COSINE}" \
  --vla.visual_token_pair_offset "${VISUAL_TOKEN_PAIR_OFFSET}" \
  --vla.future_obs_window_size 0 \
  --vla.image_sequence_len 2 \
  --debug_memory_stats False \
  --debug_embedding_viz_interval 0 \
  --debug_batch_shapes False \
  --vla.d_action 14 \
  --vla.d_proprio 14 \
  --vla.action_head_type flow_gr00t \
  --use_wrist_image True \
  --robotwin_aloha_mosaic "${ROBOTWIN_ALOHA_MOSAIC}" \
  --save_interval "${SAVE_INTERVAL}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --use_wandb "${USE_WANDB}"
)

{
  echo "Starting RoboTwin ALOHA training"
  echo "Log file: ${LOG_FILE}"
  echo "Runs: ${RUNS_DIR}"
  echo "Data mix: ${DATA_MIX}"
  echo "GPUs: ${NUM_GPUS}"
  echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
  echo "Per-device batch size: ${PER_DEVICE_BATCH_SIZE}"
  echo "Mosaic: ${ROBOTWIN_ALOHA_MOSAIC}"
  echo "Robot platform: ${VLA_ROBOT_PLATFORM} (action chunk: 25)"
  echo "Trainable: full LLM, projector, action queries, and action head; vision backbone frozen"
  echo "Learning rates: VLM/projector=${VLM_LEARNING_RATE}, action/query=${ACTION_LEARNING_RATE}"
  echo "Visual cosine: projection=mlp, layer=${VISUAL_TOKEN_COSINE_LAYER_IDX}, offset=${VISUAL_TOKEN_PAIR_OFFSET}"
  echo "Visual cosine weight: ${LAMBDA_VISUAL_TOKEN_COSINE}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"

echo "Log saved to: ${LOG_FILE}"
