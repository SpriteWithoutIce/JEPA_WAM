#!/usr/bin/env bash

# Standalone LoRA training entrypoint for the 50 RoboTwin ALOHA clean datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLA_ROBOT_PLATFORM="ALOHA"

LOG_DIR="${LOG_DIR:-./logs}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-aloha-robotwin-clean-50}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_${RUN_ID_NOTE}_$(date +%Y%m%d_%H%M%S).log"

ALOHA_DATA="${ALOHA_DATA:-/home/linyihan/datasets/aloha_dataset}"
# Fixed to prevent an inherited DATA_MIX from adding randomized_500 datasets.
DATA_MIX="aloha_robotwin_clean_50"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-1024}"
RLDS_FRAME_TRANSFORM_THREADS="${RLDS_FRAME_TRANSFORM_THREADS:-4}"
RLDS_PRIVATE_THREADPOOL_SIZE="${RLDS_PRIVATE_THREADPOOL_SIZE:-8}"
RLDS_MAX_INTRA_OP_PARALLELISM="${RLDS_MAX_INTRA_OP_PARALLELISM:-1}"
ROBOTWIN_ALOHA_MOSAIC="${ROBOTWIN_ALOHA_MOSAIC:-True}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/linyihan/jepa_copy/bin/torchrun}"

VISUAL_TOKEN_PAIR_OFFSET="${VISUAL_TOKEN_PAIR_OFFSET:-31}"
VISUAL_TOKEN_COSINE_LAYER_IDX="${VISUAL_TOKEN_COSINE_LAYER_IDX:--1}"
LAMBDA_VISUAL_TOKEN_COSINE="${LAMBDA_VISUAL_TOKEN_COSINE:-0.5}"
USE_WANDB="${USE_WANDB:-True}"

QWEN_PATH="${QWEN_PATH:-/home/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/home/linyihan/datasets/vjepa2_1_vitl_dist_vitG_384.pt}"
BASE_VLM_RUN="${BASE_VLM_RUN:-/home/linyihan/datasets/prism-qwen25-vjepa21-vitl-384px+0_5b+stage-finetune+x7}"

export VLA_RLDS_FRAME_TRANSFORM_THREADS="${RLDS_FRAME_TRANSFORM_THREADS}"
export VLA_RLDS_PRIVATE_THREADPOOL_SIZE="${RLDS_PRIVATE_THREADPOOL_SIZE}"
export VLA_RLDS_MAX_INTRA_OP_PARALLELISM="${RLDS_MAX_INTRA_OP_PARALLELISM}"

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
  --vla.learning_rate 2e-4 \
  --vla.min_learning_rate 1e-5 \
  --vla.lr_scheduler_type linear-warmup+cosine-decay \
  --vla.warmup_ratio 0.03 \
  --vla.shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --vla.shared_dataset_statistics False \
  --vla.use_lora True \
  --vla.freeze_vision_backbone True \
  --vla.lora_rank 32 \
  --vla.lora_alpha 64 \
  --vla.lora_dropout 0.1 \
  --vla.lora_unfreeze_last_n_llm_layers 0 \
  --vla.use_vlm_peft False \
  --vla.use_action_queries False \
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
  echo "Starting RoboTwin ALOHA LoRA training"
  echo "Log file: ${LOG_FILE}"
  echo "Runs: ${RUNS_DIR}"
  echo "Data mix: ${DATA_MIX} (50 clean tasks only)"
  echo "Normalization statistics: per-dataset"
  echo "Max steps: ${MAX_STEPS}"
  echo "GPUs: ${NUM_GPUS}"
  echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
  echo "Per-device batch size: ${PER_DEVICE_BATCH_SIZE}"
  echo "RLDS shuffle buffer: ${SHUFFLE_BUFFER_SIZE}"
  echo "RLDS frame transform threads: ${RLDS_FRAME_TRANSFORM_THREADS}"
  echo "RLDS private threadpool: ${RLDS_PRIVATE_THREADPOOL_SIZE}"
  echo "RLDS max intra-op parallelism: ${RLDS_MAX_INTRA_OP_PARALLELISM}"
  echo "Mosaic: ${ROBOTWIN_ALOHA_MOSAIC}"
  echo "Robot platform: ${VLA_ROBOT_PLATFORM} (action chunk: 25)"
  echo "Trainable: LLM LoRA adapters, action head/projectors, and visual cosine head"
  echo "Frozen: vision backbone, base LLM weights, VLM projector, and action queries"
  echo "Visual cosine: projection=mlp, layer=${VISUAL_TOKEN_COSINE_LAYER_IDX}, offset=${VISUAL_TOKEN_PAIR_OFFSET}"
  echo "Visual cosine weight: ${LAMBDA_VISUAL_TOKEN_COSINE}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
} 2>&1 | tee "${LOG_FILE}"

echo "Log saved to: ${LOG_FILE}"
