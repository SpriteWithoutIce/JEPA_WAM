#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=6,7

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_dinosiglip_flow_gr00t_$(date +%Y%m%d_%H%M%S).log"
LIBERO_DATA="${LIBERO_DATA:-/ssd/linyihan/datasets/modified_libero_rlds}"
QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
BASE_VLM_RUN="${BASE_VLM_RUN:-/ssd/linyihan/ckpt/prism-qwen25-extra-dinosiglip-224px-0_5b}"
DINO_LOCAL_PATH="${DINO_LOCAL_PATH:-}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dinosiglip}"

CMD=(
    torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py
    --vla.type prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90
    --vla.base_vlm "${BASE_VLM_RUN}"
    --vla.data_mix libero_4_task_suites_no_noops
    --llm_checkpoint_path "${QWEN_PATH}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir "${RUNS_DIR}"
    --run_id_note "${RUN_ID_NOTE}"
    --vla.expected_world_size 2
    --vla.global_batch_size 128
    --vla.per_device_batch_size 64
    --vla.learning_rate 2e-4
    --vla.min_learning_rate 1e-5
    --vla.lr_scheduler_type linear-warmup+cosine-decay
    --vla.warmup_ratio 0.03
    --vla.max_steps 60000
    --vla.shuffle_buffer_size 20000
    --vla.use_lora True
    --vla.freeze_vision_backbone True
    --vla.lora_rank 32
    --vla.lora_alpha 64
    --vla.lora_dropout 0.1
    --vla.action_head_type flow_gr00t
    --vla.use_aux_head False
    --vla.future_obs_window_size 0
    --use_wrist_image True
    --vla.image_sequence_len 2
    --save_interval 5000
    --seed 7
    --use_wandb True
    --debug_memory_stats False
    --debug_embedding_viz_interval 0
    --debug_batch_shapes False
)

if [[ -n "${DINO_LOCAL_PATH}" ]]; then
    CMD+=(--vla.dino_local_path "${DINO_LOCAL_PATH}")
fi

if [[ -n "${SIGLIP_LOCAL_PATH}" ]]; then
    CMD+=(--vla.siglip_local_path "${SIGLIP_LOCAL_PATH}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
