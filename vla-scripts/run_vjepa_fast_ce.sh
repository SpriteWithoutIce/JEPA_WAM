#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_vjepa_fast_ce_$(date +%Y%m%d_%H%M%S).log"
LIBERO_DATA="${LIBERO_DATA:-/home/linyihan/datasets/modified_libero_rlds}"
QWEN_PATH="${QWEN_PATH:-/home/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/home/linyihan/datasets/vjepa2_1_vitl_dist_vitG_384.pt}"  
BASE_VLM_RUN="${BASE_VLM_RUN:-/home/linyihan/datasets/prism-qwen25-extra-vjepa21-384px+0_5b+stage-finetune+x7}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/home/linyihan/ckpt/fast}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-vjepa-fast-ce}"
ROTATION_REPRESENTATION="${ROTATION_REPRESENTATION:-axis_angle}"
LAMBDA_LLM_CE="${LAMBDA_LLM_CE:-0.1}"

if [[ -z "${FAST_TOKENIZER_PATH}" ]]; then
    echo "FAST_TOKENIZER_PATH is required for FAST-token CE training."
    exit 1
fi

CMD=(
    torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py
    --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90
    --vla.base_vlm "${BASE_VLM_RUN}"
    --vla.data_mix libero_4_task_suites_no_noops
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --llm_checkpoint_path "${QWEN_PATH}"
    --fast_tokenizer_path "${FAST_TOKENIZER_PATH}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir "${RUNS_DIR}"
    --run_id_note "${RUN_ID_NOTE}"
    --vla.expected_world_size 8
    --vla.global_batch_size 128
    --vla.per_device_batch_size 16
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
    --vla.lora_unfreeze_last_n_llm_layers 0
    --vla.action_head_type flow_gr00t
    --vla.fm_state_dropout 0.5
    --vla.flow_gr00t_use_full_llm_hidden False
    --vla.use_aux_head False
    --vla.use_visual_token_cosine_head False
    --vla.use_llm_ce_loss True
    --vla.lambda_llm_ce "${LAMBDA_LLM_CE}"
    --vla.future_obs_window_size 0
    --vla.rotation_representation "${ROTATION_REPRESENTATION}"
    --use_wrist_image True
    --vla.image_sequence_len 2
    --save_interval 10000
    --seed 7
    --use_wandb True
    --debug_memory_stats False
    --debug_embedding_viz_interval 0
    --debug_batch_shapes False
)

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
