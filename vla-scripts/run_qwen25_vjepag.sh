#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_vjepa-g_qwen25_cosine_$(date +%Y%m%d_%H%M%S).log"
LIBERO_DATA="${LIBERO_DATA:-/ssd/linyihan/datasets/modified_libero_rlds}"
QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/ssd/linyihan/ckpt/vjepa2_1_vitg_384.pt}"  
BASE_VLM_RUN="${BASE_VLM_RUN:-/root/linyihan/prismatic-vlms/runs/prism-qwen25-vjepa21-vitg-384px+0_5b+stage-finetune+x7}"
RUNS_DIR="./runs"
VISUAL_TOKEN_PAIR_OFFSET="${VISUAL_TOKEN_PAIR_OFFSET:-31}"
LAMBDA_VISUAL_TOKEN_COSINE="${LAMBDA_VISUAL_TOKEN_COSINE:-0.5}"
RUN_ID_NOTE="${RUN_ID_NOTE:-vjepa-g_qwen25_cosine}"
ROTATION_REPRESENTATION="${ROTATION_REPRESENTATION:-axis_angle}"
# ROTATION_REPRESENTATION="${ROTATION_REPRESENTATION:-rot6d}"

CMD=(
    torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py
    --vla.type jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90
    --vla.base_vlm "${BASE_VLM_RUN}"
    --vla.data_mix libero_4_task_suites_no_noops
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --llm_checkpoint_path "${QWEN_PATH}"
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
    --vla.use_vlm_peft False
    --vla.use_action_queries False
    --vla.action_head_type flow_gr00t
    --vla.fm_state_dropout 0.5
    --vla.flow_gr00t_use_full_llm_hidden False
    --vla.use_aux_head False
    --vla.use_visual_token_cosine_head True
    --vla.lambda_visual_token_cosine "${LAMBDA_VISUAL_TOKEN_COSINE}"
    --vla.visual_token_pair_offset "${VISUAL_TOKEN_PAIR_OFFSET}"
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
    # --pretrained_checkpoint "${LAST_CKPT}"
    # --is_resume True
    # --resume_step 30000
    # --resume_epoch 14
)

# "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
