SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
LIBERO_DATA="/home/jwhe/linyihan/datasets/modified_libero_rlds"
QWEN_PATH="/home/jwhe/linyihan/CKPT/Qwen2.5-0.5B"
VJEPA_CKPT="/home/jwhe/linyihan/CKPT/vjepa2_1_vitl_384.pt"
BASE_VLM_RUN="/root/linyihan/prismatic-vlms/runs/prism-jepasiglip+0_5b+stage-finetune+x7"
SIGLIP_LOCAL_PATH=""
RUNS_DIR="./runs"
export CUDA_VISIBLE_DEVICES=2,3

if [[ -z "${SIGLIP_LOCAL_PATH}" ]]; then
    echo "Prewarming SigLIP weights cache..."
    python - <<'PY'
import timm

timm.create_model("vit_so400m_patch14_siglip_384", pretrained=True, num_classes=0, img_size=384)
print("SigLIP weights cache is ready.")
PY
fi

CMD=(
    torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py
    --vla.type jepavla-qwen25-jepasiglip-384px+0_5b+mx-libero-90
    --vla.base_vlm "${BASE_VLM_RUN}"
    --vla.data_mix libero_4_task_suites_no_noops
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --llm_checkpoint_path "${QWEN_PATH}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir ./runs
    --vla.expected_world_size 2
    --vla.global_batch_size 64
    --vla.per_device_batch_size 8
    --vla.learning_rate 2e-4
    --vla.lr_scheduler_type linear-warmup+cosine-decay
    --vla.warmup_ratio 0.03
    --vla.max_steps 45000
    --vla.shuffle_buffer_size 10000
    --vla.use_lora False
    --vla.freeze_vision_backbone True
    --vla.freeze_llm_backbone False
    --vla.unfreeze_last_llm_layer False
    --vla.action_head_type l1
    --vla.future_obs_window_size 8
    --vla.use_aux_head True
    --use_wrist_image False
    --save_interval 5000
    --seed 7
    --use_wandb True
    --debug_memory_stats False
    --debug_embedding_viz_interval 0
    --debug_batch_shapes False
)

if [[ -n "${SIGLIP_LOCAL_PATH}" ]]; then
    CMD+=(--vla.siglip_local_path "${SIGLIP_LOCAL_PATH}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
