#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub

CHECKPOINT="${CHECKPOINT:-/root/linyihan/JEPA-WAM/runs/prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90+n0+b64+x7--dinosiglip--20260519_122228/checkpoints/latest-checkpoint.pt}"
TRIALS=1

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
CUDA_ID="${2:-${CUDA_ID:-7}}"

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

LIBERO_PATH_VALUE="${LIBERO_PATH:-/root/linyihan/LIBERO-plus}"
QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
DINO_LOCAL_PATH="${DINO_LOCAL_PATH:-}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_plus_config}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-flow_gr00t}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dinosiglip-libero-plus}"
WANDB_PROJECT="${WANDB_PROJECT:-vla_libero_plus}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/libero_plus_dinosiglip.sh /path/to/checkpoint.pt [task_suite] [trials]" >&2
    echo "Or set CHECKPOINT=/path/to/checkpoint.pt before running." >&2
    exit 1
fi

if [[ ! -e "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -e "${QWEN_PATH}" ]]; then
    echo "LLM path not found: ${QWEN_PATH}" >&2
    exit 1
fi

if [[ ! -d "${LIBERO_PATH_VALUE}" ]]; then
    echo "LIBERO-plus path not found: ${LIBERO_PATH_VALUE}" >&2
    exit 1
fi

if [[ -n "${DINO_LOCAL_PATH}" && ! -e "${DINO_LOCAL_PATH}" ]]; then
    echo "DINO local path not found: ${DINO_LOCAL_PATH}" >&2
    exit 1
fi

if [[ -n "${SIGLIP_LOCAL_PATH}" && ! -e "${SIGLIP_LOCAL_PATH}" ]]; then
    echo "SigLIP local path not found: ${SIGLIP_LOCAL_PATH}" >&2
    exit 1
fi

case "${TASK_SUITE}" in
    libero_spatial|libero_object|libero_goal|libero_10|libero_90)
        ;;
    *)
        echo "Invalid task suite: ${TASK_SUITE}" >&2
        exit 1
        ;;
esac

case "${ACTION_HEAD_TYPE}" in
    l1|flow_gr00t|flow_gr00t_jepa)
        ;;
    *)
        echo "Unsupported action head type for LIBERO-plus eval: ${ACTION_HEAD_TYPE}" >&2
        exit 1
        ;;
esac

case "${NUM_IMAGES_IN_INPUT}" in
    1|2)
        ;;
    *)
        echo "Unsupported NUM_IMAGES_IN_INPUT for LIBERO-plus eval: ${NUM_IMAGES_IN_INPUT}" >&2
        exit 1
        ;;
esac

export LIBERO_PATH="${LIBERO_PATH_VALUE}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
mkdir -p "${LIBERO_CONFIG_DIR}"

cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${LIBERO_PATH_VALUE}/libero/libero
bddl_files: ${LIBERO_PATH_VALUE}/libero/libero/bddl_files
init_states: ${LIBERO_PATH_VALUE}/libero/libero/init_files
datasets: ${LIBERO_PATH_VALUE}/libero/datasets
assets: ${LIBERO_PATH_VALUE}/libero/libero/assets
EOF

echo "Evaluating DinoSigLIP checkpoint on LIBERO-plus with action_head_type=${ACTION_HEAD_TYPE}, num_images_in_input=${NUM_IMAGES_IN_INPUT}, trials=${TRIALS}"
echo "LIBERO-plus Sensor Noise tasks render at 224 first, then resize to the policy input size during evaluation."

CMD=(
    "${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py
    --model_family openvla
    --pretrained_checkpoint "${CHECKPOINT}"
    --llm_checkpoint_path "${QWEN_PATH}"
    --task_suite_name "${TASK_SUITE}"
    --num_trials_per_task "${TRIALS}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --use_proprio True
    --action_head_type "${ACTION_HEAD_TYPE}"
    --use_minivlm True
    --center_crop False
    --use_aux_head False
    --use_wandb True
    --wandb_project "${WANDB_PROJECT}"
    --save_version vla-adapter
    --log_per_task_metrics False
    --run_id_note "${RUN_ID_NOTE}"
)

if [[ -n "${DINO_LOCAL_PATH}" ]]; then
    CMD+=(--dino_local_path "${DINO_LOCAL_PATH}")
fi

if [[ -n "${SIGLIP_LOCAL_PATH}" ]]; then
    CMD+=(--siglip_local_path "${SIGLIP_LOCAL_PATH}")
fi

"${CMD[@]}"
