#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="/root/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b16+x7--20260514_174927/checkpoints/latest-checkpoint.pt"

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
TRIALS="${2:-${TRIALS:-1}}"
CUDA_ID="${3:-${CUDA_ID:-7}}"

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

LIBERO_PATH_VALUE="${LIBERO_PATH:-/root/linyihan/LIBERO-PRO}"
QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_pro_config}"
EVALUATION_CONFIG_PATH="${EVALUATION_CONFIG_PATH:-${LIBERO_PATH_VALUE}/evaluation_config.yaml}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-flow_gr00t}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
UNNORM_KEY="${UNNORM_KEY:-${TASK_SUITE}}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/libero_pro.sh /path/to/checkpoint.pt [task_suite] [trials]" >&2
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

if [[ ! -e "${VJEPA_CKPT}" ]]; then
    echo "V-JEPA checkpoint not found: ${VJEPA_CKPT}" >&2
    exit 1
fi

if [[ ! -d "${LIBERO_PATH_VALUE}" ]]; then
    echo "LIBERO-PRO path not found: ${LIBERO_PATH_VALUE}" >&2
    exit 1
fi

if [[ ! -e "${EVALUATION_CONFIG_PATH}" ]]; then
    echo "LIBERO-PRO evaluation config not found: ${EVALUATION_CONFIG_PATH}" >&2
    exit 1
fi

if [[ -n "${SIGLIP_LOCAL_PATH}" && ! -e "${SIGLIP_LOCAL_PATH}" ]]; then
    echo "SigLIP local path not found: ${SIGLIP_LOCAL_PATH}" >&2
    exit 1
fi

case "${ACTION_HEAD_TYPE}" in
    l1|flow_gr00t|flow_gr00t_jepa)
        ;;
    *)
        echo "Unsupported action head type for LIBERO-PRO eval: ${ACTION_HEAD_TYPE}" >&2
        exit 1
        ;;
esac

case "${NUM_IMAGES_IN_INPUT}" in
    1|2)
        ;;
    *)
        echo "Unsupported NUM_IMAGES_IN_INPUT for LIBERO-PRO eval: ${NUM_IMAGES_IN_INPUT}" >&2
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

echo "Evaluating LIBERO-PRO with action_head_type=${ACTION_HEAD_TYPE}, num_images_in_input=${NUM_IMAGES_IN_INPUT}, trials=${TRIALS}"
echo "Base suite: ${TASK_SUITE}"
echo "Perturbation config: ${EVALUATION_CONFIG_PATH}"

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
    --use_minivlm True
    --center_crop False
    --use_aux_head False
    --use_wandb True
    --wandb_project libero_pro
    --save_version vla-adapter
    --evaluation_config_path "${EVALUATION_CONFIG_PATH}"
    --unnorm_key "${UNNORM_KEY}"
)

if [[ -n "${SIGLIP_LOCAL_PATH}" ]]; then
    CMD+=(--siglip_local_path "${SIGLIP_LOCAL_PATH}")
fi

"${CMD[@]}"
