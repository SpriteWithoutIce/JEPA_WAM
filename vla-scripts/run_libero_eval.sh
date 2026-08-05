#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT="${1:-}"
TASK_SUITE="${2:-libero_goal}"
TRIALS="${3:-10}"

: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the pretrained base VLM run directory}"
: "${LLM_PATH:?Set LLM_PATH to the Qwen2.5-0.5B checkpoint directory}"
: "${VJEPA_CKPT:?Set VJEPA_CKPT to the V-JEPA 2.1 ViT-L checkpoint}"
: "${LIBERO_PATH:?Set LIBERO_PATH to the LIBERO checkout}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/run_libero_eval.sh /path/to/checkpoint.pt [task_suite] [trials]" >&2
    echo "Example: bash vla-scripts/run_libero_eval.sh ./runs/xxx/checkpoints/latest-checkpoint.pt libero_goal 10" >&2
    exit 1
fi

if [[ ! -e "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
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

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"

"${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --base_vlm "${BASE_VLM_RUN}" \
    --llm_checkpoint_path "${LLM_PATH}" \
    --vjepa_checkpoint_path "${VJEPA_CKPT}" \
    --task_suite_name "${TASK_SUITE}" \
    --num_trials_per_task "${TRIALS}"
