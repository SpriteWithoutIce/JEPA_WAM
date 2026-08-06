#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_jepa_visual_cosine_primary_$(date +%Y%m%d_%H%M%S).log"
: "${LIBERO_DATA:?Set LIBERO_DATA to the modified LIBERO RLDS dataset root}"
: "${QWEN_PATH:?Set QWEN_PATH to the downloaded Qwen2.5-0.5B directory}"
: "${VJEPA_CKPT:?Set VJEPA_CKPT to the downloaded V-JEPA 2.1 ViT-L checkpoint}"
: "${BASE_VLM_RUN:?Set BASE_VLM_RUN to the downloaded pretrained VLM run directory}"
RUNS_DIR="${RUNS_DIR:-./runs}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${TORCHRUN_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "torchrun is not available. Activate the project environment or set TORCHRUN_BIN." >&2
    exit 1
fi

for path in "${LIBERO_DATA}" "${QWEN_PATH}" "${VJEPA_CKPT}" "${BASE_VLM_RUN}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

NPROC_PER_NODE=8
RUN_ID_NOTE="visual-cosine-projector-allviews"
USE_SWANLAB=True

EXTRA_ARGS=()
if [[ "${SMOKE_TEST}" == "1" ]]; then
    NPROC_PER_NODE=1
    RUN_ID_NOTE="visual-cosine-projector-allviews-smoke"
    USE_SWANLAB=False
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    EXTRA_ARGS+=(
        --vla.expected_world_size 1
        --vla.global_batch_size 1
        --vla.per_device_batch_size 1
        --vla.max_steps 1
        --vla.shuffle_buffer_size 128
        --cpu_memory_log_interval 0
        --debug_batch_shapes True
    )
fi

CMD=(
    "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" --module prismatic.training.train
    --vla.base_vlm "${BASE_VLM_RUN}"
    --vla.vjepa_checkpoint_path "${VJEPA_CKPT}"
    --llm_checkpoint_path "${QWEN_PATH}"
    --data_root_dir "${LIBERO_DATA}"
    --run_root_dir "${RUNS_DIR}"
    --run_id_note "${RUN_ID_NOTE}"
    --save_interval 10000
    --seed 7
    --use_swanlab "${USE_SWANLAB}"
    --debug_memory_stats False
    --debug_batch_shapes False
    "${EXTRA_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}"
