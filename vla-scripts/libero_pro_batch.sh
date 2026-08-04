#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="/root/linyihan/JEPA-WAM/runs/jepavla-qwen25-vjepa-224px+0_5b+mx-libero-90+n1+b16+x7--20260516_224403/checkpoints/latest-checkpoint.pt"

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
TRIALS="${2:-${TRIALS:-1}}"
CUDA_ID="${3:-${CUDA_ID:-7}}"
DIMENSIONS_ARG="${4:-${DIMENSIONS:-semantic,object,position,task}}"

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"

LIBERO_PATH_VALUE="${LIBERO_PATH:-/root/linyihan/LIBERO-PRO}"
QWEN_PATH="${QWEN_PATH:-/ssd/linyihan/ckpt/Qwen2.5-0.5B}"
VJEPA_CKPT="${VJEPA_CKPT:-/ssd/linyihan/ckpt/vjepa2_1_vitl_dist_vitG_384.pt}"
SIGLIP_LOCAL_PATH="${SIGLIP_LOCAL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero_pro_batch_config}"
SUMMARY_LOG_DIR="${SUMMARY_LOG_DIR:-${REPO_ROOT}/experiments/logs}"
ACTION_HEAD_TYPE="${ACTION_HEAD_TYPE:-flow_gr00t}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
UNNORM_KEY="${UNNORM_KEY:-${TASK_SUITE}}"

mkdir -p "${SUMMARY_LOG_DIR}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash vla-scripts/libero_pro_batch.sh [task_suite] [trials] [cuda_id] [dimensions]" >&2
    echo "Example: bash vla-scripts/libero_pro_batch.sh libero_spatial 1 4 semantic,object,task" >&2
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

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_PATH="${SUMMARY_LOG_DIR}/LIBERO_PRO_BATCH-${TASK_SUITE}-${RUN_STAMP}.txt"

normalize_dimension() {
    local dim
    dim="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    case "${dim}" in
        semantic|sem|language|lan)
            echo "semantic"
            ;;
        object|obj)
            echo "object"
            ;;
        position|pos|swap)
            echo "position"
            ;;
        task)
            echo "task"
            ;;
        *)
            echo ""
            ;;
    esac
}

display_name() {
    case "$1" in
        semantic) echo "Semantic" ;;
        object) echo "Object" ;;
        position) echo "Position" ;;
        task) echo "Task" ;;
        *) echo "$1" ;;
    esac
}

write_eval_config() {
    local dim="$1"
    local config_path="$2"
    local use_environment=false
    local use_swap=false
    local use_object=false
    local use_language=false
    local use_task=false

    case "${dim}" in
        semantic) use_language=true ;;
        object) use_object=true ;;
        position) use_swap=true ;;
        task) use_task=true ;;
        *)
            echo "Unknown dimension: ${dim}" >&2
            exit 1
            ;;
    esac

    cat > "${config_path}" <<EOF
bddl_files_path: ""
script_path: ""
init_file_dir: ""

use_environment: ${use_environment}
use_swap: ${use_swap}
use_object: ${use_object}
use_language: ${use_language}
use_task: ${use_task}

ood_task_configs:
  environment: "./libero_ood/ood_environment.yaml"
  swap: "./libero_ood/ood_spatial_relation.yaml"
  object: "./libero_ood/ood_object.yaml"
  language: "./libero_ood/ood_language.yaml"
  task: "./libero_ood/ood_task.yaml"

perturbation_mapping:
  use_environment: env
  use_swap: swap
  use_object: object
  use_language: lan
  use_task: task
EOF
}

IFS=',' read -r -a requested_dimensions <<< "${DIMENSIONS_ARG}"
selected_dimensions=()

for raw_dim in "${requested_dimensions[@]}"; do
    canonical_dim="$(normalize_dimension "${raw_dim}")"
    if [[ -z "${canonical_dim}" ]]; then
        echo "Unsupported LIBERO-PRO dimension: ${raw_dim}" >&2
        echo "Supported dimensions: semantic, object, position, task" >&2
        exit 1
    fi

    already_added=false
    for existing_dim in "${selected_dimensions[@]:-}"; do
        if [[ "${existing_dim}" == "${canonical_dim}" ]]; then
            already_added=true
            break
        fi
    done
    if [[ "${already_added}" == false ]]; then
        selected_dimensions+=("${canonical_dim}")
    fi
done

if [[ "${#selected_dimensions[@]}" -eq 0 ]]; then
    echo "No LIBERO-PRO dimensions selected." >&2
    exit 1
fi

declare -a dimension_labels
declare -a dimension_rates
declare -a dimension_episodes
declare -a dimension_successes

total_episodes_sum=0
total_successes_sum=0
rate_sum="0.0"

{
    echo "LIBERO-PRO batch evaluation"
    echo "task_suite=${TASK_SUITE}"
    echo "trials=${TRIALS}"
    echo "cuda_id=${CUDA_ID}"
    echo "dimensions=${DIMENSIONS_ARG}"
    echo "checkpoint=${CHECKPOINT}"
    echo
} | tee "${SUMMARY_PATH}"

for dim in "${selected_dimensions[@]}"; do
    label="$(display_name "${dim}")"
    config_path="${TMP_DIR}/${dim}.yaml"
    run_log="${TMP_DIR}/${dim}.log"
    write_eval_config "${dim}" "${config_path}"

    echo "===== Running ${label} =====" | tee -a "${SUMMARY_PATH}"

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
        --evaluation_config_path "${config_path}"
        --unnorm_key "${UNNORM_KEY}"
    )

    if [[ -n "${SIGLIP_LOCAL_PATH}" ]]; then
        CMD+=(--siglip_local_path "${SIGLIP_LOCAL_PATH}")
    fi

    "${CMD[@]}" 2>&1 | tee "${run_log}"
    cat "${run_log}" >> "${SUMMARY_PATH}"
    echo >> "${SUMMARY_PATH}"

    episodes="$(grep -F "Total episodes:" "${run_log}" | tail -n1 | sed -E 's/.*Total episodes: ([0-9]+).*/\1/')"
    successes="$(grep -F "Total successes:" "${run_log}" | tail -n1 | sed -E 's/.*Total successes: ([0-9]+).*/\1/')"
    rate="$(grep -F "Overall success rate:" "${run_log}" | tail -n1 | sed -E 's/.*Overall success rate: ([0-9.]+).*/\1/')"

    if [[ -z "${episodes}" || -z "${successes}" || -z "${rate}" ]]; then
        echo "Failed to parse evaluation summary for ${label}. Check ${run_log}." | tee -a "${SUMMARY_PATH}"
        exit 1
    fi

    dimension_labels+=("${label}")
    dimension_episodes+=("${episodes}")
    dimension_successes+=("${successes}")
    dimension_rates+=("${rate}")

    total_episodes_sum=$((total_episodes_sum + episodes))
    total_successes_sum=$((total_successes_sum + successes))
    rate_sum="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1]) + float(sys.argv[2]))' "${rate_sum}" "${rate}")"
done

weighted_total_rate="$("${PYTHON_BIN}" -c 'import sys; succ=int(sys.argv[1]); eps=int(sys.argv[2]); print(f"{(succ / eps) if eps else 0.0:.4f}")' "${total_successes_sum}" "${total_episodes_sum}")"
macro_average_rate="$("${PYTHON_BIN}" -c 'import sys; total=float(sys.argv[1]); count=int(sys.argv[2]); print(f"{(total / count) if count else 0.0:.4f}")' "${rate_sum}" "${#selected_dimensions[@]}")"

{
    echo "===== Summary ====="
    for idx in "${!dimension_labels[@]}"; do
        printf "%-8s success_rate=%s successes=%s episodes=%s\n" \
            "${dimension_labels[$idx]}" "${dimension_rates[$idx]}" "${dimension_successes[$idx]}" "${dimension_episodes[$idx]}"
    done
    echo "Aggregate success_rate=${weighted_total_rate} successes=${total_successes_sum} episodes=${total_episodes_sum}"
    echo "Macro average success_rate=${macro_average_rate}"
    echo "Summary log saved to ${SUMMARY_PATH}"
} | tee -a "${SUMMARY_PATH}"
