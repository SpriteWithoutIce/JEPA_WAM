#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK_SUITE="${1:-${TASK_SUITE:-libero_spatial}}"
TRIALS="${2:-${TRIALS:-1}}"
CUDA_ID="${3:-${CUDA_ID:-7}}"

DIMENSIONS=("Semantic" "Object" "Position" "Task")
SCRIPTS=(
    "${SCRIPT_DIR}/libero_pro_semantic.sh"
    "${SCRIPT_DIR}/libero_pro_object.sh"
    "${SCRIPT_DIR}/libero_pro_position.sh"
    "${SCRIPT_DIR}/libero_pro_task.sh"
)

SUMMARY_LOG_DIR="${SUMMARY_LOG_DIR:-${REPO_ROOT}/experiments/logs}"
mkdir -p "${SUMMARY_LOG_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_PATH="${SUMMARY_LOG_DIR}/LIBERO_PRO_ALL-${TASK_SUITE}-${RUN_STAMP}.txt"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

total_episodes_sum=0
total_successes_sum=0
rate_sum="0.0"

declare -a dimension_rates
declare -a dimension_episodes
declare -a dimension_successes

{
    echo "LIBERO-PRO multi-dimension evaluation"
    echo "task_suite=${TASK_SUITE}"
    echo "trials=${TRIALS}"
    echo "cuda_id=${CUDA_ID}"
    echo
} | tee "${SUMMARY_PATH}"

for idx in "${!DIMENSIONS[@]}"; do
    dimension="${DIMENSIONS[$idx]}"
    script_path="${SCRIPTS[$idx]}"
    run_log="${TMP_DIR}/${dimension,,}.log"

    echo "===== Running ${dimension} =====" | tee -a "${SUMMARY_PATH}"
    bash "${script_path}" "${TASK_SUITE}" "${TRIALS}" "${CUDA_ID}" 2>&1 | tee "${run_log}"
    cat "${run_log}" >> "${SUMMARY_PATH}"
    echo >> "${SUMMARY_PATH}"

    episodes="$(grep -F "Total episodes:" "${run_log}" | tail -n1 | sed -E 's/.*Total episodes: ([0-9]+).*/\1/')"
    successes="$(grep -F "Total successes:" "${run_log}" | tail -n1 | sed -E 's/.*Total successes: ([0-9]+).*/\1/')"
    rate="$(grep -F "Overall success rate:" "${run_log}" | tail -n1 | sed -E 's/.*Overall success rate: ([0-9.]+).*/\1/')"

    if [[ -z "${episodes}" || -z "${successes}" || -z "${rate}" ]]; then
        echo "Failed to parse evaluation summary for ${dimension}. Check ${run_log}." | tee -a "${SUMMARY_PATH}"
        exit 1
    fi

    dimension_episodes+=("${episodes}")
    dimension_successes+=("${successes}")
    dimension_rates+=("${rate}")

    total_episodes_sum=$((total_episodes_sum + episodes))
    total_successes_sum=$((total_successes_sum + successes))
    rate_sum="$(python3 -c 'import sys; print(float(sys.argv[1]) + float(sys.argv[2]))' "${rate_sum}" "${rate}")"
done

weighted_total_rate="$(python3 -c 'import sys; succ=int(sys.argv[1]); eps=int(sys.argv[2]); print(f"{(succ / eps) if eps else 0.0:.4f}")' "${total_successes_sum}" "${total_episodes_sum}")"
macro_average_rate="$(python3 -c 'import sys; print(f"{float(sys.argv[1]) / 4.0:.4f}")' "${rate_sum}")"

{
    echo "===== Summary ====="
    for idx in "${!DIMENSIONS[@]}"; do
        printf "%-8s success_rate=%s successes=%s episodes=%s\n" \
            "${DIMENSIONS[$idx]}" "${dimension_rates[$idx]}" "${dimension_successes[$idx]}" "${dimension_episodes[$idx]}"
    done
    echo "Aggregate success_rate=${weighted_total_rate} successes=${total_successes_sum} episodes=${total_episodes_sum}"
    echo "Macro average success_rate=${macro_average_rate}"
    echo "Summary log saved to ${SUMMARY_PATH}"
} | tee -a "${SUMMARY_PATH}"
