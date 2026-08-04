#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/libero_pro_configs/semantic.yaml"

export EVALUATION_CONFIG_PATH="${EVALUATION_CONFIG_PATH:-${CONFIG_PATH}}"

exec bash "${SCRIPT_DIR}/libero_pro.sh" "$@"
