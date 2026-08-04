#!/bin/bash
#========== One-click training setup: register dataset + local Qwen + optional local model loading ==========#
#
# Usage:
#   bash setup_training.sh <dataset_name> [--local-models]
#
# Examples:
#   # Register dataset only (vision models loaded from HF Hub)
#   bash setup_training.sh bowl_stack_and_shelf_aloha_realworld_50
#
#   # Register dataset + enable local model loading (no internet needed)
#   bash setup_training.sh bowl_stack_and_shelf_aloha_realworld_50 --local-models
#
# What it does:
#   1. Adds the dataset entry to configs.py, mixtures.py, transforms.py
#   2. (--local-models) Replaces qwen25.py so qwen25-0_5b-extra loads from the local disk path
#   3. (--local-models) Replaces materialize.py and dinosiglip_vit.py with
#      local-loading versions that read vision weights from disk instead of HF Hub
#
# How to restore overwritten files with git:
#   cd <project_root>
#   git restore prismatic/models/backbones/llm/qwen25.py
#   git restore prismatic/models/materialize.py
#   git restore prismatic/models/backbones/vision/dinosiglip_vit.py
#
# How to re-enable network download instead of local files:
#   1. Restore qwen25.py to re-enable HF loading for Qwen:
#      git restore prismatic/models/backbones/llm/qwen25.py
#   2. If you previously used --local-models, also restore the two vision files:
#      git restore prismatic/models/materialize.py
#      git restore prismatic/models/backbones/vision/dinosiglip_vit.py
#
# Local path settings:
#   Edit ROOT_DIR / LOCAL_QWEN_PATH / LOCAL_TIMM_PATH below before running this script.
#   These values will be written into the overwritten training files.
#

set -euo pipefail

# User-configurable local model paths.
ROOT_DIR="${ROOT_DIR:-/path/to/root}"
LOCAL_QWEN_PATH="${LOCAL_QWEN_PATH:-${ROOT_DIR}/ai_models/Qwen/Qwen2.5-0.5B}"
LOCAL_TIMM_PATH="${LOCAL_TIMM_PATH:-${ROOT_DIR}/ai_models/timm}"

DATASET_NAME="${1:?Usage: bash setup_training.sh <dataset_name> [--local-models]}"
LOCAL_MODELS_FLAG="${2:-}"

# Auto-detect project root (script is at experiments/robot/aloha/train_files/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CONFIGS="${PROJECT_ROOT}/prismatic/vla/datasets/rlds/oxe/configs.py"
MIXTURES="${PROJECT_ROOT}/prismatic/vla/datasets/rlds/oxe/mixtures.py"
TRANSFORMS="${PROJECT_ROOT}/prismatic/vla/datasets/rlds/oxe/transforms.py"

########################################
# Step 1: Register dataset
########################################

check_exists() {
  if grep -q "\"${DATASET_NAME}\"" "$1" 2>/dev/null; then
    echo "[SKIP] ${DATASET_NAME} already exists in $(basename $1)"
    return 0
  fi
  return 1
}

render_template() {
  local src="$1"
  local dst="$2"
  local qwen_escaped="${LOCAL_QWEN_PATH//&/\\&}"
  local timm_escaped="${LOCAL_TIMM_PATH//&/\\&}"

  sed \
    -e "s|__LOCAL_QWEN_PATH__|${qwen_escaped}|g" \
    -e "s|__LOCAL_TIMM_PATH__|${timm_escaped}|g" \
    "${src}" > "${dst}"
}

# Add to configs.py
if ! check_exists "${CONFIGS}"; then
  sed -i '/^}$/i\
    "'"${DATASET_NAME}"'": {\
        "image_obs_keys": {"primary": "image", "secondary": None, "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"},\
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\
        "state_obs_keys": ["state"],\
        "state_encoding": StateEncoding.JOINT_BIMANUAL,\
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,\
    },' "${CONFIGS}"
  echo "[ADDED] ${DATASET_NAME} to configs.py"
fi

# Add to mixtures.py
if ! check_exists "${MIXTURES}"; then
  sed -i '/^# fmt: on$/i\
    "'"${DATASET_NAME}"'": [\
        ("'"${DATASET_NAME}"'", 1.0),\
    ],' "${MIXTURES}"
  echo "[ADDED] ${DATASET_NAME} to mixtures.py"
fi

# Add to transforms.py
if ! check_exists "${TRANSFORMS}"; then
  sed -i '/^}$/i\
    "'"${DATASET_NAME}"'": aloha_dataset_transform,' "${TRANSFORMS}"
  echo "[ADDED] ${DATASET_NAME} to transforms.py"
fi

echo "[OK] Dataset '${DATASET_NAME}' registered."

########################################
# Step 2: Enable local model loading
########################################

if [ "${LOCAL_MODELS_FLAG}" = "--local-models" ] || [ "${LOCAL_MODELS_FLAG}" = "--local-vision" ]; then
  QWEN25_SRC="${SCRIPT_DIR}/qwen25.py"
  QWEN25_DST="${PROJECT_ROOT}/prismatic/models/backbones/llm/qwen25.py"
  MATERIALIZE_SRC="${SCRIPT_DIR}/materialize_local_vision.py"
  DINOSIGLIP_SRC="${SCRIPT_DIR}/dinosiglip_vit_local_vision.py"
  MATERIALIZE_DST="${PROJECT_ROOT}/prismatic/models/materialize.py"
  DINOSIGLIP_DST="${PROJECT_ROOT}/prismatic/models/backbones/vision/dinosiglip_vit.py"

  if [ ! -f "${QWEN25_SRC}" ] || [ ! -f "${MATERIALIZE_SRC}" ] || [ ! -f "${DINOSIGLIP_SRC}" ]; then
    echo "[ERROR] Local model files not found in ${SCRIPT_DIR}"
    exit 1
  fi

  if [[ "${ROOT_DIR}" == "/path/to/root" ]] || [[ "${LOCAL_QWEN_PATH}" == /path/to/root/* ]] || [[ "${LOCAL_TIMM_PATH}" == /path/to/root/* ]]; then
    echo "[ERROR] Please set ROOT_DIR, or set LOCAL_QWEN_PATH and LOCAL_TIMM_PATH explicitly before using --local-models."
    exit 1
  fi

  render_template "${QWEN25_SRC}" "${QWEN25_DST}"
  render_template "${MATERIALIZE_SRC}" "${MATERIALIZE_DST}"
  cp "${DINOSIGLIP_SRC}" "${DINOSIGLIP_DST}"
  echo "[OK] Local model loading enabled (materialize.py + dinosiglip_vit.py replaced)."
  echo "     Qwen will be loaded from local disk: ${LOCAL_QWEN_PATH}"
  echo "     Vision models will be loaded from local disk: ${LOCAL_TIMM_PATH}"
else
  echo "[INFO] HF / default model loading remains enabled."
  echo "       Use --local-models if you want local Qwen + local vision loading."
fi

echo
echo "Configured local paths:"
echo "  LOCAL_QWEN_PATH=${LOCAL_QWEN_PATH}"
echo "  LOCAL_TIMM_PATH=${LOCAL_TIMM_PATH}"
echo
echo "Restore commands:"
echo "  cd ${PROJECT_ROOT}"
echo "  git restore prismatic/models/backbones/llm/qwen25.py"
echo "  git restore prismatic/models/materialize.py"
echo "  git restore prismatic/models/backbones/vision/dinosiglip_vit.py"
echo
echo "To re-enable network download:"
echo "  - Restore qwen25.py to use HF for Qwen."
echo "  - If you used --local-models, also restore materialize.py and dinosiglip_vit.py."

echo "Done."
