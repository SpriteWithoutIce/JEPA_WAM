#!/bin/bash
#========== Download pretrained models from HuggingFace ==========#
#
# Usage:
#   bash download_models.sh
#   HF_TOKEN=your_huggingface_token bash download_models.sh
#
# HF_TOKEN is optional for public repos. If provided, it will be used.
# ROOT_DIR can be edited below or overridden via environment variable.
# Skips download if model directory already exists and is non-empty.
#

# User-configurable default root path.
DEFAULT_ROOT_DIR="/path/to/root"
ROOT_DIR="${ROOT_DIR:-${DEFAULT_ROOT_DIR}}"
MODEL_DIR="${ROOT_DIR}/ai_models"
HF_TOKEN="${HF_TOKEN:-}"

download_model() {
  local repo="$1"
  local local_dir="$2"

  if [ -d "${local_dir}" ] && [ "$(ls -A "${local_dir}" 2>/dev/null)" ]; then
    echo "[SKIP] ${repo} already exists at ${local_dir}"
    return
  fi

  echo "[DOWNLOAD] ${repo} -> ${local_dir}"
  if [ -n "${HF_TOKEN}" ]; then
    huggingface-cli download --resume-download \
      "${repo}" \
      --local-dir "${local_dir}" \
      --token "${HF_TOKEN}"
  else
    huggingface-cli download --resume-download \
      "${repo}" \
      --local-dir "${local_dir}"
  fi
}

download_model "timm/vit_large_patch14_reg4_dinov2.lvd142m" \
  "${MODEL_DIR}/timm/vit_large_patch14_reg4_dinov2.lvd142m"

download_model "timm/ViT-SO400M-14-SigLIP" \
  "${MODEL_DIR}/timm/ViT-SO400M-14-SigLIP"

download_model "Qwen/Qwen2.5-0.5B" \
  "${MODEL_DIR}/Qwen/Qwen2.5-0.5B"

download_model "Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b" \
  "${MODEL_DIR}/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b"

echo "Done."
