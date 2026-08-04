#!/bin/bash
###
 # @Description: raw finetuning script for bottle_cleanup task
 # @FilePath: /github_projects/Inspire-cli/.claude/mnt/shared/vla_projects/realworld_vla_adapter/experiments/robot/aloha/train_files/train_aloha.sh
###

#========== Basic Settings ==========#
PROJECT_PATH=realworld_vla_adapter
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-/path/to/root}"
export WANDB_CONSOLE=off
export WANDB_MODE=offline
export PYTHONPATH="${PROJECT_ROOT}"
#========== Training Configuration ==========#
# Dataset and paths
data_name="${DATASET_NAME:-bowl_stack_and_shelf_aloha_realworld_50}"
data_root_dir="${DATA_ROOT_DIR:-${ROOT_DIR}/datasets/cobot_aloha/tfds}"
vlm_path="${VLM_PATH:-${ROOT_DIR}/ai_models/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b}"
config_file_path="${CONFIG_FILE_PATH:-pretrained_models/configs}"

# Training parameters
batch_size=12
grad_accumulation_steps=1
learning_rate=2e-4
max_steps=10005
num_steps_before_decay=5000
save_freq=2000
lr_warmup_steps=0

# Model configuration
num_images_in_input=3
lora_rank=64
use_film=False
use_proprio=True
use_lora=True
use_fz=False
use_minivlm=True
image_aug=True
save_latest_checkpoint_only=False
merge_lora_during_training=True
use_pro_version=True
# Wandb settings
wandb_entity="${WANDB_ENTITY:-your-wandb-entity}"
wandb_project="${WANDB_PROJECT:-vla_adapter}"

# Generate timestamp and run ID
current_time=$(date +"%Y%m%d_%H%M%S")
run_id_note="raw"

# Build MODE string with important configuration variables (excluding those already in run_id)
MODE="${run_id_note}_img${num_images_in_input}_mini${use_minivlm}_prop${use_proprio}_pro${use_pro_version}_film${use_film}"

# Build run_root_dir using MODE
run_root_dir="outputs/${data_name}/${MODE}-$current_time"

mkdir -p logs

#========== Training Execution ==========#
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path $vlm_path \
  --config_file_path $config_file_path \
  --data_root_dir $data_root_dir \
  --dataset_name $data_name \
  --run_root_dir $run_root_dir \
  --use_film $use_film \
  --num_images_in_input $num_images_in_input \
  --use_proprio $use_proprio \
  --use_lora $use_lora \
  --use_fz $use_fz \
  --use_minivlm $use_minivlm \
  --image_aug $image_aug \
  --num_steps_before_decay $num_steps_before_decay \
  --max_steps $max_steps \
  --save_freq $save_freq \
  --save_latest_checkpoint_only $save_latest_checkpoint_only \
  --merge_lora_during_training $merge_lora_during_training \
  --batch_size $batch_size \
  --grad_accumulation_steps $grad_accumulation_steps \
  --learning_rate $learning_rate \
  --lora_rank $lora_rank \
  --use_pro_version $use_pro_version \
  --wandb_entity "$wandb_entity" \
  --wandb_project "$wandb_project" \
  --run_id_note $run_id_note \
  --lr_warmup_steps $lr_warmup_steps

echo "Training started with run ID: $run_id_note"
echo "Output directory: $run_root_dir"
echo "Log file: logs/$run_id_note.log"
echo "Process ID: $!" 
