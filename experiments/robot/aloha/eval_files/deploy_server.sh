PROJECT_PATH=realworld_vla_adapter
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-/path/to/checkpoint_dir}"
port="${PORT:-8888}"
model_family="${MODEL_FAMILY:-openvla}"
device="${DEVICE:-0}"

python experiments/robot/server_deploy/deploy.py \
        --pretrained_checkpoint $pretrained_checkpoint \
        --model_family $model_family \
        --port $port \
        --device $device
