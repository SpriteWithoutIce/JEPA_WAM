PROJECT_PATH=realworld_vla_adapter
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# Allow passing the server URL as an argument or via VLA_SERVER_URL env variable.
VLA_SERVER_URL="${VLA_SERVER_URL:-http://127.0.0.1:8888}"
TASK_LABEL="${TASK_LABEL:-Use the right arm to stack the red bowl on the blue one, then use the left arm to place the stack on the shelf.}"
UNNORM_KEY="${UNNORM_KEY:-bowl_stack_and_shelf_aloha_realworld_50}"

python experiments/robot/aloha/run_cobot_client.py \
  --use_vla_server \
  --vla_server_url "${VLA_SERVER_URL}" \
  --unnorm_key "${UNNORM_KEY}" \
  --task_label "${TASK_LABEL}"
