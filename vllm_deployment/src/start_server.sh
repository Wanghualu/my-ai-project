#!/bin/bash
# 启动 vLLM OpenAI 兼容 server（AutoDL 版）
#
# 使用方式：
#   cd /root/autodl-tmp/vllm_deployment/src   # 或你的代码目录
#   bash start_server.sh
#
# 模型与日志保存在持久化挂载目录 /root/autodl-fs/vllm_deployment/

set -e

# ── AutoDL 持久化目录 ────────────────────────────────────────
DATA_ROOT="/root/autodl-fs/vllm_deployment"
MODEL_PATH="${DATA_ROOT}/Qwen2-0.5B-Instruct"
LOG_DIR="${DATA_ROOT}/logs"
SERVED_NAME="qwen2-0.5b"
PORT=8000
MAX_MODEL_LEN=2048
GPU_MEM_UTIL=0.6
DTYPE="float16"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vllm_server_$(date +%Y%m%d_%H%M%S).log"

# ── 激活 venv ────────────────────────────────────────────────
if [ -z "$VIRTUAL_ENV" ]; then
    [ -f ~/vllm_env/bin/activate ] && source ~/vllm_env/bin/activate
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export VLLM_DATA_ROOT="$DATA_ROOT"

echo "============================================"
echo "  启动 vLLM OpenAI Server"
echo "  模型路径: $MODEL_PATH"
echo "  对外名称: $SERVED_NAME"
echo "  端口:     $PORT"
echo "  日志文件: $LOG_FILE"
echo "============================================"
echo ""
echo "启动后测试："
echo "  curl http://localhost:${PORT}/v1/models"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --enforce-eager \
    --host 0.0.0.0 \
    2>&1 | tee "$LOG_FILE"
