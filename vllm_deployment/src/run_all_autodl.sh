#!/bin/bash
# AutoDL 一键跑完全部 demo + 吞吐对比（需先手动启动 vLLM server）
#
# 使用方式：
#   终端 1：bash start_server.sh
#   终端 2：bash run_all_autodl.sh
#
# 产出目录（持久化挂载，可下载到本地）：
#   /root/autodl-fs/vllm_deployment/outputs/   — JSON + PNG 结果
#   /root/autodl-fs/vllm_deployment/logs/      — 运行日志 + run_summary.json

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/root/autodl-fs/vllm_deployment"

if [ -z "$VIRTUAL_ENV" ]; then
    [ -f ~/vllm_env/bin/activate ] && source ~/vllm_env/bin/activate
fi

export VLLM_DATA_ROOT="$DATA_ROOT"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  数据目录: $DATA_ROOT"
echo "  日志目录: $DATA_ROOT/logs"
echo "  结果目录: $DATA_ROOT/outputs"
echo "============================================"

check_server() {
    if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[ERROR] vLLM server 未启动，请先在另一终端运行: bash start_server.sh"
        exit 1
    fi
}

echo ""
echo ">>> [1/6] demo_guided_choice.py"
check_server
python demo_guided_choice.py

echo ""
echo ">>> [2/6] demo_guided_regex.py"
python demo_guided_regex.py

echo ""
echo ">>> [3/6] demo_guided_json.py"
python demo_guided_json.py

echo ""
echo ">>> [4/6] demo_response_format.py"
python demo_response_format.py

echo ""
echo ">>> [5/6] demo_function_call.py"
python demo_function_call.py

echo ""
echo ">>> [6/6] bench_throughput.py（需先停 server 释放显存）"
echo "    正在停止 vLLM server..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 2
python bench_throughput.py

echo ""
echo "============================================"
echo "  全部完成！请从以下目录下载到本地："
echo "  $DATA_ROOT/outputs/"
echo "  $DATA_ROOT/logs/"
echo "============================================"
