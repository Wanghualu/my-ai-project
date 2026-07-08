#!/bin/bash
# AutoDL 首次初始化：创建持久化目录并下载模型
#
# 使用方式（在 AutoDL 实例上执行一次）：
#   bash setup_autodl.sh

set -e

DATA_ROOT="/root/autodl-fs/vllm_deployment"
MODEL_DIR="${DATA_ROOT}/Qwen2-0.5B-Instruct"

mkdir -p "${DATA_ROOT}/outputs"
mkdir -p "${DATA_ROOT}/logs"

echo "数据根目录: ${DATA_ROOT}"

if [ -f "${MODEL_DIR}/model.safetensors" ] || ls "${MODEL_DIR}"/*.safetensors 1>/dev/null 2>&1; then
    echo "模型已存在: ${MODEL_DIR}"
else
    echo "正在下载 Qwen2-0.5B-Instruct ..."
    pip install modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2-0.5B-Instruct', local_dir='${MODEL_DIR}')
"
    echo "模型下载完成: ${MODEL_DIR}"
fi

echo ""
echo "目录结构："
echo "  ${DATA_ROOT}/"
echo "  ├── Qwen2-0.5B-Instruct/   模型权重"
echo "  ├── outputs/               JSON + PNG 结果"
echo "  └── logs/                  运行日志"
echo ""
echo "下一步："
echo "  1. cd src && bash start_server.sh"
echo "  2. 另开终端：bash run_all_autodl.sh"
