"""
AutoDL / 本地通用路径配置。

AutoDL 持久化挂载目录：/root/autodl-fs
  项目数据、模型、outputs、logs 均保存在该目录下，关机不丢失，方便下载到本地。

可通过环境变量 VLLM_DATA_ROOT 覆盖数据根目录。
"""

import os

AUTODL_FS = "/root/autodl-fs"
PROJECT_NAME = "vllm_deployment"
MODEL_DIRNAME = "Qwen2-0.5B-Instruct"
SERVED_MODEL_NAME = "qwen2-0.5b"
VLLM_BASE_URL = "http://localhost:8000/v1"


def get_data_root() -> str:
    if os.environ.get("VLLM_DATA_ROOT"):
        return os.environ["VLLM_DATA_ROOT"]
    autodl_root = os.path.join(AUTODL_FS, PROJECT_NAME)
    if os.path.isdir(AUTODL_FS):
        return autodl_root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


DATA_ROOT = get_data_root()
MODEL_PATH = os.path.join(DATA_ROOT, MODEL_DIRNAME)
OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
LOG_DIR = os.path.join(DATA_ROOT, "logs")


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def output_path(filename: str) -> str:
    ensure_dirs()
    return os.path.join(OUTPUT_DIR, filename)
