"""项目路径与本地资源统一配置。"""

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
QWEN_PATH = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"
SFT_ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
SFT_FULL_CKPT_DIR = ROOT / "outputs" / "sft_full_ckpt"


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def check_bert_path(path: PathLike = BERT_PATH) -> Path:
    path = _as_path(path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"本地 BERT 模型不存在或不完整: {path}\n"
            f"请确认目录下有 config.json、vocab.txt 等文件。"
        )
    return path


def check_qwen_path(path: PathLike = QWEN_PATH) -> Path:
    path = _as_path(path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"本地 Qwen 模型不存在或不完整: {path}\n"
            f"请确认目录下有 config.json、tokenizer.json 等文件。"
        )
    return path


def check_data_dir(path: PathLike = DATA_DIR) -> Path:
    path = _as_path(path)
    train_file = path / "train.json"
    if not train_file.exists():
        raise FileNotFoundError(
            f"数据集不存在: {train_file}\n"
            f"请先运行: python download_data.py --skip_cluener"
        )
    return path
