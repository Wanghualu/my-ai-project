"""CPU / GPU 运行时优化（联想小新等笔记本 CPU 训练）。"""

import os
from argparse import Namespace

import torch

# BERT NER：CPU 快速模式默认值（与 train.py 一致，不影响评估指标计算方式）
CPU_NER_NUM_TRAIN = 8000
CPU_NER_EPOCHS = 3
CPU_NER_HEAD_LR = 1e-3
CPU_NER_BATCH = 16
CPU_EVAL_BATCH = 32

# LLM SFT：CPU 快速模式默认值
CPU_SFT_NUM_TRAIN = 2000
CPU_SFT_EPOCHS = 1
CPU_SFT_VAL_SAMPLES = 100
CPU_SFT_BATCH = 2
CPU_SFT_GRAD_ACCUM = 8


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_cpu_threads(verbose: bool = True) -> int:
    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(min(4, max(1, n // 2)))
    except RuntimeError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    if verbose:
        print(f"CPU 优化：PyTorch 线程数={n}")
    return n


def setup_runtime(device: torch.device | None = None, verbose: bool = True) -> torch.device:
    device = device or get_device()
    if device.type == "cpu":
        setup_cpu_threads(verbose=verbose)
    elif verbose:
        print(f"GPU 训练：{torch.cuda.get_device_name(0)}")
    return device


def default_batch_size(device: torch.device) -> int:
    return 32 if device.type == "cuda" else CPU_NER_BATCH


def default_eval_batch_size(device: torch.device) -> int:
    """评估无反向传播，CPU 可用更大 batch。"""
    return 64 if device.type == "cuda" else CPU_EVAL_BATCH


def default_sft_batch_size(device: torch.device) -> int:
    return 4 if device.type == "cuda" else CPU_SFT_BATCH


def apply_cpu_ner_defaults(args: Namespace, device: torch.device) -> None:
    """CPU 上 BERT NER 训练默认快速模式（--full 关闭）。"""
    if device.type != "cpu" or getattr(args, "full", False):
        return
    if getattr(args, "num_train", -1) == -1:
        args.num_train = CPU_NER_NUM_TRAIN
    if getattr(args, "epochs", 3) == 3:
        args.epochs = CPU_NER_EPOCHS
    if not getattr(args, "freeze_bert", False):
        args.freeze_bert = True
    if getattr(args, "head_lr", None) is None and getattr(args, "freeze_bert", False):
        args.head_lr = CPU_NER_HEAD_LR
    ep = getattr(args, "epochs", 3)
    if ep < 3:
        print(
            f"⚠ 警告：epochs={ep} 过少，entity F1 通常很低。"
            f"CPU 推荐至少 --epochs 3（当前配置约需 {ep * 15}~{ep * 25} 分钟/次）"
        )
    print(
        "CPU 快速模式（完整训练请加 --full）：\n"
        f"  num_train={args.num_train}，epochs={args.epochs}，"
        f"freeze_bert={args.freeze_bert}，head_lr={getattr(args, 'head_lr', None)}"
    )


def apply_cpu_sft_defaults(args: Namespace, device: torch.device) -> None:
    """CPU 上 LLM SFT 默认快速模式（--full 关闭）。"""
    if device.type != "cpu" or getattr(args, "full", False):
        return
    if getattr(args, "num_train", -1) == -1:
        args.num_train = CPU_SFT_NUM_TRAIN
    if getattr(args, "epochs", 3) == 3:
        args.epochs = CPU_SFT_EPOCHS
    if getattr(args, "val_samples", 300) == 300:
        args.val_samples = CPU_SFT_VAL_SAMPLES
    print(
        "CPU 快速模式（完整 SFT 请加 --full）：\n"
        f"  num_train={args.num_train}，epochs={args.epochs}，"
        f"val_samples={args.val_samples}"
    )


def ckpt_use_dynamic_padding(ckpt_args: dict) -> bool:
    """从 checkpoint 读取训练时的 padding 策略，保证评估与训练一致。"""
    if not ckpt_args:
        return True
    return not ckpt_args.get("no_dynamic_padding", False)
