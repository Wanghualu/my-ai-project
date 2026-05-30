"""
模型评估脚本

使用方式：
  python evaluate.py --pool cls
  python evaluate.py --pool mean --fast
  python evaluate.py --all_pools --fast    # 一次对比 cls / mean / max

  from evaluate import evaluate_model
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR = ROOT / "outputs" / "figures"
POOL_OPTIONS = ("cls", "mean", "max")


def evaluate_model(
    model,
    loader,
    device: torch.device,
    id2name: dict,
    print_report: bool = True,
    max_batches: int = 0,
) -> dict:
    """在给定 DataLoader 上评估；max_batches>0 时只跑前 N 个 batch。"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_batches > 0 and step >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    valid_mask = all_labels != -1
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]

    if len(all_labels) == 0:
        raise ValueError("没有有效标签样本，请检查数据集或 max_batches 设置")

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if print_report:
        label_ids = sorted(id2name.keys())
        target_names = [id2name[i] for i in label_ids]
        print("\n分类报告：")
        print(classification_report(
            all_labels, all_preds,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        ))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
    }


def resolve_ckpt_path(
    pool: str,
    ckpt_dir: Path,
    ckpt_path: str | None = None,
    use_class_weight: bool = False,
) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    tag = f"{pool}_weighted" if use_class_weight else pool
    return ckpt_dir / f"best_{tag}.pt"


def load_eval_model(ckpt_path: Path, bert_path: str, num_labels: int, device: torch.device):
    """加载 checkpoint；推理时 freeze_bert 加速 CPU。"""
    from model import build_model

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pool = ckpt.get("pool", "cls")

    model = build_model(bert_path, num_labels, pool=pool)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze_bert()
    model.to(device)
    return model, ckpt, pool


def infer_max_length(ckpt_path: Path, fallback: int) -> int:
    """优先使用训练时保存的 max_length，保证与 checkpoint 一致。"""
    if not ckpt_path.exists():
        return fallback
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") or {}
    return int(ckpt_args.get("max_length", fallback))


def build_val_loader(data_dir, bert_path, max_length, batch_size, max_val_samples):
    from transformers import BertTokenizer
    from dataset import build_dataloaders

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    _, val_loader, _ = build_dataloaders(
        data_dir, tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    if max_val_samples > 0:
        val_loader.dataset.data = val_loader.dataset.data[:max_val_samples]
    return val_loader


def plot_confusion_matrix(preds, labels, id2name, save_path: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns

    matplotlib.rcParams["axes.unicode_minus"] = False
    candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break

    label_ids = sorted(id2name.keys())
    class_names = [id2name[i] for i in label_ids]
    cm = confusion_matrix(labels, preds, labels=label_ids)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 7})
    axes[0].set_title("混淆矩阵（绝对计数）")
    axes[0].tick_params(axis="x", rotation=40)

    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 7}, vmin=0, vmax=1)
    axes[1].set_title("混淆矩阵（按行归一化）")
    axes[1].tick_params(axis="x", rotation=40)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存 → {save_path}")


def eval_single_pool(args, device, id2name, num_labels, pool: str) -> dict:
    ckpt_path = resolve_ckpt_path(
        pool, Path(args.ckpt_dir), args.ckpt_path, args.use_class_weight
    )
    max_length = infer_max_length(ckpt_path, args.max_length)
    if max_length != args.max_length:
        print(f"使用 checkpoint 中的 max_length={max_length}")

    val_loader = build_val_loader(
        Path(args.data_dir), args.bert_path,
        max_length, args.batch_size, args.max_val_samples,
    )

    model, ckpt, ckpt_pool = load_eval_model(
        ckpt_path, args.bert_path, num_labels, device
    )
    if pool != ckpt_pool:
        print(f"注意: 请求 pool={pool}，checkpoint 实际为 pool={ckpt_pool}")

    print(f"Checkpoint: {ckpt_path}")
    print(f"  pool={ckpt_pool}, epoch={ckpt.get('epoch')}, "
          f"训练时 val_acc={ckpt.get('val_acc', 0):.4f}")
    print(f"  验证样本: {len(val_loader.dataset)} 条, max_length={max_length}")

    metrics = evaluate_model(
        model, val_loader, device, id2name,
        print_report=not args.quiet,
        max_batches=args.max_batches,
    )
    print(f"\n[{ckpt_pool}] val accuracy : {metrics['accuracy']:.4f}")
    print(f"[{ckpt_pool}] val macro F1 : {metrics['macro_f1']:.4f}")

    if not args.no_plot:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            metrics["preds"], metrics["labels"], id2name,
            FIG_DIR / f"confusion_matrix_{ckpt_pool}.png",
        )

    return {
        "pool": ckpt_pool,
        "val_acc": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="加载 checkpoint 并评估")
    parser.add_argument("--pool", default="cls", choices=list(POOL_OPTIONS))
    parser.add_argument("--all_pools", action="store_true",
                        help="一次评估 cls / mean / max")
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--ckpt_dir", default=str(CKPT_DIR), type=str)
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--max_val_samples", default=0, type=int)
    parser.add_argument("--max_batches", default=0, type=int)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="CPU 快速：2000 条验证集、跳过混淆矩阵")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def apply_fast_preset(args):
    if not args.fast:
        return args
    if args.max_val_samples == 0:
        args.max_val_samples = 2000
    args.no_plot = True
    print(f"快速评估: max_val_samples={args.max_val_samples}, no_plot=True")
    return args


def main():
    args = apply_fast_preset(parse_args())
    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    if args.all_pools:
        if args.ckpt_path:
            raise ValueError("--all_pools 不能与 --ckpt_path 同时使用")
        print(f"\n{'池化':<6} {'val_acc':>8} {'macro_f1':>9}")
        print("-" * 30)
        rows = []
        for pool in POOL_OPTIONS:
            try:
                row = eval_single_pool(args, device, id2name, num_labels, pool)
                rows.append(row)
            except FileNotFoundError as e:
                print(f"{pool:<6} {'—':>8} {'—':>9}  {e}")
        for row in rows:
            print(f"{row['pool']:<6} {row['val_acc']:>8.4f} {row['val_macro_f1']:>9.4f}")
        if len(rows) >= 2:
            best = max(rows, key=lambda r: r["val_macro_f1"])
            print(f"\nmacro F1 最优: {best['pool']}  (F1={best['val_macro_f1']:.4f})")
        return

    eval_single_pool(args, device, id2name, num_labels, args.pool)


if __name__ == "__main__":
    main()
