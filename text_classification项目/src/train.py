"""
BERT 文本分类训练

教学重点：
  1. fine-tuning 的学习率设置：BERT 层用较小 lr（1e-5 ~ 3e-5），分类头可稍大
  2. 类别不均衡的处理：class_weight → 加权 CrossEntropyLoss
  3. GPU/CPU 自动兼容：torch.device 的标准用法
  4. 训练 checkpoint 保存策略：只保留验证集最优的模型
  5. 梯度累积（可选）：显存不足时等效扩大 batch size

使用方式：
  # 默认参数（CLS 池化，不加权 loss）
  python train.py

  # 使用均值池化 + 加权 loss（处理类别不均衡）
  python train.py --pool mean --use_class_weight

  # 自定义参数
  python train.py --pool max --epochs 5 --batch_size 16 --lr 2e-5

  # CPU 快速试跑（冻结 BERT + 限制训练/验证规模，约快 10~30 倍）
  python train.py --fast
  python train.py --fast --epochs 1 --max_length 32 --batch_size 16

依赖：
  pip install torch==2.4.1 torchvision==0.19.1 transformers==4.44.2 scikit-learn tqdm numpy
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
# ✅ 修复：分开导入，避免 transformers 自动加载不必要的模块
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from evaluate import evaluate_model

# ─────────────────── 默认路径（相对于 src/ 目录）────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
BERT_PATH     = ROOT / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR    = ROOT / "outputs"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 文本分类训练")
    parser.add_argument("--bert_path",      default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir",       default=str(DATA_DIR),  type=str)
    parser.add_argument("--output_dir",     default=str(OUTPUT_DIR), type=str)
    parser.add_argument("--pool",           default="cls",
                        choices=["cls", "mean", "max"],
                        help="向量提取策略：cls / mean / max")
    parser.add_argument("--epochs",         default=3,   type=int)
    parser.add_argument("--batch_size",     default=32,  type=int)
    parser.add_argument("--max_length",     default=64, type=int)
    parser.add_argument("--lr",             default=2e-5, type=float,
                        help="BERT 层学习率")
    parser.add_argument("--head_lr_mult",   default=5.0,  type=float,
                        help="分类头学习率倍数（head_lr = lr * head_lr_mult）")
    parser.add_argument("--dropout",        default=0.1,  type=float)
    parser.add_argument("--warmup_ratio",   default=0.1,  type=float,
                        help="warmup 步数占总步数的比例")
    parser.add_argument("--grad_accum",     default=1,    type=int,
                        help="梯度累积步数，显存不足时设为 2/4")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="使用加权 CrossEntropyLoss 处理类别不均衡")
    # ── 加速选项 ─────────────────────────────────────────────────────────────
    parser.add_argument("--fast", action="store_true",
                        help="快速试跑：冻结BERT + 每epoch最多1250步 + 验证2000条")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="冻结 BERT，只训练分类头（CPU 上通常快 3~5 倍）")
    parser.add_argument("--max_train_steps", default=0, type=int,
                        help="每 epoch 最多训练步数，0 表示跑完全部数据")
    parser.add_argument("--max_val_samples", default=0, type=int,
                        help="验证集最多样本数，0 表示全部")
    parser.add_argument("--head_lr", default=None, type=float,
                        help="冻结 BERT 时分类头学习率，默认 1e-3")
    return parser.parse_args()


def apply_fast_preset(args):
    """--fast 时启用一组适合 CPU 试跑的默认值（可被显式参数覆盖）。"""
    if not args.fast:
        return args
    args.freeze_bert = True
    if args.max_train_steps == 0:
        args.max_train_steps = 1250
    if args.max_val_samples == 0:
        args.max_val_samples = 2000
    if args.max_length == 64:
        args.max_length = 32
    print("快速模式: freeze_bert=True, "
          f"max_train_steps={args.max_train_steps}, "
          f"max_val_samples={args.max_val_samples}, "
          f"max_length={args.max_length}")
    return args


def compute_loss_weights(data_dir: Path, num_labels: int, device: torch.device):
    """根据训练集类别频次计算 inverse-frequency 权重。"""
    # ✅ 增加文件存在性检查
    train_file = data_dir / "train.json"
    label_file = data_dir / "label_map.json"
    
    if not train_file.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"标签映射文件不存在: {label_file}")

    with open(train_file, encoding="utf-8") as f:
        train_data = json.load(f)
    
    labels = np.array([item["label"] for item in train_data])
    classes = np.arange(num_labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    
    print("类别权重（用于加权 loss）：")
    with open(label_file, encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}
    
    for i, w in enumerate(weights):
        print(f"  {i:2d} {id2name[i]:4s}: {w:.3f}")
    
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    device, epoch, total_epochs, grad_accum, max_train_steps=0,
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        if max_train_steps > 0 and step >= max_train_steps:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)  # [B, C]
        loss   = criterion(logits, labels)

        # 梯度累积：loss 除以累积步数，等效于更大 batch
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                         acc=f"{total_correct/total_samples:.4f}")

    # 提前结束时，处理未更新的累积梯度
    if max_train_steps > 0 and total_samples > 0:
        steps_done = min(max_train_steps, len(loader))
        if steps_done % grad_accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc  = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def main():
    args = apply_fast_preset(parse_args())
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 增加 BERT 模型路径检查
    bert_path = Path(args.bert_path)
    if not bert_path.exists():
        raise FileNotFoundError(
            f"BERT 预训练模型路径不存在: {bert_path}\n"
            f"请下载 bert-base-chinese 并放到正确位置，或通过 --bert_path 指定路径"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载 label_map ───────────────────────────────────────────────────────
    label_file = data_dir / "label_map.json"
    if not label_file.exists():
        raise FileNotFoundError(f"标签映射文件不存在: {label_file}")
    
    with open(label_file, encoding="utf-8") as f:
        label_map = json.load(f)
    
    num_labels = label_map["num_labels"]
    id2name    = {int(k): v for k, v in label_map["id2name"].items()}
    print(f"类别数: {num_labels}")

    # ── Tokenizer & DataLoader ───────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_dataloaders(
        data_dir, tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    if args.max_val_samples > 0:
        val_loader.dataset.data = val_loader.dataset.data[:args.max_val_samples]
        print(f"验证集截断为 {len(val_loader.dataset)} 条")

    # ── 模型 ────────────────────────────────────────────────────────────────
    model = build_model(
        args.bert_path, num_labels, pool=args.pool, dropout=args.dropout
    )
    model = model.to(device)

    if args.freeze_bert:
        model.freeze_bert()
        print("已冻结 BERT 主干，仅训练分类头（forward 使用 no_grad 加速）")

    # ── Loss ────────────────────────────────────────────────────────────────
    if args.use_class_weight:
        weights = compute_loss_weights(data_dir, num_labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("使用加权 CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用普通 CrossEntropyLoss")

    # ── 优化器 ───────────────────────────────────────────────────────────────
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "bert" not in n]

    if args.freeze_bert:
        head_lr = args.head_lr if args.head_lr is not None else 1e-3
        optimizer = AdamW(head_params, lr=head_lr, weight_decay=0.01)
        print(f"分类头学习率: {head_lr}")
    else:
        bert_params = [p for n, p in model.named_parameters() if "bert" in n]
        optimizer = AdamW([
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ], weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    if args.max_train_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)
    total_steps  = steps_per_epoch * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"总训练步数: {total_steps}, warmup: {warmup_steps}")

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    log_records  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum,
            max_train_steps=args.max_train_steps,
        )
        val_metrics = evaluate_model(model, val_loader, device, id2name,
                                     print_report=(epoch == args.epochs))
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["macro_f1"]
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1, "elapsed_s": elapsed,
        })

        # 只保存验证集最优的 checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
            ckpt_path = ckpt_dir / f"best_{run_tag}.pt"
            torch.save({
                "epoch":           epoch,
                "pool":            args.pool,
                "use_class_weight": args.use_class_weight,
                "state_dict":      model.state_dict(),
                "val_acc":         val_acc,
                "val_macro_f1":    val_f1,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_acc={val_acc:.4f})")

    # ── 保存训练日志 ─────────────────────────────────────────────────────────
    run_tag  = f"{args.pool}_weighted" if args.use_class_weight else args.pool
    log_path = output_dir / f"train_log_{run_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}")
    print(f"训练日志 → {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 训练出错: {type(e).__name__}: {e}")
        exit(1)