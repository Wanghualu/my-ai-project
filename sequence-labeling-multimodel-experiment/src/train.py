"""
BERT NER 训练脚本（peoples_daily + 本地 bert-base-chinese）

使用方式（在 src/ 目录下）：
  python train.py                 # CPU 自动快速模式（约 30~60 min）
  python train.py --full          # 全量数据 + 全参数微调（CPU 很慢，数小时）
  python train.py --use_crf       # BERT+CRF
  python train.py --num_train 5000 --epochs 2   # 自定义规模
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_label_schema, build_dataloaders, load_records, compute_class_weights
from model import build_model
from paths import BERT_PATH, DATA_DIR, CKPT_DIR, LOG_DIR, check_bert_path, check_data_dir
from runtime import setup_runtime, default_batch_size, apply_cpu_ner_defaults

def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                _, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()
            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                for j, gold_id in enumerate(labels_np[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_ids = pred_ids_list[i]
                    pred_seq.append(
                        id2label.get(pred_ids[j] if j < len(pred_ids) else 0, "O")
                    )
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return total_loss / len(loader), seqeval_f1(all_golds, all_preds)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    args = parse_args()
    device = setup_runtime()
    apply_cpu_ner_defaults(args, device)
    check_data_dir(DATA_DIR)
    bert_path = check_bert_path(args.bert_path)

    if args.batch_size is None:
        args.batch_size = default_batch_size(device)

    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"数据集：peoples_daily（{DATA_DIR}）")
    print(f"本地模型：{bert_path}")
    print(f"BIO 标签数：{num_labels}")

    tokenizer = BertTokenizer.from_pretrained(str(bert_path), local_files_only=True)

    train_records = load_records("train", DATA_DIR)
    if args.num_train > 0:
        random.seed(42)
        train_records = random.sample(
            train_records, min(args.num_train, len(train_records))
        )
    class_weights = compute_class_weights(train_records, label2id).to(device)
    print(f"类别权重（缓解 O 标签过多）：{class_weights.tolist()}")

    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
        preprocess=not args.no_preprocess,
        dynamic_padding=not args.no_dynamic_padding,
        num_train=args.num_train,
    )

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
        class_weights=class_weights,
    ).to(device)

    if args.freeze_bert:
        if args.head_lr is None:
            args.head_lr = 1e-3
        for p in model.bert.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        head_lr = args.head_lr if args.head_lr is not None else 1e-3
        optimizer = AdamW(trainable, lr=head_lr, weight_decay=0.01)
        n_trainable = sum(p.numel() for p in trainable)
        print(f"冻结 BERT，仅训练分类头（可训练参数 {n_trainable / 1e6:.2f}M，lr={head_lr}）")
    else:
        bert_params = list(model.bert.parameters())
        head_params = (
            list(model.classifier.parameters())
            + list(model.dropout.parameters())
            + (list(model.crf.parameters()) if args.use_crf else [])
        )
        optimizer = AdamW(
            [
                {"params": bert_params, "lr": args.lr},
                {"params": head_params, "lr": args.lr * args.head_lr_mult},
            ],
            weight_decay=0.01,
        )

    total_steps = max(1, len(train_loader) * args.epochs // args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"batch_size={args.batch_size}，max_length={args.max_length}，"
          f"训练步数={total_steps}，预热={warmup_steps}")

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum,
        )
        val_loss, val_f1 = evaluate_epoch(
            model, val_loader, id2label, device, args.use_crf
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python evaluate.py {'--use_crf' if args.use_crf else ''}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型（peoples_daily）")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层")
    parser.add_argument("--bert_path", type=str, default=str(BERT_PATH))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="默认 CPU=16，GPU=32")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_train", type=int, default=-1,
                        help="训练样本数，-1=全部；CPU 快速模式默认 8000")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="全参数微调时 BERT 学习率")
    parser.add_argument("--head_lr", type=float, default=None,
                        help="冻结 BERT 时分类头学习率，默认 1e-3")
    parser.add_argument("--head_lr_mult", type=float, default=5.0,
                        help="全参数微调时分类头 lr = lr * mult")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_bert", action="store_true",
                        help="冻结 BERT，只训练分类头（更快，F1 通常仍可达 90%+）")
    parser.add_argument("--full", action="store_true",
                        help="全量训练：20864 条 + 全参数微调 + 3 epoch（CPU 很慢）")
    parser.add_argument("--no_preprocess", action="store_true")
    parser.add_argument("--no_dynamic_padding", action="store_true",
                        help="关闭动态 padding（更慢，一般不需要）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
