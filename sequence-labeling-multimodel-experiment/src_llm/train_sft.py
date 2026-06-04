"""
LLM SFT 训练（peoples_daily + 本地 Qwen2-0.5B-Instruct）

使用方式（在 src_llm/ 目录下）：
  python train_sft.py                 # CPU 自动快速模式
  python train_sft.py --full          # 全量数据 + 3 epoch（CPU 极慢）
"""

import os
import sys
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import record_to_target, record_to_text
from paths import DATA_DIR, QWEN_PATH, check_data_dir, check_qwen_path
from runtime import (
    setup_runtime, default_sft_batch_size,
    apply_cpu_sft_defaults, CPU_SFT_GRAD_ACCUM,
)

OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def _encode_sft_item(item: dict, tokenizer, max_length: int) -> dict:
    target = record_to_target(item)
    text = record_to_text(item)
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = (
        tokenizer.encode(target, add_special_tokens=False)
        + [tokenizer.eos_token_id]
    )
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels":    torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256, preprocess=True, desc="SFT预处理"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = None
        if preprocess:
            self.features = [
                _encode_sft_item(item, tokenizer, max_length)
                for item in tqdm(data, desc=desc, leave=False)
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.features is not None:
            return self.features[idx]
        return _encode_sft_item(self.data[idx], self.tokenizer, self.max_length)


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n, pad = item["input_ids"].size(0), max_len - item["input_ids"].size(0)
        input_ids_list.append(torch.cat([
            item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long),
        ]))
        labels_list.append(torch.cat([
            item["labels"], torch.full((pad,), -100, dtype=torch.long),
        ]))
        mask_list.append(torch.cat([
            torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long),
        ]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER（peoples_daily）")
    parser.add_argument("--model_path",  default=str(QWEN_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1, type=int)
    parser.add_argument("--val_samples", default=300, type=int,
                        help="验证集采样数；CPU 快速模式默认 100")
    parser.add_argument("--epochs",      default=3, type=int)
    parser.add_argument("--batch_size",  default=None, type=int)
    parser.add_argument("--grad_accum",  default=None, type=int)
    parser.add_argument("--lr",          default=None, type=float)
    parser.add_argument("--max_length",  default=256, type=int)
    parser.add_argument("--full_ft",     action="store_true")
    parser.add_argument("--full",        action="store_true",
                        help="全量 SFT：20864 条 + 3 epoch + 300 条验证")
    parser.add_argument("--lora_r",      default=8, type=int)
    parser.add_argument("--lora_alpha",  default=16, type=int)
    parser.add_argument("--seed",        default=42, type=int)
    parser.add_argument("--no_preprocess", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = setup_runtime()
    apply_cpu_sft_defaults(args, device)
    check_data_dir(Path(args.data_dir))
    model_path = check_qwen_path(Path(args.model_path))

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4
    if args.batch_size is None:
        args.batch_size = default_sft_batch_size(device)
    if args.grad_accum is None:
        args.grad_accum = CPU_SFT_GRAD_ACCUM if device.type == "cpu" else 4

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        random.seed(args.seed)
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    val_raw = val_raw[: args.val_samples]

    print(f"数据集：peoples_daily（{data_dir}）")
    print(f"本地模型：{model_path}")
    print(f"微调模式：{'全量微调' if args.full_ft else 'LoRA 微调'}")
    print(f"训练集: {len(train_raw)} 条 | 验证集: {len(val_raw)} 条")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    preprocess = not args.no_preprocess
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length, preprocess, "缓存SFT训练集")
    val_dataset   = SFTDataset(val_raw, tokenizer, args.max_length, preprocess, "缓存SFT验证集")

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float32,
        trust_remote_code=True, local_files_only=True,
    )

    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} (100%)")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 需要 peft：pip install peft>=0.14.0")
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none",
        ))
        model.print_trainable_parameters()

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs // args.grad_accum)
    print(f"batch={args.batch_size}, grad_accum={args.grad_accum}, steps={total_steps}\n")

    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            (outputs.loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n
            total_tokens += n

        if len(train_loader) % args.grad_accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train = total_loss / max(total_tokens, 1)

        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n
                val_tokens += n
        avg_val = val_loss / max(val_tokens, 1)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train={avg_train:.4f} val={avg_val:.4f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "elapsed_s": elapsed})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  ✓ 已保存 → {ckpt_dir}")

    log_path = output_dir / "logs" / f"train_{'full_ft' if args.full_ft else 'sft'}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。下一步：python evaluate_sft.py")


if __name__ == "__main__":
    main()
