"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct

使用方式：
  python train_sft.py --demo              # CPU 冒烟（约 20 步，最快）
  python train_sft.py --fast              # CPU 试跑（50 步、200 条）
  python train_sft.py --num_train 5000    # LoRA 默认演示
  python train_sft.py --model_path "E:/.../pretrain_models/Qwen2-0.5B-Instruct"

  模型目录需含 model.safetensors（约 1GB），不能只有 config/tokenizer。
  CPU 不要用 --full_ft。

依赖：
  pip install torch transformers peft tqdm
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM

from qwen_tokenizer import load_qwen_tokenizer
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from classify_llm import check_model_path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"


def load_label_names(data_dir: Path) -> list[str]:
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        m = json.load(f)
    return [m["id2name"][str(i)] for i in range(m["num_labels"])]


def build_system_prompt(label_names: list[str], short: bool = False) -> str:
    labels = "、".join(label_names)
    if short:
        return f"新闻分类，只输出类别名。类别：{labels}"
    return (
        "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
        "只输出类别名称，不要输出任何其他内容。\n"
        f"可选类别：{labels}"
    )


class SFTDataset(Dataset):
    """SFT：仅对 assistant 回复（类别名）计算 loss。"""

    def __init__(self, data, tokenizer, label_names, system_prompt, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.system_prompt = system_prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_name = self.label_names[item["label"]]

        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "新闻标题：" + item["sentence"] + "\n类别："},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = (
            self.tokenizer.encode(label_name, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        max_len = self.max_length
        if len(response_ids) >= max_len:
            input_ids = response_ids[:max_len]
            labels = list(input_ids)
            prompt_len = 0
        else:
            max_prompt_len = max_len - len(response_ids)
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]
            prompt_len = len(prompt_ids)
            input_ids = prompt_ids + response_ids
            labels = [-100] * prompt_len + response_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([
            item["input_ids"],
            torch.full((pad,), pad_id, dtype=torch.long),
        ]))
        labels_list.append(torch.cat([
            item["labels"],
            torch.full((pad,), -100, dtype=torch.long),
        ]))
        mask_list.append(torch.cat([
            torch.ones(n, dtype=torch.long),
            torch.zeros(pad, dtype=torch.long),
        ]))
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT 文本分类（LoRA / 全量）")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num_train", default=5000, type=int,
                        help="-1 表示全部训练集")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--max_train_steps", default=0, type=int,
                        help="每 epoch 最多步数，0=不限制")
    parser.add_argument("--max_val_samples", default=500, type=int)
    parser.add_argument("--max_val_steps", default=0, type=int,
                        help="验证最多 batch 数，0=不限制")
    parser.add_argument("--full_ft", action="store_true")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--fast", action="store_true",
                        help="CPU 试跑：200 条、50 步、max_length=64")
    parser.add_argument("--demo", action="store_true",
                        help="CPU 冒烟：50 条、20 步，比 --fast 更快")
    parser.add_argument("--short_prompt", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def apply_fast_preset(args):
    if args.demo:
        args.fast = True
        args.num_train = 50
        args.max_train_steps = 20
        args.max_val_samples = 20
        args.max_val_steps = 10
        args.max_length = min(args.max_length, 48)
        args.lora_r = min(args.lora_r, 4)
    if not args.fast:
        return args
    if args.num_train == 5000 and not args.demo:
        args.num_train = 200
    args.epochs = 1
    args.batch_size = 1
    args.grad_accum = 2
    args.max_length = min(args.max_length, 64)
    if not args.demo:
        args.max_val_samples = min(args.max_val_samples, 100)
    if args.max_train_steps == 0:
        args.max_train_steps = 50
    if args.max_val_steps == 0:
        args.max_val_steps = 25
    args.lora_r = min(args.lora_r, 4)
    args.short_prompt = True
    tag = "demo" if args.demo else "fast"
    print(
        f"[{tag}] num_train={args.num_train}, epochs={args.epochs}, "
        f"batch={args.batch_size}, max_length={args.max_length}, "
        f"train_steps={args.max_train_steps}, val={args.max_val_samples}, "
        f"val_steps={args.max_val_steps}, lora_r={args.lora_r}"
    )
    return args


def load_base_model(model_path: str, device: torch.device):
    path = str(Path(model_path).resolve())
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    return model


def main():
    args = apply_fast_preset(parse_args())
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA"
    print(f"使用设备: {device}  |  模式: {mode_str}")

    if args.full_ft and device.type == "cpu":
        print("警告: CPU 全量微调极慢且占内存，建议去掉 --full_ft，仅用 LoRA + --fast")

    if device.type == "cpu" and not args.fast:
        print("提示: CPU 训练较慢，建议加 --demo 或 --fast")

    if device.type == "cpu":
        n = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(n)
        print(f"CPU 线程数: {torch.get_num_threads()}")

    check_model_path(Path(args.model_path))

    label_names = load_label_names(data_dir)
    system_prompt = build_system_prompt(label_names, short=args.short_prompt)

    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    val_raw = val_raw[: args.max_val_samples]
    print(f"训练集: {len(train_raw)} 条 | 验证集: {len(val_raw)} 条")

    tokenizer = load_qwen_tokenizer(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = SFTDataset(
        train_raw, tokenizer, label_names, system_prompt, args.max_length
    )
    val_dataset = SFTDataset(
        val_raw, tokenizer, label_names, system_prompt, args.max_length
    )

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, collate_fn=_collate, num_workers=0,
    )

    model = load_base_model(args.model_path, device)

    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} (100%)")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 需要 peft: pip install peft")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader)
    if args.max_train_steps > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)
    total_steps = steps_per_epoch * args.epochs // args.grad_accum
    print(f"每 epoch 约 {steps_per_epoch} 步, 总优化步数: {total_steps}, lr={args.lr}\n")

    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        steps_done = min(len(train_loader), args.max_train_steps or len(train_loader))
        if steps_done % args.grad_accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / max(total_tokens, 1)

        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.inference_mode():
            for vstep, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
                if args.max_val_steps > 0 and vstep >= args.max_val_steps:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            tag = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{tag} → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    log_path = output_dir / f"train_log_{'full_ft' if args.full_ft else 'sft'}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"日志 → {log_path}")
    print(f"权重 → {ckpt_dir}")
    print("下一步: python evaluate_sft.py --demo")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        raise SystemExit(1) from e
