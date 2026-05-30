"""
加载 SFT LoRA adapter 或全量 checkpoint，在验证集上评估分类准确率

使用方式：
  python evaluate_sft.py --demo              # 5 条，最快
  python evaluate_sft.py --fast              # 10 条，CPU 试跑
  python evaluate_sft.py --text "今天股市大幅下跌"

  需先 train_sft.py 生成 outputs/sft_adapter/
  本地模型: pretrain_models/Qwen2-0.5B-Instruct/（含 model.safetensors）

依赖：
  pip install torch transformers peft
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from qwen_tokenizer import load_qwen_tokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from classify_llm import (
    MODEL_PATH,
    check_model_path,
    load_label_names,
    build_system_prompt,
    run_batch,
    parse_prediction,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
FULL_CKPT_DIR = ROOT / "outputs" / "sft_full_ckpt"
OUTPUT_DIR = ROOT / "outputs"


def load_sft_model(model_path: str, ckpt_dir: str, device: torch.device):
    """自动识别 LoRA adapter 或全量微调 checkpoint。"""
    base_path = str(Path(model_path).resolve())
    ckpt_path = Path(ckpt_dir)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    load_kw = dict(
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cuda":
        load_kw["device_map"] = "auto"

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 评估需要 peft: pip install peft")
        print(f"LoRA: base={base_path}")
        print(f"      adapter={ckpt_dir}")
        tokenizer = load_qwen_tokenizer(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_path, **load_kw)
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"全量 checkpoint: {ckpt_dir}")
        tokenizer = load_qwen_tokenizer(ckpt_dir)
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), **load_kw)

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print("模型加载完成\n")
    return model, tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="LLM SFT 分类评估")
    p.add_argument("--model_path", default=str(MODEL_PATH))
    p.add_argument("--ckpt_dir", default=str(ADAPTER_DIR))
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--num_samples", default=200, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--demo", action="store_true", help="5 条")
    p.add_argument("--fast", action="store_true", help="10 条 + 短 prompt")
    p.add_argument("--text", default=None, type=str, help="单条推理")
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--max_new_tokens", default=4, type=int)
    p.add_argument("--max_chars", default=80, type=int)
    p.add_argument("--max_input_tokens", default=256, type=int)
    p.add_argument("--short_prompt", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def apply_fast_preset(args):
    if not args.fast:
        return args
    if args.num_samples == 200:
        args.num_samples = 10
    args.max_new_tokens = min(args.max_new_tokens, 3)
    args.max_chars = min(args.max_chars, 60)
    args.max_input_tokens = min(args.max_input_tokens, 128)
    args.short_prompt = True
    args.batch_size = min(args.batch_size, 2)
    print(
        f"快速模式: samples={args.num_samples}, batch={args.batch_size}, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    return args


def run_eval(samples, id2name, label_names, model, tokenizer, device, args, system_prompt):
    correct, total, unparseable = 0, 0, 0
    results = []
    t0 = time.time()
    bs = max(1, args.batch_size)

    for start in range(0, len(samples), bs):
        batch_items = samples[start: start + bs]
        batch_texts = [x["sentence"] for x in batch_items]
        batch_out = run_batch(
            batch_texts, model, tokenizer, device,
            system_prompt, label_names,
            args.max_new_tokens, args.max_chars, args.max_input_tokens,
        )

        for i, (item, (raw, pred_name)) in enumerate(zip(batch_items, batch_out)):
            idx = start + i
            true_name = id2name[item["label"]]
            is_correct = pred_name == true_name
            if pred_name is None:
                unparseable += 1
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "text": item["sentence"],
                "true_label": true_name,
                "pred_label": pred_name,
                "raw_output": raw,
                "correct": is_correct,
            })

            if not args.quiet:
                status = "✓" if is_correct else ("?" if pred_name is None else "✗")
                print(
                    f"[{idx + 1:3d}/{len(samples)}] {status} "
                    f"真实:{true_name:4s} 预测:{str(pred_name):4s} | "
                    f"{item['sentence'][:35]}"
                )

    elapsed = time.time() - t0
    acc = correct / total if total else 0.0
    per_item = elapsed / total if total else 0.0
    unpct = (unparseable / total * 100) if total else 0.0

    zero_shot_acc_str = "（未运行 classify_llm.py）"
    zs_path = OUTPUT_DIR / "llm_zero_shot_results.json"
    if zs_path.exists():
        with open(zs_path, encoding="utf-8") as f:
            zs = json.load(f)
        zero_shot_acc_str = f"{zs['accuracy']:.4f}（{zs['total']} 条）"

    print(f"\n{'=' * 50}")
    print("LLM SFT 分类结果")
    print(f"{'=' * 50}")
    print(f"  样本数   : {total}")
    print(f"  准确率   : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析 : {unparseable} 条 ({unpct:.1f}%)")
    print(f"  总耗时   : {elapsed:.1f}s, 均值 {per_item:.2f}s/条")
    print(f"\n对比: BERT(--fast)≈0.52 | zero-shot={zero_shot_acc_str} | SFT={acc:.4f}")

    out_path = OUTPUT_DIR / "llm_sft_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc, "total": total, "correct": correct,
            "unparseable": unparseable, "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")
    return acc


def main():
    args = apply_fast_preset(parse_args())
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if device.type == "cpu":
        n = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(n)
        print(f"CPU 线程数: {torch.get_num_threads()}")

    check_model_path(Path(args.model_path))

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在: {ckpt_dir}\n"
            "请先运行: python train_sft.py --demo  或  python train_sft.py --fast"
        )

    label_names = load_label_names(data_dir)
    system_prompt = build_system_prompt(label_names, short=args.short_prompt)

    model, tokenizer = load_sft_model(args.model_path, str(ckpt_dir), device)

    if args.text:
        t0 = time.time()
        raw, pred = run_batch(
            [args.text], model, tokenizer, device,
            system_prompt, label_names,
            args.max_new_tokens, args.max_chars, args.max_input_tokens,
        )[0]
        print(f"文本：{args.text}")
        print(f"预测：{pred}  (原始: {raw!r})")
        print(f"耗时：{time.time() - t0:.2f}s")
        return

    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    run_eval(samples, id2name, label_names, model, tokenizer, device, args, system_prompt)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        raise SystemExit(1) from e
