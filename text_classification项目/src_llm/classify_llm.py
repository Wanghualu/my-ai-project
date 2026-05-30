"""
用本地 Qwen2-0.5B-Instruct 做 zero-shot 分类，与 BERT fine-tune 对比

使用方式：
  python classify_llm.py --demo              # 5 条，最快演示
  python classify_llm.py --fast              # 10 条，CPU 试跑
  python classify_llm.py --text "今天股市大幅下跌"
  python classify_llm.py --num_samples 50 --batch_size 2

模型目录（需完整下载，约 1GB 权重）：
  pretrain_models/Qwen2-0.5B-Instruct/
    config.json, tokenizer.json, model.safetensors
  若只有 config (1).json，脚本会自动复制为 config.json
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from qwen_tokenizer import load_qwen_tokenizer

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"


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


def build_prompt(text: str, max_chars: int = 80) -> str:
    return f"新闻标题：{text.strip()[:max_chars]}\n类别："


def prepare_model_dir(model_path: Path) -> None:
    config = model_path / "config.json"
    if config.exists():
        return
    for alt in (model_path / "config (1).json", model_path / "config(1).json"):
        if alt.exists():
            shutil.copy(alt, config)
            print(f"已复制 {alt.name} → config.json")
            return


def find_weight_files(model_path: Path) -> list[Path]:
    names = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    found = []
    for pattern in names:
        found.extend(model_path.glob(pattern))
    return [p for p in found if p.is_file()]


def check_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在: {model_path}\n"
            "请创建目录并放入 Qwen2-0.5B-Instruct 完整文件"
        )

    prepare_model_dir(model_path)

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"缺少 config.json: {model_path}")

    weights = find_weight_files(model_path)
    if not weights:
        existing = sorted(p.name for p in model_path.iterdir() if p.is_file())
        raise FileNotFoundError(
            f"未找到模型权重（需要 model.safetensors 等）\n"
            f"目录: {model_path}\n"
            f"当前仅有: {existing or '（空）'}\n"
            "请从 HuggingFace 下载 Qwen2-0.5B-Instruct 的权重到该目录，例如：\n"
            "  huggingface-cli download Qwen/Qwen2-0.5B-Instruct "
            f"--local-dir \"{model_path}\""
        )


def load_model(model_path: str, device: torch.device):
    path = Path(model_path)
    check_model_path(path)

    if device.type == "cpu":
        n = min(os.cpu_count() or 4, 8)
        torch.set_num_threads(n)
        print(f"CPU 线程数: {torch.get_num_threads()}")

    print(f"加载模型: {path}")
    tokenizer = load_qwen_tokenizer(path)

    load_kwargs = dict(
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    print("模型加载完成")
    return model, tokenizer


def encode_batch(
    texts: list[str],
    tokenizer,
    system_prompt: str,
    device: torch.device,
    max_chars: int,
    max_input_tokens: int,
):
    ids_list, mask_list = [], []
    for text in texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(text, max_chars)},
        ]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        ids_list.append(enc["input_ids"].squeeze(0))
        mask_list.append(enc["attention_mask"].squeeze(0))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        mask_list, batch_first=True, padding_value=0
    ).to(device)
    return input_ids, attention_mask


def run_batch(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    system_prompt: str,
    label_names: list[str],
    max_new_tokens: int,
    max_chars: int,
    max_input_tokens: int,
) -> list[tuple[str, str | None]]:
    if not texts:
        return []

    input_ids, attention_mask = encode_batch(
        texts, tokenizer, system_prompt, device, max_chars, max_input_tokens
    )
    seq_len = input_ids.shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    results = []
    for i in range(len(texts)):
        gen_ids = output_ids[i, seq_len:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append((raw, parse_prediction(raw, label_names)))
    return results


# 模型常输出描述性短语而非标准类名，用关键词回退到 TNEWS 15 类
_LABEL_HINTS: tuple[tuple[str, str], ...] = (
    ("股市", "财经"), ("股票", "财经"), ("A股", "财经"), ("沪指", "财经"), ("金融", "财经"),
    ("证券", "证券"), ("基金", "证券"),
    ("足球", "体育"), ("篮球", "体育"), ("奥运", "体育"), ("比赛", "体育"),
    ("房产", "房产"), ("楼市", "房产"), ("房价", "房产"),
    ("汽车", "汽车"), ("新能源", "汽车"),
    ("学校", "教育"), ("高考", "教育"), ("大学", "教育"),
    ("手机", "科技"), ("芯片", "科技"), ("互联网", "科技"), ("AI", "科技"),
    ("战争", "军事"), ("军演", "军事"), ("部队", "军事"),
    ("旅游", "旅游"), ("景区", "旅游"),
    ("美国", "国际"), ("日本", "国际"), ("外交", "国际"),
    ("农药", "农业"), ("种植", "农业"),
    ("电竞", "电竞"), ("游戏", "电竞"),
    ("电影", "娱乐"), ("明星", "娱乐"), ("综艺", "娱乐"),
    ("文化", "文化"), ("艺术", "文化"),
    ("故事", "故事"),
)


def parse_prediction(raw_output: str, label_names: list[str]) -> str | None:
    text = raw_output.strip().replace("。", "").replace("：", "").replace(" ", "")
    if not text:
        return None
    label_set = set(label_names)

    for name in label_names:
        if text == name:
            return name
    for name in sorted(label_names, key=len, reverse=True):
        if name in text:
            return name
    for kw, label in _LABEL_HINTS:
        if kw in text and label in label_set:
            return label
    # 只生成了半个词时：输出是某类名的前缀（如「财」→ 财经）
    for name in sorted(label_names, key=len, reverse=True):
        if text in name or name.startswith(text):
            return name
    return None


def parse_args():
    p = argparse.ArgumentParser(description="LLM Zero-Shot 分类")
    p.add_argument("--model_path", default=str(MODEL_PATH))
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--num_samples", default=200, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--demo", action="store_true", help="5 条示例")
    p.add_argument("--fast", action="store_true", help="10 条 + 短 prompt，CPU 试跑")
    p.add_argument("--text", default=None, type=str, help="单条文本，不读 val.json")
    p.add_argument("--batch_size", default=1, type=int, help="CPU 建议 1~2")
    p.add_argument("--max_new_tokens", default=4, type=int)
    p.add_argument("--max_chars", default=80, type=int)
    p.add_argument("--max_input_tokens", default=256, type=int,
                   help="输入最大 token 数，越小越快")
    p.add_argument("--short_prompt", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def apply_fast_preset(args):
    if not args.fast:
        return args
    if args.num_samples == 200:
        args.num_samples = 10
    args.max_new_tokens = min(args.max_new_tokens, 8)
    args.max_chars = min(args.max_chars, 60)
    args.max_input_tokens = min(args.max_input_tokens, 128)
    args.short_prompt = True
    args.batch_size = min(args.batch_size, 2)
    print(
        f"快速模式: samples={args.num_samples}, batch={args.batch_size}, "
        f"max_new_tokens={args.max_new_tokens}, max_input_tokens={args.max_input_tokens}"
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

    print(f"\n{'=' * 50}")
    print("Zero-Shot LLM 分类结果")
    print(f"{'=' * 50}")
    print(f"  样本数   : {total}")
    print(f"  准确率   : {correct}/{total} = {acc:.4f}")
    print(f"  无法解析 : {unparseable} 条 ({unpct:.1f}%)")
    print(f"  总耗时   : {elapsed:.1f}s, 均值 {per_item:.2f}s/条")

    out_path = ROOT / "outputs" / "llm_zero_shot_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    label_names = load_label_names(data_dir)
    system_prompt = build_system_prompt(label_names, short=args.short_prompt)

    model, tokenizer = load_model(args.model_path, device)

    if args.text:
        t0 = time.time()
        raw, pred = run_batch(
            [args.text], model, tokenizer, device,
            system_prompt, label_names,
            args.max_new_tokens, args.max_chars, args.max_input_tokens,
        )[0]
        print(f"\n文本：{args.text}")
        print(f"预测：{pred}  (原始输出: {raw!r})")
        print(f"耗时：{time.time() - t0:.2f}s")
        return

    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}")

    run_eval(samples, id2name, label_names, model, tokenizer, device, args, system_prompt)


if __name__ == "__main__":
    main()
