"""
单条和批量推理脚本

使用方式：
  python predict.py --text "苹果发布了最新的 iPhone 17 系列手机"
  python predict.py --pool mean --text "今天股市大幅下跌"
  python predict.py --input_file ../data/val.json --fast --max_samples 500
  python predict.py --input_file ../data/val.json --output_file ../outputs/val_predictions.json

依赖：
  pip install torch transformers
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from evaluate import infer_max_length, load_eval_model, resolve_ckpt_path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"


def load_for_predict(
    bert_path: str,
    ckpt_path: Path,
    num_labels: int,
    device: torch.device,
):
    model, ckpt, pool = load_eval_model(ckpt_path, bert_path, num_labels, device)
    max_length = infer_max_length(ckpt_path, fallback=128)
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    return model, tokenizer, pool, max_length, ckpt


def predict_single(
    text: str,
    model,
    tokenizer,
    id2name: dict,
    max_length: int,
    device: torch.device,
    top_k: int = 3,
) -> dict:
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    top_probs, top_ids = probs.topk(min(top_k, len(id2name)))
    results = [
        {"label_id": int(lid), "label_name": id2name[int(lid)], "prob": float(p)}
        for lid, p in zip(top_ids, top_probs)
    ]
    return {"text": text, "prediction": results[0], "top_k": results}


def predict_batch(
    texts: list[str],
    model,
    tokenizer,
    id2name: dict,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)

        with torch.inference_mode():
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(1, dim=-1)

        for text, lid, prob in zip(batch_texts, top_ids[:, 0], top_probs[:, 0]):
            all_results.append({
                "text": text,
                "label_id": int(lid),
                "label_name": id2name[int(lid)],
                "prob": float(prob),
            })
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 文本分类推理")
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--ckpt_dir", default=str(CKPT_DIR), type=str)
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--max_length", default=0, type=int,
                        help="0=从 checkpoint 自动读取")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--text", default=None, type=str)
    parser.add_argument("--input_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--max_samples", default=0, type=int,
                        help="批量推理最多条数，0=全部")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="加载 best_{pool}_weighted.pt")
    parser.add_argument("--fast", action="store_true",
                        help="批量试跑：最多 500 条")
    parser.add_argument("--demo", action="store_true",
                        help="无参数时跑内置示例（默认不跑，避免 CPU 空等）")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fast and args.max_samples == 0:
        args.max_samples = 500

    data_dir = Path(args.data_dir)
    ckpt_path = resolve_ckpt_path(
        args.pool, Path(args.ckpt_dir), args.ckpt_path, args.use_class_weight
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在: {ckpt_path}\n"
            f"请先训练: python train.py --fast --epochs 1 --pool {args.pool}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    model, tokenizer, pool, ckpt_max_len, ckpt = load_for_predict(
        args.bert_path, ckpt_path, num_labels, device
    )
    max_length = args.max_length if args.max_length > 0 else ckpt_max_len
    print(f"模型加载完成 | pool={pool} | max_length={max_length}")
    if ckpt.get("val_acc") is not None:
        print(f"  checkpoint val_acc={ckpt['val_acc']:.4f}")

    if args.text:
        result = predict_single(
            args.text, model, tokenizer, id2name, max_length, device, args.top_k
        )
        print(f"\n文本：{result['text']}")
        print(f"预测：{result['prediction']['label_name']} "
              f"(置信度 {result['prediction']['prob']:.4f})")
        print(f"Top-{args.top_k}：")
        for r in result["top_k"]:
            print(f"  [{r['label_id']:2d}] {r['label_name']:4s}  {r['prob']:.4f}")
        return

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            data = json.load(f)
        if args.max_samples > 0:
            data = data[: args.max_samples]
        texts = [item["sentence"] for item in data]
        print(f"批量推理 {len(texts)} 条 ...")
        results = predict_batch(
            texts, model, tokenizer, id2name,
            max_length, args.batch_size, device,
        )

        true_labels = [item["label"] for item in data]
        correct = sum(
            1 for r, t in zip(results, true_labels)
            if r["label_id"] == t and t != -1
        )
        valid = sum(1 for t in true_labels if t != -1)
        if valid > 0:
            print(f"准确率: {correct}/{valid} = {correct / valid:.4f}")

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存 → {out_path}")
        return

    if args.demo:
        examples = [
            "苹果发布了最新的 iPhone 17，搭载 A19 芯片",
            "今天 A 股市场全线下跌，沪指跌幅超过 2%",
            "梅西在比赛中打入一粒世界波，全场沸腾",
            "教育部出台新政策，要求减轻学生课业负担",
        ]
        for text in examples:
            result = predict_single(
                text, model, tokenizer, id2name, max_length, device, top_k=3
            )
            top1 = result["prediction"]
            print(f"  [{top1['label_name']}] ({top1['prob']:.3f}) {text[:30]}")
        return

    print("请使用 --text 单条推理，或 --input_file 批量推理")
    print("示例:")
    print('  python predict.py --pool cls --text "今天股市大幅下跌"')
    print("  python predict.py --pool mean --input_file ../data/val.json --fast")
    print("  python predict.py --demo   # 运行内置示例")


if __name__ == "__main__":
    main()
