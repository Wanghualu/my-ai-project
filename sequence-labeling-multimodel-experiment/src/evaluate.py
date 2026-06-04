"""
在测试集上评估 BERT NER 模型，并统计非法序列

使用方式（在 src/ 目录下）：
  python evaluate.py              # BERT+Linear，test 集
  python evaluate.py --use_crf    # BERT+CRF
  python evaluate.py --split validation
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse

import torch
from transformers import BertTokenizer
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)

from dataset import build_label_schema, build_dataloaders
from model import build_model
from paths import BERT_PATH, DATA_DIR, CKPT_DIR, LOG_DIR, check_bert_path, check_data_dir
from runtime import setup_runtime, default_eval_batch_size, ckpt_use_dynamic_padding


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev.startswith("B-") or prev.startswith("I-"):
                    if prev[2:] != curr_type:
                        stats["illegal_transition"] += 1

    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    model.eval()
    all_preds, all_golds = [], []

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                for j, gold_id in enumerate(labels_list[i]):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_ids = pred_ids_list[i]
                    pred_seq.append(
                        id2label.get(pred_ids[j] if j < len(pred_ids) else 0, "O")
                    )
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def main():
    args = parse_args()
    device = setup_runtime()
    check_data_dir(DATA_DIR)
    bert_path = check_bert_path(args.bert_path)

    if args.batch_size is None:
        args.batch_size = default_eval_batch_size(device)

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}")
        print(f"请先运行：python train.py {'--use_crf' if args.use_crf else ''}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    labels, label2id, id2label = build_label_schema()

    dynamic_padding = ckpt_use_dynamic_padding(ckpt_args)
    if args.no_dynamic_padding:
        dynamic_padding = False

    state = ckpt["state_dict"]
    class_weights = state["class_weight"] if "class_weight" in state else None

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(bert_path),
        num_labels=len(labels),
        class_weights=class_weights,
    ).to(device)
    model.load_state_dict(state, strict=True)
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")
    if ckpt_args.get("freeze_bert"):
        print("  训练配置：冻结 BERT + 分类头（CPU 快速模式）")
    if ckpt_args.get("num_train", -1) > 0:
        print(f"  训练样本数：{ckpt_args['num_train']}")

    tokenizer = BertTokenizer.from_pretrained(str(bert_path), local_files_only=True)
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=ckpt_args.get("max_length", 128),
        data_dir=DATA_DIR,
        preprocess=not args.no_preprocess,
        dynamic_padding=dynamic_padding,
        include_train=False,
    )
    loader = val_loader if args.split == "validation" else test_loader
    split_name = args.split

    print(f"\n正在 [{split_name}] 集上推理（batch={args.batch_size}）...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}  |  评估集：{split_name}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))

    illegal_stats = count_illegal_sequences(all_preds)
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
    if args.use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  → CRF Viterbi 解码：非法序列 0 条 ✓")
        else:
            print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
    else:
        print(f"  → 线性头约 {pct:.1f}% 的序列含非法转移，CRF 可完全消除")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": split_name,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
    }
    out_path = LOG_DIR / f"eval_{run_tag}_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")
    print("\n下一步：python compare_results.py")


def parse_args():
    parser = argparse.ArgumentParser(description="评估 BERT NER 模型（peoples_daily）")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=str, default=str(BERT_PATH))
    parser.add_argument("--batch_size", type=int, default=None,
                        help="默认 CPU=32（评估可用更大 batch）")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--no_preprocess", action="store_true")
    parser.add_argument("--no_dynamic_padding", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
