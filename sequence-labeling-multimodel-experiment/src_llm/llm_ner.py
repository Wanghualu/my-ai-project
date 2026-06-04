"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. LLM 的 span 级别 F1 计算（与 BERT 保持可比性）
  4. 成本控制：只采样 100 条，不跑完整验证集

使用方式：
  # DeepSeek（默认）
  set DEEPSEEK_API_KEY=sk-xxx
  python llm_ner.py
  python llm_ner.py --n_samples 50 --model deepseek-chat

  # 通义千问（可选）
  set DASHSCOPE_API_KEY=sk-xxx
  python llm_ner.py --provider dashscope --model qwen-plus

依赖：
  pip install openai
"""

import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import ENTITY_TYPES, gold_spans_from_record, iter_bio_entities, record_to_text
from paths import DATA_DIR, check_data_dir, LOG_DIR
from runtime import setup_runtime

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地名",
}

API_PRESETS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
    },
}


def build_client(base_url: str, api_key_env: str) -> OpenAI:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(f"请设置环境变量 {api_key_env}")
    return OpenAI(api_key=api_key, base_url=base_url)


def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理中文文本。
请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：
- PER：人名
- ORG：组织机构名称
- LOC：地名

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}

如果没有实体，输出：{"entities": []}"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "海钓比赛地点在厦门与金门之间的海域。",
        "output": '{"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}'
    },
    {
        "text": "这座依山傍水的博物馆由国内一流的设计师主持设计，整个建筑群精美而恢宏。",
        "output": '{"entities": []}'
    },
    {
        "text": "毛泽东主席在北京中南海会见了来访的美国总统尼克松。",
        "output": '{"entities": [{"text": "毛泽东", "type": "PER"}, {"text": "北京", "type": "LOC"}, {"text": "中南海", "type": "LOC"}, {"text": "美国", "type": "ORG"}, {"text": "尼克松", "type": "PER"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""


def sample_records(n: int, seed: int = 42) -> list[dict]:
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    by_type = defaultdict(list)
    for r in records:
        for _, etype, _, _ in iter_bio_entities(r["tokens"], r["ner_tags"]):
            by_type[etype].append(r)

    selected = set()
    selected_list = []

    per_type = max(1, n // len(ENTITY_TYPES))
    for etype in ENTITY_TYPES:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected:
                selected.add(id(r))
                selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def main():
    args = parse_args()
    device = setup_runtime()
    check_data_dir(DATA_DIR)

    if args.n_samples is None:
        args.n_samples = 50 if device.type == "cpu" else 100

    preset = API_PRESETS[args.provider]
    base_url = args.base_url or preset["base_url"]
    api_key_env = args.api_key_env or preset["api_key_env"]
    model = args.model or preset["default_model"]

    client = build_client(base_url, api_key_env)
    records = sample_records(args.n_samples)
    print(f"API：{args.provider}（{base_url}，模型={model}）")
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        text = record_to_text(record)
        gold = gold_spans_from_record(record)

        zs_resp = call_api(client, zero_shot_prompt(text), model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        fs_resp = call_api(client, few_shot_prompt(text), model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（{args.provider} / {model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "provider": args.provider,
        "model": model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print("\n下一步：train_sft.py")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="采样数；CPU 默认 50，GPU 默认 100")
    parser.add_argument(
        "--provider", type=str, default="deepseek", choices=list(API_PRESETS),
        help="API 提供商（默认 deepseek）",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="模型名；默认 deepseek-chat 或 qwen-plus")
    parser.add_argument("--base_url", type=str, default=None,
                        help="自定义 API base_url（覆盖预设）")
    parser.add_argument("--api_key_env", type=str, default=None,
                        help="自定义密钥环境变量名（覆盖预设）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
