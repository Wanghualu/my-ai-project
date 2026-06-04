"""
NER 数据集类：人民日报 NER（BIO 格式）+ BERT 子词对齐

优化点：
  - 预处理缓存：启动时一次性 tokenize
  - 动态 padding：按 batch 内最长句 padding，CPU 训练显著加速（均值约 47 字 vs 固定 128）
"""

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from paths import DATA_DIR

ENTITY_TYPES = ["PER", "ORG", "LOC"]
DEFAULT_MAX_LENGTH = 128


def build_label_schema(
    data_dir: Optional[Path] = None,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    d = data_dir or DATA_DIR
    label_path = d / "label_names.json"
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = ["O"]
        for etype in ENTITY_TYPES:
            labels.append(f"B-{etype}")
            labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def record_to_text(record: dict) -> str:
    return "".join(record["tokens"])


def iter_bio_entities(
    tokens: list[str], ner_tags: list[str]
) -> Iterator[tuple[str, str, int, int]]:
    i = 0
    n = len(tokens)
    while i < n:
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < n and ner_tags[i] == f"I-{etype}":
                i += 1
            yield "".join(tokens[start:i]), etype, start, i - 1
        else:
            i += 1


def bio_record_to_entities(record: dict) -> list[dict]:
    return [
        {"text": surface, "type": etype}
        for surface, etype, _, _ in iter_bio_entities(
            record["tokens"], record["ner_tags"]
        )
    ]


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    return {
        (surface, etype, start, end)
        for surface, etype, start, end in iter_bio_entities(
            record["tokens"], record["ner_tags"]
        )
    }


def record_to_target(record: dict) -> str:
    entities = bio_record_to_entities(record)
    return json.dumps({"entities": entities}, ensure_ascii=False)


def bio_tags_to_ids(ner_tags: list[str], label2id: dict) -> list[int]:
    return [label2id.get(tag, 0) for tag in ner_tags]


def _encode_record(
    row: dict,
    tokenizer: BertTokenizer,
    label2id: dict,
    max_length: int,
    pad_to_max: bool = False,
) -> dict:
    tokens: list[str] = row["tokens"]
    ner_tags: list[str] = row["ner_tags"]
    char_labels = bio_tags_to_ids(ner_tags, label2id)

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding="max_length" if pad_to_max else False,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids(batch_index=0)
    aligned_labels = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            aligned_labels.append(-100)
        elif wid != prev_word_id:
            aligned_labels.append(char_labels[wid] if wid < len(char_labels) else -100)
            prev_word_id = wid
        else:
            aligned_labels.append(-100)

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "token_type_ids": encoding["token_type_ids"].squeeze(0),
        "labels": torch.tensor(aligned_labels, dtype=torch.long),
    }


def ner_collate_fn(batch: list[dict]) -> dict:
    """按 batch 内最长序列 padding，避免短句浪费算力。"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    pad_id = 0

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids.append(torch.cat([
            item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long),
        ]))
        attention_mask.append(torch.cat([
            item["attention_mask"], torch.zeros(pad, dtype=torch.long),
        ]))
        token_type_ids.append(torch.cat([
            item["token_type_ids"], torch.zeros(pad, dtype=torch.long),
        ]))
        labels.append(torch.cat([
            item["labels"], torch.full((pad,), -100, dtype=torch.long),
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "token_type_ids": torch.stack(token_type_ids),
        "labels": torch.stack(labels),
    }


class PeoplesDailyDataset(Dataset):
    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = DEFAULT_MAX_LENGTH,
        preprocess: bool = True,
        dynamic_padding: bool = True,
        desc: str = "预处理",
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding
        self.features: list[dict] | None = None

        pad_to_max = not dynamic_padding
        if preprocess:
            self.features = [
                _encode_record(row, tokenizer, label2id, max_length, pad_to_max)
                for row in tqdm(records, desc=desc, leave=False)
            ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if self.features is not None:
            return self.features[idx]
        return _encode_record(
            self.records[idx], self.tokenizer, self.label2id,
            self.max_length, pad_to_max=not self.dynamic_padding,
        )


def compute_class_weights(
    records: list,
    label2id: dict,
    max_weight: float = 15.0,
) -> torch.Tensor:
    """按标签逆频率计算 loss 权重，缓解 O 标签占比过高（~88%）导致 F1 为 0。

    NER 常见现象：模型全预测 O 时 CE loss 仍较低，但 entity-level F1 接近 0。
    """
    from collections import Counter

    counts = Counter()
    for row in records:
        for tag in row["ner_tags"]:
            counts[tag] += 1

    total = sum(counts.values()) or 1
    weights = torch.ones(len(label2id), dtype=torch.float32)
    for tag, idx in label2id.items():
        freq = counts.get(tag, 1) / total
        weights[idx] = min(max_weight, (1.0 / freq) ** 0.5)

    weights = weights / weights[0].clamp(min=1e-6)
    return weights.clamp(max=max_weight)


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    data_dir: Optional[Path] = None,
    preprocess: bool = True,
    dynamic_padding: bool = True,
    num_train: int = -1,
    include_train: bool = True,
) -> tuple[DataLoader | None, DataLoader, DataLoader]:
    """构建 DataLoader。evaluate 时可设 include_train=False 跳过训练集缓存。"""
    d = data_dir or DATA_DIR
    val_records = load_records("validation", d)
    test_records = load_records("test", d)

    train_loader = None
    if include_train:
        train_records = load_records("train", d)
        if num_train > 0:
            random.seed(42)
            n = min(num_train, len(train_records))
            train_records = random.sample(train_records, n)
            print(f"训练集子采样：{n} 条（--num_train={num_train}）")

        train_ds = PeoplesDailyDataset(
            train_records, tokenizer, label2id, max_length,
            preprocess=preprocess, dynamic_padding=dynamic_padding, desc="缓存训练集",
        )
        collate = ner_collate_fn if dynamic_padding else None
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=False, collate_fn=collate,
        )
        print(f"数据集规模：训练={len(train_ds)}，验证={len(val_records)}，测试={len(test_records)}")
    else:
        print(f"评估模式：验证={len(val_records)}，测试={len(test_records)}（跳过训练集）")

    if dynamic_padding:
        print("动态 padding：已开启（按 batch 内实际长度计算，CPU 更快）")

    collate = ner_collate_fn if dynamic_padding else None
    loader_kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False, collate_fn=collate)

    val_ds = PeoplesDailyDataset(
        val_records, tokenizer, label2id, max_length,
        preprocess=preprocess, dynamic_padding=dynamic_padding, desc="缓存验证集",
    )
    test_ds = PeoplesDailyDataset(
        test_records, tokenizer, label2id, max_length,
        preprocess=preprocess, dynamic_padding=dynamic_padding, desc="缓存测试集",
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
