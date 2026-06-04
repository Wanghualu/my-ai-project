"""
下载 NER 数据集并保存为本地 JSON 文件

教学重点：
  1. 人民日报 NER 的 CoNLL 格式（BIO，3 类实体：PER/ORG/LOC）
  2. cluener2020 的 span 标注格式（可选下载，作对比参考）
  3. 数据来源：
     - 人民日报 NER：GitHub 原始文件（无需账号）
     - cluener2020：CLUE benchmark 官方 Google Storage（无需账号）

使用方式：
  python download_data.py                       # 下载人民日报 NER（默认主数据集）
  python download_data.py --skip_cluener          # 只下载人民日报 NER
  python download_data.py --skip_peoples_daily  # 只下载 cluener2020
"""

import io
import json
import zipfile
import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

CLUENER_URL = "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip"

PEOPLES_DAILY_URLS = {
    "train":      "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/example.train",
    "validation": "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/example.dev",
    "test":       "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/People%27s%20Daily/example.test",
}

PEOPLES_DAILY_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def _parse_conll(raw_text: str) -> list[dict]:
    """解析 CoNLL 格式：每行 '字符 BIO标签'，空行分隔句子。"""
    records = []
    tokens, tags = [], []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            if tokens:
                records.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
        else:
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        records.append({"tokens": tokens, "ner_tags": tags})
    return records


def download_peoples_daily(save_dir: Path):
    """从 GitHub 下载人民日报 NER 数据集（CoNLL 格式，3 类实体：PER/ORG/LOC）。"""
    print("=" * 60)
    print("正在下载人民日报 NER 数据集（项目主数据集）...")
    print("  数据集：3 类实体（PER 人名 / ORG 机构 / LOC 地名）")
    print("  规模：训练集 ~20864 句，验证集 ~2318 句，测试集 ~4636 句")
    print("  来源：GitHub OYE93/Chinese-NLP-Corpus（无需账号）")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, url in PEOPLES_DAILY_URLS.items():
        print(f"  下载 {split_name} ← {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw_text = resp.read().decode("utf-8")

        records = _parse_conll(raw_text)
        out_path = save_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  [{split_name}] {len(records)} 条 → {out_path}")

    label_path = save_dir / "label_names.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(PEOPLES_DAILY_LABELS, f, ensure_ascii=False, indent=2)
    print(f"\n人民日报 NER 标签体系（共 {len(PEOPLES_DAILY_LABELS)} 个）：{PEOPLES_DAILY_LABELS}")
    print()


def download_cluener(save_dir: Path):
    """从 CLUE 官方 Google Storage 下载 cluener2020，解析为 JSON（可选）。"""
    print("=" * 60)
    print("正在下载 cluener2020（可选对比数据集）...")
    print("  来源：Google Storage（无需 HuggingFace 账号）")
    print("  数据集：10 类细粒度中文 NER（人名/公司/政府机构等）")
    print("  规模：训练集 10748 条，验证集 1343 条，测试集 1345 条")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  下载 {CLUENER_URL} ...")
    with urllib.request.urlopen(CLUENER_URL, timeout=60) as resp:
        zip_bytes = resp.read()
    print(f"  下载完成（{len(zip_bytes) / 1024:.0f} KB）")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"  zip 内文件：{names}")

        split_map = {
            "train.json": "train",
            "dev.json": "validation",
            "test.json": "test",
        }
        all_records = {}
        for zip_name, split_name in split_map.items():
            match = next((n for n in names if n.endswith(zip_name)), None)
            if match is None:
                print(f"  警告：zip 中未找到 {zip_name}")
                continue
            raw = zf.read(match)
            records = []
            for line in raw.decode("utf-8").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            out_path = save_dir / f"{split_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"  [{split_name}] {len(records)} 条 → {out_path}")
            all_records[split_name] = records

    print()
    print("cluener2020 数据格式示例（span 标注）：")
    sample = all_records.get("train", [{}])[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print()


def main():
    args = parse_args()

    cluener_dir = DATA_DIR / "cluener"
    peoples_daily_dir = DATA_DIR / "peoples_daily"

    if not args.skip_peoples_daily:
        try:
            download_peoples_daily(peoples_daily_dir)
        except Exception as e:
            print(f"  人民日报 NER 下载失败（{e}）")
            raise
    else:
        print("跳过人民日报 NER 下载（--skip_peoples_daily）")

    if not args.skip_cluener:
        try:
            download_cluener(cluener_dir)
        except Exception as e:
            print(f"  cluener2020 下载失败（{e}），跳过，不影响主项目")
    else:
        print("跳过 cluener2020 下载（--skip_cluener）")

    print("=" * 60)
    print("数据下载完成！")
    print(f"  主数据集: {peoples_daily_dir}")
    print()
    print("下一步：python explore_data.py")


def parse_args():
    parser = argparse.ArgumentParser(description="下载 NER 数据集")
    parser.add_argument(
        "--skip_peoples_daily",
        action="store_true",
        help="跳过人民日报 NER 数据集下载",
    )
    parser.add_argument(
        "--skip_cluener",
        action="store_true",
        help="跳过 cluener2020 数据集下载",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
