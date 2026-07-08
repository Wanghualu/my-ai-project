"""
文档分块脚本：对解析后的论文/技术文档做分块处理

教学重点（三种分块策略的对比）：
  策略A  固定大小分块  —— 最简单，但会切断句子/表格
  策略B  语义分块      —— 按段落/章节边界切，保留语义完整性
  策略C  层级分块      —— 父子块：父块用于召回上下文，子块用于精确匹配

企业级 RAG 通常用 B 或 C，
让学生先跑通 A，再体会 B/C 在召回效果上的区别。

输出格式说明：
  每个 chunk 是一个 dict，包含：
    - chunk_id      唯一标识
    - content       文本内容（供 embedding）
    - metadata      元信息（供过滤/溯源）
      - doc_type    paper/book
      - title       文档标题
      - topic       主题分类
      - year        年份
      - page_num    来源页码
      - section     章节路径（字符串）
      - block_type  text/table/title
      - is_ocr      是否 OCR 结果
      - strategy    分块策略名
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ── 策略 A：固定大小分块 ──────────────────────────────────────────────────────

def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


# ── 策略 B：语义分块 ──────────────────────────────────────────────────────────

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> Iterator[dict]:
    buffer_blocks = []
    buffer_len    = 0
    buffer_meta   = {}

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        meta = {
            "page_num":   buf[0]["page_num"],
            "section":    " > ".join(buf[0]["section_path"]) if buf[0]["section_path"] else "",
            "block_types": list({b["block_type"] for b in buf}),
            "is_ocr":     any(b["is_ocr"] for b in buf),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])

        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0

        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": block["page_num"],
                    "section":  " > ".join(block["section_path"]),
                    "block_types": ["table"],
                    "is_ocr":   block["is_ocr"],
                }
            }
            continue

        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len    = 0

        buffer_blocks.append(block)
        buffer_len += blen

    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


# ── 策略 C：层级分块（父子块） ────────────────────────────────────────────────

def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size:  int = 400,
    overlap:     int = 50,
) -> Iterator[dict]:
    full_text  = "\n\n".join(b["content"] for b in blocks if b["content"].strip())

    parents = []
    start   = 0
    while start < len(full_text):
        end     = min(start + parent_size, len(full_text))
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content":   content,
            "start":     start,
            "end":       end,
        })
        start += parent_size - overlap

    for parent in parents:
        p_content = parent["content"]
        p_id      = parent["parent_id"]
        c_start   = 0
        while c_start < len(p_content):
            c_end     = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content":  child_content,
                "metadata": {
                    "parent_id":   p_id,
                    "parent_content": p_content,
                    "block_types": ["text"],
                    "is_ocr":      False,
                    "section":     "",
                    "page_num":    -1,
                }
            }
            c_start += child_size - overlap


# ── 主流程 ────────────────────────────────────────────────────────────────────

STRATEGY = "semantic"

def build_chunk_id(title: str, idx: int) -> str:
    safe_title = "".join(c for c in title if c.isalnum() or c in ["-", "_"])[:30]
    return f"{safe_title}_{idx:05d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    meta   = data.get("meta", {})
    blocks = data.get("blocks", [])

    doc_type = meta.get("doc_type", "paper")
    title    = meta.get("title", "unknown")
    topic    = meta.get("topic", "")
    year     = meta.get("year", "")

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content":  text_chunk,
                "metadata": {"block_types": ["text"], "is_ocr": False, "section": "", "page_num": -1}
            })

    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    else:
        raise ValueError(f"未知策略: {strategy}")

    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(title, idx)
        chunk["chunk_id"]              = chunk_id
        chunk["metadata"]["doc_type"]   = doc_type
        chunk["metadata"]["title"]      = title
        chunk["metadata"]["topic"]      = topic
        chunk["metadata"]["year"]       = year
        chunk["metadata"]["strategy"]   = strategy
        chunk["metadata"]["source_file"] = parsed_path.name
        result.append(chunk)

    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  → {len(result)} 个 chunk，已保存 {out_path.name}")
    return result


def main():
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error("没有找到解析结果，请先运行 parse_pdf.py")
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)

    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    ocr_count   = sum(1 for c in all_chunks if c["metadata"].get("is_ocr"))
    logger.info(f"其中表格块: {table_count}  OCR块: {ocr_count}")


if __name__ == "__main__":
    main()