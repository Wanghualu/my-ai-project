"""
PDF 解析脚本：将论文/技术文档 PDF 转换为结构化文本

教学重点（企业级 RAG 的真实挑战）：
  1. 数字 PDF vs 扫描件：处理方式完全不同
  2. 表格提取：论文里的实验结果表格，直接按文字流提取会乱序
  3. 页眉/页脚噪声：每页都有标题、页码，必须去除
  4. 章节识别：利用字体大小/加粗识别标题层级，支持中英文
  5. 输出格式：保留元信息（页码、章节路径），供后续溯源用

依赖安装：
  pip install pdfplumber pymupdf pytesseract pillow
  # tesseract-ocr 需要单独安装并配置 PATH
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber
import fitz

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_pdf"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int
    section_path: list[str]      # ["Abstract", "1. Introduction", "1.1 Background"]
    is_ocr:       bool = False
    raw_table:    Optional[list] = field(default=None, repr=False)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

CHINESE_TITLE_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]"),
    re.compile(r"^[一二三四五六七八九十]、"),
    re.compile(r"^\d+\.\s"),
]

ENGLISH_TITLE_PATTERNS = [
    re.compile(r"^(Abstract|Introduction|Related Work|Background|Method|Methods|Experiments?|Experimental Results?|Results?|Discussion|Conclusion|Conclusions?|References?|Appendix)[\s\.:]*$", re.IGNORECASE),
    re.compile(r"^\d+(\.\d+)*\s+[A-Z][a-zA-Z][^\n]{0,80}$"),
    re.compile(r"^\d+\.\d+(\.\d+)*\s"),
    re.compile(r"^[A-Z]{2,}[^\n]{0,100}$"),
]

NOISE_PATTERNS = [
    re.compile(r"^.{1,60}(年度报告|Report|Proceedings|arXiv)\s*$"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^—\s*\d+\s*—$"),
    re.compile(r"^-\s*\d+\s*-?$"),
    re.compile(r"^(\d+\s+)?[A-Z][a-zA-Z]+\s+[0-9]+$"),
]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    if line.isdigit():
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    line_stripped = line.strip()
    if not line_stripped:
        return False
    
    if fontsize and fontsize >= 14:
        return True
    if is_bold and len(line_stripped) < 100:
        return True
    
    for p in CHINESE_TITLE_PATTERNS:
        if p.match(line_stripped):
            return True
    for p in ENGLISH_TITLE_PATTERNS:
        if p.match(line_stripped):
            return True
    
    return False


def table_to_markdown(table: list[list]) -> str:
    if not table:
        return ""

    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def detect_if_scanned(page: fitz.Page, text: str) -> bool:
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images(full=True)
    return len(image_list) > 0


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用（未安装 pytesseract/tesseract），内容跳过]"
    try:
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        clip = page.rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        logger.warning(f"  OCR 失败，跳过此页: {e}")
        return "[扫描页，OCR 失败，内容跳过]"


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

class TechnicalDocParser:
    def __init__(self, pdf_path: Path, meta: dict = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []

    def _update_section(self, title: str):
        title_stripped = title.strip()
        
        numbered_match = re.match(r"^(\d+)(\.\d+)*\s+(.+)$", title_stripped)
        if numbered_match:
            levels = numbered_match.group(1).split('.')
            main_num = int(numbered_match.group(1))
            
            if len(self._section_stack) == 0:
                self._section_stack = [title_stripped]
            else:
                prev_num_match = re.match(r"^(\d+)", self._section_stack[-1])
                if prev_num_match:
                    prev_num = int(prev_num_match.group(1))
                    if main_num > prev_num:
                        self._section_stack.append(title_stripped)
                    else:
                        while self._section_stack:
                            last_num_match = re.match(r"^(\d+)", self._section_stack[-1])
                            if not last_num_match or int(last_num_match.group(1)) < main_num:
                                break
                            self._section_stack.pop()
                        self._section_stack.append(title_stripped)
                else:
                    self._section_stack.append(title_stripped)
        else:
            if re.match(r"^(Abstract|Introduction|Related Work|Conclusion|References?|Appendix)$", 
                       title_stripped, re.IGNORECASE):
                self._section_stack = [title_stripped]
            else:
                if len(self._section_stack) >= 2:
                    self._section_stack = self._section_stack[:2] + [title_stripped]
                else:
                    self._section_stack = self._section_stack + [title_stripped]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page   = fitz_doc[page_num]
            plumb_page  = plumber_doc.pages[page_num]

            raw_text = fitz_page.get_text("text")
            is_scanned = detect_if_scanned(fitz_page, raw_text)

            if is_scanned:
                logger.debug(f"  第{page_num+1}页：检测到扫描件，启动 OCR")
                ocr_text = ocr_page(fitz_page)
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content=ocr_text,
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                    is_ocr=True,
                ))
                continue

            table_bboxes = []
            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md:
                        self.blocks.append(ParsedBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                            raw_table=table,
                        ))
            for table_obj in plumb_page.find_tables():
                table_bboxes.append(table_obj.bbox)

            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            current_para_lines = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()

                    if not line_text or is_noise_line(line_text):
                        continue

                    spans    = line.get("spans", [])
                    fontsize = spans[0].get("size", 0) if spans else 0
                    is_bold  = any("Bold" in span.get("font", "") for span in spans)

                    if is_title_line(line_text, fontsize, is_bold):
                        if current_para_lines:
                            self.blocks.append(ParsedBlock(
                                block_type="text",
                                content="\n".join(current_para_lines),
                                page_num=page_num + 1,
                                section_path=list(self._section_stack),
                            ))
                            current_para_lines = []

                        self._update_section(line_text)
                        self.blocks.append(ParsedBlock(
                            block_type="title",
                            content=line_text,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                    else:
                        current_para_lines.append(line_text)

            if current_para_lines:
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content="\n".join(current_para_lines),
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                ))

        plumber_doc.close()
        fitz_doc.close()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def save(self):
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "meta":   self.meta,
            "source": str(self.pdf_path),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    manifest_path = RAW_DIR.parent / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = [
            {"filename": p.name, "doc_type": "paper", "title": p.stem, "topic": ""}
            for p in RAW_DIR.glob("*.pdf")
        ]

    if not manifest:
        logger.error("没有找到任何 PDF，请先将 PDF 放入 data/raw_pdf/")
        return

    for item in manifest:
        pdf_path = RAW_DIR / item["filename"]
        if not pdf_path.exists():
            logger.warning(f"文件不存在，跳过: {pdf_path}")
            continue

        parser = TechnicalDocParser(pdf_path, meta=item)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()