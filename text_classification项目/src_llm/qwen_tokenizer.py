"""
Qwen2 本地 tokenizer 加载（兼容仅有 tokenizer.json 而无 tokenizer_config.json 的情况）
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

# 与 HuggingFace Qwen2-0.5B-Instruct 官方 tokenizer_config.json 一致
QWEN2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.\n' }}"
    "{% endif %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def ensure_chat_template(tokenizer, model_path: Path | None = None) -> None:
    """若本地缺少 chat_template，从 tokenizer_config.json 或内置模板补齐。"""
    if getattr(tokenizer, "chat_template", None):
        return

    if model_path is not None:
        cfg_file = Path(model_path) / "tokenizer_config.json"
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("chat_template"):
                tokenizer.chat_template = cfg["chat_template"]
                return

    tokenizer.chat_template = QWEN2_CHAT_TEMPLATE
    if model_path is not None:
        _write_tokenizer_config_if_missing(Path(model_path))
    else:
        print("提示: 已使用内置 Qwen2 chat_template。")


def _write_tokenizer_config_if_missing(model_path: Path) -> None:
    """写入最小 tokenizer_config.json，避免每次提示且无需 huggingface-cli。"""
    cfg_file = model_path / "tokenizer_config.json"
    if cfg_file.exists():
        return
    try:
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_max_length": 32768,
                    "tokenizer_class": "Qwen2Tokenizer",
                    "chat_template": QWEN2_CHAT_TEMPLATE,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"已生成 {cfg_file.name}（内置 chat_template）")
    except OSError as e:
        print(f"提示: 无法写入 tokenizer_config.json: {e}")


def load_qwen_tokenizer(model_path: str | Path) -> AutoTokenizer:
    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    ensure_chat_template(tokenizer, path)
    return tokenizer
