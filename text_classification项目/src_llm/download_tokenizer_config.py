"""下载 Qwen2 tokenizer_config.json（无需 huggingface-cli 命令）"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        raise SystemExit(1)

    path = hf_hub_download(
        repo_id="Qwen/Qwen2-0.5B-Instruct",
        filename="tokenizer_config.json",
        local_dir=str(MODEL_DIR),
    )
    print(f"已下载 → {path}")


if __name__ == "__main__":
    main()
