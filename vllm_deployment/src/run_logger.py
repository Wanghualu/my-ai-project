"""
运行日志工具：将终端 print 输出同时写入 logs/ 目录，便于 AutoDL 运行后下载到本地。

用法：
    from run_logger import run_with_log

    def main():
        ...

    if __name__ == "__main__":
        run_with_log("bench_throughput", main)
"""

import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Callable

from config import LOG_DIR, ensure_dirs


class _Tee:
    """同时写入终端和日志文件。"""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data: str) -> None:
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()


@contextmanager
def capture_run_log(script_name: str):
    """捕获 stdout，返回本次日志文件路径。"""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{script_name}_{ts}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# script: {script_name}\n")
        log_file.write(f"# started: {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write("=" * 70 + "\n\n")

        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, log_file)
        try:
            yield log_path
            log_file.write(f"\n\n# finished: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write("# status: success\n")
        except Exception:
            log_file.write(f"\n\n# finished: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write("# status: failed\n")
            log_file.write(traceback.format_exc())
            raise
        finally:
            sys.stdout = old_stdout


def append_run_summary(entry: dict) -> str:
    """追加一条运行记录到 logs/run_summary.json。"""
    ensure_dirs()
    summary_path = os.path.join(LOG_DIR, "run_summary.json")

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = []

    records.append(entry)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return summary_path


def run_with_log(script_name: str, main_fn: Callable[[], None]) -> None:
    """包装 main()：捕获日志 + 写入 run_summary.json。"""
    started = datetime.now()
    log_path = ""
    status = "success"
    error_msg = None

    try:
        with capture_run_log(script_name) as path:
            log_path = path
            main_fn()
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()
        raise
    finally:
        finished = datetime.now()
        entry = {
            "script": script_name,
            "status": status,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "duration_sec": round((finished - started).total_seconds(), 2),
            "log_file": os.path.basename(log_path) if log_path else None,
            "error": error_msg,
        }
        summary_path = append_run_summary(entry)
        if log_path:
            print(f"\n运行日志已保存：{log_path}")
        print(f"运行摘要已更新：{summary_path}")
