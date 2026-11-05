"""运行目录的读写。一次运行 = 一个目录：

    runs/<run_id>/
      config.json     配置快照：评测集指纹、各变体参数、打分器、题目 id、代码版本
      results.jsonl   每题一行，跑完立即追加；续跑时同一 key 以最后一行为准
      summary.json    汇总（运行结束时生成，也可以用 `evalkit report` 重算）
      traces/<trace_id>.json  每题的完整过程：输入、输出、中间步骤、各打分器结果

结果行的 key 是 (item_id, variant, repeat)。
"""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

RESULTS = "results.jsonl"
CONFIG = "config.json"
SUMMARY = "summary.json"
TRACES = "traces"


def new_run_id(tag: str) -> str:
    return f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def row_key(row: dict) -> tuple:
    return (row["item_id"], row["variant"], row.get("repeat", 0))


def trace_id(item_id: Any, variant: str, repeat: int) -> str:
    return hashlib.sha1(f"{item_id}|{variant}|{repeat}".encode()).hexdigest()[:12]


def load_rows(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / RESULTS
    if not path.exists():
        return []
    latest: dict[tuple, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                latest[row_key(r)] = r
    return list(latest.values())


def append_row(run_dir: Path, row: dict) -> None:
    with (run_dir / RESULTS).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_rows(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / RESULTS).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_config(run_dir: str | Path) -> dict:
    p = Path(run_dir) / CONFIG
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_trace(run_dir: Path, tid: str, trace: dict) -> None:
    write_json(run_dir / TRACES / f"{tid}.json", trace)


def load_trace(run_dir: str | Path, tid: str) -> dict | None:
    p = Path(run_dir) / TRACES / f"{tid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_runs(root: str | Path) -> list[Path]:
    """root 下所有含 results.jsonl 的运行目录（root 本身是运行目录时只返回它）。"""
    root = Path(root)
    if (root / RESULTS).exists():
        return [root]
    return sorted(p for p in root.iterdir() if (p / RESULTS).exists())


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
