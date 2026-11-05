"""线上反馈：用户对模型输出打标（如"有用 / 没用 / 不是风险 / 看不懂"），按最后一次为准汇总。

每次打标追加一行 JSONL（只追加、不改写，保留改标历史）；汇总时同一个目标以最后一次为准。
目标由 key 里的字段确定，例如合同审查是 (会话, 审查, 第几条意见)。

隐私：只存调用方传进来的字段。建议只记 id、标签和输出摘要，不记用户原文。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


def get_path(row: dict, path: str) -> Any:
    """按点号路径取嵌套字段：get_path(r, "finding.level")。"""
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class FeedbackStore:
    def __init__(
        self,
        path: str | Path,
        labels: Sequence[str] | None = None,
        required: Sequence[str] = (),
    ):
        self.path = Path(path)
        self.labels = tuple(labels) if labels else None
        self.required = tuple(required)
        self._lock = threading.Lock()

    def add(self, entry: dict) -> dict:
        if self.labels is not None and entry.get("label") not in self.labels:
            raise ValueError(f"未知标签 {entry.get('label')!r}，可选 {self.labels}")
        missing = [k for k in self.required if get_path(entry, k) is None]
        if missing:
            raise ValueError(f"缺少字段 {missing}")
        row = {"ts": time.time(), **entry}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def latest(rows: Iterable[dict], key: Sequence[str]) -> list[dict]:
    """同一目标只保留最后一次打标。文件追加顺序即时间顺序；有 ts 时按 ts 稳定排序。"""
    ordered = sorted(enumerate(rows), key=lambda x: (x[1].get("ts", 0), x[0]))
    out: dict[tuple, dict] = {}
    for _, r in ordered:
        out[tuple(get_path(r, k) for k in key)] = r
    return list(out.values())


def _rate(n: int, total: int) -> float | None:
    return n / total if total else None


def summarize(
    rows: Iterable[dict],
    key: Sequence[str],
    labels: Sequence[str],
    positive: str | None = None,
    negative: str | None = None,
    where: Callable[[dict], bool] | None = None,
    by: str | None = None,
    session_key: str = "session_id",
) -> dict:
    """汇总反馈。

    positive / negative：主指标对应的标签（如 useful → 采纳率，not_risk → 误报率）；
    where：只统计满足条件的打标（如只算高、中等级的意见）；
    by：按某个字段拆分（点号路径，如 "finding.risk_type"）。
    """
    cur = [r for r in latest(rows, key) if where is None or where(r)]
    counts = {lab: sum(r.get("label") == lab for r in cur) for lab in labels}
    out: dict[str, Any] = {
        "rated": len(cur),
        "sessions": len({get_path(r, session_key) for r in cur} - {None}),
        "labels": counts,
        "rates": {lab: _rate(c, len(cur)) for lab, c in counts.items()},
    }
    if positive:
        out["positive_rate"] = _rate(counts.get(positive, 0), len(cur))
    if negative:
        out["negative_rate"] = _rate(counts.get(negative, 0), len(cur))
    if by:
        groups: dict[str, dict] = {}
        for r in cur:
            g = groups.setdefault(str(get_path(r, by)), {"rated": 0, **dict.fromkeys(labels, 0)})
            g["rated"] += 1
            if r.get("label") in g:
                g[r["label"]] += 1
        out["by"] = {"field": by, "groups": groups}
    return out
