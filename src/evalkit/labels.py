"""人工标注导入：CSV 或 JSONL，每行至少有样本 id 和标签。

同一样本被标多次时以文件里最后一次为准（与线上反馈一致）。
导入后可以：当打分器用（HumanLabelScorer），或与 LLM 裁判做一致率校准（judge.calibrate）。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.dataset import Item
from evalkit.scorers import Score


def _coerce_id(v: str) -> Any:
    return int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else v


def load_labels(
    path: str | Path, id_field: str = "id", label_field: str = "label", annotator: str | None = None
) -> dict:
    """返回 {样本 id: 标签}。annotator 不为空时只取该标注者的行（需要 annotator 列）。"""
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    out: dict = {}
    for r in rows:
        if annotator is not None and r.get("annotator") != annotator:
            continue
        if r.get(label_field) in (None, ""):
            continue
        out[_coerce_id(r[id_field])] = r[label_field]
    return out


@dataclass
class HumanLabelScorer:
    """把人工标注当打分器：标签在 positive 里算对；没标注的题记为 status=unlabeled、不计入。"""

    labels: dict
    positive: tuple = ("correct", "pass", "1", 1, True)
    name: str = "human"

    def __call__(self, item: Item, output: Any) -> Score:
        if item.id not in self.labels:
            return Score(0.0, None, "未标注", "unlabeled")
        lab = self.labels[item.id]
        ok = lab in self.positive
        return Score(float(ok), ok, {"label": lab})
