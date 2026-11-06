"""入门示例：一个"回答首都"的假应用，两个版本，用来演示运行、对比和门禁（不调用任何模型）。

v1 查表回答；v2 模拟一次"改坏了"的发布：对非洲和美洲的问题会答错一部分。
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit.runner import TaskResult

_ANSWERS = {}
for line in (Path(__file__).with_name("capitals.jsonl")).read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    ans = d["expected"][0] if isinstance(d["expected"], list) else d["expected"]
    _ANSWERS[d["input"]] = ans


def answer_v1(item):
    return TaskResult(output=_ANSWERS[item.input], cost_usd=0.0001, trace=[{"step": "lookup"}])


def answer_v2(item):
    out = _ANSWERS[item.input]
    if item.tags["region"] in ("africa", "americas") and int(item.id[-2:]) % 2 == 0:
        out = "不知道"
    return TaskResult(output=out, cost_usd=0.0002, trace=[{"step": "lookup"}, {"step": "rewrite"}])
