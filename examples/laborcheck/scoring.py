"""劳动法评测（laborcheck）的打分规则，用 evalkit 的 RuleScorer 重写。

模型回答末尾输出一个 JSON 对象：verdict（合法 / 违法）、amount、key_value、rewrite、basis。
导入运行记录时先用 extract_answer 取出这个对象作为 output，打分只看它：
- 判断题：结论一致；
- 计算题：金额在每题的误差容忍内；
- 改写题：结论一致、关键数值满足约束（如试用期 <= 2 个月）、给出了改写文本。
依据命中（模型列出的依据里至少命中一条标准依据的条号）单独一个打分器，只作参考，不计入准确率。
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalkit.dataset import Item
from evalkit.scorers import RuleScorer, Score

ARTICLE = re.compile(r"第[一二三四五六七八九十百零〇]+条")


def extract_answer(text: str | None) -> dict | None:
    """取回答里最后一个能解析成 JSON 对象的 {...}（允许嵌套、允许包在代码块里）。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    for end in (i for i in range(len(text) - 1, -1, -1) if text[i] == "}"):
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


def number(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    s = re.sub(r"[,，元天]", "", str(v))
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def verdict(v: Any) -> str | None:
    s = "" if v is None else str(v)
    if any(w in s for w in ("不合法", "违法", "无效")):
        return "违法"
    if any(w in s for w in ("合法", "有效")):
        return "合法"
    return None


def _fmt(item: Item) -> str:
    return item.input["format"]


def _parsed(item: Item, out: Any) -> bool:
    return isinstance(out, dict)


def _verdict_ok(item: Item, out: Any) -> bool:
    if _fmt(item) == "amount":
        return True
    return verdict(out.get("verdict")) == item.expected["verdict"]


def _amount_ok(item: Item, out: Any) -> bool:
    if _fmt(item) != "amount":
        return True
    a = number(out.get("amount"))
    exp = item.expected
    return a is not None and abs(a - exp["amount"]) <= exp["tolerance"] + 1e-9


def _key_value_ok(item: Item, out: Any) -> bool:
    if _fmt(item) != "rewrite":
        return True
    v, c = number(out.get("key_value")), item.expected["constraint"]
    if v is None:
        return False
    eps = 1e-9
    ok = v <= c["value"] + eps if c["op"] == "<=" else v >= c["value"] - eps
    return ok and v >= c.get("min", float("-inf")) - eps and v <= c.get("max", float("inf")) + eps


def _rewrite_text_ok(item: Item, out: Any) -> bool:
    return _fmt(item) != "rewrite" or bool(str(out.get("rewrite") or "").strip())


def answer_scorer() -> RuleScorer:
    """主指标：各项检查全部通过才算答对；没解析出 JSON 时其余检查按不通过处理。"""
    return RuleScorer(
        "laborcheck",
        [
            ("parsed", _parsed),
            ("verdict", _verdict_ok),
            ("amount", _amount_ok),
            ("key_value", _key_value_ok),
            ("rewrite_text", _rewrite_text_ok),
        ],
    )


class BasisScorer:
    """依据命中：标准依据里没有条号的题不参与（passed=None）。"""

    name = "basis_hit"

    def __call__(self, item: Item, out: Any) -> Score:
        want = {a for cite in item.expected.get("basis", []) for a in ARTICLE.findall(cite)}
        if not want:
            return Score(0.0, None, "标准依据没有条号")
        got = out.get("basis") if isinstance(out, dict) else None
        got = got or []
        text = " ".join(map(str, got)) if isinstance(got, list) else str(got)
        hit = any(a in text for a in want)
        return Score(float(hit), hit)
