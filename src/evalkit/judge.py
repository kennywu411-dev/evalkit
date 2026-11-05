"""LLM 当裁判：逐题打分（pointwise）和两两比较（pairwise，交换位置各判一次）。

裁判模型通过 `client` 注入：一个函数，接收提示词字符串，返回模型原话，或 (原话, 花费美元)。
本模块不依赖任何模型 SDK，测试时用假的 client，接入时包一层真实 API 调用即可。

已知偏差和对策：
- 位置偏差：两两比较时把 A、B 交换再判一次，两次结论不一致记为平局（consistent=False），
  并统计位置一致率，一致率低说明裁判在这个任务上不可靠。
- 与人工不一致：用 calibrate() 在一批人工标注过的样本上算一致率和 Cohen's κ，
  一致率达标之前，裁判分数只作参考，不进门禁。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evalkit.dataset import Item
from evalkit.scorers import Score
from evalkit.stats import cohen_kappa

Client = Callable[[str], "str | tuple[str, float]"]

POINTWISE_PROMPT = """你是严格的评测员。按评分标准给回答打分。

评分标准：
{rubric}

问题：
{input}

参考答案：
{expected}

待评回答：
{output}

只输出一个 JSON 对象：{{"score": 1 到 {max_score} 的整数, "reason": "一句话理由"}}"""

PAIRWISE_PROMPT = """你是严格的评测员。比较两个回答哪个更好。

评判标准：
{rubric}

问题：
{input}

回答 A：
{a}

回答 B：
{b}

只输出一个 JSON 对象：{{"winner": "A" 或 "B" 或 "tie", "reason": "一句话理由"}}"""


def _call(client: Client, prompt: str) -> tuple[str, float]:
    out = client(prompt)
    if isinstance(out, tuple):
        return out[0], float(out[1])
    return out, 0.0


def parse_json_object(text: str) -> dict:
    """从模型原话里取第一个 JSON 对象（允许前后有多余文字或代码块标记）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"裁判输出里没有 JSON：{text[:200]}")
    return json.loads(m.group(0))


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


@dataclass
class PointwiseJudge:
    """逐题打分，可直接当打分器用。value = score / max_score，passed = score >= pass_score。"""

    client: Client
    rubric: str
    name: str = "llm_judge"
    max_score: int = 5
    pass_score: int = 4
    prompt: str = POINTWISE_PROMPT

    def __call__(self, item: Item, output: Any) -> Score:
        p = self.prompt.format(
            rubric=self.rubric,
            input=_text(item.input),
            expected=_text(item.expected),
            output=_text(output),
            max_score=self.max_score,
        )
        raw, cost = _call(self.client, p)
        try:
            d = parse_json_object(raw)
            s = int(d["score"])
        except (ValueError, KeyError, TypeError) as e:
            return Score(0.0, False, f"裁判输出无法解析：{e}", "judge_error", cost)
        s = max(1, min(self.max_score, s))
        return Score(
            s / self.max_score,
            s >= self.pass_score,
            {"score": s, "reason": d.get("reason", "")},
            cost_usd=cost,
        )


@dataclass
class PairwiseJudge:
    """两两比较，交换位置各判一次。"""

    client: Client
    rubric: str
    prompt: str = PAIRWISE_PROMPT

    def _once(self, question: Any, a: Any, b: Any) -> tuple[str, float]:
        raw, cost = _call(
            self.client,
            self.prompt.format(rubric=self.rubric, input=_text(question), a=_text(a), b=_text(b)),
        )
        try:
            w = str(parse_json_object(raw).get("winner", "")).strip().upper()
        except ValueError:
            w = ""
        return (w if w in ("A", "B", "TIE") else "INVALID"), cost

    def compare(self, question: Any, a: Any, b: Any) -> dict:
        first, c1 = self._once(question, a, b)
        second, c2 = self._once(question, b, a)
        swapped = {"A": "B", "B": "A"}.get(second, second)  # 第二次的 A 是原来的 B
        consistent = first == swapped and first != "INVALID"
        winner = {"A": "a", "B": "b", "TIE": "tie"}[first] if consistent else "tie"
        return {
            "winner": winner,
            "consistent": consistent,
            "votes": [first, swapped],
            "cost_usd": c1 + c2,
        }

    def compare_many(self, triples: list[tuple[Any, Any, Any]]) -> dict:
        """批量比较 (问题, A, B)：胜 / 负 / 平计数、胜率（平局各算半场）、位置一致率。"""
        results = [self.compare(q, a, b) for q, a, b in triples]
        c = Counter(r["winner"] for r in results)
        n = len(results)
        return {
            "n": n,
            "a_wins": c["a"],
            "b_wins": c["b"],
            "ties": c["tie"],
            "b_win_rate": (c["b"] + c["tie"] / 2) / n if n else 0.0,
            "position_consistency": sum(r["consistent"] for r in results) / n if n else 0.0,
            "cost_usd": sum(r["cost_usd"] for r in results),
            "results": results,
        }


def calibrate(judge_labels: dict, human_labels: dict) -> dict:
    """裁判标签与人工标签的一致率、Cohen's κ、混淆矩阵和不一致的样本 id（只比两边都有的样本）。"""
    ids = sorted(set(judge_labels) & set(human_labels), key=str)
    j = [judge_labels[i] for i in ids]
    h = [human_labels[i] for i in ids]
    n = len(ids)
    confusion: dict[str, dict[str, int]] = {}
    for a, b in zip(h, j, strict=True):
        confusion.setdefault(str(b), {}).setdefault(str(a), 0)
        confusion[str(b)][str(a)] += 1
    return {
        "n": n,
        "agreement": sum(a == b for a, b in zip(j, h, strict=True)) / n if n else 0.0,
        "kappa": cohen_kappa(j, h) if n else 0.0,
        "confusion": confusion,  # confusion[裁判标签][人工标签] = 数量
        "disagreements": [i for i, a, b in zip(ids, j, h, strict=True) if a != b],
        "missing_human": len(set(judge_labels) - set(human_labels)),
    }
