"""打分器：输入一道题和模型输出，返回 Score。

约定：打分器是一个可调用对象 `scorer(item, output) -> Score`，并带 `name` 属性。
- value：0–1 的分数（规则打分可以是通过比例）；
- passed：是否判对，二值指标和配对检验用它；
- detail：判分依据，失败回放时展示，规则打分器可以放逐项结果（dict）。

LLM 裁判在 judge.py，人工标注导入在 labels.py。
"""

from __future__ import annotations

import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.dataset import Item


@dataclass
class Score:
    value: float
    passed: bool | None = None
    detail: Any = ""
    status: str = "ok"
    cost_usd: float = 0.0  # 打分本身的花费（LLM 裁判），计入该题总花费

    def to_dict(self) -> dict:
        d = {"value": self.value, "passed": self.passed, "detail": self.detail}
        if self.status != "ok":
            d["status"] = self.status
        if self.cost_usd:
            d["cost_usd"] = self.cost_usd
        return d


@dataclass
class FunctionScorer:
    """把普通函数包装成打分器：函数返回 bool、数字或 Score 都可以。"""

    name: str
    fn: Callable[[Item, Any], Any]

    def __call__(self, item: Item, output: Any) -> Score:
        out = self.fn(item, output)
        if isinstance(out, Score):
            return out
        if isinstance(out, bool):
            return Score(float(out), out)
        return Score(float(out), None)


def scorer(name: str):
    """装饰器写法：@scorer("has_citation") def f(item, output): ..."""

    def wrap(fn):
        return FunctionScorer(name, fn)

    return wrap


def normalize_text(s: str) -> str:
    """全角转半角、去首尾空白、合并空白、转小写。"""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    return re.sub(r"\s+", " ", s)


@dataclass
class ExactMatch:
    """精确匹配。expected 可以是单个答案或候选答案列表，命中任意一个即算对。"""

    name: str = "exact_match"
    normalize: bool = True
    field: str | None = None  # 输出是 dict 时取其中一个字段比较

    def __call__(self, item: Item, output: Any) -> Score:
        got = output.get(self.field) if self.field and isinstance(output, dict) else output
        cands = item.expected if isinstance(item.expected, list) else [item.expected]
        f = normalize_text if self.normalize else str
        ok = got is not None and any(f(got) == f(c) for c in cands)
        return Score(float(ok), ok, "" if ok else f"期望 {cands!r}，得到 {got!r}")


# ---------- 执行结果比对（SQL） ----------


class SQLTimeout(Exception):
    pass


def run_sql_readonly(db_path: str | Path, sql: str, deadline: float) -> list[tuple]:
    """只读连接执行 SQL；超过 deadline（time.monotonic 时间点）就中断。"""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)

    def check():
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(check, 10_000)
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise SQLTimeout from e
        raise
    finally:
        conn.close()


@dataclass
class SQLResultMatch:
    """执行结果比对：分别执行预测 SQL 和标准 SQL，比较结果集。

    默认口径与 BIRD 官方 EX 一致：set(pred_rows) == set(gold_rows)，忽略行顺序和重复行，
    列顺序和取值必须一致；预测与标准共用一个超时预算。数据库一律只读打开。
    db: 题目 → 数据库路径的函数；sql_of: 从模型输出里取 SQL（默认输出本身就是 SQL 字符串）。
    """

    db: Callable[[Item], str | Path]
    name: str = "execution_match"
    timeout_s: float = 30.0
    ordered: bool = False
    sql_of: Callable[[Any], str | None] = field(default=lambda out: out)

    def __call__(self, item: Item, output: Any) -> Score:
        pred = self.sql_of(output)
        if not pred:
            return Score(0.0, False, "没有 SQL", "no_output")
        path = self.db(item)
        deadline = time.monotonic() + self.timeout_s
        try:
            pred_rows = run_sql_readonly(path, pred, deadline)
        except SQLTimeout:
            return Score(0.0, False, "预测 SQL 超时", "pred_timeout")
        except sqlite3.Error as e:
            return Score(0.0, False, f"预测 SQL 出错：{e}", "pred_error")
        try:
            gold_rows = run_sql_readonly(path, item.expected, deadline)
        except SQLTimeout:
            return Score(0.0, False, "标准 SQL 超时（与预测共用预算）", "timeout")
        except sqlite3.Error as e:
            return Score(0.0, False, f"标准 SQL 出错：{e}", "gold_error")
        ok = pred_rows == gold_rows if self.ordered else set(pred_rows) == set(gold_rows)
        detail = "" if ok else f"预测 {len(pred_rows)} 行，标准 {len(gold_rows)} 行"
        return Score(float(ok), ok, detail)


# ---------- 规则打分 ----------


Check = Callable[[Item, Any], bool]


@dataclass
class RuleScorer:
    """规则打分：多条具名检查，value = 通过比例，passed = 全部通过（或达到 threshold）。

    检查函数抛异常按不通过处理，并把异常写进 detail，便于发现规则本身的问题。
    """

    name: str
    checks: list[tuple[str, Check]]
    threshold: float = 1.0

    def __call__(self, item: Item, output: Any) -> Score:
        results: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for cname, fn in self.checks:
            try:
                results[cname] = bool(fn(item, output))
            except Exception as e:  # 规则写错不应中断整次评测
                results[cname] = False
                errors[cname] = f"{type(e).__name__}: {e}"
        value = sum(results.values()) / len(results) if results else 0.0
        detail = {"checks": results, **({"errors": errors} if errors else {})}
        return Score(value, value >= self.threshold, detail)


def contains_all(keywords: Iterable[str], field: str | None = None) -> Check:
    kws = list(keywords)

    def check(item: Item, output: Any) -> bool:
        text = output.get(field) if field and isinstance(output, dict) else output
        return all(k in str(text) for k in kws)

    return check


def matches_regex(pattern: str, field: str | None = None) -> Check:
    rx = re.compile(pattern)

    def check(item: Item, output: Any) -> bool:
        text = output.get(field) if field and isinstance(output, dict) else output
        return rx.search(str(text)) is not None

    return check


def max_length(n: int, field: str | None = None) -> Check:
    def check(item: Item, output: Any) -> bool:
        text = output.get(field) if field and isinstance(output, dict) else output
        return len(str(text)) <= n

    return check
