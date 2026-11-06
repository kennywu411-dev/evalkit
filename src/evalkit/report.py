"""汇总：通过率 + Wilson 置信区间、各打分器均分、按子集拆分、状态分布、成本、延迟、多次运行。

除了"每题一个对错"的通过率，很多任务的指标是跨题累加的比例（微平均）：
例如合同审查的检出率 = 所有合同里检出的风险数 / 风险总数，而不是各合同检出率的平均。
用 Ratio 声明这类指标，aggregate() 统一计算，并给出按题重抽样的 bootstrap 区间。
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from evalkit.stats import mean_std, percentile, stability, wilson

GROUP_KEYS = ("variant",)


def _rate_block(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(bool(r["passed"]) for r in rows)
    lo, hi = wilson(k, n)
    return {"n": n, "passed": k, "rate": k / n if n else 0.0, "ci95": [lo, hi]}


def tag_value(row: dict, key: str):
    if key == "split":
        return row.get("split")
    return (row.get("tags") or {}).get(key)


def summarize(
    rows: list[dict], by: Sequence[str] = GROUP_KEYS, slices: Iterable[str] = ()
) -> list[dict]:
    """按 by 分组汇总；slices 里的标签各自再拆一层（如难度、开发 / 留出集）。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        costs = [r.get("cost_usd", 0.0) for r in rs]
        lat = [r.get("latency_s", 0.0) for r in rs]
        score_names = dict.fromkeys(n for r in rs for n in (r.get("scores") or {}))
        item = {
            **dict(zip(by, key, strict=True)),
            **_rate_block(rs),
            "repeats": len({r.get("repeat", 0) for r in rs}),
            "scores": {
                name: statistics.fmean(
                    r["scores"][name]["value"] for r in rs if name in (r.get("scores") or {})
                )
                for name in score_names
            },
            "status": dict(Counter(r["status"] for r in rs)),
            "cost_total": sum(costs),
            "cost_per_item": sum(costs) / len(rs),
            "cost_p50": percentile(costs, 0.5),
            "latency_p50": percentile(lat, 0.5),
            "latency_p95": percentile(lat, 0.95),
        }
        if slices:
            item["slices"] = {
                s: {
                    str(v): _rate_block([r for r in rs if tag_value(r, s) == v])
                    for v in sorted({tag_value(r, s) for r in rs}, key=str)
                }
                for s in slices
            }
        out.append(item)
    return out


def repeat_stats(rows: list[dict]) -> dict:
    """同一变体跑多次（repeat 不同）：每次的通过率、均值 ± 标准差、极差、稳定性。"""
    by_rep: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_rep[r.get("repeat", 0)].append(r)
    rates = [_rate_block(by_rep[k])["rate"] for k in sorted(by_rep)]
    m, s = mean_std(rates)
    outcomes: dict = defaultdict(list)
    for k in sorted(by_rep):
        for r in by_rep[k]:
            outcomes[r["item_id"]].append(bool(r["passed"]))
    return {
        "runs": len(rates),
        "rates": rates,
        "mean": m,
        "std": s,
        "range": (max(rates) - min(rates)) if rates else 0.0,
        "stability": stability(outcomes),
    }


@dataclass
class Ratio:
    """跨题累加的比例指标：sum(num(row)) / sum(den(row))。"""

    name: str
    num: Callable[[dict], float]
    den: Callable[[dict], float]

    def value(self, rows: Sequence[dict]) -> float | None:
        d = sum(self.den(r) for r in rows)
        return sum(self.num(r) for r in rows) / d if d else None

    def counts(self, rows: Sequence[dict]) -> tuple[float, float]:
        return sum(self.num(r) for r in rows), sum(self.den(r) for r in rows)


def ratio_ci(
    metric: Ratio, rows: Sequence[dict], n_resamples: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    """按题重抽样的 bootstrap 区间（题内的多个计数一起抽，保留题内相关性）。"""
    rows = list(rows)
    if not rows:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_resamples):
        v = metric.value(rng.choices(rows, k=len(rows)))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))])


def aggregate(rows: Sequence[dict], metrics: Iterable[Ratio], ci: bool = False) -> dict:
    out = {}
    for m in metrics:
        num, den = m.counts(rows)
        entry = {"value": num / den if den else None, "num": num, "den": den}
        if ci:
            entry["ci95"] = ratio_ci(m, rows)
        out[m.name] = entry
    return out


def render(summary: list[dict], by: Sequence[str] = GROUP_KEYS) -> str:
    head = [*by, "题次", "通过率", "95% CI", "单题成本", "总成本", "P50 / P95 延迟", "状态"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summary:
        lo, hi = s["ci95"]
        cells = [str(s.get(k)) for k in by] + [
            str(s["n"]),
            f"{s['rate']:.1%}",
            f"{lo:.1%}–{hi:.1%}",
            f"${s['cost_per_item']:.4f}",
            f"${s['cost_total']:.3f}",
            f"{s['latency_p50']:.1f}s / {s['latency_p95']:.1f}s",
            ", ".join(f"{k} {v}" for k, v in s["status"].items()),
        ]
        lines.append("| " + " | ".join(cells) + " |")
        for sl, vals in (s.get("slices") or {}).items():
            parts = [f"{v} {b['passed']}/{b['n']}（{b['rate']:.1%}）" for v, b in vals.items()]
            lines.append(f"|  ↳ {sl} | " + "；".join(parts) + " |" + " |" * (len(head) - 2))
    return "\n".join(lines)
