"""两次运行的配对比较：同一批题上，A → B 的主指标变化是否显著。

只比较两边都有的题（按 item_id 对齐），报告：
- 两边通过率、差值和配对 bootstrap 区间；
- McNemar 精确检验：只看"A 对 B 错"和"B 对 A 错"的题，比较两个总分更严谨；
- 成本、延迟变化；可选按标签拆分（子集上分别检验）。
"""

from __future__ import annotations

from collections.abc import Iterable

from evalkit.report import tag_value
from evalkit.stats import mcnemar_exact, paired_bootstrap_diff, percentile


def select(rows: list[dict], variant: str | None = None, repeat: int | None = 0) -> list[dict]:
    """取出某个变体、某次重复的结果行。variant 为空且只有一个变体时直接用它。"""
    variants = sorted({r["variant"] for r in rows})
    if variant is None:
        if len(variants) != 1:
            raise ValueError(f"运行里有多个变体 {variants}，需要指定 variant")
        variant = variants[0]
    out = [
        r
        for r in rows
        if r["variant"] == variant and (repeat is None or r.get("repeat", 0) == repeat)
    ]
    if not out:
        raise ValueError(f"没有变体 {variant!r}（可选 {variants}）")
    return out


def _paired(a: list[dict], b: list[dict]) -> tuple[list, dict, dict]:
    da = {r["item_id"]: r for r in a}
    db = {r["item_id"]: r for r in b}
    ids = sorted(set(da) & set(db), key=str)
    return ids, da, db


def _block(ids: list, da: dict, db: dict) -> dict:
    n = len(ids)
    xa = [float(bool(da[i]["passed"])) for i in ids]
    xb = [float(bool(db[i]["passed"])) for i in ids]
    only_a = [i for i in ids if da[i]["passed"] and not db[i]["passed"]]
    only_b = [i for i in ids if db[i]["passed"] and not da[i]["passed"]]
    boot = paired_bootstrap_diff(xa, xb) if n else {"diff": 0.0, "ci": (0.0, 0.0)}
    return {
        "n": n,
        "rate_a": sum(xa) / n if n else 0.0,
        "rate_b": sum(xb) / n if n else 0.0,
        "delta": boot["diff"],
        "delta_ci95": list(boot["ci"]),
        "a_only": only_a,
        "b_only": only_b,
        "p_value": mcnemar_exact(len(only_a), len(only_b)),
    }


def compare(
    rows_a: list[dict],
    rows_b: list[dict],
    variant_a: str | None = None,
    variant_b: str | None = None,
    repeat: int | None = 0,
    slices: Iterable[str] = (),
) -> dict:
    a = select(rows_a, variant_a, repeat)
    b = select(rows_b, variant_b, repeat)
    ids, da, db = _paired(a, b)
    if not ids:
        raise ValueError("两次运行没有共同的题目")
    out = {
        "variant_a": a[0]["variant"],
        "variant_b": b[0]["variant"],
        **_block(ids, da, db),
        "only_in_a": len(set(da) - set(db)),
        "only_in_b": len(set(db) - set(da)),
        "cost_a": sum(da[i].get("cost_usd", 0.0) for i in ids),
        "cost_b": sum(db[i].get("cost_usd", 0.0) for i in ids),
        "latency_p50_a": percentile([da[i].get("latency_s", 0.0) for i in ids], 0.5),
        "latency_p50_b": percentile([db[i].get("latency_s", 0.0) for i in ids], 0.5),
        "errors_a": sum(da[i]["status"] == "error" for i in ids),
        "errors_b": sum(db[i]["status"] == "error" for i in ids),
    }
    if slices:
        out["slices"] = {}
        for s in slices:
            vals = sorted({tag_value(da[i], s) for i in ids}, key=str)
            out["slices"][s] = {
                str(v): _block([i for i in ids if tag_value(da[i], s) == v], da, db) for v in vals
            }
    return out


def render(c: dict) -> str:
    lo, hi = c["delta_ci95"]
    lines = [
        f"{c['variant_a']} → {c['variant_b']}：共同题目 {c['n']}",
        f"通过率 {c['rate_a']:.1%} → {c['rate_b']:.1%}，差值 {c['delta']:+.1%}"
        f"（95% CI {lo:+.1%} ~ {hi:+.1%}）",
        f"A 独对 {len(c['a_only'])}，B 独对 {len(c['b_only'])}，McNemar p = {c['p_value']:.3f}",
        f"花费 ${c['cost_a']:.4f} → ${c['cost_b']:.4f}；"
        f"P50 延迟 {c['latency_p50_a']:.2f}s → {c['latency_p50_b']:.2f}s",
    ]
    for s, vals in (c.get("slices") or {}).items():
        for v, b in vals.items():
            lines.append(
                f"  [{s}={v}] n={b['n']} {b['rate_a']:.1%} → {b['rate_b']:.1%}"
                f"（{len(b['a_only'])}/{len(b['b_only'])}，p={b['p_value']:.3f}）"
            )
    return "\n".join(lines)
