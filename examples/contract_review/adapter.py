"""合同风险审查运行记录 → evalkit 运行目录。

原项目每次运行是 runs/<run_id>/：
    review_results.jsonl   每份合同一行：运行时算好的逐项计数（检出、误报……）、花费、延迟、trace_id
    traces/<trace_id>.json 完整过程：每步模型调用（usage、耗时、工具调用）、最终审查结果、引用

映射：item_id = 合同 id，output = 审查结果（意见、引用、我方身份），
scores["recorded"] = 运行时记下的逐项计数（复现"跑的时候的口径"），
重新判分时用 ContractScorer 从 output 算（复现"金标准订正后的口径"）。
评测集只导出金标准和标签，不导出合同正文。
"""

from __future__ import annotations

import json
from pathlib import Path

from scoring import dev_ids

from evalkit import store
from evalkit.dataset import Dataset, Item
from evalkit.stats import percentile

EXPECTED_KEYS = (
    "gold",
    "borderline",
    "favorable",
    "acceptable",
    "distractors",
    "clarify",
    "our_side",
)
FINDING_KEYS = (
    "clause",
    "risk_type",
    "level",
    "needs_human_review",
    "citations",
    "quote_found",
    "quote",
    "reason",
)


def _group(c: dict) -> str:
    if c["clarify"]:
        return "clarify"
    if c.get("favorable"):
        return "favorable"
    if not c["gold"]:
        return "clean"
    return "borderline" if c.get("borderline") else "risky"


def load_contracts(path: Path, name: str = "contract-review", version: str = "1.0.0") -> Dataset:
    raw = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    dev = dev_ids(raw, 20)
    items = [
        Item(
            id=c["id"],
            input={k: c[k] for k in ("tenant", "contract_type", "title") if k in c},
            expected={
                k: c.get(k, [] if k not in ("clarify", "our_side") else None) for k in EXPECTED_KEYS
            },
            tags={"contract_type": c["contract_type"], "tenant": c["tenant"], "group": _group(c)},
            split="dev" if c["id"] in dev else "holdout",
        )
        for c in raw
    ]
    return Dataset(name, items, version)


def _citation(c: dict) -> dict:
    if c.get("kind") == "table":
        return {"kind": "table", "rows": len(c.get("rows") or [])}
    return {"kind": c.get("kind"), "doc_id": c.get("doc_id"), "articles": c.get("articles", [])}


def trace_output(t: dict) -> dict:
    res = t["result"]
    return {
        "kind": res["kind"],
        "our_side": res.get("our_side"),
        "findings": [
            {k: f.get(k) for k in FINDING_KEYS if k in f} for f in res.get("findings", [])
        ],
        "citations": {cid: _citation(c) for cid, c in t.get("citations", {}).items()},
    }


def trace_steps(t: dict) -> list[dict]:
    """回放用的精简步骤：每步的停止原因、token、花费、模型耗时、调用了哪些工具。"""
    return [
        {
            "step": s.get("step"),
            "stop_reason": s.get("stop_reason"),
            "usage": s.get("usage"),
            "cost_usd": s.get("cost_usd"),
            "llm_ms": s.get("llm_ms"),
            "tools": [c.get("name") for c in s.get("tool_calls") or []],
        }
        for s in t.get("steps", [])
    ]


def _recorded_score(r: dict) -> dict:
    keys = (
        "clarify",
        "clean",
        "kind",
        "side_ok",
        "clarify_ok",
        "gold",
        "borderline_n",
        "borderline_flagged",
        "borderline_human",
        "favorable_n",
        "favorable_flagged",
        "fp",
        "fp_hm",
        "n_findings",
        "matched_findings",
        "n_findings_hm",
        "matched_findings_hm",
        "strict_hits",
        "human_findings",
        "quote_found",
    )
    d = {k: r[k] for k in keys if k in r}
    d["gold"] = [
        {k: g[k] for k in ("clause", "type", "detected", "type_ok", "basis_ok", "human") if k in g}
        for g in r["gold"]
    ]
    bad = d["fp"] + d["favorable_flagged"] > 0
    if d["clarify"]:
        ok = bool(d["clarify_ok"])
    elif d["clean"]:
        ok = not bad
    else:
        ok = all(g["detected"] for g in d["gold"]) and not bad
    return {"value": float(ok), "passed": ok, "detail": d}


def import_run(src: Path, dst: Path, variant: str, ds: Dataset) -> list[dict]:
    split = {it.id: it for it in ds}
    rows = []
    for line in (src / "review_results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = json.loads((src / "traces" / f"{r['trace_id']}.json").read_text(encoding="utf-8"))
        rec = _recorded_score(r)
        it = split[r["id"]]
        rows.append(
            {
                "item_id": r["id"],
                "variant": variant,
                "repeat": 0,
                "split": it.split,
                "tags": it.tags,
                "status": "ok" if r["kind"] != "fallback" else "fallback",
                "passed": rec["passed"],
                "scores": {"recorded": rec},
                "output": trace_output(t),
                "cost_usd": r["cost_usd"],
                "latency_s": round(r["latency_ms"] / 1000, 3),
                # 模型耗时（各步 llm_ms 之和）：不受本机检索负载影响，原项目用它比较模型
                "llm_s": round(sum(s.get("llm_ms", 0) for s in t["steps"]) / 1000, 3),
                "steps": len(t["steps"]),
                "trace_id": r["trace_id"],
                "trace": trace_steps(t),
            }
        )
    store.write_rows(dst, rows)
    store.write_json(
        dst / store.CONFIG,
        {
            "run_id": dst.name,
            "tag": dst.name,
            "dataset": ds.fingerprint,
            "variants": {variant: {"model": t.get("model"), "task": t.get("task")}},
            "scorers": ["recorded"],
            "primary_metric": "recorded",
            "imported_from": "contract-review",
        },
    )
    return rows


def cost_stats(rows: list[dict]) -> tuple[float, float]:
    costs = [r["cost_usd"] for r in rows]
    return percentile(costs, 0.5), sum(costs)
