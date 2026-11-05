"""DataPilot（NL2SQL 问数）运行记录 → evalkit 运行目录。

DataPilot 每次运行是 runs/<run_id>/results.jsonl，每行一题一个配置：
    question_id, db_id, difficulty, model, pipeline, effort, repeat, pred_sql, correct, status,
    cost_usd, latency_s, llm_calls, usage, trace, first_sql, first_correct ...

映射：item_id = question_id，variant = model，repeat 原样，passed = correct，
tags = {difficulty, db_id}，主打分器记为 execution_match（BIRD 官方口径的执行结果比对）。
续跑时出错的题会被重跑，同一 key 以最后一行为准，与 DataPilot 的 load_rows 一致。
"""

from __future__ import annotations

import json
from pathlib import Path

from evalkit import store
from evalkit.dataset import from_records

SCORER = "execution_match"


def load_source_rows(run_dir: Path) -> list[dict]:
    rows = [
        json.loads(x) for x in (run_dir / "results.jsonl").read_text().splitlines() if x.strip()
    ]
    latest = {
        (r["question_id"], r["model"], r["pipeline"], r["effort"], r["repeat"]): r for r in rows
    }
    return list(latest.values())


def convert_row(r: dict, repeat_offset: int = 0, keep_output: bool = True) -> dict:
    row = {
        "item_id": int(r["question_id"]),
        "variant": r["model"],
        "repeat": r["repeat"] + repeat_offset,
        "tags": {"difficulty": r["difficulty"], "db_id": r["db_id"]},
        "status": r["status"],
        "passed": bool(r["correct"]),
        "scores": {SCORER: {"value": float(r["correct"]), "passed": bool(r["correct"])}},
        "output": r.get("pred_sql") if keep_output else None,
        "cost_usd": round(r["cost_usd"], 8),
        "latency_s": round(r["latency_s"], 4),
        "trace_id": store.trace_id(r["question_id"], r["model"], r["repeat"] + repeat_offset),
        "llm_calls": r.get("llm_calls", 1),
    }
    if keep_output and r.get("trace"):
        row["trace"] = r["trace"]  # DataPilot 把中间步骤记在结果行里，看板直接读这里
    if r.get("first_sql") and r["first_sql"] != r.get("pred_sql"):
        row["first_correct"] = r.get("first_correct")
    return row


def import_runs(
    sources: list[tuple[Path, int]],
    dst: Path,
    pipeline_label: str,
    keep_output: bool = True,
) -> list[dict]:
    """把一个或多个 DataPilot 运行合成一个 evalkit 运行。

    sources：[(运行目录, repeat 偏移)]，多次重复分散在不同运行时用偏移把 repeat 接起来。
    """
    rows = []
    for src, offset in sources:
        rows += [convert_row(r, offset, keep_output) for r in load_source_rows(src)]
    store.write_rows(dst, rows)
    store.write_json(
        dst / store.CONFIG,
        {
            "run_id": dst.name,
            "tag": dst.name,
            "dataset": {"name": "bird-minidev", "n": len({r["item_id"] for r in rows})},
            "variants": {
                v: {"model": v, "pipeline": pipeline_label}
                for v in sorted({r["variant"] for r in rows})
            },
            "scorers": [SCORER],
            "primary_metric": SCORER,
            "imported_from": "datapilot",
            "pipeline": pipeline_label,
        },
    )
    return rows


def load_questions(path: Path):
    """BIRD Mini-Dev 题目 → evalkit Dataset（input = 问题 + evidence + 库名，expected = 标准 SQL）。"""
    qs = json.loads(path.read_text(encoding="utf-8"))
    for q in qs:
        q["question_id"] = int(q["question_id"])
    return from_records(
        "bird-minidev",
        qs,
        id_key="question_id",
        expected_key="SQL",
        tag_keys=["difficulty", "db_id"],
    )
