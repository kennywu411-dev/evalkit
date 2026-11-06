"""劳动法评测（laborcheck）运行记录 → evalkit 运行目录。

laborcheck 每次运行是 runs/<run_id>/：
    config.json    模型（name、display）、模式（closed 仅模型 / tools 加规则引擎工具）、提示词……
    results.jsonl  每题一行：passed、basis_hit、detail、output（模型原话）、token、花费、耗时、trace（工具调用）

映射：item_id 原样，variant = <模型名>-<模式>，output = 回答末尾的 JSON 对象（extract_answer），
scores["recorded"] = 运行时的判分（复现"跑的时候的口径"），重新判分时用 scoring.answer_scorer。
tool_calls = trace 里的工具调用次数。评测集只导出标准答案和标签，题面换成哈希（用来判断两版之间改没改）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scoring import extract_answer

from evalkit import store
from evalkit.dataset import Dataset, Item

TOPIC_LABEL = {
    "probation_term": "试用期期限",
    "probation_wage": "试用期工资",
    "penalty_kind": "违约金情形",
    "service_penalty": "服务期违约金",
    "noncompete": "竞业限制",
    "overtime": "加班费",
    "severance": "经济补偿",
    "double_wage": "二倍工资",
    "annual_leave": "年休假",
    "social_insurance": "社保",
    "housing_fund": "公积金",
    "bonus_tax": "年终奖个税",
    "withholding": "累计预扣",
    "take_home": "税后到手",
    "offer_compare": "多 offer 对比",
    "si_waiver": "放弃社保约定",
}
CITY_LABEL = {"beijing": "北京", "shanghai": "上海", "shenzhen": "深圳", "hangzhou": "杭州"}


def _strip(d: dict) -> Item:
    q = d["input"]["question"]
    return Item(
        d["id"],
        {
            "format": d["input"]["format"],
            "city": d["input"].get("city"),
            "question_sha": hashlib.sha256(q.encode()).hexdigest()[:12],
        },
        {k: v for k, v in d["expected"].items() if k != "explanation"},
        d["tags"],
        d["split"],
    )


def evalset_from_jsonl(text: str, version: str) -> Dataset:
    items = [_strip(json.loads(x)) for x in text.splitlines() if x.strip()]
    return Dataset("laborcheck", items, version)


def evalsets_from_repo(repo: Path, versions: tuple[str, ...]) -> dict[str, Dataset]:
    """从 laborcheck 的 git 历史里取各版评测集（同一文件 data/laborcheck-v1.jsonl 的不同提交）。"""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout

    out: dict[str, Dataset] = {}
    for sha in git("log", "--format=%H", "--", "data/laborcheck-v1.meta.json").split():
        v = json.loads(git("show", f"{sha}:data/laborcheck-v1.meta.json"))["version"]
        if v in versions and v not in out:  # git log 从新到旧，取每个版本最后一次提交
            out[v] = evalset_from_jsonl(git("show", f"{sha}:data/laborcheck-v1.jsonl"), v)
    missing = set(versions) - set(out)
    if missing:
        raise SystemExit(f"{repo} 的历史里找不到评测集版本 {sorted(missing)}")
    return out


def latest_rows(run_dir: Path) -> list[dict]:
    """续跑时出错的题会被重跑，同一题以最后一条为准（与 laborcheck 的 latest_records 一致）。"""
    rows: dict[str, dict] = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["item_id"]] = r
    return [rows[k] for k in sorted(rows)]


def convert_row(r: dict, variant: str) -> dict:
    return {
        "item_id": r["item_id"],
        "variant": variant,
        "repeat": 0,
        "split": r["split"],
        "tags": r["tags"],
        "status": r["status"],
        "passed": bool(r["passed"]),
        "scores": {
            "recorded": {
                "value": float(r["passed"]),
                "passed": bool(r["passed"]),
                "detail": {**r["detail"], "basis_hit": r["basis_hit"]},
            }
        },
        "output": extract_answer(r.get("output")),
        "cost_usd": r["cost_usd"],
        "latency_s": r["latency_s"],
        "usage": {k: r.get(k, 0) for k in ("tokens_in", "tokens_out", "cache_read", "cache_write")},
        "tool_calls": sum(1 for x in r.get("trace") or [] if "tool" in x),
        **({"error": r["error"]} if r.get("error") else {}),
    }


def import_run(src: Path, dst: Path, evalset_version: str) -> list[dict]:
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    model, mode = cfg["model"], cfg["mode"]
    variant = f"{model['name']}-{mode}"
    rows = [convert_row(r, variant) for r in latest_rows(src)]
    store.write_rows(dst, rows)
    store.write_json(
        dst / store.CONFIG,
        {
            "run_id": dst.name,
            "tag": dst.name,
            "source_run": src.name,
            "dataset": {"name": "laborcheck", "version": evalset_version, "n": len(rows)},
            "variants": {
                variant: {
                    "model": model["model"],
                    "display": model["display"],
                    "mode": mode,
                    "effort": model.get("effort", ""),
                }
            },
            "scorers": ["recorded"],
            "primary_metric": "recorded",
            "repeats": 1,
            "item_ids": [r["item_id"] for r in rows],
        },
    )
    return rows
