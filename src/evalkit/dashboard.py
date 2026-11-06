"""静态看板：把若干运行目录汇总成一个自包含的 HTML 文件（数据内嵌，不依赖网络和构建工具）。

页面：实验列表、指标对比（配对检验在浏览器里现算）、失败题逐条回放、成本与延迟。

为控制体积，只有未通过的题保留输出和过程（截断到 OUTPUT_LIMIT 字符），通过的题只保留判分和用量。
运行目录里如果有 metrics.json（{变体: {指标名: 数值}}），会作为自定义指标显示在实验列表里，
适合合同审查这类"检出率、精确率"等跨题累加的指标。
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from evalkit import store
from evalkit.report import summarize

OUTPUT_LIMIT = 4000
MAX_SLICE_VALUES = 12


def _clip(x, limit: int = OUTPUT_LIMIT):
    if x is None:
        return None
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1, default=str)
    return s if len(s) <= limit else s[:limit] + f"\n…（截断，原长 {len(s)} 字符）"


def slice_keys(rows: list[dict]) -> list[str]:
    keys = []
    if any(r.get("split") for r in rows):
        keys.append("split")
    tag_keys = dict.fromkeys(k for r in rows for k in (r.get("tags") or {}))
    for k in tag_keys:
        vals = {str((r.get("tags") or {}).get(k)) for r in rows}
        if 2 <= len(vals) <= MAX_SLICE_VALUES:
            keys.append(k)
    return keys


def run_payload(run_dir: Path) -> dict:
    rows = store.load_rows(run_dir)
    cfg = store.load_config(run_dir)
    keys = slice_keys(rows)
    compact = []
    for r in rows:
        c = {
            "i": r["item_id"],
            "v": r["variant"],
            "r": r.get("repeat", 0),
            "p": 1 if r["passed"] else 0,
            "s": r["status"],
            "c": r.get("cost_usd", 0.0),
            "l": r.get("latency_s", 0.0),
            "t": {
                k: (r.get("split") if k == "split" else (r.get("tags") or {}).get(k)) for k in keys
            },
        }
        if not r["passed"]:
            trace = r.get("trace")
            if trace is None and r.get("trace_id"):
                t = store.load_trace(run_dir, r["trace_id"])
                trace = (t or {}).get("steps")
            c["o"] = _clip(r.get("output"))
            c["tr"] = _clip(trace)
            c["sc"] = _clip(
                {
                    k: {kk: vv for kk, vv in v.items() if kk != "detail" or len(str(vv)) < 2000}
                    for k, v in (r.get("scores") or {}).items()
                },
                3000,
            )
            if r.get("error"):
                c["e"] = r["error"]
        compact.append(c)
    metrics_path = run_dir / "metrics.json"
    return {
        "id": run_dir.name,
        # 运行通常放在 <项目>/runs/ 下，分组名取项目名
        "group": run_dir.parent.parent.name
        if run_dir.parent.name == "runs"
        else run_dir.parent.name,
        "dataset": (cfg.get("dataset") or {}).get("name", "?"),
        "dataset_hash": (cfg.get("dataset") or {}).get("hash"),
        "primary": cfg.get("primary_metric"),
        "variants": cfg.get("variants", {}),
        "code_version": cfg.get("code_version"),
        "slices": keys,
        "summary": summarize(rows, slices=keys),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {},
        "rows": compact,
    }


def build(paths: list[Path], output: Path, title: str = "evalkit 评测看板") -> Path:
    run_dirs: list[Path] = []
    for p in paths:
        run_dirs += store.list_runs(p)
    if not run_dirs:
        raise SystemExit("没有找到运行目录（需要包含 results.jsonl）")
    data = {"title": title, "runs": [run_payload(d) for d in run_dirs]}
    template = resources.files("evalkit").joinpath("dashboard.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = template.replace("__TITLE__", title).replace("__DATA__", blob)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
