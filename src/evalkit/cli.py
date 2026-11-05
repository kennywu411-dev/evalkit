"""命令行入口：evalkit <子命令>。

evalkit run       --dataset ds.jsonl --task mypkg.app:answer --scorer exact_match [--repeats 3 --max-cost 2]
evalkit report    runs/<run_id> [--slice difficulty]
evalkit compare   runs/<A> runs/<B> [--variant-a x --variant-b y] [--slice difficulty]
evalkit repeats   runs/<run_id> --variant x
evalkit gate      --baseline runs/<A> --candidate runs/<B> [--max-drop 0.02 --alpha 0.05]
evalkit rescore   runs/<run_id> --dataset ds.jsonl --scorer mypkg.scoring:scorer
evalkit feedback  summary feedback.jsonl --key session_id,target_id --labels useful,not_risk
evalkit dashboard runs/ [更多运行目录...] -o site/index.html
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from evalkit import dataset as ds_mod
from evalkit import store

BUILTIN_SCORERS = {"exact_match": "evalkit.scorers:ExactMatch"}


def load_object(spec: str):
    """'包.模块:对象' → 对象；当前目录加入 sys.path，方便引用项目里的模块。"""
    if ":" not in spec:
        raise SystemExit(f"格式应为 模块:对象，得到 {spec!r}")
    mod, _, attr = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def make_scorer(spec: str):
    obj = load_object(BUILTIN_SCORERS.get(spec, spec))
    return obj() if isinstance(obj, type) else obj


def _rows(path: str) -> list[dict]:
    rows = store.load_rows(path)
    if not rows:
        raise SystemExit(f"{path} 下没有 results.jsonl")
    return rows


def cmd_run(a) -> int:
    from evalkit.runner import AbortRun, Variant, run

    ds = ds_mod.load(a.dataset)
    if a.split:
        ds = ds.split(a.split)
    if a.sample:
        ds = ds_mod.Dataset(
            ds.name,
            ds_mod.stratified_sample(ds.items, a.sample, a.stratify or (lambda it: None), a.seed),
            ds.version,
        )
    task = load_object(a.task)
    variants = [
        Variant(
            a.variant or a.task.rsplit(":", 1)[-1], task, json.loads(a.params) if a.params else {}
        )
    ]
    scorers = [make_scorer(s) for s in a.scorer]
    try:
        r = run(
            ds,
            variants,
            scorers,
            runs_dir=a.runs_dir,
            tag=a.tag,
            repeats=a.repeats,
            concurrency=a.concurrency,
            max_cost=a.max_cost,
            resume=a.resume,
        )
    except AbortRun as e:
        print(e, file=sys.stderr)
        return 1
    from evalkit.report import render, summarize

    print(render(summarize(r.rows)))
    return 0 if r.complete else 1


def cmd_report(a) -> int:
    from evalkit.report import render, summarize

    by = tuple(a.by.split(","))
    s = summarize(_rows(a.run), by=by, slices=a.slice or ())
    print(json.dumps(s, ensure_ascii=False, indent=2) if a.json else render(s, by))
    return 0


def cmd_compare(a) -> int:
    from evalkit.compare import compare, render

    c = compare(_rows(a.run_a), _rows(a.run_b), a.variant_a, a.variant_b, a.repeat, a.slice or ())
    print(json.dumps(c, ensure_ascii=False, indent=2) if a.json else render(c))
    return 0


def cmd_repeats(a) -> int:
    from evalkit.report import repeat_stats

    rows = [r for r in _rows(a.run) if a.variant is None or r["variant"] == a.variant]
    s = repeat_stats(rows)
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        st = s["stability"]
        print(f"{s['runs']} 次：" + " / ".join(f"{x:.1%}" for x in s["rates"]))
        print(
            f"均值 {s['mean']:.1%} ± {s['std'] * 100:.2f}（标准差，百分点），极差 {s['range'] * 100:.1f} 个百分点"
        )
        print(
            f"每次都对 {st['always']}，每次都错 {st['never']}，时对时错 {st['flaky']}（{st['flaky_rate']:.1%}）"
        )
    return 0


def cmd_gate(a) -> int:
    from evalkit.gate import gate

    try:
        g = gate(
            _rows(a.baseline),
            _rows(a.candidate),
            a.variant_baseline,
            a.variant_candidate,
            a.max_drop,
            a.alpha,
            a.max_error_rate,
            a.max_cost_ratio,
        )
    except ValueError as e:
        print(f"[gate] {e}", file=sys.stderr)
        return 2
    if a.json:
        print(
            json.dumps(
                {"passed": g.passed, "reasons": g.reasons, **g.comparison},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(g.render())
    if a.summary_file:
        Path(a.summary_file).write_text(g.render() + "\n", encoding="utf-8")
    return 0 if g.passed else 1


def cmd_rescore(a) -> int:
    from evalkit.report import render, summarize
    from evalkit.runner import rescore

    rows = rescore(a.run, ds_mod.load(a.dataset), [make_scorer(s) for s in a.scorer], a.out)
    print(render(summarize(rows)))
    return 0


def cmd_feedback(a) -> int:
    from evalkit.feedback import FeedbackStore, summarize

    rows = FeedbackStore(a.file).load()
    s = summarize(
        rows,
        key=a.key.split(","),
        labels=a.labels.split(","),
        positive=a.positive,
        negative=a.negative,
        by=a.by,
    )
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def cmd_dashboard(a) -> int:
    from evalkit.dashboard import build

    out = build([Path(p) for p in a.runs], Path(a.output), title=a.title)
    print(f"看板已生成：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="evalkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑一次实验")
    r.add_argument("--dataset", required=True)
    r.add_argument("--task", required=True, help="被评测的函数，模块:函数")
    r.add_argument(
        "--scorer", action="append", default=[], help="内置名或 模块:对象，可多次；第一个是主指标"
    )
    r.add_argument("--variant", help="变体名（默认取函数名）")
    r.add_argument("--params", help="变体参数快照，JSON")
    r.add_argument("--split", choices=ds_mod.SPLITS)
    r.add_argument("--sample", type=int)
    r.add_argument("--stratify", help="分层抽样用的标签")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--max-cost", type=float)
    r.add_argument("--tag", default="run")
    r.add_argument("--resume")
    r.add_argument("--runs-dir", default="runs")
    r.set_defaults(fn=cmd_run)

    r = sub.add_parser("report", help="汇总一次运行")
    r.add_argument("run")
    r.add_argument("--by", default="variant")
    r.add_argument("--slice", action="append")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_report)

    r = sub.add_parser("compare", help="两次运行配对比较")
    r.add_argument("run_a")
    r.add_argument("run_b")
    r.add_argument("--variant-a")
    r.add_argument("--variant-b")
    r.add_argument("--repeat", type=int, default=0)
    r.add_argument("--slice", action="append")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_compare)

    r = sub.add_parser("repeats", help="多次运行的均值、方差和稳定性")
    r.add_argument("run")
    r.add_argument("--variant")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_repeats)

    r = sub.add_parser("gate", help="回归门禁，不通过时退出码 1")
    r.add_argument("--baseline", required=True)
    r.add_argument("--candidate", required=True)
    r.add_argument("--variant-baseline")
    r.add_argument("--variant-candidate")
    r.add_argument("--max-drop", type=float, default=0.02)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--max-error-rate", type=float)
    r.add_argument("--max-cost-ratio", type=float)
    r.add_argument("--summary-file", help="把 Markdown 结论另存一份（如 $GITHUB_STEP_SUMMARY）")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_gate)

    r = sub.add_parser("rescore", help="用保存的输出重新打分，不调用模型")
    r.add_argument("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--scorer", action="append", required=True)
    r.add_argument("--out")
    r.set_defaults(fn=cmd_rescore)

    r = sub.add_parser("feedback", help="汇总线上反馈")
    fsub = r.add_subparsers(dest="action", required=True)
    f = fsub.add_parser("summary")
    f.add_argument("file")
    f.add_argument("--key", required=True, help="确定一个打标目标的字段，逗号分隔")
    f.add_argument("--labels", required=True)
    f.add_argument("--positive")
    f.add_argument("--negative")
    f.add_argument("--by")
    f.set_defaults(fn=cmd_feedback)

    r = sub.add_parser("dashboard", help="生成静态看板")
    r.add_argument("runs", nargs="+", help="运行目录，或包含多个运行目录的文件夹")
    r.add_argument("-o", "--output", default="site/index.html")
    r.add_argument("--title", default="evalkit 评测看板")
    r.set_defaults(fn=cmd_dashboard)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
