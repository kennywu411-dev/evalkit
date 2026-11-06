"""命令行入口：evalkit <子命令>。

run        跑一次实验：evalkit run --dataset ds.jsonl --task mypkg.app:answer --scorer exact_match
report     汇总一次运行：evalkit report runs/<id> --slice difficulty
compare    配对比较：evalkit compare runs/<A> runs/<B> --slice difficulty
repeats    多次运行的均值、方差、稳定性：evalkit repeats runs/<id> [runs/<id2> ...]
gate       回归门禁：evalkit gate --baseline runs/<A> --candidate runs/<B>
rescore    用保存的输出重新打分：evalkit rescore runs/<id> --dataset ds.jsonl --scorer exact_match
dataset    评测集管理：info / validate / diff / split / sample
calibrate  裁判与人工一致率：evalkit calibrate --judge j.csv --human h.csv
feedback   线上反馈：summary / add / serve
dashboard  静态看板：evalkit dashboard runs/ -o site/index.html
"""

from __future__ import annotations

import argparse
import contextlib
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

    rows, offset = [], 0
    for run in a.runs:  # 多个运行目录：把各自的 repeat 依次接起来
        rs = [r for r in _rows(run) if a.variant is None or r["variant"] == a.variant]
        if not rs:
            raise SystemExit(f"{run} 里没有变体 {a.variant!r}")
        rows += [{**r, "repeat": r.get("repeat", 0) + offset} for r in rs]
        offset = max(r["repeat"] for r in rows) + 1
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


def _summary_kw(a) -> dict:
    return {
        "key": a.key.split(","),
        "labels": a.labels.split(","),
        "positive": a.positive,
        "negative": a.negative,
        "by": a.by,
    }


def cmd_feedback(a) -> int:
    from evalkit.feedback import FeedbackStore, make_server, summarize

    store = FeedbackStore(a.file, labels=a.labels.split(","), required=a.key.split(","))
    if a.action == "summary":
        print(json.dumps(summarize(store.load(), **_summary_kw(a)), ensure_ascii=False, indent=2))
    elif a.action == "add":
        try:
            store.add(json.loads(a.entry))
        except ValueError as e:
            print(f"[feedback] {e}", file=sys.stderr)
            return 2
    else:
        srv = make_server(store, a.host, a.port, **_summary_kw(a))
        print(f"反馈服务：http://{a.host}:{srv.server_port}  POST /feedback  GET /feedback/summary")
        with contextlib.suppress(KeyboardInterrupt):
            srv.serve_forever()
    return 0


def cmd_dataset(a) -> int:
    ds = ds_mod.load(a.file)
    if a.action == "info":
        print(json.dumps(ds_mod.describe(ds), ensure_ascii=False, indent=2))
    elif a.action == "validate":
        problems = ds_mod.validate(ds)
        for x in problems:
            print(f"✗ {x}")
        print(
            f"{ds.name} v{ds.version}（{ds.content_hash}）：{len(ds)} 题，问题 {len(problems)} 个"
        )
        return 1 if problems else 0
    elif a.action == "diff":
        print(json.dumps(ds_mod.diff(ds, ds_mod.load(a.other)), ensure_ascii=False, indent=2))
    elif a.action == "split":
        out = ds_mod.assign_split(ds, a.holdout, by=a.by, seed=a.seed)
        out.version = a.version or ds.version
        out.save(Path(a.output))
        print(json.dumps(ds_mod.describe(out), ensure_ascii=False, indent=2))
    elif a.action == "sample":
        items = ds_mod.stratified_sample(ds.items, a.n, a.by or (lambda it: None), a.seed)
        out = ds_mod.Dataset(f"{ds.name}-sample{a.n}", items, ds.version)
        out.save(Path(a.output))
        print(f"抽样 {len(items)} 题 → {a.output}")
    return 0


def cmd_calibrate(a) -> int:
    from evalkit.judge import calibrate
    from evalkit.labels import load_labels

    judge = load_labels(a.judge, a.id_field, a.label_field)
    human = load_labels(a.human, a.id_field, a.label_field)
    c = calibrate(judge, human)
    if a.json:
        print(json.dumps(c, ensure_ascii=False, indent=2))
    else:
        print(f"共同样本 {c['n']}：一致率 {c['agreement']:.1%}，Cohen's κ = {c['kappa']:.3f}")
        print("混淆矩阵（行 = 裁判，列 = 人工）：" + json.dumps(c["confusion"], ensure_ascii=False))
        if c["disagreements"]:
            print("不一致：" + ", ".join(map(str, c["disagreements"][:50])))
    ok = a.min_agreement is None or c["agreement"] >= a.min_agreement
    return 0 if ok else 1


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
    r.add_argument("runs", nargs="+", help="一个含多次重复的运行，或多个运行目录")
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

    fb_common = argparse.ArgumentParser(add_help=False)
    fb_common.add_argument("file", help="反馈 JSONL 文件")
    fb_common.add_argument("--key", required=True, help="确定一个打标目标的字段，逗号分隔")
    fb_common.add_argument("--labels", required=True, help="允许的标签，逗号分隔")
    fb_common.add_argument("--positive", help="正向标签（如 useful → 采纳率）")
    fb_common.add_argument("--negative", help="负向标签（如 not_risk → 误报率）")
    fb_common.add_argument("--by", help="按字段拆分，点号路径如 finding.risk_type")
    r = sub.add_parser("feedback", help="线上反馈：汇总、追加、接收服务")
    fsub = r.add_subparsers(dest="action", required=True)
    fsub.add_parser("summary", parents=[fb_common], help="按最后一次打标汇总").set_defaults(
        fn=cmd_feedback
    )
    f = fsub.add_parser("add", parents=[fb_common], help="追加一条打标")
    f.add_argument("--entry", required=True, help="打标内容，JSON 对象")
    f.set_defaults(fn=cmd_feedback)
    f = fsub.add_parser("serve", parents=[fb_common], help="启动 HTTP 接收服务")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--port", type=int, default=8787)
    f.set_defaults(fn=cmd_feedback)

    r = sub.add_parser("dataset", help="评测集管理：信息、校验、版本差异、划分、抽样")
    dsub = r.add_subparsers(dest="action", required=True)
    for name, help_ in (("info", "题数、划分、标签分布"), ("validate", "体检，有问题退出码 1")):
        d = dsub.add_parser(name, help=help_)
        d.add_argument("file")
        d.set_defaults(fn=cmd_dataset)
    d = dsub.add_parser("diff", help="两个版本的差异")
    d.add_argument("file")
    d.add_argument("other")
    d.set_defaults(fn=cmd_dataset)
    d = dsub.add_parser("split", help="划分开发集 / 留出集")
    d.add_argument("file")
    d.add_argument("--holdout", type=float, default=0.3)
    d.add_argument("--by", help="分层标签")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--version", help="新版本号")
    d.add_argument("-o", "--output", required=True)
    d.set_defaults(fn=cmd_dataset)
    d = dsub.add_parser("sample", help="分层抽样出一个小评测集（如冒烟集）")
    d.add_argument("file")
    d.add_argument("-n", type=int, required=True)
    d.add_argument("--by", help="分层标签")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("-o", "--output", required=True)
    d.set_defaults(fn=cmd_dataset)

    r = sub.add_parser("calibrate", help="LLM 裁判与人工标注的一致率校准")
    r.add_argument("--judge", required=True, help="裁判标签 CSV / JSONL")
    r.add_argument("--human", required=True, help="人工标签 CSV / JSONL")
    r.add_argument("--id-field", default="id")
    r.add_argument("--label-field", default="label")
    r.add_argument("--min-agreement", type=float, help="一致率低于该值时退出码 1")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_calibrate)

    r = sub.add_parser("dashboard", help="生成静态看板")
    r.add_argument("runs", nargs="+", help="运行目录，或包含多个运行目录的文件夹")
    r.add_argument("-o", "--output", default="site/index.html")
    r.add_argument("--title", default="evalkit 评测看板")
    r.set_defaults(fn=cmd_dashboard)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
