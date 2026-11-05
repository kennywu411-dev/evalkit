"""用 evalkit 复算 DataPilot docs/numbers.md 里的关键数字。

    uv run python examples/datapilot/reproduce.py                       # 用仓库自带的样例运行记录
    uv run python examples/datapilot/reproduce.py --repo ~/Documents/datapilot   # 直接读 DataPilot 仓库（只读）
    uv run python examples/datapilot/reproduce.py --repo ~/Documents/datapilot --export  # 刷新样例
    uv run python examples/datapilot/reproduce.py --repo ~/Documents/datapilot --verify-sql 100

--verify-sql N：在 N 道题上用 evalkit 的 SQLResultMatch 重新执行预测 SQL 和标准 SQL，
检查判分与 DataPilot 记录的逐题一致（需要本地有 BIRD 数据库）。

每一项按 numbers.md 的精度格式化后逐字比较，全部一致退出码 0，否则 1。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapter import import_runs, load_questions  # noqa: E402

from evalkit import store  # noqa: E402
from evalkit.compare import compare  # noqa: E402
from evalkit.dataset import stratified_sample  # noqa: E402
from evalkit.report import repeat_stats, summarize  # noqa: E402

BUNDLED = HERE / "runs"

# numbers.md 与复算不一致、且已查明原因的项：单独列出，不计入失败
KNOWN: dict[str, str] = {}

# evalkit 运行名 → [(DataPilot 运行目录前缀, repeat 偏移)]、流程名、是否保留 SQL 和中间步骤
RUNS = {
    "e0-baseline": ([("baseline-", 0)], "baseline", False),
    "e1-cached": ([("e1-", 0)], "e1_cached", False),
    "e2-enhanced": ([("e2-", 0)], "e2_enhanced", False),
    "e3-values": ([("e3-", 0)], "e3_values", False),
    "e4-correct": ([("e4-", 0)], "e4_correct", True),
    "agent": ([("agent-", 0)], "agent", False),
    "abl-retrieval": ([("abl-", 0)], "abl_retrieval", False),
    "e4-x3": ([("e4-", 0), ("e4rep-", 1)], "e4_correct", False),
    "e4-opus": ([("e4opus-", 0)], "e4_correct", False),
}


def find_run(repo_runs: Path, prefix: str) -> Path:
    hits = [p for p in repo_runs.glob(f"{prefix}[0-9]*") if (p / "results.jsonl").exists()]
    if len(hits) != 1:
        raise SystemExit(f"{repo_runs} 下前缀 {prefix!r} 匹配到 {len(hits)} 个运行：{hits}")
    return hits[0]


def import_all(repo: Path, dst_root: Path) -> None:
    for name, (sources, pipeline, keep) in RUNS.items():
        srcs = [(find_run(repo / "runs", pre), off) for pre, off in sources]
        import_runs(srcs, dst_root / name, pipeline, keep_output=keep)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def compute(root: Path) -> list[tuple[str, str, str]]:
    """返回 [(项目, numbers.md 里的值, evalkit 复算值)]。"""
    rows = {name: store.load_rows(root / name) for name in RUNS}
    out: list[tuple[str, str, str]] = []

    # 1. 500 题全量基线：EX、Wilson 区间、按难度、成本、延迟
    expected_base = {
        "haiku": (
            "63.0%",
            "58.7%–67.1%",
            "77.0% / 59.2% / 52.0%",
            "$0.0003",
            "$0.171",
            "2.0s / 4.2s",
        ),
        "sonnet": (
            "71.8%",
            "67.7%–75.6%",
            "83.1% / 68.0% / 64.7%",
            "$0.0054",
            "$2.687",
            "1.7s / 2.7s",
        ),
        "opus": (
            "86.6%",
            "83.3%–89.3%",
            "94.6% / 85.2% / 78.4%",
            "$0.0104",
            "$5.203",
            "2.2s / 3.3s",
        ),
    }
    for s in summarize(rows["e0-baseline"], slices=["difficulty"]):
        m = s["variant"]
        d = s["slices"]["difficulty"]
        got = (
            pct(s["rate"]),
            f"{pct(s['ci95'][0])}–{pct(s['ci95'][1])}",
            " / ".join(pct(d[k]["rate"]) for k in ("simple", "moderate", "challenging")),
            f"${s['cost_per_item']:.4f}",
            f"${s['cost_total']:.3f}",
            f"{s['latency_p50']:.1f}s / {s['latency_p95']:.1f}s",
        )
        labels = (
            "EX",
            "95% CI",
            "simple / moderate / challenging",
            "单题成本",
            "总成本",
            "P50 / P95 延迟",
        )
        for lab, e, g in zip(labels, expected_base[m], got, strict=True):
            out.append((f"基线 {m} {lab}", e, g))

    # 2. 逐项实验各配置 EX
    ladder = {
        "e1-cached": ("70.8%", "64.2%"),
        "e2-enhanced": ("71.6%", "65.4%"),
        "e3-values": ("72.8%", "67.0%"),
        "e4-correct": ("72.0%", "67.4%"),
        "agent": ("72.0%", "65.8%"),
    }
    for name, (es, eh) in ladder.items():
        rates = {s["variant"]: s["rate"] for s in summarize(rows[name])}
        out.append((f"{name} Sonnet EX", es, pct(rates["sonnet"])))
        out.append((f"{name} Haiku EX", eh, pct(rates["haiku"])))

    # 3. 配对检验（McNemar 精确检验）：A 独对 / B 独对 / p
    pairs = [
        ("e0-baseline", "e1-cached", "sonnet", "20 / 15 / 0.50"),
        ("e1-cached", "e2-enhanced", "sonnet", "13 / 17 / 0.58"),
        ("e0-baseline", "e1-cached", "haiku", "22 / 28 / 0.48"),
        ("e1-cached", "e2-enhanced", "haiku", "22 / 28 / 0.48"),
        ("e0-baseline", "e2-enhanced", "haiku", "24 / 36 / 0.16"),
        ("e0-baseline", "e4-correct", "haiku", "21 / 43 / 0.008"),
        ("e1-cached", "e4-correct", "haiku", "21 / 37 / 0.048"),
        ("e2-enhanced", "e3-values", "haiku", "13 / 21 / 0.23"),
        ("e0-baseline", "e4-correct", "sonnet", "17 / 18 / 1.00"),
        ("e2-enhanced", "e3-values", "sonnet", "6 / 12 / 0.24"),
        ("e4-correct", "agent", "sonnet", "15 / 15 / 1.00"),
        ("e4-correct", "agent", "haiku", "31 / 23 / 0.34"),
    ]
    for a, b, m, exp in pairs:
        c = compare(rows[a], rows[b], m, m)
        digits = len(exp.rsplit(".", 1)[1])
        got = f"{len(c['a_only'])} / {len(c['b_only'])} / {c['p_value']:.{digits}f}"
        out.append((f"{a} → {b}（{m}）", exp, got))

    # 4. 检索消融：只完成 373 题，在共同题目上与 E4 配对比较
    c = compare(rows["e4-correct"], rows["abl-retrieval"], "sonnet", "sonnet")
    out.append(("消融 共同题数", "373", str(c["n"])))
    out.append(("消融 E4 / 检索 EX", "72.9% / 70.0%", f"{pct(c['rate_a'])} / {pct(c['rate_b'])}"))
    out.append(
        (
            "消融 E4 独对 / 检索独对 / p",
            "14 / 3 / 0.013",
            f"{len(c['a_only'])} / {len(c['b_only'])} / {c['p_value']:.3f}",
        )
    )

    # 5. 最终配置跑 3 次：均值 ± 标准差、3 次都对 / 都错 / 不稳定
    exp_rep = {
        "sonnet": ("72.0% / 72.6% / 72.6%", "72.4% ± 0.35", "351 / 127 / 22"),
        "haiku": ("67.4% / 67.8% / 68.8%", "68.0% ± 0.72", "322 / 141 / 37"),
    }
    for m, (e_rates, e_ms, e_stab) in exp_rep.items():
        rs = repeat_stats([r for r in rows["e4-x3"] if r["variant"] == m])
        st = rs["stability"]
        out.append((f"E4×3 {m} 每次 EX", e_rates, " / ".join(pct(x) for x in rs["rates"])))
        out.append((f"E4×3 {m} 均值 ± 标准差", e_ms, f"{pct(rs['mean'])} ± {rs['std'] * 100:.2f}"))
        out.append(
            (
                f"E4×3 {m} 都对 / 都错 / 不稳定",
                e_stab,
                f"{st['always']} / {st['never']} / {st['flaky']}",
            )
        )

    # 6. Opus：E0 → E4
    c = compare(rows["e0-baseline"], rows["e4-opus"], "opus", "opus")
    out.append(("Opus E0 → E4 EX", "86.6% → 86.4%", f"{pct(c['rate_a'])} → {pct(c['rate_b'])}"))
    out.append(
        (
            "Opus E0 独对 / E4 独对 / p",
            "12 / 11 / 1.00",
            f"{len(c['a_only'])} / {len(c['b_only'])} / {c['p_value']:.2f}",
        )
    )
    return out


def verify_trial_sample(repo: Path) -> tuple[str, str, str]:
    """evalkit 的分层抽样与 DataPilot 20 题试跑（seed=0）抽到的题目是否相同。"""
    import json

    ds = load_questions(repo / "data" / "mini_dev_sqlite.json")
    picked = sorted(it.id for it in stratified_sample(ds.items, 20, "difficulty", seed=0))
    trial = find_run(repo / "runs", "trial-")
    ids = sorted(int(i) for i in json.loads((trial / "config.json").read_text())["question_ids"])
    return (
        "分层抽样 20 题（seed=0）与试跑题目一致",
        "一致",
        "一致" if picked == ids else f"不一致 {picked}",
    )


def verify_sql(repo: Path, root: Path, n: int) -> tuple[str, str, str]:
    from evalkit.scorers import SQLResultMatch

    ds = load_questions(repo / "data" / "mini_dev_sqlite.json")
    by_id = {it.id: it for it in ds}
    rows = [r for r in store.load_rows(root / "e4-correct") if r["variant"] == "haiku"]
    # 超时类题目判分受机器负载影响（numbers.md：518、701），不纳入逐题比对
    rows = [r for r in rows if r["status"] in ("ok", "pred_error") and r["output"]]
    sample = stratified_sample(
        [by_id[r["item_id"]] for r in rows], min(n, len(rows)), "difficulty", seed=0
    )
    recorded = {r["item_id"]: r["passed"] for r in rows}
    out_of = {r["item_id"]: r["output"] for r in rows}
    m = SQLResultMatch(
        db=lambda it: (
            repo / "data" / "dev_databases" / it.tags["db_id"] / f"{it.tags['db_id']}.sqlite"
        )
    )
    diff = [it.id for it in sample if m(it, out_of[it.id]).passed != recorded[it.id]]
    return (
        f"SQLResultMatch 重判 {len(sample)} 题与记录逐题一致",
        f"0 / {len(sample)} 不一致",
        f"{len(diff)} / {len(sample)} 不一致",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", type=Path, help="DataPilot 仓库路径（只读）；不给则用自带样例")
    ap.add_argument("--export", action="store_true", help="把导入结果写成仓库自带样例")
    ap.add_argument("--verify-sql", type=int, default=0, metavar="N")
    args = ap.parse_args()

    if args.repo:
        repo = args.repo.expanduser()
        root = BUNDLED if args.export else Path(tempfile.mkdtemp(prefix="evalkit-datapilot-"))
        import_all(repo, root)
    else:
        root = BUNDLED
    table = compute(root)
    if args.repo:
        table.append(verify_trial_sample(repo))
        if args.verify_sql:
            table.append(verify_sql(repo, root, args.verify_sql))

    print("| 项目 | numbers.md | evalkit 复算 | 一致 |\n|---|---|---|---|")
    bad, known = 0, []
    for label, exp, got in table:
        if exp == got:
            mark = "✓"
        elif label in KNOWN:
            mark = "注"
            known.append(f"- {label}：{KNOWN[label]}")
        else:
            mark = "✗"
            bad += 1
        print(f"| {label} | {exp} | {got} | {mark} |")
    print(
        f"\n{len(table) - bad - len(known)} / {len(table)} 项一致，{len(known)} 项已知差异，{bad} 项不一致"
    )
    if known:
        print("\n".join(known))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
