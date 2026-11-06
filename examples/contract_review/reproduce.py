"""用 evalkit 复算合同风险审查项目 docs/numbers.md 里的关键数字。

    uv run python examples/contract_review/reproduce.py                    # 用仓库自带的样例
    uv run python examples/contract_review/reproduce.py --repo ~/Documents/docpilot           # 只读原仓库
    uv run python examples/contract_review/reproduce.py --repo ~/Documents/docpilot --export  # 刷新样例

两种口径都复算：
1. 运行时口径：直接汇总运行时记下的逐项计数（scores["recorded"]），验证比例指标的累加方式；
2. 订正后口径：用 evalkit.runner.rescore + ContractScorer 从保存的审查结果重新判分，
   验证规则打分器与原项目判分逐份一致。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapter import cost_stats, import_run, load_contracts  # noqa: E402
from scoring import ContractScorer, metrics  # noqa: E402

from evalkit import dataset as ek_dataset  # noqa: E402
from evalkit import store  # noqa: E402
from evalkit.report import aggregate  # noqa: E402
from evalkit.runner import rescore  # noqa: E402

BUNDLED = HERE / "runs"
EVALSET = HERE / "data" / "contracts.jsonl"

# evalkit 运行名 → (原项目运行目录前缀, 变体名)
RUNS = {
    "single-haiku-1": ("review-full-", "haiku-single"),
    "single-haiku-2": ("review-haiku-v2-", "haiku-single"),
    "single-deepseek": ("review-ds-full-", "deepseek-single"),
    "multi-haiku-1": ("review-multi-full-", "haiku-multi"),
    "multi-haiku-2": ("review-multi-full2-", "haiku-multi"),
}


def find_run(runs: Path, prefix: str) -> Path:
    hits = [p for p in runs.glob(f"{prefix}[0-9]*") if (p / "review_results.jsonl").exists()]
    if len(hits) != 1:
        raise SystemExit(f"{runs} 下前缀 {prefix!r} 匹配到 {len(hits)} 个运行：{hits}")
    return hits[0]


def import_all(repo: Path, root: Path, evalset_out: Path) -> None:
    ds = load_contracts(repo / "evals" / "contracts.jsonl")
    ds.save(evalset_out)
    for name, (prefix, variant) in RUNS.items():
        import_run(find_run(repo / "runs", prefix), root / name, variant, ds)


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def m(rows: list[dict], scorer: str) -> dict:
    return {k: v["value"] for k, v in aggregate(rows, metrics(scorer)).items()}


def splits(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "全部": rows,
        "调试小样": [r for r in rows if r["split"] == "dev"],
        "留出集": [r for r in rows if r["split"] == "holdout"],
    }


def compute(root: Path, evalset: Path) -> list[tuple[str, str, str]]:
    ds = ek_dataset.load(evalset)
    out: list[tuple[str, str, str]] = []

    # 1. Haiku 基线全量，运行时口径，全部 / 调试小样 / 留出集
    recorded = store.load_rows(root / "single-haiku-1")
    exp = {
        "风险检出率": ("recall", "100.0%", "100.0%", "100.0%"),
        "类型判对率": ("type_accuracy", "97.3%", "100.0%", "96.4%"),
        "依据引用准确率": ("basis_accuracy", "97.3%", "100.0%", "96.4%"),
        "意见精确率（全部）": ("precision", "68.0%", "51.2%", "77.2%"),
        "意见精确率（高中）": ("precision_hm", "72.6%", "56.4%", "81.1%"),
        "干净合同误报率（全部）": ("clean_fp_rate", "83.3%", "87.5%", "75.0%"),
        "干净合同误报率（高中）": ("clean_fp_rate_hm", "66.7%", "75.0%", "50.0%"),
        "视角错误率": ("favorable_flag_rate", "0.0%", "0.0%", "0.0%"),
        "我方身份判断正确率": ("side_accuracy", "100.0%", "100.0%", "100.0%"),
        "临界条款转人工率": ("borderline_human_rate", "66.7%", "50.0%", "100.0%"),
        "摘录与原文一致率": ("quote_found_rate", "100.0%", "100.0%", "100.0%"),
    }
    per_split = {k: m(v, "recorded") for k, v in splits(recorded).items()}
    for label, (key, *vals) in exp.items():
        for (sname, ms), e in zip(per_split.items(), vals, strict=True):
            out.append((f"基线 运行时口径 {sname} {label}", e, pct(ms[key])))
    agg = aggregate(recorded, metrics("recorded"))
    out.append(("基线 检出 风险数", "73/73", f"{agg['recall']['num']}/{agg['recall']['den']}"))
    out.append(
        (
            "基线 反问率",
            "100.0%（4/4）",
            f"{pct(agg['clarify_rate']['value'])}（{agg['clarify_rate']['num']}/{agg['clarify_rate']['den']}）",
        )
    )
    for (sname, rs), e in zip(
        splits(recorded).items(),
        ("$0.0038 / $0.1726", "$0.0038 / $0.0648", "$0.0039 / $0.1079"),
        strict=True,
    ):
        p50, total = cost_stats(rs)
        out.append((f"基线 {sname} 单份花费 P50 / 合计", e, f"${p50:.4f} / ${total:.4f}"))

    # 2. 订正后口径：rescore 重新判分
    tmp = Path(tempfile.mkdtemp(prefix="evalkit-contract-"))
    rescored = {
        name: rescore(root / name, ds, [ContractScorer()], out_dir=tmp / name) for name in RUNS
    }
    s = {k: m(v, "contract_review") for k, v in splits(rescored["single-haiku-1"]).items()}
    for sname, e_all, e_hm in (
        ("全部", "71.3%", "76.1%"),
        ("调试小样", "58.1%", "64.1%"),
        ("留出集", "78.5%", "82.4%"),
    ):
        out.append(
            (
                f"基线 订正后 {sname} 精确率（全部 / 高中）",
                f"{e_all} / {e_hm}",
                f"{pct(s[sname]['precision'])} / {pct(s[sname]['precision_hm'])}",
            )
        )
    out.append(
        (
            "基线 订正后 严格精确率（全部 / 留出）",
            "58.2% / 67.1%",
            f"{pct(s['全部']['precision_strict'])} / {pct(s['留出集']['precision_strict'])}",
        )
    )
    for key in (
        "recall",
        "type_accuracy",
        "basis_accuracy",
        "clean_fp_rate",
        "clean_fp_rate_hm",
        "favorable_flag_rate",
        "borderline_human_rate",
        "clarify_rate",
    ):
        out.append((f"基线 订正前后不变：{key}", pct(per_split["全部"][key]), pct(s["全部"][key])))

    # 3. 模型对比与多 Agent（numbers.md 两张对比表，都按订正后金标准从 trace 重算）
    table = {
        "single-haiku-1": (
            "100.0%",
            "97.3% / 97.3%",
            "71.3% / 76.1% / 58.2%",
            "66.7%",
            "82.4% / 67.1%",
            None,
        ),
        "single-haiku-2": (
            "98.6%",
            "100.0% / 100.0%",
            "73.8% / 76.3% / 59.0%",
            "83.3%",
            "82.9% / 70.1%",
            "$0.0038 / $0.1781",
        ),
        "single-deepseek": (
            "98.6%",
            "100.0% / 100.0%",
            "66.4% / 70.5% / 55.0%",
            "75.0%",
            None,
            "$0.0092 / $0.4361",
        ),
        "multi-haiku-1": (
            "100.0%",
            "100.0% / 100.0%",
            "78.8% / 82.3% / 61.9%",
            "41.7%",
            "84.6% / 67.1%",
            "$0.0062 / $0.2738",
        ),
        "multi-haiku-2": (
            "98.6%",
            "100.0% / 97.2%",
            "79.7% / 83.9% / 61.0%",
            "33.3%",
            "82.9% / 64.3%",
            "$0.0055 / $0.2688",
        ),
    }
    for name, (e_rec, e_tb, e_prec, e_clean, e_hold, e_cost) in table.items():
        rs = rescored[name]
        a = m(rs, "contract_review")
        h = m([r for r in rs if r["split"] == "holdout"], "contract_review")
        out.append((f"{name} 检出率", e_rec, pct(a["recall"])))
        out.append(
            (
                f"{name} 类型判对 / 依据正确",
                e_tb,
                f"{pct(a['type_accuracy'])} / {pct(a['basis_accuracy'])}",
            )
        )
        out.append(
            (
                f"{name} 精确率 全部 / 高中 / 严格",
                e_prec,
                f"{pct(a['precision'])} / {pct(a['precision_hm'])} / {pct(a['precision_strict'])}",
            )
        )
        out.append((f"{name} 干净合同误报（高中）", e_clean, pct(a["clean_fp_rate_hm"])))
        if e_hold:
            out.append(
                (
                    f"{name} 留出集精确率 高中 / 严格",
                    e_hold,
                    f"{pct(h['precision_hm'])} / {pct(h['precision_strict'])}",
                )
            )
        if name == "single-deepseek":
            out.append((f"{name} 留出集严格精确率", "61.4%", pct(h["precision_strict"])))
        if e_cost:
            p50, total = cost_stats(rs)
            out.append((f"{name} 单份花费 P50 / 合计", e_cost, f"${p50:.4f} / ${total:.4f}"))
    return out


METRIC_LABELS = {
    "recall": "检出率",
    "type_accuracy": "类型判对率",
    "basis_accuracy": "依据正确率",
    "precision_hm": "精确率（高中）",
    "precision_strict": "精确率（严格）",
    "clean_fp_rate_hm": "干净合同误报率（高中）",
    "borderline_human_rate": "临界条款转人工率",
    "clarify_rate": "反问率",
}


def write_metrics(root: Path, evalset: Path) -> None:
    """给看板用：每个运行按订正后金标准算出的自定义指标，写到 metrics.json。"""
    ds = ek_dataset.load(evalset)
    tmp = Path(tempfile.mkdtemp(prefix="evalkit-contract-metrics-"))
    for name, (_, variant) in RUNS.items():
        a = m(rescore(root / name, ds, [ContractScorer()], out_dir=tmp / name), "contract_review")
        store.write_json(
            root / name / "metrics.json", {variant: {v: a[k] for k, v in METRIC_LABELS.items()}}
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", type=Path, help="合同审查项目仓库路径（只读）；不给则用自带样例")
    ap.add_argument("--export", action="store_true", help="把导入结果写成仓库自带样例")
    args = ap.parse_args()
    if args.repo:
        repo = args.repo.expanduser()
        if args.export:
            root, evalset = BUNDLED, EVALSET
        else:
            tmp = Path(tempfile.mkdtemp(prefix="evalkit-contract-src-"))
            root, evalset = tmp / "runs", tmp / "contracts.jsonl"
        import_all(repo, root, evalset)
        if args.export:
            write_metrics(root, evalset)
    else:
        root, evalset = BUNDLED, EVALSET
    table = compute(root, evalset)
    print("| 项目 | numbers.md | evalkit 复算 | 一致 |\n|---|---|---|---|")
    bad = 0
    for label, exp, got in table:
        ok = exp == got
        bad += not ok
        print(f"| {label} | {exp} | {got} | {'✓' if ok else '✗'} |")
    print(f"\n{len(table) - bad} / {len(table)} 项一致，{bad} 项不一致")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
