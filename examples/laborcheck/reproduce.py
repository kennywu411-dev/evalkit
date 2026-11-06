"""用 evalkit 复算劳动法评测（laborcheck）docs/results.md 里的结果表。

    uv run python examples/laborcheck/reproduce.py                                 # 用仓库自带的样例
    uv run python examples/laborcheck/reproduce.py --repo ~/Documents/laborcheck            # 只读原仓库
    uv run python examples/laborcheck/reproduce.py --repo ~/Documents/laborcheck --export   # 刷新样例

做法：
1. 把 18 次运行导入成 evalkit 运行目录，output 只留回答末尾的 JSON 对象；
2. 用 evalkit.runner.rescore + scoring.answer_scorer / BasisScorer 重新判分，检查与运行时的判分逐题一致；
3. 用重新判分的结果算出 results.md 里每张结果表的每个格子（准确率、Wilson 区间、拆分、McNemar、
   花费延迟、两轮之间的稳定性），与文档逐字比较。

文档里的表格由 --repo 现读；自带样例模式读 results-tables.json（--export 时从文档摘录）。
运行登记里的"题目范围""评测集版本"这类描述性的列不复算。全部一致退出码 0，否则 1。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adapter import (  # noqa: E402
    CITY_LABEL,
    TOPIC_LABEL,
    evalsets_from_repo,
    import_run,
)
from scoring import BasisScorer, answer_scorer  # noqa: E402

from evalkit import dataset as ek_dataset  # noqa: E402
from evalkit import store  # noqa: E402
from evalkit.compare import compare  # noqa: E402
from evalkit.runner import rescore  # noqa: E402
from evalkit.stats import wilson  # noqa: E402

BUNDLED = HERE / "runs"
DATA = HERE / "data"
TABLES = HERE / "results-tables.json"
VERSIONS = ("1.0.1", "1.0.2")

# 运行 → 评测集版本；三组分别对应 results.md 的试跑、第一轮全量、主结果
GROUPS = {
    "pilot": [
        f"pilot-{m}-{mode}"
        for m in ("claude", "deepseek", "qwen", "claude-haiku")
        for mode in ("closed", "tools")
    ],
    "full": [
        f"full-{m}-{mode}" for m in ("claude", "claude-haiku") for mode in ("closed", "tools")
    ],
    "v102": [
        f"v102-{m}-{mode}"
        for m in ("claude", "claude-sonnet", "claude-haiku")
        for mode in ("closed", "tools")
    ],
}
RUN_VERSION = {"pilot": "1.0.0", "full": "1.0.1", "v102": "1.0.2"}
# results.md 的章节 → 组
SECTION_GROUP = {"试跑结果": "pilot", "第一轮全量": "full", "主结果": "v102"}
SECTIONS = ("运行登记", *SECTION_GROUP)

# 文档与复算不一致、且已查明原因的项：单独列出，不计入失败
KNOWN = {
    "第一轮全量（评测集 1.0.1，工具修复前） / 花费与工具使用 | `full-claude-tools` | 每题平均花费 (USD)": (
        "合计 5.9829 ÷ 339 题 = 0.01765，应为 0.0176；文档的 0.0177 是除以 338 得到的"
    ),
}


# ---------- 文档表格 ----------


def parse_tables(md: str) -> list[dict]:
    """按章节摘出 Markdown 表格：[{section, header, rows}]，section 是各级标题拼起来的路径。"""
    heads: dict[int, str] = {}
    tables: list[dict] = []
    cur: dict | None = None
    for line in md.splitlines():
        m = re.match(r"^(#{2,4}) (.+)", line)
        if m:
            level = len(m.group(1))
            heads = {k: v for k, v in heads.items() if k < level} | {level: m.group(2).strip()}
            cur = None
            continue
        if not line.startswith("|"):
            cur = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cur is None:
            section = " / ".join(heads[k] for k in sorted(heads))
            cur = {"section": section, "header": cells, "rows": []}
            tables.append(cur)
        elif not set("".join(cells)) <= set("-: "):
            cur["rows"].append(cells)
    return [t for t in tables if any(s in t["section"] for s in SECTIONS)]


# ---------- 复算 ----------


def pct(k: int, n: int) -> str:
    return f"{k / n * 100:.1f}%（{k}/{n}）"


def data_kind(r: dict) -> str:
    t = r["tags"]
    if t["params_given"]:
        return "题面给出地方参数"
    return "需要模型自己知道地方数据" if t["needs_local_data"] else "不依赖地方数据"


class Calc:
    def __init__(self, root: Path, evalsets: dict[str, ek_dataset.Dataset]):
        self.evalsets = evalsets
        tmp = Path(tempfile.mkdtemp(prefix="evalkit-laborcheck-"))
        ds = evalsets["1.0.2"]  # 三版的标准答案相同（只改过题面），都用最新版判分
        self.recorded = {name: store.load_rows(root / name) for g in GROUPS.values() for name in g}
        self.rows = {
            name: rescore(root / name, ds, [answer_scorer(), BasisScorer()], out_dir=tmp / name)
            for name in self.recorded
        }
        self.cfg = {
            name: next(iter(store.load_config(root / name)["variants"].values()))
            for name in self.recorded
        }

    # 一次运行
    def acc(self, name: str, pred=lambda r: True) -> str:
        rs = [r for r in self.rows[name] if pred(r)]
        return pct(sum(r["passed"] for r in rs), len(rs)) if rs else "—"

    def find(self, group: str, display_or_key: str, mode: str) -> str:
        hits = [
            n
            for n in GROUPS[group]
            if display_or_key in self.cfg[n]["display"] and self.cfg[n]["mode"] == mode
        ]
        if len(hits) != 1:
            raise KeyError(f"{group} 里按 {display_or_key!r}/{mode} 找到 {hits}")
        return hits[0]

    def column_run(self, group: str, col: str) -> str:
        """'Claude Opus 5.5（仅模型）' / 'Claude Opus 5.5（工具）' → 运行名。"""
        display, mode = re.fullmatch(r"(.+)（(仅模型|工具)）", col).groups()
        return self.find(group, display, "closed" if mode == "仅模型" else "tools")

    def overall(self, name: str, col: str) -> str | None:
        rs, cfg = self.rows[name], self.cfg[name]
        k, n = sum(r["passed"] for r in rs), len(rs)
        types = {"合法性判断": "legality", "金额计算": "calc", "条款改写": "rewrite"}
        if col == "模型":
            return cfg["display"]
        if col == "模式":
            return "模型 + 规则引擎工具" if cfg["mode"] == "tools" else "仅模型"
        if col == "总准确率":
            return pct(k, n)
        if col == "95% 区间":
            lo, hi = wilson(k, n)
            return f"{lo * 100:.0f}%–{hi * 100:.0f}%"
        if col in types:
            return self.acc(name, lambda r: r["tags"]["type"] == types[col])
        if col == "依据命中":
            hits = [r["scores"]["basis_hit"]["passed"] for r in rs]
            hits = [h for h in hits if h is not None]
            return f"{sum(hits) / len(hits) * 100:.0f}%"
        if col == "花费 (USD)":
            return f"{sum(r['cost_usd'] for r in rs):.4f}"
        return None

    def registry(self, name: str, col: str) -> str | None:
        rs, cfg = self.rows[name], self.cfg[name]
        if col == "模型":
            return cfg["display"]
        if col == "模式":
            return "模型 + 工具" if cfg["mode"] == "tools" else "仅模型"
        if col == "题数":
            return str(len(rs))
        if col == "花费 (USD)":
            return f"{sum(r['cost_usd'] for r in rs):.2f}"
        return None  # 题目范围、评测集版本是描述，不复算

    def split(self, name: str, col: str) -> str | None:
        if col.startswith("开发集"):
            return self.acc(name, lambda r: r["split"] == "dev")
        if col.startswith("留出集（"):
            return self.acc(name, lambda r: r["split"] == "holdout")
        if col == "留出集中需地方数据的题":
            rs = [
                r
                for r in self.rows[name]
                if r["split"] == "holdout" and data_kind(r) == "需要模型自己知道地方数据"
            ]
            return f"{sum(r['passed'] for r in rs)}/{len(rs)}"
        return None

    def cost(self, name: str, col: str) -> str | None:
        rs, tools = self.rows[name], self.cfg[name]["mode"] == "tools"
        n = len(rs)
        if col == "每题平均花费 (USD)":
            avg = sum(r["cost_usd"] for r in rs) / n
            return f"{avg:.5f}" if avg < 0.001 else f"{avg:.4f}"
        if col == "平均延迟 (秒)":
            return f"{sum(r['latency_s'] for r in rs) / n:.1f}"
        if col == "平均输出 token":
            return f"{sum(r['usage']['tokens_out'] for r in rs) / n:.0f}"
        if not tools:
            return "—"
        none = [r for r in rs if r["tool_calls"] == 0]
        if col.startswith("每题工具调用"):
            return f"{sum(r['tool_calls'] for r in rs) / n:.2f}"
        if col == "未调用工具的题数":
            return str(len(none))
        if col == "未调用工具的题（其中答对）":
            return f"{len(none)}（{sum(r['passed'] for r in none)}）"
        return None

    def mcnemar(self, group: str, model: str, col: str) -> str:
        c = compare(
            self.rows[self.find(group, model, "closed")],
            self.rows[self.find(group, model, "tools")],
        )
        return {
            "仅加工具后答对": str(len(c["b_only"])),
            "仅不加工具时答对": str(len(c["a_only"])),
            "p 值": f"{c['p_value']:.1e}",
        }[col]

    def local_errors(self, group: str, col: str, pred) -> str:
        model = re.match(r"(Opus|Sonnet|Haiku)", col).group(1)
        name = self.find(group, model, "closed")
        rs = [r for r in self.rows[name] if data_kind(r) == "需要模型自己知道地方数据" and pred(r)]
        return f"{sum(not r['passed'] for r in rs)}/{len(rs)}"

    def stability(self, row: str, col: str) -> str:
        """1.0.1 和 1.0.2 两轮全量之间，题面未改动的题上同一模型同一模式的结果对比。"""
        model, mode = re.fullmatch(r"(\w+) (仅模型|加工具)", row).groups()
        mode = "closed" if mode == "仅模型" else "tools"
        a = {r["item_id"]: r["passed"] for r in self.rows[self.find("full", model, mode)]}
        b = {r["item_id"]: r["passed"] for r in self.rows[self.find("v102", model, mode)]}
        changed = set(ek_dataset.diff(self.evalsets["1.0.1"], self.evalsets["1.0.2"])["changed"])
        same = [i for i in b if i in a and i not in changed]
        return {
            "不一致题数": str(sum(a[i] != b[i] for i in same)),
            "第一轮答对": str(sum(a[i] for i in same)),
            "第二轮答对": str(sum(b[i] for i in same)),
        }[col]

    def cell(self, section: str, header: list[str], row: list[str], col: str) -> str | None:
        group = next((g for s, g in SECTION_GROUP.items() if s in section), None)
        first, label = header[0], row[0]
        run = label.strip("`")
        if first == "运行 ID":
            if "题数" in header:
                return self.registry(run, col)
            if "总准确率" in header:
                return self.overall(run, col)
            if header[1].startswith("开发集"):
                return self.split(run, col)
            if header[1].startswith("每题平均花费"):
                return self.cost(run, col)
        if first == "模型" and "p 值" in header:
            return self.mcnemar(group, label, col)
        if first == "地方数据":
            return self.acc(self.column_run(group, col), lambda r: data_kind(r) == label)
        if first == "主题" and col.endswith("错题"):
            topic = {v: k for k, v in TOPIC_LABEL.items()}[label.split("（")[0]]
            return self.local_errors(group, col, lambda r: r["tags"]["topic"] == topic)
        if first == "主题":
            topic = {v: k for k, v in TOPIC_LABEL.items()}[label]
            return self.acc(self.column_run(group, col), lambda r: r["tags"]["topic"] == topic)
        if first == "城市":
            city = {v: k for k, v in CITY_LABEL.items()}[label]
            return self.local_errors(group, col, lambda r: r["tags"]["city"] == city)
        if first == "模型与模式":
            return self.stability(label, col)
        return None

    def agreement(self) -> list[tuple[str, str, str]]:
        """重新判分与运行时判分逐题比较：主指标和依据命中。"""
        out = []
        for name, rec in self.recorded.items():
            old = {r["item_id"]: r["scores"]["recorded"] for r in rec}
            new = {r["item_id"]: r["scores"] for r in self.rows[name]}
            same = sum(
                old[i]["passed"] == new[i]["laborcheck"]["passed"]
                and old[i]["detail"]["basis_hit"] == new[i]["basis_hit"]["passed"]
                for i in old
            )
            out.append(
                (f"{name} 重新判分与运行时逐题一致", f"{len(old)}/{len(old)}", f"{same}/{len(old)}")
            )
        return out


def compute(root: Path, evalsets: dict, tables: list[dict]) -> list[tuple[str, str, str]]:
    calc = Calc(root, evalsets)
    out = calc.agreement()
    for t in tables:
        for row in t["rows"]:
            for col, exp in zip(t["header"][1:], row[1:], strict=True):
                got = calc.cell(t["section"], t["header"], row, col)
                if got is not None:
                    out.append((f"{t['section']} | {row[0]} | {col}", exp, got))
    return out


def load_bundled_evalsets() -> dict[str, ek_dataset.Dataset]:
    return {v: ek_dataset.load(DATA / f"laborcheck-{v}.jsonl") for v in VERSIONS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", type=Path, help="laborcheck 仓库路径（只读）；不给则用自带样例")
    ap.add_argument("--export", action="store_true", help="把导入结果写成仓库自带样例")
    args = ap.parse_args()
    if args.repo:
        repo = args.repo.expanduser()
        evalsets = evalsets_from_repo(repo, VERSIONS)
        tables = parse_tables((repo / "docs" / "results.md").read_text(encoding="utf-8"))
        root = BUNDLED if args.export else Path(tempfile.mkdtemp(prefix="evalkit-lc-src-")) / "runs"
        for group, names in GROUPS.items():
            for name in names:
                import_run(repo / "runs" / name, root / name, RUN_VERSION[group])
        if args.export:
            for v, ds in evalsets.items():
                ds.save(DATA / f"laborcheck-{v}.jsonl")
            TABLES.write_text(json.dumps(tables, ensure_ascii=False, indent=1) + "\n", "utf-8")
    else:
        root, evalsets = BUNDLED, load_bundled_evalsets()
        tables = json.loads(TABLES.read_text(encoding="utf-8"))
    table = compute(root, evalsets, tables)
    print("| 项目 | results.md | evalkit 复算 | 一致 |\n|---|---|---|---|")
    bad = known = 0
    for label, exp, got in table:
        ok = exp == got
        mark = "✓" if ok else ("已知差异" if label in KNOWN else "✗")
        bad += mark == "✗"
        known += mark == "已知差异"
        print(f"| {label} | {exp} | {got} | {mark} |")
    for label, why in KNOWN.items():
        print(f"\n已知差异：{label}\n  {why}")
    print(
        f"\n{len(table) - bad - known} / {len(table)} 项一致，已知差异 {known} 项，{bad} 项不一致"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
