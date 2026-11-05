"""合同风险审查的规则打分器，以及跨合同累加的比例指标。

一份合同 = 一道题。expected 是金标准：
    gold        植入的风险：clause（缺失型为 None）/ clauses（组合风险）/ type / alt_types / level / basis
    borderline  临界条款：期望标"转人工复核"
    favorable   对我方有利的单方条款：报出算视角错误
    acceptable  金标准没植入、但按租户规范报出也合理的意见：不算误报
    distractors 干净合同里的干扰项（v2）
    clarify     当事人空白：期望反问
    our_side    我方是甲方还是乙方

输出是审查结果：kind（review / clarify / fallback）、our_side、findings、citations。

判分口径（与原项目一致）：
- 检出：金标准风险所在条款上有一条意见（缺失型：有一条 clause 为空且类型一致的意见）；
- 类型判对：检出的意见里有一条类型一致（或属于可接受的替代类型）；
- 依据正确：检出的意见引用里至少一条落在合法依据集合（文档 + 条号）；单价风险引用有数据的查表结果也算；
- 误报：既不对应金标准、也不是临界或合理意见的意见，按条款去重；落在有利条款上的单独记为视角错误；
- 反问：当事人空白的合同应当反问，不计入检出；其余合同反问算多余。

ContractScorer 对每份合同产出逐项计数（detail），METRICS 把这些计数跨合同累加成比例（微平均）。
"""

from __future__ import annotations

from typing import Any

from evalkit.dataset import Item
from evalkit.report import Ratio
from evalkit.scorers import Score


def _basis_ok(f: dict, gold: dict, citations: dict) -> bool:
    allowed = {(d, a) for d, a in gold["basis"]}
    for cid in f["citations"]:
        c = citations.get(cid)
        if not c:
            continue
        if c["kind"] == "table":
            if gold["type"] == "unit_price" and c.get("rows"):
                return True
        elif any((c["doc_id"], a) in allowed for a in c.get("articles", [])):
            return True
    return False


def _matches(f: dict, gold: dict) -> bool:
    if gold["clause"] is None:
        return f["clause"] is None and f["risk_type"] == gold["type"]
    return f["clause"] in (gold.get("clauses") or [gold["clause"]])


def _type_ok(f: dict, gold: dict) -> bool:
    return f["risk_type"] == gold["type"] or f["risk_type"] in gold.get("alt_types", [])


def score_contract(exp: dict, out: dict) -> dict:
    """一份合同的逐项计数。exp = 金标准，out = 审查结果。"""
    findings = out["findings"] if out["kind"] == "review" else []
    citations = out.get("citations", {})
    used: set[int] = set()
    gold_rows = []
    for g in exp["gold"]:
        hits = [i for i, f in enumerate(findings) if _matches(f, g)]
        used.update(hits)
        fs = [findings[i] for i in hits]
        gold_rows.append(
            {
                "clause": g["clause"],
                "type": g["type"],
                "category": g.get("category"),
                "detected": bool(fs),
                "type_ok": any(_type_ok(f, g) for f in fs),
                "basis_ok": any(_basis_ok(f, g, citations) for f in fs),
                "human": any(f["needs_human_review"] for f in fs),
            }
        )
    strict_hits = sum(
        1 for g in exp["gold"] if any(_matches(f, g) and _type_ok(f, g) for f in findings)
    )
    border = []
    for b in exp.get("borderline", []):
        hits = [i for i, f in enumerate(findings) if f["clause"] == b["clause"]]
        used.update(hits)
        border.append(
            {"flagged": bool(hits), "human": any(findings[i]["needs_human_review"] for i in hits)}
        )
    for i, f in enumerate(findings):
        if i not in used and any(
            f["clause"] in a["clauses"] and f["risk_type"] in a["types"]
            for a in exp.get("acceptable", [])
        ):
            used.add(i)
    favorable = {f["clause"] for f in exp.get("favorable", [])}
    rest = [f for i, f in enumerate(findings) if i not in used]
    fav_flagged = {f["clause"] for f in rest if f["clause"] in favorable}
    fp = {f["clause"] or f"缺失:{f['risk_type']}" for f in rest} - fav_flagged
    fp_hm = {
        f["clause"] or f"缺失:{f['risk_type']}" for f in rest if f["level"] != "低"
    } - fav_flagged
    return {
        "clarify": exp["clarify"],
        "clean": not exp["gold"],
        "kind": out["kind"],
        "side_ok": out.get("our_side") == exp["our_side"] if out["kind"] == "review" else None,
        "clarify_ok": out["kind"] == "clarify" if exp["clarify"] else False,
        "gold": gold_rows,
        "borderline_n": len(border),
        "borderline_flagged": sum(b["flagged"] for b in border),
        "borderline_human": sum(b["human"] for b in border),
        "favorable_n": len(favorable),
        "favorable_flagged": len(fav_flagged),
        "fp": len(fp),
        "fp_hm": len(fp_hm),
        "n_findings": len(findings),
        "matched_findings": len(used),
        "n_findings_hm": sum(1 for f in findings if f["level"] != "低"),
        "matched_findings_hm": sum(1 for i in used if findings[i]["level"] != "低"),
        "strict_hits": strict_hits,
        "human_findings": sum(f["needs_human_review"] for f in findings),
        "quote_found": sum(f.get("quote_found", True) for f in findings),
    }


class ContractScorer:
    """规则打分器：value = 该合同风险检出比例（干净合同为 1 − 是否误报），
    passed = 风险全部检出且没有误报（反问合同：是否正确反问）。逐项计数放在 detail。"""

    name = "contract_review"

    def __call__(self, item: Item, output: Any) -> Score:
        d = score_contract(item.expected, output)
        if d["clarify"]:
            ok = d["clarify_ok"]
            return Score(float(ok), ok, d)
        bad = d["fp"] + d["favorable_flagged"] > 0
        if d["clean"]:
            return Score(float(not bad), not bad, d)
        det = sum(g["detected"] for g in d["gold"]) / len(d["gold"])
        return Score(det, det == 1 and not bad, d)


def detail(row: dict, scorer: str = ContractScorer.name) -> dict:
    return row["scores"][scorer]["detail"]


def metrics(scorer: str = ContractScorer.name) -> list[Ratio]:
    """跨合同累加的比例指标。反问合同只进反问率。"""

    def D(r):  # noqa: N802
        return detail(r, scorer)

    def scored(r):
        return not D(r)["clarify"]

    def golds(r):
        return D(r)["gold"] if scored(r) else []

    def clean(r):
        return scored(r) and D(r)["clean"]

    def s(key):
        return lambda r: D(r).get(key, 0) if scored(r) else 0

    return [
        Ratio("recall", lambda r: sum(g["detected"] for g in golds(r)), lambda r: len(golds(r))),
        Ratio(
            "type_accuracy",
            lambda r: sum(g["detected"] and g["type_ok"] for g in golds(r)),
            lambda r: sum(g["detected"] for g in golds(r)),
        ),
        Ratio(
            "basis_accuracy",
            lambda r: sum(g["detected"] and g["basis_ok"] for g in golds(r)),
            lambda r: sum(g["detected"] for g in golds(r)),
        ),
        Ratio("precision", s("matched_findings"), s("n_findings")),
        Ratio("precision_hm", s("matched_findings_hm"), s("n_findings_hm")),
        Ratio("precision_strict", s("strict_hits"), s("n_findings")),
        Ratio(
            "clean_fp_rate",
            lambda r: clean(r) and D(r)["fp"] + D(r)["favorable_flagged"] > 0,
            lambda r: clean(r),
        ),
        Ratio(
            "clean_fp_rate_hm",
            lambda r: clean(r) and D(r)["fp_hm"] + D(r)["favorable_flagged"] > 0,
            lambda r: clean(r),
        ),
        Ratio("favorable_flag_rate", s("favorable_flagged"), s("favorable_n")),
        Ratio(
            "side_accuracy",
            lambda r: scored(r) and D(r)["kind"] == "review" and bool(D(r)["side_ok"]),
            lambda r: scored(r) and D(r)["kind"] == "review",
        ),
        Ratio("borderline_human_rate", s("borderline_human"), s("borderline_n")),
        Ratio(
            "clarify_rate",
            lambda r: D(r)["clarify"] and D(r)["clarify_ok"],
            lambda r: D(r)["clarify"],
        ),
        Ratio("quote_found_rate", s("quote_found"), s("n_findings")),
    ]


def dev_ids(contracts: list[dict], n: int = 20) -> set[str]:
    """调试小样（原项目的 sample_contracts）：按子集轮流取、组内优先补齐合同类型。

    小样参与过提示词调试，所以其余合同作为留出集单独报告。
    """

    def group(c: dict) -> str:
        if c["clarify"]:
            return "clarify"
        if c.get("favorable"):
            return "favorable"
        if not c["gold"]:
            return "clean"
        return "borderline" if c.get("borderline") else "risky"

    buckets: dict[str, list[dict]] = {}
    for c in sorted(contracts, key=lambda c: (c["contract_type"], c["id"])):
        buckets.setdefault(group(c), []).append(c)
    out: list[dict] = []
    seen: dict[str, int] = {}
    while len(out) < min(n, len(contracts)):
        for k in ["clarify", "favorable", "clean", "borderline", "risky"]:
            items = buckets.get(k)
            if not items or len(out) >= n:
                continue
            pick = min(items, key=lambda c: seen.get(c["contract_type"], 0))
            items.remove(pick)
            seen[pick["contract_type"]] = seen.get(pick["contract_type"], 0) + 1
            out.append(pick)
    return {c["id"] for c in out}
