import pytest

from evalkit.compare import compare, render
from evalkit.feedback import FeedbackStore, latest, summarize
from evalkit.gate import gate
from evalkit.report import Ratio, aggregate, repeat_stats
from evalkit.report import render as render_summary
from evalkit.report import summarize as summarize_runs


def rows(passed, variant="v", repeat=0, cost=0.01, tags=None):
    return [
        {
            "item_id": i,
            "variant": variant,
            "repeat": repeat,
            "passed": p,
            "status": "ok",
            "cost_usd": cost,
            "latency_s": 1.0 + i,
            "tags": tags[i] if tags else {"d": "easy" if i % 2 else "hard"},
            "scores": {"m": {"value": float(p), "passed": p}},
        }
        for i, p in enumerate(passed)
    ]


def test_summarize_and_slices():
    s = summarize_runs(rows([True, True, False, True]), slices=["d"])
    assert len(s) == 1
    s = s[0]
    assert s["n"] == 4 and s["rate"] == 0.75 and s["scores"]["m"] == 0.75
    assert s["slices"]["d"]["hard"] == {
        "n": 2,
        "passed": 1,
        "rate": 0.5,
        "ci95": pytest.approx(s["slices"]["d"]["hard"]["ci95"]),
    }
    assert s["latency_p50"] == 3.0 and s["cost_total"] == pytest.approx(0.04)
    assert "75.0%" in render_summary([s])


def test_repeat_stats():
    rs = rows([True, True, False, False]) + rows([True, False, False, True], repeat=1)
    out = repeat_stats(rs)
    assert out["rates"] == [0.5, 0.5] and out["std"] == 0.0
    assert out["stability"] == {"n": 4, "always": 1, "never": 1, "flaky": 2, "flaky_rate": 0.5}


def test_ratio_metric_is_micro_average():
    data = [{"hit": 1, "total": 1}, {"hit": 0, "total": 3}]
    recall = Ratio("recall", lambda r: r["hit"], lambda r: r["total"])
    assert recall.value(data) == 0.25
    out = aggregate(data, [recall], ci=True)
    assert out["recall"]["num"] == 1 and out["recall"]["den"] == 4
    lo, hi = out["recall"]["ci95"]
    assert 0 <= lo <= hi <= 1
    assert Ratio("x", lambda r: 0, lambda r: 0).value(data) is None


def test_compare_counts_and_mcnemar():
    a = rows([True] * 10 + [False] * 10)
    b = rows([False] * 2 + [True] * 8 + [True] * 8 + [False] * 2, variant="w")
    c = compare(a, b, slices=["d"])
    assert c["n"] == 20 and len(c["a_only"]) == 2 and len(c["b_only"]) == 8
    assert c["delta"] == pytest.approx(0.3)
    assert round(c["p_value"], 3) == 0.109
    assert sum(v["n"] for v in c["slices"]["d"].values()) == 20
    assert "McNemar" in render(c)


def test_compare_requires_variant_when_ambiguous():
    with pytest.raises(ValueError, match="多个变体"):
        compare(rows([True]) + rows([True], variant="w"), rows([True]))


def test_gate_significant_drop_fails():
    base = rows([True] * 100)
    cand = rows([True] * 85 + [False] * 15)
    g = gate(base, cand, max_drop=0.5)
    assert not g.passed and any("显著" in r for r in g.reasons)
    assert "门禁不通过" in g.render()


def test_gate_small_drop_on_small_set_passes_but_threshold_catches_big_drop():
    base = rows([True] * 20)
    assert gate(base, rows([True] * 19 + [False]), max_drop=0.05).passed
    g = gate(base, rows([True] * 17 + [False] * 3), max_drop=0.10)  # p=0.25 不显著，但降 15 点
    assert not g.passed and any("超过" in r for r in g.reasons)


def test_gate_improvement_passes_and_coverage_cost_checks():
    base = rows([False] * 10)
    assert gate(base, rows([True] * 10)).passed
    g = gate(base, rows([False] * 8))
    assert not g.passed and any("缺少" in r for r in g.reasons)
    g = gate(base, rows([False] * 10, cost=0.05), max_cost_ratio=2.0)
    assert not g.passed and any("花费" in r for r in g.reasons)


def fb(label, index=0, level="高", review_id="rev1", risk_type="inspection", session="s1"):
    return {
        "session_id": session,
        "review_id": review_id,
        "finding_index": index,
        "label": label,
        "finding": {"clause": "第四条", "risk_type": risk_type, "level": level},
    }


LABELS = ("useful", "not_useful", "not_risk", "unclear")
KEY = ("session_id", "review_id", "finding_index")


def test_feedback_latest_label_wins(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl", labels=LABELS)
    store.add(fb("not_risk", 0))
    store.add(fb("useful", 0))  # 改了主意，以最后一次为准
    store.add(fb("not_risk", 1))
    store.add(fb("unclear", 2, level="中"))
    store.add(fb("useful", 3, level="低"))  # 低等级不进主指标
    s = summarize(
        store.load(),
        key=KEY,
        labels=LABELS,
        positive="useful",
        negative="not_risk",
        where=lambda r: r["finding"]["level"] in ("高", "中"),
        by="finding.risk_type",
    )
    assert s["rated"] == 3 and s["sessions"] == 1
    assert s["labels"] == {"useful": 1, "not_useful": 0, "not_risk": 1, "unclear": 1}
    assert s["positive_rate"] == pytest.approx(1 / 3) and s["negative_rate"] == pytest.approx(1 / 3)
    assert s["by"]["groups"]["inspection"]["rated"] == 3
    assert len(store.load()) == 5  # 改标历史保留


def test_feedback_validation_and_empty(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl", labels=LABELS, required=["session_id"])
    with pytest.raises(ValueError, match="未知标签"):
        store.add(fb("great"))
    with pytest.raises(ValueError, match="缺少"):
        store.add({"label": "useful"})
    s = summarize(store.load(), key=KEY, labels=LABELS, positive="useful")
    assert s["rated"] == 0 and s["positive_rate"] is None


def test_latest_orders_by_ts_when_present():
    rs = [{"k": 1, "label": "b", "ts": 2}, {"k": 1, "label": "a", "ts": 1}]
    assert latest(rs, ["k"])[0]["label"] == "b"
