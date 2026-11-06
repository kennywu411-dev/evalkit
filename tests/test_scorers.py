import sqlite3

import pytest

from evalkit.dataset import Item
from evalkit.judge import PairwiseJudge, PointwiseJudge, calibrate, parse_json_object
from evalkit.labels import HumanLabelScorer, load_labels
from evalkit.scorers import (
    ExactMatch,
    RuleScorer,
    Score,
    SQLResultMatch,
    contains_all,
    matches_regex,
    max_length,
    scorer,
)


def test_exact_match_normalizes_and_accepts_candidates():
    em = ExactMatch()
    assert em(Item(1, expected="Paris"), "  paris ").passed
    assert em(Item(1, expected=["北京", "Beijing"]), "ＢＥＩＪＩＮＧ").passed
    s = em(Item(1, expected="a"), "b")
    assert not s.passed and "期望" in s.detail
    assert not em(Item(1, expected="a"), None).passed
    assert ExactMatch(field="answer")(Item(1, expected="x"), {"answer": "X"}).passed


def test_function_scorer_decorator():
    @scorer("short")
    def short(item, out):
        return len(out) < 5

    assert short.name == "short"
    assert short(Item(1), "abc") == Score(1.0, True)


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
    conn.commit()
    conn.close()
    return p


def test_sql_result_match_is_order_insensitive_by_default(db):
    m = SQLResultMatch(db=lambda it: db)
    item = Item(1, expected="SELECT id FROM t ORDER BY id")
    assert m(item, "SELECT id FROM t ORDER BY id DESC").passed
    assert not SQLResultMatch(db=lambda it: db, ordered=True)(
        item, "SELECT id FROM t ORDER BY id DESC"
    ).passed
    wrong = m(item, "SELECT id FROM t WHERE id > 1")
    assert not wrong.passed and "2 行" in wrong.detail


def test_sql_result_match_statuses(db):
    m = SQLResultMatch(db=lambda it: db)
    item = Item(1, expected="SELECT 1")
    assert m(item, "").status == "no_output"
    assert m(item, "SELEC nonsense").status == "pred_error"
    assert m(Item(1, expected="SELECT * FROM missing"), "SELECT 1").status == "gold_error"


def test_sql_result_match_is_readonly(db):
    m = SQLResultMatch(db=lambda it: db)
    s = m(Item(1, expected="SELECT 1"), "DELETE FROM t")
    assert s.status == "pred_error"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3


def test_sql_result_match_timeout(db):
    slow = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) "
        "SELECT COUNT(*) FROM c"
    )
    m = SQLResultMatch(db=lambda it: db, timeout_s=0.2)
    assert m(Item(1, expected="SELECT 1"), slow).status == "pred_timeout"
    assert m(Item(1, expected=slow), "SELECT 1").status == "timeout"


def test_rule_scorer():
    r = RuleScorer(
        "format",
        [
            ("has_basis", contains_all(["依据"], field="text")),
            ("cites_article", matches_regex(r"第[一二三四五六七八九十百]+条", field="text")),
            ("short", max_length(50, field="text")),
            ("broken", lambda item, out: out["missing"]),
        ],
        threshold=0.75,
    )
    s = r(Item(1), {"text": "依据第五百八十五条"})
    assert s.value == 0.75 and s.passed
    assert s.detail["checks"]["broken"] is False and "KeyError" in s.detail["errors"]["broken"]


def fake_client(answers):
    it = iter(answers)
    calls = []

    def client(prompt):
        calls.append(prompt)
        return next(it)

    client.calls = calls
    return client


def test_parse_json_object():
    assert parse_json_object('```json\n{"score": 4}\n```') == {"score": 4}
    with pytest.raises(ValueError):
        parse_json_object("没有")


def test_pointwise_judge():
    j = PointwiseJudge(
        fake_client(['{"score": 4, "reason": "ok"}', ('好的 {"score": 9}', 0.01), "乱码"]), "准确"
    )
    s = j(Item(1, "问", "答"), "回答")
    assert s.passed and s.value == 0.8 and s.detail["reason"] == "ok"
    s = j(Item(1), "x")
    assert s.detail["score"] == 5 and s.cost_usd == 0.01  # 越界分数截到 max
    assert j(Item(1), "x").status == "judge_error"


def test_pairwise_judge_swaps_positions():
    # 第一次判 A（原 A），交换后判 B（即原 A）→ 一致，原 A 胜
    j = PairwiseJudge(fake_client(['{"winner": "A"}', '{"winner": "B"}']), "更准确")
    r = j.compare("q", "ans1", "ans2")
    assert r == {"winner": "a", "consistent": True, "votes": ["A", "A"], "cost_usd": 0.0}
    assert "ans1" in j.client.calls[0].split("回答 A")[1].split("回答 B")[0]
    assert "ans2" in j.client.calls[1].split("回答 A")[1].split("回答 B")[0]


def test_pairwise_position_bias_becomes_tie():
    # 两次都选 A：说明只看位置，记为平局
    j = PairwiseJudge(fake_client(['{"winner": "A"}', '{"winner": "A"}'] * 2), "x")
    out = j.compare_many([("q", "a", "b"), ("q2", "a", "b")])
    assert out["ties"] == 2 and out["position_consistency"] == 0.0 and out["b_win_rate"] == 0.5


def test_calibrate():
    judge = {1: "pass", 2: "fail", 3: "pass", 4: "pass"}
    human = {1: "pass", 2: "fail", 3: "fail"}
    c = calibrate(judge, human)
    assert c["n"] == 3 and c["agreement"] == pytest.approx(2 / 3)
    assert c["disagreements"] == [3] and c["missing_human"] == 1
    assert c["confusion"]["pass"] == {"pass": 1, "fail": 1}


def test_load_labels_csv_and_jsonl(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text(
        "id,label,annotator\n1,correct,u1\n2,wrong,u1\n1,wrong,u2\n3,,u1\n", encoding="utf-8"
    )
    assert load_labels(p) == {1: "wrong", 2: "wrong"}
    assert load_labels(p, annotator="u1") == {1: "correct", 2: "wrong"}
    q = tmp_path / "l.jsonl"
    q.write_text('{"qid": "a", "verdict": "pass"}\n', encoding="utf-8")
    assert load_labels(q, id_field="qid", label_field="verdict") == {"a": "pass"}


def test_human_label_scorer():
    s = HumanLabelScorer({1: "correct", 2: "wrong"})
    assert s(Item(1), None).passed
    assert s(Item(2), None).passed is False
    assert s(Item(3), None).status == "unlabeled"
