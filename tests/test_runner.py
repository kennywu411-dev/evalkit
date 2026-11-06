import asyncio
import json

import pytest

from evalkit import store
from evalkit.dataset import Dataset, Item
from evalkit.runner import AbortRun, TaskResult, Variant, rescore, run
from evalkit.scorers import ExactMatch


def ds(n=10):
    return Dataset("toy", [Item(i, f"q{i}", str(i), {"parity": i % 2}) for i in range(n)], "1.0.0")


def echo(item):
    return TaskResult(output=item.expected, cost_usd=0.01, trace=[{"step": "answer"}])


async def async_wrong_on_odd(item):
    await asyncio.sleep(0)
    return str(item.id) if item.id % 2 == 0 else "wrong"


def test_run_writes_rows_config_traces_summary(tmp_path):
    r = run(
        ds(),
        [Variant("echo", echo, {"model": "m1"}), Variant("half", async_wrong_on_odd)],
        [ExactMatch()],
        runs_dir=tmp_path,
        tag="t",
        repeats=2,
        progress=False,
    )
    assert len(r.rows) == 40 and r.complete
    assert r.spent_usd == pytest.approx(0.2)
    cfg = store.load_config(r.run_dir)
    assert cfg["dataset"]["hash"] == ds().content_hash
    assert cfg["variants"] == {"echo": {"model": "m1"}, "half": {}}
    assert cfg["primary_metric"] == "exact_match" and len(cfg["item_ids"]) == 10
    row = next(x for x in r.rows if x["variant"] == "half" and x["item_id"] == 3)
    assert row["passed"] is False and row["tags"] == {"parity": 1}
    t = store.load_trace(r.run_dir, row["trace_id"])
    assert t["input"] == "q3" and t["output"] == "wrong"
    summary = json.loads((r.run_dir / "summary.json").read_text())
    rates = {s["variant"]: s["rate"] for s in summary}
    assert rates == {"echo": 1.0, "half": 0.5}


def test_errors_are_recorded_and_retried_on_resume(tmp_path):
    calls = {"n": 0}

    def flaky(item):
        calls["n"] += 1
        if item.id == 3 and calls["n"] < 20:
            raise RuntimeError("网络超时")
        return item.expected

    r = run(ds(), [Variant("v", flaky)], [ExactMatch()], runs_dir=tmp_path, progress=False)
    err = [x for x in r.rows if x["status"] == "error"]
    assert len(err) == 1 and "网络超时" in err[0]["error"] and not r.complete
    calls["n"] = 100
    r2 = run(
        ds(),
        [Variant("v", flaky)],
        [ExactMatch()],
        runs_dir=tmp_path,
        resume=r.run_id,
        progress=False,
    )
    assert r2.run_id == r.run_id and all(x["passed"] for x in r2.rows) and len(r2.rows) == 10
    assert calls["n"] == 101  # 只重跑了出错的那一题


def test_budget_cap_stops_and_resume_finishes(tmp_path):
    r = run(
        ds(),
        [Variant("v", echo)],
        [ExactMatch()],
        runs_dir=tmp_path,
        concurrency=1,
        max_cost=0.035,
        progress=False,
    )
    assert len(r.rows) == 4 and r.skipped == 6  # 花到 0.04 ≥ 0.035 后停止
    r2 = run(
        ds(),
        [Variant("v", echo)],
        [ExactMatch()],
        runs_dir=tmp_path,
        resume=r.run_id,
        max_cost=1.0,
        progress=False,
    )
    assert len(r2.rows) == 10 and r2.spent_usd == pytest.approx(0.1)


def test_resume_rejects_changed_dataset(tmp_path):
    r = run(ds(), [Variant("v", echo)], [ExactMatch()], runs_dir=tmp_path, progress=False)
    changed = ds()
    changed.items[0].expected = "x"
    with pytest.raises(ValueError, match="被改动"):
        run(
            changed,
            [Variant("v", echo)],
            [ExactMatch()],
            runs_dir=tmp_path,
            resume=r.run_id,
            progress=False,
        )


def test_abort_run_stops(tmp_path):
    def dead(item):
        raise AbortRun("密钥失效")

    with pytest.raises(AbortRun, match="续跑"):
        run(ds(), [Variant("v", dead)], [ExactMatch()], runs_dir=tmp_path, progress=False)


def test_scorer_error_does_not_crash(tmp_path):
    def bad(item, out):
        raise ZeroDivisionError

    bad.name = "bad"
    r = run(ds(2), [Variant("v", echo)], [bad], runs_dir=tmp_path, progress=False)
    assert all(x["status"] == "scorer_error" and not x["passed"] for x in r.rows)


def test_rescore_uses_saved_outputs(tmp_path):
    r = run(
        ds(),
        [Variant("half", async_wrong_on_odd)],
        [ExactMatch()],
        runs_dir=tmp_path,
        progress=False,
    )
    fixed = ds()
    for it in fixed.items:  # 订正标准答案：奇数题的正确答案就是 "wrong"
        if it.id % 2:
            it.expected = "wrong"
    rows = rescore(r.run_dir, fixed, [ExactMatch()])
    assert all(x["passed"] for x in rows)
    out = tmp_path / f"{r.run_id}-rescored"
    assert store.load_config(out)["rescored_from"] == r.run_id
    assert (out / "traces").is_dir()
    assert sum(x["passed"] for x in store.load_rows(r.run_dir)) == 5  # 原结果不动


def test_load_rows_last_wins(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    store.append_row(d, {"item_id": 1, "variant": "v", "repeat": 0, "passed": False})
    store.append_row(d, {"item_id": 1, "variant": "v", "repeat": 0, "passed": True})
    assert store.load_rows(d) == [{"item_id": 1, "variant": "v", "repeat": 0, "passed": True}]
    assert store.list_runs(tmp_path) == [d]


def test_sync_tasks_run_concurrently(tmp_path):
    import threading
    import time

    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow(item):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
        return item.expected

    r = run(
        ds(8),
        [Variant("v", slow)],
        [ExactMatch()],
        runs_dir=tmp_path,
        concurrency=4,
        progress=False,
    )
    assert all(x["passed"] for x in r.rows)
    assert active["max"] > 1


def test_dict_trace_is_flattened_and_custom_trace_id(tmp_path):
    def app(item):
        own = {"trace_id": f"t{item.id}", "steps": [{"llm_ms": 10}], "result": {"kind": "review"}}
        return TaskResult(output=item.expected, trace=own, trace_id=f"t{item.id}")

    r = run(ds(2), [Variant("v", app)], [ExactMatch()], runs_dir=tmp_path, progress=False)
    assert {x["trace_id"] for x in r.rows} == {"t0", "t1"}
    t = store.load_trace(r.run_dir, "t1")
    assert t["result"] == {"kind": "review"} and t["steps"] == [{"llm_ms": 10}]
    assert t["item_id"] == 1 and t["scores"]["exact_match"]["passed"]


def test_resume_can_add_items(tmp_path):
    r = run(ds(3), [Variant("v", echo)], [ExactMatch()], runs_dir=tmp_path, progress=False)
    r2 = run(
        ds(5),
        [Variant("v", echo)],
        [ExactMatch()],
        runs_dir=tmp_path,
        resume=r.run_id,
        progress=False,
    )
    assert len(r2.rows) == 5 and r2.spent_usd == pytest.approx(0.05)


def test_row_fields_and_no_trace_files(tmp_path):
    def fields(item, row):
        return {"question_id": item.id, "correct": row["passed"], "item_id": "不能改"}

    r = run(
        ds(3),
        [Variant("v", echo)],
        [ExactMatch()],
        runs_dir=tmp_path,
        progress=False,
        row_fields=fields,
        save_traces=False,
    )
    assert all(x["question_id"] == x["item_id"] and x["correct"] for x in r.rows)
    assert not (r.run_dir / "traces").exists()
    fixed = ds(3)
    fixed.items[0].expected = "x"
    rows = rescore(r.run_dir, fixed, [ExactMatch()], row_fields=fields)
    assert [x["correct"] for x in sorted(rows, key=lambda x: x["item_id"])] == [False, True, True]
