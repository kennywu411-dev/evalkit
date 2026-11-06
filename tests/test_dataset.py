import json

import pytest

from evalkit.dataset import (
    Dataset,
    Item,
    assign_split,
    from_records,
    load,
    slice_counts,
    stratified_sample,
)


def make(n=30):
    diffs = ["simple"] * 9 + ["moderate"] * 15 + ["challenging"] * 6
    return Dataset(
        "toy", [Item(i, f"q{i}", f"a{i}", {"difficulty": diffs[i % 30]}) for i in range(n)]
    )


def test_save_load_roundtrip_and_hash(tmp_path):
    ds = make()
    ds.version = "1.2.0"
    p = tmp_path / "toy.jsonl"
    ds.save(p)
    back = load(p)
    assert back.name == "toy" and back.version == "1.2.0"
    assert back.content_hash == ds.content_hash
    back.items[0].expected = "changed"
    assert back.content_hash != ds.content_hash


def test_load_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text(json.dumps({"id": 1}) + "\n" + json.dumps({"id": 1}) + "\n")
    with pytest.raises(ValueError):
        load(p)


def test_stratified_sample_keeps_proportions_and_is_reproducible():
    ds = make()
    s = stratified_sample(ds.items, 10, "difficulty", seed=0)
    assert len(s) == 10
    assert slice_counts(s, "difficulty") == {"challenging": 2, "moderate": 5, "simple": 3}
    assert [it.id for it in s] == [it.id for it in stratified_sample(ds.items, 10, "difficulty")]
    assert [it.id for it in s] == sorted(it.id for it in s)
    with pytest.raises(ValueError):
        stratified_sample(ds.items, 31, "difficulty")


def test_largest_remainder_total_is_exact():
    ds = make()
    for n in range(1, 31):
        assert len(stratified_sample(ds.items, n, "difficulty")) == n


def test_assign_split_by_ratio_and_by_ids():
    ds = assign_split(make(), 0.3, by="difficulty")
    hold = ds.split("holdout")
    assert len(hold) == 9 and len(ds.split("dev")) == 21
    assert slice_counts(hold, "difficulty") == {"challenging": 2, "moderate": 4, "simple": 3}
    ds2 = assign_split(make(), holdout=[0, 1])
    assert {it.id for it in ds2.split("holdout")} == {0, 1}
    with pytest.raises(ValueError):
        ds.split("test")


def test_from_records():
    ds = from_records(
        "bird",
        [{"question_id": 7, "question": "x", "SQL": "SELECT 1", "difficulty": "simple"}],
        id_key="question_id",
        expected_key="SQL",
        tag_keys=["difficulty"],
    )
    it = ds.get(7)
    assert it.expected == "SELECT 1" and it.tags == {"difficulty": "simple"}
    assert it.input == {"question": "x", "difficulty": "simple"}


def test_diff_validate_describe():
    from evalkit.dataset import describe, diff, validate

    a = make()
    b = make()
    b.items[0].expected = "new"
    b.items = b.items[1:5] + [b.items[0], Item(99, "q", "a")]
    d = diff(a, b)
    assert d["added"] == [99] and 5 in d["removed"] and d["changed"] == {0: ["expected"]}
    bad = Dataset("x", [Item(1, "q", None, split="test"), Item(2, "q", "a", split="dev")])
    probs = validate(bad)
    assert any("没有标准答案" in p for p in probs) and any("未知划分" in p for p in probs)
    assert validate(assign_split(make(), 0.3, by="difficulty")) == []
    info = describe(assign_split(make(), 0.3, by="difficulty"))
    assert info["splits"] == {"dev": 21, "holdout": 9}
    assert info["tags"]["difficulty"]["moderate"] == 15
