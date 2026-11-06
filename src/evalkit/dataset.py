"""评测集：题目、标准答案、标签、开发集 / 留出集划分、分层抽样、版本号。

文件格式：一个 JSONL 文件，每行一题：
    {"id": "q1", "input": ..., "expected": ..., "tags": {"difficulty": "simple"}, "split": "dev"}

旁边可以放一个同名的 `.meta.json` 记名称和语义化版本号；内容哈希由题目算出，
题目、答案、标签、划分任何一处改动哈希都会变，运行记录里同时存两者，避免"同一版本号、不同题目"。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPLITS = ("dev", "holdout")


@dataclass
class Item:
    id: str | int
    input: Any = None
    expected: Any = None
    tags: dict[str, Any] = field(default_factory=dict)
    split: str | None = None


@dataclass
class Dataset:
    name: str
    items: list[Item]
    version: str = "0.0.0"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        for it in sorted(self.items, key=lambda i: str(i.id)):
            h.update(json.dumps(asdict(it), ensure_ascii=False, sort_keys=True).encode())
            h.update(b"\n")
        return h.hexdigest()[:12]

    @property
    def fingerprint(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "hash": self.content_hash,
            "n": len(self.items),
        }

    def get(self, item_id) -> Item:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)

    def filter(self, pred: Callable[[Item], bool]) -> Dataset:
        return Dataset(self.name, [it for it in self.items if pred(it)], self.version)

    def split(self, name: str) -> Dataset:
        if name not in SPLITS:
            raise ValueError(f"未知划分 {name}，可选 {SPLITS}")
        return self.filter(lambda it: it.split == name)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
        _meta_path(path).write_text(
            json.dumps({"name": self.name, "version": self.version}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _meta_path(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


def load(path: str | Path, name: str | None = None, version: str | None = None) -> Dataset:
    path = Path(path)
    meta = {}
    if _meta_path(path).exists():
        meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                items.append(
                    Item(
                        **{k: d[k] for k in ("id", "input", "expected", "tags", "split") if k in d}
                    )
                )
    ids = [it.id for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} 里有重复的题目 id")
    return Dataset(
        name or meta.get("name") or path.stem,
        items,
        version or meta.get("version", "0.0.0"),
    )


def from_records(
    name: str,
    records: Iterable[dict],
    id_key: str = "id",
    expected_key: str | None = None,
    tag_keys: Iterable[str] = (),
    version: str = "0.0.0",
) -> Dataset:
    """把已有项目的题目列表转成 Dataset：指定 id、答案、标签字段，其余字段整体作为 input。"""
    tag_keys = tuple(tag_keys)
    items = []
    for r in records:
        tags = {k: r[k] for k in tag_keys if k in r}
        items.append(
            Item(
                id=r[id_key],
                input={k: v for k, v in r.items() if k not in (id_key, expected_key)},
                expected=r.get(expected_key) if expected_key else None,
                tags=tags,
            )
        )
    return Dataset(name, items, version)


def _tag(it: Item, key: str | Callable[[Item], Any]):
    return key(it) if callable(key) else it.tags.get(key)


def _quota(sizes: list[int], n: int) -> list[int]:
    """按各层大小比例分配 n 个名额（最大余数法，总数恰为 n）。"""
    total = sum(sizes)
    exact = [n * s / total for s in sizes]
    quota = [int(x) for x in exact]
    by_remainder = sorted(range(len(sizes)), key=lambda i: exact[i] - quota[i], reverse=True)
    for i in by_remainder[: n - sum(quota)]:
        quota[i] += 1
    return quota


def stratified_sample(
    items: list[Item], n: int, by: str | Callable[[Item], Any], seed: int = 0
) -> list[Item]:
    """按标签分层抽样：各层按原比例分配名额，层内用固定种子随机抽，结果按 id 排序。"""
    if n > len(items):
        raise ValueError(f"抽样数 {n} 大于题目总数 {len(items)}")
    rng = random.Random(seed)
    strata: dict[Any, list[Item]] = {}
    for it in items:
        strata.setdefault(_tag(it, by), []).append(it)
    groups = sorted(strata.items(), key=lambda kv: str(kv[0]))
    picked: list[Item] = []
    for (_, its), k in zip(groups, _quota([len(g) for _, g in groups], n), strict=True):
        picked += rng.sample(its, k)
    return sorted(picked, key=lambda it: it.id)


def assign_split(
    ds: Dataset,
    holdout: float | Iterable = 0.3,
    by: str | Callable[[Item], Any] | None = None,
    seed: int = 0,
) -> Dataset:
    """划分开发集和留出集。holdout 为比例时按 by 分层随机划分；为 id 集合时直接指定。

    留出集只在最终报告时使用：调提示词、改规则时只看开发集，避免把留出集"调"进去。
    """
    if isinstance(holdout, float):
        n = round(len(ds) * holdout)
        chosen = stratified_sample(ds.items, n, by or (lambda it: None), seed)
        hold_ids = {it.id for it in chosen}
    else:
        hold_ids = set(holdout)
    items = [
        Item(it.id, it.input, it.expected, dict(it.tags), "holdout" if it.id in hold_ids else "dev")
        for it in ds.items
    ]
    return Dataset(ds.name, items, ds.version)


def slice_counts(items: Iterable[Item], key: str) -> dict:
    out: dict = {}
    for it in items:
        k = it.tags.get(key)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def diff(old: Dataset, new: Dataset) -> dict:
    """两个版本的评测集差异：新增、删除、改动的题目 id，以及改动了哪些字段。"""
    a = {it.id: asdict(it) for it in old}
    b = {it.id: asdict(it) for it in new}
    changed = {}
    for i in sorted(set(a) & set(b), key=str):
        fields = [k for k in ("input", "expected", "tags", "split") if a[i][k] != b[i][k]]
        if fields:
            changed[i] = fields
    return {
        "old": old.fingerprint,
        "new": new.fingerprint,
        "added": sorted(set(b) - set(a), key=str),
        "removed": sorted(set(a) - set(b), key=str),
        "changed": changed,
    }


def validate(ds: Dataset) -> list[str]:
    """评测集体检：返回问题清单（空表示通过）。"""
    problems = []
    for it in ds:
        if it.expected is None:
            problems.append(f"{it.id}：没有标准答案")
        if it.split not in (None, *SPLITS):
            problems.append(f"{it.id}：未知划分 {it.split!r}")
        if not isinstance(it.tags, dict):
            problems.append(f"{it.id}：tags 应为对象")
    splits = {it.split for it in ds}
    if None in splits and len(splits) > 1:
        problems.append("部分题目没有划分 dev / holdout")
    return problems


def describe(ds: Dataset) -> dict:
    """题数、划分、各标签的取值分布（取值过多的标签只报种类数）。"""
    tag_keys = dict.fromkeys(k for it in ds for k in it.tags)
    tags = {}
    for k in tag_keys:
        counts = slice_counts(ds.items, k)
        tags[k] = counts if len(counts) <= 20 else {"distinct": len(counts)}
    return {
        **ds.fingerprint,
        "splits": slice_counts([Item(it.id, tags={"s": it.split}) for it in ds], "s"),
        "tags": tags,
    }
