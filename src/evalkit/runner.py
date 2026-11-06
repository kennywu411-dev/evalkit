"""实验运行：评测集 × 变体 × 重复次数，并发执行、逐题落盘、断点续跑、花费硬上限。

被评测的应用以 Task 的形式接入：一个（同步或异步）函数，接收一道题，返回 TaskResult
（或直接返回输出，此时花费记 0）。变体 = 名字 + Task + 参数快照，例如同一流程换模型、换提示词。

    variants = [Variant("haiku", make_task("haiku"), {"model": "haiku", "prompt": "v2"})]
    run = evalkit.run(dataset, variants, [ExactMatch()], tag="e1", max_cost=2.0)

约定：
- 第一个打分器是主指标，它的 passed 写进结果行的 passed，配对检验和门禁默认看它；
- 任务抛异常记为 status=error，续跑时会重试；抛 AbortRun（如密钥失效、额度用完）立即停止整次运行；
- max_cost 是本次运行目录的累计花费上限（续跑时已花的钱也算在内）：达到后不再发起新任务，
  已在途的任务会跑完，所以可能略超；
- row_fields(item, row) 返回的字段并入结果行，用于沿用项目原有的字段名，让旧代码照常读取；
- save_traces=False 时不单独写 trace 文件（过程已经放在结果行里、或运行目录要入 git 时）。
"""

from __future__ import annotations

import asyncio
import inspect
import platform
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit import store
from evalkit.dataset import Dataset, Item
from evalkit.scorers import Score


class AbortRun(Exception):
    """继续跑只会让剩下的题全部失败时抛出（鉴权失败、欠费等）。"""


@dataclass
class TaskResult:
    output: Any
    cost_usd: float = 0.0
    usage: dict | None = None
    # 中间步骤（工具调用、检索结果、重试……）。列表等记在 trace 文件的 steps 字段；
    # dict 则原样展开到 trace 文件顶层，应用自己的 trace 格式和读取它的代码都不用改
    trace: Any = None
    meta: dict = field(default_factory=dict)  # 额外字段，写进结果行
    trace_id: str | None = None  # 应用自己的 trace id；不给则由 (题目, 变体, 重复) 生成


Task = Callable[[Item], Any]


@dataclass
class Variant:
    name: str
    task: Task
    params: dict = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    rows: list[dict]
    spent_usd: float
    skipped: int  # 因花费上限没跑的任务数

    @property
    def complete(self) -> bool:
        return self.skipped == 0 and not any(r["status"] == "error" for r in self.rows)


def _is_async(task: Task) -> bool:
    # 带 async __call__ 的可调用对象也算异步任务
    return inspect.iscoroutinefunction(task) or inspect.iscoroutinefunction(
        type(task).__call__ if not inspect.isfunction(task) else None
    )


async def _call_task(task: Task, item: Item) -> TaskResult:
    """异步任务直接 await；同步任务放进线程执行，否则会阻塞事件循环、并发形同虚设。"""
    out = await task(item) if _is_async(task) else await asyncio.to_thread(task, item)
    if inspect.isawaitable(out):
        out = await out
    return out if isinstance(out, TaskResult) else TaskResult(output=out)


def _score_all(scorers: Sequence, item: Item, output: Any) -> dict[str, Score]:
    scores = {}
    for s in scorers:
        name = getattr(s, "name", None) or getattr(s, "__name__", "score")
        try:
            scores[name] = s(item, output)
        except Exception as e:  # 打分器出错：该项记失败，保留输出，不中断整次运行
            scores[name] = Score(0.0, False, f"{type(e).__name__}: {e}", "scorer_error")
    return scores


def build_row(
    item: Item,
    variant: str,
    repeat: int,
    res: TaskResult | None,
    scores: dict[str, Score],
    latency_s: float,
    error: str | None = None,
) -> dict:
    primary = next(iter(scores.values()), None)
    if error is not None:
        status = "error"
    elif primary is not None and primary.status != "ok":
        status = primary.status
    else:
        status = "ok"
    return {
        "item_id": item.id,
        "variant": variant,
        "repeat": repeat,
        "split": item.split,
        "tags": item.tags,
        "status": status,
        "passed": bool(primary.passed) if primary is not None and error is None else False,
        "scores": {k: v.to_dict() for k, v in scores.items()},
        "output": res.output if res else None,
        "cost_usd": (res.cost_usd if res else 0.0) + sum(s.cost_usd for s in scores.values()),
        "usage": res.usage if res else None,
        "latency_s": latency_s,
        "trace_id": (res.trace_id if res and res.trace_id else None)
        or store.trace_id(item.id, variant, repeat),
        **({"error": error} if error else {}),
        **(res.meta if res else {}),
    }


KEY_FIELDS = ("item_id", "variant", "repeat")


def _with_fields(row: dict, item: Item, row_fields) -> dict:
    """项目自定义字段并入结果行（如沿用原有的字段名），但不能改 key。"""
    if row_fields is None:
        return row
    return {**row, **row_fields(item, row), **{k: row[k] for k in KEY_FIELDS}}


def trace_record(
    item: Item, variant: str, repeat: int, res: TaskResult | None, row: dict, err: str | None
) -> dict:
    rec = {
        "trace_id": row["trace_id"],
        "item_id": item.id,
        "variant": variant,
        "repeat": repeat,
        "input": item.input,
        "expected": item.expected,
        "output": row["output"],
        "scores": row["scores"],
        "error": err,
    }
    trace = res.trace if res else None
    if isinstance(trace, dict):
        return {**rec, **trace}  # 应用的字段优先
    return {**rec, "steps": trace}


async def arun(
    dataset: Dataset,
    variants: Sequence[Variant],
    scorers: Sequence,
    *,
    runs_dir: str | Path = "runs",
    tag: str = "run",
    repeats: int = 1,
    concurrency: int = 8,
    max_cost: float | None = None,
    resume: str | None = None,
    extra_config: dict | None = None,
    progress: bool = True,
    row_fields: Callable[[Item, dict], dict] | None = None,
    save_traces: bool = True,
) -> RunResult:
    if not variants:
        raise ValueError("至少需要一个变体")
    if len({v.name for v in variants}) != len(variants):
        raise ValueError("变体名不能重复")
    run_id = resume or store.new_run_id(tag)
    run_dir = Path(runs_dir) / run_id
    prev_cfg = store.load_config(run_dir)
    hashes = dataset.item_hashes()
    if resume and prev_cfg:
        # 续跑可以加题，但已跑过的题内容不能变（标准答案改了要用 rescore 或新开运行）
        old = prev_cfg.get("item_hashes") or {}
        changed = [i for i, h in old.items() if i in hashes and hashes[i] != h]
        if changed:
            raise ValueError(f"续跑的评测集里有 {len(changed)} 道已跑过的题被改动：{changed[:5]}")
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = store.load_rows(run_dir)
    done = {store.row_key(r) for r in existing if r["status"] != "error"}
    spent = sum(r.get("cost_usd", 0.0) for r in existing)
    store.write_json(
        run_dir / store.CONFIG,
        {
            "run_id": run_id,
            "tag": tag,
            "created_at": prev_cfg.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": dataset.fingerprint,
            "variants": {v.name: v.params for v in variants},
            "scorers": [getattr(s, "name", type(s).__name__) for s in scorers],
            "primary_metric": getattr(scorers[0], "name", None) if scorers else None,
            "repeats": repeats,
            "concurrency": concurrency,
            "max_cost": max_cost,
            "item_ids": [it.id for it in dataset],
            "item_hashes": hashes,
            "code_version": store.git_commit(),
            "python": platform.python_version(),
            **(extra_config or {}),
        },
    )

    jobs = [
        (it, v, r)
        for v in variants
        for r in range(repeats)
        for it in dataset
        if (it.id, v.name, r) not in done
    ]
    if progress and done:
        print(f"续跑 {run_id}：已完成 {len(done)}，本次 {len(jobs)}")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    finished = 0
    new_rows: list[dict] = []

    async def one(item: Item, v: Variant, repeat: int):
        nonlocal finished, spent
        async with sem:
            if max_cost is not None and spent >= max_cost:
                return
            start = time.perf_counter()
            res, err = None, None
            try:
                res = await _call_task(v.task, item)
            except AbortRun:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - start
            scores = {} if err else await asyncio.to_thread(_score_all, scorers, item, res.output)
            row = _with_fields(
                build_row(item, v.name, repeat, res, scores, latency, err), item, row_fields
            )
        async with lock:
            spent += row["cost_usd"]
            store.append_row(run_dir, row)
            if save_traces:
                store.save_trace(
                    run_dir, row["trace_id"], trace_record(item, v.name, repeat, res, row, err)
                )
            new_rows.append(row)
            finished += 1
            if progress:
                mark = "✓" if row["passed"] else f"✗ {row['status']}"
                print(f"[{finished}/{len(jobs)}] {v.name} r{repeat} {item.id} {mark}", flush=True)

    try:
        await asyncio.gather(*(one(*j) for j in jobs))
    except AbortRun as e:
        raise AbortRun(f"{e}（已停止，处理后用 resume={run_id!r} 续跑）") from e

    rows = store.load_rows(run_dir)
    from evalkit.report import summarize

    store.write_json(run_dir / store.SUMMARY, summarize(rows))
    skipped = len(jobs) - len(new_rows)
    if progress:
        if skipped:
            print(f"达到花费上限 ${max_cost:.2f}，跳过 {skipped} 个任务，可用 resume 续跑")
        print(f"run_id={run_id}  累计花费 ${spent:.4f}")
    return RunResult(run_id, run_dir, rows, spent, skipped)


def run(dataset: Dataset, variants: Sequence[Variant], scorers: Sequence, **kw) -> RunResult:
    """arun 的同步版本。"""
    return asyncio.run(arun(dataset, variants, scorers, **kw))


def rescore(
    run_dir: str | Path,
    dataset: Dataset,
    scorers: Sequence,
    out_dir: str | Path | None = None,
    row_fields: Callable[[Item, dict], dict] | None = None,
) -> list[dict]:
    """用已保存的输出重新打分，不再调用模型。标准答案订正、打分规则修改后用。

    默认写到 <run_dir>-rescored，原始结果保留不动。花费 = 原花费 + 本次打分花费（LLM 裁判）。
    """
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir.with_name(run_dir.name + "-rescored")
    by_id = {it.id: it for it in dataset}
    rows = []
    for r in store.load_rows(run_dir):
        item = by_id.get(r["item_id"])
        if item is None or r["status"] == "error":
            rows.append(r)
            continue
        scores = _score_all(scorers, item, r["output"])
        res = TaskResult(r["output"], r.get("cost_usd", 0.0), r.get("usage"))
        new = build_row(
            item, r["variant"], r.get("repeat", 0), res, scores, r.get("latency_s", 0.0)
        )
        new["trace_id"] = r.get("trace_id", new["trace_id"])
        # 原行里平台不认识的字段（项目自定义字段）原样保留，再按 row_fields 重算
        new = {**{k: v for k, v in r.items() if k not in new}, **new}
        rows.append(_with_fields(new, item, row_fields))
    store.write_rows(out, rows)
    if (run_dir / store.TRACES).is_dir():
        shutil.copytree(run_dir / store.TRACES, out / store.TRACES, dirs_exist_ok=True)
    cfg = store.load_config(run_dir) | {
        "run_id": out.name,
        "rescored_from": run_dir.name,
        "dataset": dataset.fingerprint,
        "scorers": [getattr(s, "name", type(s).__name__) for s in scorers],
    }
    store.write_json(out / store.CONFIG, cfg)
    from evalkit.report import summarize

    store.write_json(out / store.SUMMARY, summarize(rows))
    return rows
