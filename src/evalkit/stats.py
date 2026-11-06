"""统计工具：置信区间、配对检验、多次运行的均值与方差、稳定性。

全部只用标准库。随机方法（bootstrap）一律接收 seed，结果可复现。
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from math import comb


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间；样本小、比例接近 0 或 1 时比正态近似可靠。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def percentile(values: Sequence[float], q: float) -> float:
    """取排序后第 int(q·n) 个值（不插值）。延迟、花费的 P50 / P95 都用这个口径。"""
    xs = sorted(values)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar 检验。b = 只有 A 对的题数，c = 只有 B 对的题数。

    同一批题上比较两个配置时，两边都对、都错的题不提供差异信息，只看不一致的题。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = statistics.fmean,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """非参数 bootstrap 百分位区间，适合均分、评分等非二值指标。"""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    xs = list(values)
    stats = sorted(stat(rng.choices(xs, k=len(xs))) for _ in range(n_resamples))
    lo = stats[int(alpha / 2 * n_resamples)]
    hi = stats[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """配对差值 mean(b - a) 的 bootstrap 区间；a、b 按同一题对齐。"""
    if len(a) != len(b):
        raise ValueError("配对比较要求两组长度相同（同一批题）")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)
    return {"diff": statistics.fmean(diffs) if diffs else 0.0, "ci": (lo, hi)}


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """多次运行的均值和样本标准差（n-1）；只有一次时标准差记 0。"""
    if not values:
        return (0.0, 0.0)
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return (m, s)


def stability(outcomes: dict[object, Sequence[bool]]) -> dict:
    """同一批题跑多次：每次都对 / 每次都错 / 时对时错 的题数。

    outcomes：题目 id → 各次运行是否答对。
    """
    always = sum(1 for xs in outcomes.values() if xs and all(xs))
    never = sum(1 for xs in outcomes.values() if xs and not any(xs))
    flaky = sum(1 for xs in outcomes.values() if xs and any(xs) and not all(xs))
    n = always + never + flaky
    return {
        "n": n,
        "always": always,
        "never": never,
        "flaky": flaky,
        "flaky_rate": flaky / n if n else 0.0,
    }


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """两位标注者（如 LLM 裁判与人工）在同一批样本上的 Cohen's κ。"""
    if len(a) != len(b):
        raise ValueError("两组标注长度不同")
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    pe = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)
