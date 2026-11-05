import math

import pytest

from evalkit.stats import (
    bootstrap_ci,
    cohen_kappa,
    mcnemar_exact,
    mean_std,
    paired_bootstrap_diff,
    percentile,
    stability,
    wilson,
)


def test_wilson_known_values():
    lo, hi = wilson(17, 20)
    assert round(lo, 3) == 0.640 and round(hi, 3) == 0.948
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.35


def test_percentile_is_lower_nearest_rank():
    xs = [5, 1, 4, 2, 3]
    assert percentile(xs, 0.5) == 3
    assert percentile(xs, 0.95) == 5
    assert percentile([], 0.5) == 0.0


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert round(mcnemar_exact(21, 43), 3) == 0.008
    assert mcnemar_exact(15, 15) == 1.0
    # 对称
    assert mcnemar_exact(3, 14) == mcnemar_exact(14, 3)


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lo, hi = bootstrap_ci(xs, seed=1)
    assert lo < 5.5 < hi
    assert bootstrap_ci(xs, seed=1) == (lo, hi)


def test_paired_bootstrap_diff():
    a = [0, 0, 1, 1, 0, 1, 0, 0]
    b = [1, 1, 1, 1, 0, 1, 1, 0]
    out = paired_bootstrap_diff(a, b)
    assert out["diff"] == pytest.approx(3 / 8)
    assert out["ci"][0] <= out["diff"] <= out["ci"][1]
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 0])


def test_mean_std():
    m, s = mean_std([0.72, 0.726, 0.726])
    assert m == pytest.approx(0.724)
    assert s == pytest.approx(0.003464, abs=1e-5)
    assert mean_std([0.5]) == (0.5, 0.0)


def test_stability():
    out = stability({1: [True, True], 2: [False, False], 3: [True, False], 4: [True, True]})
    assert out == {"n": 4, "always": 2, "never": 1, "flaky": 1, "flaky_rate": 0.25}


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    k = cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"])
    assert math.isclose(k, 0.0, abs_tol=1e-12)
