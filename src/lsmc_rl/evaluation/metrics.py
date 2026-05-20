"""Pathwise and paired metrics for frozen-policy comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_policy_metrics(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    name_a: str = "policy_a",
    name_b: str = "policy_b",
    bootstrap_seed: int = 12345,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Compute paired metrics for ``delta_i = value_A_i - value_B_i``."""

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must have the same shape")
    if a.ndim != 1:
        raise ValueError("values_a and values_b must be one-dimensional")
    if a.size == 0:
        raise ValueError("paired metrics require at least one path")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired values must be finite")

    delta = a - b
    q05, q50, q95 = np.quantile(delta, [0.05, 0.50, 0.95])
    tail = delta[delta <= q05]
    ci_low, ci_high = _bootstrap_mean_ci(delta, bootstrap_seed, n_bootstrap)
    return {
        "policy_a": name_a,
        "policy_b": name_b,
        "n_paths": int(delta.size),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "share_delta_positive": float(np.mean(delta > 0.0)),
        "q05_delta": float(q05),
        "q50_delta": float(q50),
        "q95_delta": float(q95),
        "bootstrap_mean_delta_ci95": [float(ci_low), float(ci_high)],
        "cvar_5_delta": float(np.mean(tail)) if tail.size else float(q05),
        "standard_error": _standard_error(delta),
    }


def path_value_summary(values: np.ndarray) -> dict[str, float]:
    """Compact distribution summary for one policy's path values."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("path values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("path values must be finite")
    q05, q50, q95 = np.quantile(array, [0.05, 0.50, 0.95])
    tail = array[array <= q05]
    losses = array[array < 0.0]
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    downside = array[array < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": std,
        "standard_error": _standard_error(array),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "cvar_5": float(np.mean(tail)) if tail.size else float(q05),
        "probability_positive": float(np.mean(array > 0.0)),
        "probability_loss": float(np.mean(array < 0.0)),
        "mean_loss": float(np.mean(losses)) if losses.size else 0.0,
        "sharpe_like": float(np.mean(array) / std) if std > 0.0 else float("nan"),
        "sortino_like": float(np.mean(array) / downside_std) if downside_std > 0.0 else float("nan"),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    if values.size == 1:
        mean = float(values[0])
        return mean, mean
    if n_bootstrap <= 0:
        mean = float(np.mean(values))
        return mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sample = values[rng.integers(0, values.size, size=values.size)]
        means[index] = np.mean(sample)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _standard_error(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))
