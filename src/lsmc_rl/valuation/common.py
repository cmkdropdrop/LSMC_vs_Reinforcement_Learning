"""Shared helpers for path-based valuation algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PathMatrices:
    """Dense path matrices derived from the repository long-form path format."""

    prices: np.ndarray
    variances: np.ndarray | None
    steps: np.ndarray
    path_ids: np.ndarray
    times: pd.Series | None = None


@dataclass(frozen=True)
class ContinuationModel:
    """Small ridge-regression model with stored feature scaling."""

    coefficients: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    feature_names: tuple[str, ...]
    ridge_alpha: float
    row_count: int
    r2: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.coefficients):
            raise ValueError("features must be a 2D array with the fitted column count")
        x_scaled = (x - self.center) / self.scale
        return x_scaled @ self.coefficients


def paths_frame_to_matrices(
    frame: pd.DataFrame,
    price_col: str = "price",
    variance_col: str = "variance",
) -> PathMatrices:
    """Convert the common long-form path DataFrame to dense matrices."""

    required = {"path", "step", price_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"path frame is missing required columns: {missing}")
    if frame.duplicated(subset=["path", "step"]).any():
        raise ValueError("path frame contains duplicate path/step rows")

    ordered = frame.sort_values(["path", "step"], kind="mergesort").copy()
    price_pivot = ordered.pivot(index="path", columns="step", values=price_col).sort_index(axis=0).sort_index(axis=1)
    if price_pivot.isna().any().any():
        raise ValueError("path frame must contain a complete price grid")
    prices = price_pivot.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("all path prices must be finite and positive")

    variances: np.ndarray | None = None
    if variance_col in ordered.columns:
        var_pivot = (
            ordered.pivot(index="path", columns="step", values=variance_col)
            .reindex(index=price_pivot.index, columns=price_pivot.columns)
        )
        variances = var_pivot.to_numpy(dtype=float)

    times: pd.Series | None = None
    if "time" in ordered.columns:
        step_times = ordered.drop_duplicates("step").sort_values("step")[["step", "time"]]
        times = step_times.set_index("step").reindex(price_pivot.columns)["time"]

    return PathMatrices(
        prices=prices,
        variances=variances,
        steps=price_pivot.columns.to_numpy(dtype=int),
        path_ids=price_pivot.index.to_numpy(),
        times=times,
    )


def fit_ridge_regression(
    features: np.ndarray,
    target: np.ndarray,
    feature_names: tuple[str, ...],
    ridge_alpha: float = 1e-8,
) -> ContinuationModel:
    """Fit a numerically stable ridge regression for continuation values."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and target must have the same row count")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match features")

    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[mask]
    y = y[mask]
    if len(y) == 0:
        raise ValueError("no finite regression rows available")

    center = np.zeros(x.shape[1], dtype=float)
    scale = np.ones(x.shape[1], dtype=float)
    for column in range(x.shape[1]):
        values = x[:, column]
        std = float(np.std(values, ddof=0))
        if std > 1e-12 and not np.allclose(values, values[0]):
            center[column] = float(np.mean(values))
            scale[column] = std

    x_scaled = (x - center) / scale
    penalty = float(ridge_alpha) * np.eye(x_scaled.shape[1])
    if feature_names and feature_names[0] == "constant":
        penalty[0, 0] = 0.0
    lhs = x_scaled.T @ x_scaled + penalty
    rhs = x_scaled.T @ y
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs

    fitted = x_scaled @ coefficients
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    residual_ss = float(np.sum((y - fitted) ** 2))
    r2 = float(1.0 - residual_ss / total_ss) if total_ss > 0.0 else float("nan")
    return ContinuationModel(
        coefficients=coefficients,
        center=center,
        scale=scale,
        feature_names=feature_names,
        ridge_alpha=float(ridge_alpha),
        row_count=int(len(y)),
        r2=r2,
    )


def clean_variance_feature(variances: np.ndarray | None, step: int, n_paths: int) -> np.ndarray:
    """Return a finite variance vector for regression features."""

    if variances is None:
        return np.zeros(n_paths, dtype=float)
    values = np.asarray(variances[:, step], dtype=float)
    finite = values[np.isfinite(values) & (values >= 0.0)]
    fill = float(np.median(finite)) if finite.size else 0.0
    return np.where(np.isfinite(values) & (values >= 0.0), values, fill)


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        mean = float(np.mean(array)) if array.size else float("nan")
        return mean, mean
    mean = float(np.mean(array))
    half_width = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(array.size)
    return mean - half_width, mean + half_width


def standard_error(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size <= 1:
        return 0.0
    return float(np.std(array, ddof=1) / np.sqrt(array.size))


def json_ready(value: Any) -> Any:
    """Convert common numpy/pandas values to JSON-serializable objects."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
