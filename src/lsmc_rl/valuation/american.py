"""Least-squares Monte-Carlo valuation for American options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from lsmc_rl.valuation.common import (
    PathMatrices,
    clean_variance_feature,
    confidence_interval_95,
    fit_ridge_regression,
    paths_frame_to_matrices,
    standard_error,
)

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class RegressionConfig:
    """Regression controls shared by path-based LSMC routines."""

    degree: int = 3
    ridge_alpha: float = 1e-6
    itm_only: bool = True
    min_regression_paths: int = 20
    include_log_moneyness: bool = True
    include_intrinsic: bool = True
    include_variance: bool = True
    clip_negative_continuation: bool = True


@dataclass(frozen=True)
class AmericanOptionContract:
    """American vanilla option specification on one simulated price series."""

    strike: float
    option_type: OptionType = "put"
    risk_free_rate: float = 0.0
    time_step_years: float = 1.0 / 252.0
    maturity_step: int | None = None
    exercise_start_step: int = 1
    exercise_step_interval: int = 1

    def validate(self, n_steps: int) -> int:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.time_step_years <= 0.0:
            raise ValueError("time_step_years must be positive")
        if self.exercise_start_step < 0:
            raise ValueError("exercise_start_step must be non-negative")
        if self.exercise_step_interval <= 0:
            raise ValueError("exercise_step_interval must be positive")
        maturity = n_steps - 1 if self.maturity_step is None else int(self.maturity_step)
        if maturity <= 0 or maturity >= n_steps:
            raise ValueError("maturity_step must be between 1 and the last path step")
        if self.exercise_start_step > maturity:
            raise ValueError("exercise_start_step cannot exceed maturity_step")
        return maturity

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        values = np.asarray(prices, dtype=float)
        if self.option_type == "call":
            return np.maximum(values - self.strike, 0.0)
        if self.option_type == "put":
            return np.maximum(self.strike - values, 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def is_exercise_step(self, step: int, maturity_step: int) -> bool:
        if step == maturity_step:
            return True
        if step < self.exercise_start_step:
            return False
        return (step - self.exercise_start_step) % self.exercise_step_interval == 0


@dataclass(frozen=True)
class AmericanLSMCResult:
    price: float
    stderr: float
    confidence_interval_95: tuple[float, float]
    path_values: np.ndarray
    exercise_steps: np.ndarray
    exercise_payoffs: np.ndarray
    exercise_profile: pd.DataFrame
    regression_diagnostics: pd.DataFrame
    european_value: float
    european_stderr: float
    contract: AmericanOptionContract


def value_european_option(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    variance_paths: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Value the matching European payoff on the same Monte-Carlo paths."""

    prices, _, _ = _coerce_paths(paths, variance_paths)
    maturity = contract.validate(prices.shape[1])
    discount = np.exp(-contract.risk_free_rate * contract.time_step_years * maturity)
    discounted_payoffs = discount * contract.payoff(prices[:, maturity])
    return float(np.mean(discounted_payoffs)), standard_error(discounted_payoffs), discounted_payoffs


def value_american_option_lsmc(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    regression: RegressionConfig | None = None,
    variance_paths: np.ndarray | None = None,
) -> AmericanLSMCResult:
    """Run Longstaff-Schwartz backward induction on simulated paths.

    The result is an in-sample LSMC policy estimate on the supplied paths. For
    production comparisons, the fitted exercise policy should later be evaluated
    on independent out-of-sample paths.
    """

    regression = regression or RegressionConfig()
    prices, variances, _ = _coerce_paths(paths, variance_paths)
    n_paths, n_steps = prices.shape
    maturity = contract.validate(n_steps)
    prices = prices[:, : maturity + 1]
    variances = None if variances is None else variances[:, : maturity + 1]

    payoff = contract.payoff(prices)
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    exercise_steps = np.full(n_paths, maturity, dtype=int)
    exercise_payoffs = payoff[:, maturity].copy()
    diagnostics: list[dict[str, float | int | str]] = []

    for step in range(maturity - 1, contract.exercise_start_step - 1, -1):
        if not contract.is_exercise_step(step, maturity):
            diagnostics.append(_skip_diagnostic(step, "not_exercise_date", n_paths))
            continue

        immediate = payoff[:, step]
        candidates = immediate > 0.0 if regression.itm_only else np.ones(n_paths, dtype=bool)
        discounted_future = exercise_payoffs * np.power(step_discount, exercise_steps - step)
        continuation = np.full(n_paths, float(np.mean(discounted_future[candidates])) if candidates.any() else 0.0)
        model_status = "constant_fallback"
        r2 = float("nan")
        rows = int(candidates.sum())

        if rows >= regression.min_regression_paths and np.std(discounted_future[candidates]) > 1e-12:
            features, feature_names = american_regression_features(
                prices=prices[candidates, step],
                variances=None if variances is None else variances[candidates, step],
                contract=contract,
                regression=regression,
            )
            model = fit_ridge_regression(
                features=features,
                target=discounted_future[candidates],
                feature_names=feature_names,
                ridge_alpha=regression.ridge_alpha,
            )
            all_features, _ = american_regression_features(
                prices=prices[:, step],
                variances=clean_variance_feature(variances, step, n_paths),
                contract=contract,
                regression=regression,
            )
            continuation = model.predict(all_features)
            r2 = model.r2
            model_status = "ridge"

        if regression.clip_negative_continuation:
            continuation = np.maximum(continuation, 0.0)
        exercise_now = candidates & (immediate > continuation)
        exercise_steps[exercise_now] = step
        exercise_payoffs[exercise_now] = immediate[exercise_now]
        diagnostics.append(
            {
                "step": int(step),
                "status": model_status,
                "candidate_paths": rows,
                "exercised_paths": int(exercise_now.sum()),
                "mean_immediate": float(np.mean(immediate[candidates])) if candidates.any() else 0.0,
                "mean_continuation": float(np.mean(continuation[candidates])) if candidates.any() else 0.0,
                "regression_r2": r2,
            }
        )

    path_values = exercise_payoffs * np.power(step_discount, exercise_steps)
    if contract.exercise_start_step == 0 and contract.is_exercise_step(0, maturity):
        initial_payoff = float(payoff[0, 0])
        continuation_value = float(np.mean(path_values))
        if initial_payoff > continuation_value:
            path_values = np.full(n_paths, initial_payoff, dtype=float)
            exercise_steps = np.zeros(n_paths, dtype=int)
            exercise_payoffs = np.full(n_paths, initial_payoff, dtype=float)

    european_value, european_stderr, _ = value_european_option(prices, contract)
    price = float(np.mean(path_values))
    return AmericanLSMCResult(
        price=price,
        stderr=standard_error(path_values),
        confidence_interval_95=confidence_interval_95(path_values),
        path_values=path_values,
        exercise_steps=exercise_steps,
        exercise_payoffs=exercise_payoffs,
        exercise_profile=_american_exercise_profile(exercise_steps, exercise_payoffs, maturity),
        regression_diagnostics=pd.DataFrame(diagnostics).sort_values("step").reset_index(drop=True),
        european_value=european_value,
        european_stderr=european_stderr,
        contract=contract,
    )


def american_regression_features(
    prices: np.ndarray,
    variances: np.ndarray | None,
    contract: AmericanOptionContract,
    regression: RegressionConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build continuation-regression features from current path state."""

    current_prices = np.asarray(prices, dtype=float)
    if (current_prices <= 0.0).any():
        raise ValueError("prices must be positive")
    moneyness = current_prices / contract.strike - 1.0
    columns = [np.ones_like(moneyness)]
    names = ["constant"]
    degree = max(1, int(regression.degree))
    for power in range(1, degree + 1):
        columns.append(moneyness**power)
        names.append(f"moneyness_power_{power}")
    if regression.include_log_moneyness:
        columns.append(np.log(current_prices / contract.strike))
        names.append("log_moneyness")
    if regression.include_intrinsic:
        columns.append(contract.payoff(current_prices) / contract.strike)
        names.append("scaled_intrinsic")
    if regression.include_variance:
        if variances is None:
            variance = np.zeros_like(current_prices)
        else:
            variance = np.asarray(variances, dtype=float)
            finite = variance[np.isfinite(variance) & (variance >= 0.0)]
            fill = float(np.median(finite)) if finite.size else 0.0
            variance = np.where(np.isfinite(variance) & (variance >= 0.0), variance, fill)
        columns.append(np.sqrt(np.maximum(variance, 0.0)))
        names.append("step_volatility")
    return np.column_stack(columns), tuple(names)


def _coerce_paths(
    paths: pd.DataFrame | np.ndarray,
    variance_paths: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, PathMatrices | None]:
    if isinstance(paths, pd.DataFrame):
        matrices = paths_frame_to_matrices(paths)
        return matrices.prices, matrices.variances, matrices
    prices = np.asarray(paths, dtype=float)
    if prices.ndim != 2:
        raise ValueError("paths must be a 2D price array or a long-form DataFrame")
    variances = None if variance_paths is None else np.asarray(variance_paths, dtype=float)
    if variances is not None and variances.shape != prices.shape:
        raise ValueError("variance_paths must have the same shape as prices")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("all path prices must be finite and positive")
    return prices, variances, None


def _american_exercise_profile(
    exercise_steps: np.ndarray,
    exercise_payoffs: np.ndarray,
    maturity: int,
) -> pd.DataFrame:
    frame = pd.DataFrame({"step": exercise_steps, "payoff": exercise_payoffs})
    profile = (
        frame.groupby("step", sort=True)
        .agg(exercise_count=("step", "size"), mean_payoff=("payoff", "mean"))
        .reset_index()
    )
    profile["exercise_probability"] = profile["exercise_count"] / len(frame)
    profile["is_maturity"] = profile["step"] == maturity
    return profile


def _skip_diagnostic(step: int, status: str, n_paths: int) -> dict[str, float | int | str]:
    return {
        "step": int(step),
        "status": status,
        "candidate_paths": 0,
        "exercised_paths": 0,
        "mean_immediate": 0.0,
        "mean_continuation": 0.0,
        "regression_r2": float("nan"),
    }
