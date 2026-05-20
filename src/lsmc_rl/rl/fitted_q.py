"""Finite-horizon fitted Q iteration for American option exercise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from lsmc_rl.evaluation.evaluator import AmericanPolicyEvaluation, evaluate_american_policy
from lsmc_rl.evaluation.policies import AmericanPolicyState
from lsmc_rl.valuation.american import AmericanOptionContract
from lsmc_rl.valuation.common import (
    ContinuationModel,
    PathMatrices,
    clean_variance_feature,
    confidence_interval_95,
    fit_ridge_regression,
    paths_frame_to_matrices,
    standard_error,
)


@dataclass(frozen=True)
class FittedQConfig:
    """Controls for linear fitted Q iteration.

    The algorithm uses one ridge model per exercise step to approximate the
    continuation action value. The exercise action value is the immediate
    intrinsic payoff, so no model is fitted for that action.
    """

    degree: int = 3
    ridge_alpha: float = 1e-6
    min_regression_paths: int = 20
    fit_itm_only: bool = False
    include_log_moneyness: bool = True
    include_intrinsic: bool = True
    include_variance: bool = True
    include_time_features: bool = True
    clip_negative_continuation: bool = True
    exercise_only_itm: bool = True
    exercise_tolerance: float = 1e-12


@dataclass(frozen=True)
class AmericanFittedQPolicy:
    """Frozen American-option policy learned by fitted Q iteration."""

    contract: AmericanOptionContract
    config: FittedQConfig
    maturity_step: int
    continuation_models: Mapping[int, ContinuationModel]
    fallback_continuation: Mapping[int, float]
    name: str = "american_fitted_q"

    def decide_exercise(self, state: AmericanPolicyState) -> bool:
        if not self.contract.is_exercise_step(state.step, self.maturity_step):
            return False
        if state.step >= self.maturity_step:
            return True
        if self.config.exercise_only_itm and state.intrinsic_value <= self.config.exercise_tolerance:
            return False
        continuation = self.predict_continuation(state)
        return bool(state.intrinsic_value > continuation + self.config.exercise_tolerance)

    def predict_continuation(self, state: AmericanPolicyState) -> float:
        model = self.continuation_models.get(state.step)
        value = float(self.fallback_continuation.get(state.step, 0.0))
        if model is not None:
            variance = None if state.variance is None else np.asarray([state.variance], dtype=float)
            features, _ = american_fitted_q_features(
                prices=np.asarray([state.current_price], dtype=float),
                variances=variance,
                steps=np.asarray([state.step], dtype=int),
                maturity_step=self.maturity_step,
                contract=self.contract,
                config=self.config,
            )
            value = float(model.predict(features)[0])
        if self.config.clip_negative_continuation:
            value = max(value, 0.0)
        return value


@dataclass(frozen=True)
class AmericanFittedQResult:
    """Training output for an American fitted-Q policy."""

    price: float
    stderr: float
    confidence_interval_95: tuple[float, float]
    path_values: np.ndarray
    exercise_steps: np.ndarray
    exercise_payoffs: np.ndarray
    policy: AmericanFittedQPolicy
    in_sample_evaluation: AmericanPolicyEvaluation
    training_diagnostics: pd.DataFrame
    contract: AmericanOptionContract
    config: FittedQConfig


def train_american_fitted_q(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    config: FittedQConfig | None = None,
    variance_paths: np.ndarray | None = None,
) -> AmericanFittedQResult:
    """Train a finite-horizon fitted-Q exercise policy.

    This is an offline RL baseline for the American optimal-stopping problem.
    Training uses only current-step state features and one-step Bellman targets
    from the supplied paths. The returned policy is frozen and can be evaluated
    out of sample with ``evaluate_american_policy``.
    """

    config = config or FittedQConfig()
    _validate_config(config)
    matrices = _coerce_paths(paths, variance_paths)
    prices = matrices.prices
    n_paths, n_steps = prices.shape
    maturity = contract.validate(n_steps)
    prices = prices[:, : maturity + 1]
    variances = None if matrices.variances is None else matrices.variances[:, : maturity + 1]
    payoff = contract.payoff(prices)
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))

    next_values = payoff[:, maturity].copy()
    models: dict[int, ContinuationModel] = {}
    fallback_continuation: dict[int, float] = {}
    diagnostics: list[dict[str, float | int | str | bool]] = []

    for step in range(maturity - 1, contract.exercise_start_step - 1, -1):
        target_continue = step_discount * next_values
        if not contract.is_exercise_step(step, maturity):
            next_values = target_continue
            diagnostics.append(
                {
                    "step": int(step),
                    "status": "not_exercise_date",
                    "regression_rows": 0,
                    "exercised_paths": 0,
                    "mean_immediate": 0.0,
                    "mean_continuation": float(np.mean(target_continue)),
                    "regression_r2": float("nan"),
                    "exercise_date": False,
                }
            )
            continue

        immediate = payoff[:, step]
        train_mask = np.isfinite(target_continue)
        if config.fit_itm_only:
            train_mask &= immediate > config.exercise_tolerance
        rows = int(train_mask.sum())
        fallback = float(np.mean(target_continue[train_mask])) if rows else float(np.mean(target_continue))
        fallback_continuation[step] = fallback
        continuation = np.full(n_paths, fallback, dtype=float)
        status = "constant_fallback"
        r2 = float("nan")

        if rows >= config.min_regression_paths and np.std(target_continue[train_mask]) > 1e-12:
            step_variance = clean_variance_feature(variances, step, n_paths)
            features, names = american_fitted_q_features(
                prices=prices[train_mask, step],
                variances=step_variance[train_mask],
                steps=np.full(rows, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=config,
            )
            model = fit_ridge_regression(
                features=features,
                target=target_continue[train_mask],
                feature_names=names,
                ridge_alpha=config.ridge_alpha,
            )
            all_features, _ = american_fitted_q_features(
                prices=prices[:, step],
                variances=step_variance,
                steps=np.full(n_paths, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=config,
            )
            continuation = model.predict(all_features)
            models[step] = model
            status = "ridge"
            r2 = model.r2

        raw_continuation = np.asarray(continuation, dtype=float)
        negative_continuation_share = float(np.mean(raw_continuation < 0.0))
        if config.clip_negative_continuation:
            continuation = np.maximum(raw_continuation, 0.0)
        exercise_candidates = np.ones(n_paths, dtype=bool)
        if config.exercise_only_itm:
            exercise_candidates &= immediate > config.exercise_tolerance
        exercise_now = exercise_candidates & (immediate > continuation + config.exercise_tolerance)
        next_values = np.where(exercise_now, immediate, target_continue)
        residual = target_continue - continuation
        diagnostics.append(
            {
                "step": int(step),
                "status": status,
                "regression_rows": rows,
                "exercised_paths": int(exercise_now.sum()),
                "mean_immediate": float(np.mean(immediate[exercise_candidates])) if exercise_candidates.any() else 0.0,
                "mean_continuation": float(np.mean(continuation[exercise_candidates])) if exercise_candidates.any() else 0.0,
                "std_continuation": float(np.std(continuation[exercise_candidates])) if exercise_candidates.any() else 0.0,
                "mean_target_continue": float(np.mean(target_continue)),
                "std_target_continue": float(np.std(target_continue)),
                "mean_bellman_residual": float(np.mean(residual)),
                "mean_abs_bellman_residual": float(np.mean(np.abs(residual))),
                "negative_continuation_share_before_clip": negative_continuation_share,
                "regression_r2": r2,
                "exercise_date": True,
            }
        )

    policy = AmericanFittedQPolicy(
        contract=contract,
        config=config,
        maturity_step=maturity,
        continuation_models=models,
        fallback_continuation=fallback_continuation,
    )
    in_sample = evaluate_american_policy(paths, contract, policy, variance_paths=variance_paths)
    return AmericanFittedQResult(
        price=float(np.mean(in_sample.path_values)),
        stderr=standard_error(in_sample.path_values),
        confidence_interval_95=confidence_interval_95(in_sample.path_values),
        path_values=in_sample.path_values,
        exercise_steps=in_sample.exercise_steps,
        exercise_payoffs=in_sample.exercise_payoffs,
        policy=policy,
        in_sample_evaluation=in_sample,
        training_diagnostics=pd.DataFrame(diagnostics).sort_values("step").reset_index(drop=True),
        contract=contract,
        config=config,
    )


def value_american_option_fitted_q(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    config: FittedQConfig | None = None,
    variance_paths: np.ndarray | None = None,
) -> AmericanFittedQResult:
    """Alias for ``train_american_fitted_q`` for valuation-style call sites."""

    return train_american_fitted_q(paths, contract, config=config, variance_paths=variance_paths)


def american_fitted_q_features(
    prices: np.ndarray,
    variances: np.ndarray | None,
    steps: np.ndarray,
    maturity_step: int,
    contract: AmericanOptionContract,
    config: FittedQConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build Markov state features for American fitted Q iteration."""

    current_prices = np.asarray(prices, dtype=float).reshape(-1)
    if current_prices.size == 0:
        raise ValueError("prices must contain at least one row")
    if not np.isfinite(current_prices).all() or (current_prices <= 0.0).any():
        raise ValueError("prices must be finite and positive")
    step_array = np.asarray(steps, dtype=float).reshape(-1)
    if step_array.size == 1 and current_prices.size != 1:
        step_array = np.full(current_prices.size, float(step_array[0]), dtype=float)
    if step_array.shape != current_prices.shape:
        raise ValueError("steps must be scalar-like or have the same length as prices")
    if maturity_step <= 0:
        raise ValueError("maturity_step must be positive")

    moneyness = current_prices / contract.strike - 1.0
    columns = [np.ones_like(moneyness)]
    names = ["constant"]
    for power in range(1, max(1, int(config.degree)) + 1):
        columns.append(moneyness**power)
        names.append(f"moneyness_power_{power}")
    if config.include_log_moneyness:
        columns.append(np.log(current_prices / contract.strike))
        names.append("log_moneyness")
    if config.include_intrinsic:
        columns.append(contract.payoff(current_prices) / contract.strike)
        names.append("scaled_intrinsic")
    if config.include_time_features:
        remaining_fraction = np.maximum(maturity_step - step_array, 0.0) / maturity_step
        step_fraction = step_array / maturity_step
        columns.extend([remaining_fraction, step_fraction])
        names.extend(["remaining_time_fraction", "step_fraction"])
    if config.include_variance:
        variance = _feature_variance(variances, current_prices.size)
        columns.append(np.sqrt(np.maximum(variance, 0.0)))
        names.append("step_volatility")
    return np.column_stack(columns), tuple(names)


def _feature_variance(variances: np.ndarray | None, row_count: int) -> np.ndarray:
    if variances is None:
        return np.zeros(row_count, dtype=float)
    values = np.asarray(variances, dtype=float).reshape(-1)
    if values.size == 1 and row_count != 1:
        values = np.full(row_count, float(values[0]), dtype=float)
    if values.shape != (row_count,):
        raise ValueError("variances must be scalar-like or have the same length as prices")
    finite = values[np.isfinite(values) & (values >= 0.0)]
    fill = float(np.median(finite)) if finite.size else 0.0
    return np.where(np.isfinite(values) & (values >= 0.0), values, fill)


def _coerce_paths(paths: pd.DataFrame | np.ndarray, variance_paths: np.ndarray | None) -> PathMatrices:
    if isinstance(paths, pd.DataFrame):
        return paths_frame_to_matrices(paths)
    prices = np.asarray(paths, dtype=float)
    if prices.ndim != 2:
        raise ValueError("paths must be a 2D price array or a long-form DataFrame")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("all path prices must be finite and positive")
    variances = None if variance_paths is None else np.asarray(variance_paths, dtype=float)
    if variances is not None and variances.shape != prices.shape:
        raise ValueError("variance_paths must have the same shape as prices")
    return PathMatrices(
        prices=prices,
        variances=variances,
        steps=np.arange(prices.shape[1], dtype=int),
        path_ids=np.arange(prices.shape[0], dtype=int),
        times=None,
    )


def _validate_config(config: FittedQConfig) -> None:
    if config.degree < 1:
        raise ValueError("degree must be at least 1")
    if config.ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if config.min_regression_paths < 1:
        raise ValueError("min_regression_paths must be positive")
    if config.exercise_tolerance < 0.0:
        raise ValueError("exercise_tolerance must be non-negative")
