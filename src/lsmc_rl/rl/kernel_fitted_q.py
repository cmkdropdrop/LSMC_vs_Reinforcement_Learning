"""Kernelized fitted Q iteration for American option exercise."""

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
class KernelFittedQConfig:
    """Controls for random-Fourier-feature fitted Q iteration."""

    n_rff_features: int = 96
    length_scale: float = 1.0
    ridge_alpha: float = 1e-5
    min_regression_paths: int = 30
    fit_itm_only: bool = False
    include_linear_features: bool = True
    include_variance: bool = True
    clip_negative_continuation: bool = True
    exercise_only_itm: bool = True
    exercise_tolerance: float = 1e-12
    feature_seed: int = 20260522


@dataclass(frozen=True)
class RandomFourierFeatureMap:
    """Frozen RBF-kernel approximation used by fitted-Q policies."""

    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    phase: np.ndarray
    include_linear_features: bool
    feature_names: tuple[str, ...]

    def transform(self, base_features: np.ndarray) -> np.ndarray:
        values = np.asarray(base_features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.center.size:
            raise ValueError("base_features must match the fitted feature-map width")
        standardized = (values - self.center) / self.scale
        random_projection = standardized @ self.weights + self.phase
        rff = np.sqrt(2.0 / self.weights.shape[1]) * np.cos(random_projection)
        columns = [np.ones(values.shape[0], dtype=float)]
        if self.include_linear_features:
            columns.extend(standardized[:, index] for index in range(standardized.shape[1]))
        columns.extend(rff[:, index] for index in range(rff.shape[1]))
        return np.column_stack(columns)


@dataclass(frozen=True)
class AmericanKernelFittedQPolicy:
    """Frozen American-option policy learned by kernelized fitted Q."""

    contract: AmericanOptionContract
    config: KernelFittedQConfig
    maturity_step: int
    feature_map: RandomFourierFeatureMap
    continuation_models: Mapping[int, ContinuationModel]
    fallback_continuation: Mapping[int, float]
    name: str = "american_kernel_fitted_q"

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
            base = american_kernel_base_features(
                prices=np.asarray([state.current_price], dtype=float),
                variances=variance,
                steps=np.asarray([state.step], dtype=int),
                maturity_step=self.maturity_step,
                contract=self.contract,
                config=self.config,
            )
            value = float(model.predict(self.feature_map.transform(base))[0])
        if self.config.clip_negative_continuation:
            value = max(value, 0.0)
        return value


@dataclass(frozen=True)
class AmericanKernelFittedQResult:
    """Training output for a kernelized American fitted-Q policy."""

    price: float
    stderr: float
    confidence_interval_95: tuple[float, float]
    path_values: np.ndarray
    exercise_steps: np.ndarray
    exercise_payoffs: np.ndarray
    policy: AmericanKernelFittedQPolicy
    in_sample_evaluation: AmericanPolicyEvaluation
    training_diagnostics: pd.DataFrame
    feature_map: RandomFourierFeatureMap
    contract: AmericanOptionContract
    config: KernelFittedQConfig


def train_american_kernel_fitted_q(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    config: KernelFittedQConfig | None = None,
    variance_paths: np.ndarray | None = None,
) -> AmericanKernelFittedQResult:
    """Train a nonlinear finite-horizon fitted-Q exercise policy.

    The continuation value is approximated by ridge regression on a frozen
    random-Fourier-feature map, which gives a lightweight RBF-kernel fitted-Q
    baseline without adding a deep-learning dependency.
    """

    config = config or KernelFittedQConfig()
    _validate_config(config)
    matrices = _coerce_paths(paths, variance_paths)
    prices = matrices.prices
    n_paths, n_steps = prices.shape
    maturity = contract.validate(n_steps)
    prices = prices[:, : maturity + 1]
    variances = None if matrices.variances is None else matrices.variances[:, : maturity + 1]
    payoff = contract.payoff(prices)
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    feature_map = _fit_feature_map(prices, variances, maturity, contract, config)

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
            base = american_kernel_base_features(
                prices=prices[train_mask, step],
                variances=step_variance[train_mask],
                steps=np.full(rows, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=config,
            )
            features = feature_map.transform(base)
            model = fit_ridge_regression(
                features=features,
                target=target_continue[train_mask],
                feature_names=feature_map.feature_names,
                ridge_alpha=config.ridge_alpha,
            )
            all_base = american_kernel_base_features(
                prices=prices[:, step],
                variances=step_variance,
                steps=np.full(n_paths, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=config,
            )
            continuation = model.predict(feature_map.transform(all_base))
            models[step] = model
            status = "kernel_ridge"
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

    policy = AmericanKernelFittedQPolicy(
        contract=contract,
        config=config,
        maturity_step=maturity,
        feature_map=feature_map,
        continuation_models=models,
        fallback_continuation=fallback_continuation,
    )
    in_sample = evaluate_american_policy(paths, contract, policy, variance_paths=variance_paths)
    return AmericanKernelFittedQResult(
        price=float(np.mean(in_sample.path_values)),
        stderr=standard_error(in_sample.path_values),
        confidence_interval_95=confidence_interval_95(in_sample.path_values),
        path_values=in_sample.path_values,
        exercise_steps=in_sample.exercise_steps,
        exercise_payoffs=in_sample.exercise_payoffs,
        policy=policy,
        in_sample_evaluation=in_sample,
        training_diagnostics=pd.DataFrame(diagnostics).sort_values("step").reset_index(drop=True),
        feature_map=feature_map,
        contract=contract,
        config=config,
    )


def value_american_option_kernel_fitted_q(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    config: KernelFittedQConfig | None = None,
    variance_paths: np.ndarray | None = None,
) -> AmericanKernelFittedQResult:
    """Alias for ``train_american_kernel_fitted_q`` for valuation call sites."""

    return train_american_kernel_fitted_q(paths, contract, config=config, variance_paths=variance_paths)


def american_kernel_base_features(
    prices: np.ndarray,
    variances: np.ndarray | None,
    steps: np.ndarray,
    maturity_step: int,
    contract: AmericanOptionContract,
    config: KernelFittedQConfig,
) -> np.ndarray:
    """Build current-state features before random Fourier expansion."""

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
    remaining_fraction = np.maximum(maturity_step - step_array, 0.0) / maturity_step
    columns = [
        moneyness,
        np.log(current_prices / contract.strike),
        contract.payoff(current_prices) / contract.strike,
        remaining_fraction,
        step_array / maturity_step,
        moneyness * remaining_fraction,
    ]
    if config.include_variance:
        variance = _feature_variance(variances, current_prices.size)
        columns.append(np.sqrt(np.maximum(variance, 0.0)))
    return np.column_stack(columns)


def _fit_feature_map(
    prices: np.ndarray,
    variances: np.ndarray | None,
    maturity: int,
    contract: AmericanOptionContract,
    config: KernelFittedQConfig,
) -> RandomFourierFeatureMap:
    rows = []
    n_paths = prices.shape[0]
    for step in range(contract.exercise_start_step, maturity):
        if not contract.is_exercise_step(step, maturity):
            continue
        rows.append(
            american_kernel_base_features(
                prices=prices[:, step],
                variances=clean_variance_feature(variances, step, n_paths),
                steps=np.full(n_paths, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=config,
            )
        )
    if not rows:
        rows.append(
            american_kernel_base_features(
                prices=prices[:, maturity],
                variances=clean_variance_feature(variances, maturity, n_paths),
                steps=np.full(n_paths, maturity, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=config,
            )
        )
    base = np.vstack(rows)
    center = np.nanmean(base, axis=0)
    scale = np.nanstd(base, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)

    rng = np.random.default_rng(config.feature_seed)
    weights = rng.normal(
        loc=0.0,
        scale=1.0 / config.length_scale,
        size=(base.shape[1], config.n_rff_features),
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=config.n_rff_features)
    names = ["constant"]
    if config.include_linear_features:
        names.extend(f"standardized_state_{index}" for index in range(base.shape[1]))
    names.extend(f"rff_{index}" for index in range(config.n_rff_features))
    return RandomFourierFeatureMap(
        center=center,
        scale=scale,
        weights=weights,
        phase=phase,
        include_linear_features=config.include_linear_features,
        feature_names=tuple(names),
    )


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


def _validate_config(config: KernelFittedQConfig) -> None:
    if config.n_rff_features < 1:
        raise ValueError("n_rff_features must be positive")
    if config.length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    if config.ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if config.min_regression_paths < 1:
        raise ValueError("min_regression_paths must be positive")
    if config.exercise_tolerance < 0.0:
        raise ValueError("exercise_tolerance must be non-negative")
