"""State-aware LSMC valuation for gas-style swing options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from lsmc_rl.valuation.american import RegressionConfig
from lsmc_rl.valuation.common import (
    ContinuationModel,
    PathMatrices,
    clean_variance_feature,
    confidence_interval_95,
    fit_ridge_regression,
    paths_frame_to_matrices,
    standard_error,
)

SwingPayoffType = Literal["call", "put"]


@dataclass(frozen=True)
class SwingOptionContract:
    """Discrete-volume swing contract on one simulated price series."""

    strike: float
    payoff_type: SwingPayoffType = "call"
    risk_free_rate: float = 0.0
    time_step_years: float = 1.0 / 252.0
    maturity_step: int | None = None
    exercise_start_step: int = 1
    exercise_step_interval: int = 1
    min_exercise_volume: float = 0.0
    max_exercise_volume: float = 1.0
    min_total_volume: float = 0.0
    max_total_volume: float = 5.0
    volume_step: float = 1.0
    variable_cost_per_unit: float = 0.0
    shortfall_penalty_per_unit: float = 0.0
    enforce_min_total_volume: bool = True

    def validate(self, n_steps: int) -> "SwingContractGrid":
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.payoff_type not in {"call", "put"}:
            raise ValueError("payoff_type must be 'call' or 'put'")
        if self.time_step_years <= 0.0:
            raise ValueError("time_step_years must be positive")
        if self.exercise_start_step < 0:
            raise ValueError("exercise_start_step must be non-negative")
        if self.exercise_step_interval <= 0:
            raise ValueError("exercise_step_interval must be positive")
        if self.volume_step <= 0.0:
            raise ValueError("volume_step must be positive")
        if self.max_exercise_volume <= 0.0 or self.max_total_volume <= 0.0:
            raise ValueError("max exercise and total volumes must be positive")
        if self.min_exercise_volume < 0.0 or self.min_total_volume < 0.0:
            raise ValueError("minimum volumes must be non-negative")
        if self.min_exercise_volume > self.max_exercise_volume:
            raise ValueError("min_exercise_volume cannot exceed max_exercise_volume")
        if self.min_total_volume > self.max_total_volume:
            raise ValueError("min_total_volume cannot exceed max_total_volume")
        maturity = n_steps - 1 if self.maturity_step is None else int(self.maturity_step)
        if maturity <= 0 or maturity >= n_steps:
            raise ValueError("maturity_step must be between 1 and the last path step")
        if self.exercise_start_step > maturity:
            raise ValueError("exercise_start_step cannot exceed maturity_step")

        return SwingContractGrid(
            maturity_step=maturity,
            min_action_units=_volume_to_units(self.min_exercise_volume, self.volume_step, "min_exercise_volume"),
            max_action_units=_volume_to_units(self.max_exercise_volume, self.volume_step, "max_exercise_volume"),
            min_total_units=_volume_to_units(self.min_total_volume, self.volume_step, "min_total_volume"),
            max_total_units=_volume_to_units(self.max_total_volume, self.volume_step, "max_total_volume"),
        )

    def margin(self, prices: np.ndarray) -> np.ndarray:
        values = np.asarray(prices, dtype=float)
        if self.payoff_type == "call":
            return values - self.strike - self.variable_cost_per_unit
        if self.payoff_type == "put":
            return self.strike - values - self.variable_cost_per_unit
        raise ValueError("payoff_type must be 'call' or 'put'")

    def is_exercise_step(self, step: int, maturity_step: int) -> bool:
        if step == maturity_step:
            return True
        if step < self.exercise_start_step:
            return False
        return (step - self.exercise_start_step) % self.exercise_step_interval == 0


@dataclass(frozen=True)
class SwingContractGrid:
    maturity_step: int
    min_action_units: int
    max_action_units: int
    min_total_units: int
    max_total_units: int


@dataclass(frozen=True)
class SwingLSMCResult:
    price: float
    stderr: float
    confidence_interval_95: tuple[float, float]
    path_values: np.ndarray
    policy: pd.DataFrame
    exercise_profile: pd.DataFrame
    regression_diagnostics: pd.DataFrame
    volume_summary: dict[str, float]
    policy_model: "SwingLSMCPolicy"
    contract: SwingOptionContract


@dataclass(frozen=True)
class SwingLSMCPolicy:
    """Frozen swing nomination policy learned by LSMC/ADP."""

    contract: SwingOptionContract
    regression: RegressionConfig
    grid: SwingContractGrid
    continuation_models: Mapping[int, ContinuationModel]
    fallback_continuation: Mapping[tuple[int, int], float]
    name: str = "swing_lsmc"

    def nominate(self, state: Any) -> float:
        remaining_units = _volume_to_units(
            float(state.remaining_volume),
            self.contract.volume_step,
            "remaining_volume",
        )
        action_units = self.nominate_units(
            step=int(state.step),
            price=float(state.current_price),
            variance=getattr(state, "variance", None),
            remaining_units=remaining_units,
        )
        return float(action_units * self.contract.volume_step)

    def nominate_units(
        self,
        step: int,
        price: float,
        variance: float | None,
        remaining_units: int,
    ) -> int:
        actions = _feasible_action_units(
            remaining_units=remaining_units,
            step=step,
            contract=self.contract,
            grid=self.grid,
        )
        if len(actions) == 1:
            return int(actions[0])

        margin = float(self.contract.margin(np.asarray([price], dtype=float))[0])
        best_value = -np.inf
        best_action = int(actions[0])
        for action_units in actions:
            after_units = int(remaining_units - action_units)
            immediate = float(action_units * self.contract.volume_step * margin)
            if step == self.grid.maturity_step:
                continuation = -_terminal_shortfall_penalty(after_units, self.contract, self.grid)
            else:
                continuation = float(
                    self.predict_continuation_value(
                        step=step,
                        price=price,
                        variance=variance,
                        remaining_units=after_units,
                    )
                )
            total = immediate + continuation
            if total > best_value:
                best_value = total
                best_action = int(action_units)
        return best_action

    def predict_continuation_value(
        self,
        step: int,
        price: float,
        variance: float | None,
        remaining_units: int,
    ) -> float:
        fallback = float(self.fallback_continuation.get((int(step), int(remaining_units)), 0.0))
        model = self.continuation_models.get(int(step))
        if model is None:
            value = fallback
        else:
            variance_array = None if variance is None else np.asarray([variance], dtype=float)
            features, _ = swing_regression_features(
                prices=np.asarray([price], dtype=float),
                variances=variance_array,
                remaining_units=np.asarray([remaining_units], dtype=float),
                contract=self.contract,
                grid=self.grid,
                regression=self.regression,
            )
            value = float(model.predict(features)[0])
        if self.regression.clip_negative_continuation:
            value = max(value, 0.0)
        return value


def value_swing_option_lsmc(
    paths: pd.DataFrame | np.ndarray,
    contract: SwingOptionContract,
    regression: RegressionConfig | None = None,
    variance_paths: np.ndarray | None = None,
) -> SwingLSMCResult:
    """Value a swing option using LSMC over price and remaining-volume state.

    The volume state is discretized by ``volume_step``. At each exercise date,
    the algorithm compares all feasible nominations and estimates continuation
    values from a ridge regression over price, variance and remaining volume.
    """

    regression = regression or RegressionConfig(
        itm_only=False,
        min_regression_paths=20,
        clip_negative_continuation=False,
    )
    prices, variances, matrices = _coerce_paths(paths, variance_paths)
    n_paths, n_steps = prices.shape
    grid = contract.validate(n_steps)
    maturity = grid.maturity_step
    prices = prices[:, : maturity + 1]
    variances = None if variances is None else variances[:, : maturity + 1]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))

    next_values = np.zeros((n_paths, grid.max_total_units + 1), dtype=float)
    continuation_models: dict[int, ContinuationModel] = {}
    fallback_continuation: dict[tuple[int, int], float] = {}
    diagnostics: list[dict[str, float | int | str]] = []

    for step in range(maturity, contract.exercise_start_step - 1, -1):
        margin = contract.margin(prices[:, step])
        current_values = np.empty_like(next_values)
        model = None
        status = "terminal" if step == maturity else "constant_fallback"
        r2 = float("nan")
        rows = 0

        if step < maturity:
            discounted_next_values = step_discount * next_values
            for remaining_units in range(grid.max_total_units + 1):
                fallback_continuation[(step, remaining_units)] = float(
                    np.mean(discounted_next_values[:, remaining_units])
                )
            features, feature_names, target = _stack_swing_regression_rows(
                prices=prices[:, step],
                variances=clean_variance_feature(variances, step, n_paths),
                next_values=discounted_next_values,
                contract=contract,
                grid=grid,
                regression=regression,
            )
            rows = int(len(target))
            if rows >= regression.min_regression_paths and np.std(target) > 1e-12:
                model = fit_ridge_regression(
                    features=features,
                    target=target,
                    feature_names=feature_names,
                    ridge_alpha=regression.ridge_alpha,
                )
                r2 = model.r2
                continuation_models[step] = model
                status = "ridge"

        for remaining_units in range(grid.max_total_units + 1):
            actions = _feasible_action_units(
                remaining_units=remaining_units,
                step=step,
                contract=contract,
                grid=grid,
            )
            best_value = np.full(n_paths, -np.inf, dtype=float)
            for action_units in actions:
                after_units = remaining_units - action_units
                immediate = action_units * contract.volume_step * margin
                if step == maturity:
                    penalty = _terminal_shortfall_penalty(after_units, contract, grid)
                    total = immediate - penalty
                else:
                    continuation = _predict_swing_continuation(
                        model=model,
                        prices=prices[:, step],
                        variances=clean_variance_feature(variances, step, n_paths),
                        remaining_units=after_units,
                        fallback_values=step_discount * next_values[:, after_units],
                        contract=contract,
                        grid=grid,
                        regression=regression,
                    )
                    if regression.clip_negative_continuation:
                        continuation = np.maximum(continuation, 0.0)
                    total = immediate + continuation
                take = total > best_value
                best_value[take] = total[take]
            current_values[:, remaining_units] = best_value

        next_values = current_values
        diagnostics.append(
            {
                "step": int(step),
                "status": status,
                "regression_rows": rows,
                "regression_r2": r2,
                "mean_start_state_value": float(np.mean(current_values[:, grid.max_total_units])),
                "exercise_date": bool(contract.is_exercise_step(step, maturity)),
            }
        )

    policy_model = SwingLSMCPolicy(
        contract=contract,
        regression=regression,
        grid=grid,
        continuation_models=continuation_models,
        fallback_continuation=fallback_continuation,
    )
    policy, path_values = _forward_apply_swing_policy_model(
        prices=prices,
        variances=variances,
        contract=contract,
        grid=grid,
        policy_model=policy_model,
        matrices=matrices,
    )
    profile = _swing_exercise_profile(policy)
    volume_summary = _volume_summary(policy, path_values)
    price = float(np.mean(path_values))
    return SwingLSMCResult(
        price=price,
        stderr=standard_error(path_values),
        confidence_interval_95=confidence_interval_95(path_values),
        path_values=path_values,
        policy=policy,
        exercise_profile=profile,
        regression_diagnostics=pd.DataFrame(diagnostics).sort_values("step").reset_index(drop=True),
        volume_summary=volume_summary,
        policy_model=policy_model,
        contract=contract,
    )


def _stack_swing_regression_rows(
    prices: np.ndarray,
    variances: np.ndarray,
    next_values: np.ndarray,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
    regression: RegressionConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    rows = []
    targets = []
    for remaining_units in range(grid.max_total_units + 1):
        features, names = swing_regression_features(
            prices=prices,
            variances=variances,
            remaining_units=np.full(len(prices), remaining_units, dtype=float),
            contract=contract,
            grid=grid,
            regression=regression,
        )
        rows.append(features)
        targets.append(next_values[:, remaining_units])
    return np.vstack(rows), names, np.concatenate(targets)


def swing_regression_features(
    prices: np.ndarray,
    variances: np.ndarray | None,
    remaining_units: np.ndarray,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
    regression: RegressionConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    current_prices = np.asarray(prices, dtype=float)
    remaining = np.asarray(remaining_units, dtype=float)
    if current_prices.shape[0] != remaining.shape[0]:
        raise ValueError("prices and remaining_units must have the same length")
    moneyness = current_prices / contract.strike - 1.0
    remaining_fraction = remaining / max(grid.max_total_units, 1)
    exercised_fraction = (grid.max_total_units - remaining) / max(grid.max_total_units, 1)
    min_shortfall_units = np.maximum(grid.min_total_units - (grid.max_total_units - remaining), 0.0)
    min_shortfall_fraction = min_shortfall_units / max(grid.max_total_units, 1)
    columns = [np.ones_like(moneyness)]
    names = ["constant"]
    for power in range(1, max(1, int(regression.degree)) + 1):
        columns.append(moneyness**power)
        names.append(f"moneyness_power_{power}")
    if regression.include_log_moneyness:
        columns.append(np.log(current_prices / contract.strike))
        names.append("log_moneyness")
    columns.extend([remaining_fraction, remaining_fraction**2, exercised_fraction, min_shortfall_fraction])
    names.extend(["remaining_fraction", "remaining_fraction_sq", "exercised_fraction", "min_shortfall_fraction"])
    columns.append((contract.margin(current_prices) / contract.strike) * remaining_fraction)
    names.append("scaled_margin_x_remaining")
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


def _predict_swing_continuation(
    model,
    prices: np.ndarray,
    variances: np.ndarray,
    remaining_units: int,
    fallback_values: np.ndarray,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
    regression: RegressionConfig,
) -> np.ndarray:
    if model is None:
        return np.full(len(prices), float(np.mean(fallback_values)), dtype=float)
    features, _ = swing_regression_features(
        prices=prices,
        variances=variances,
        remaining_units=np.full(len(prices), remaining_units, dtype=float),
        contract=contract,
        grid=grid,
        regression=regression,
    )
    return model.predict(features)


def _feasible_action_units(
    remaining_units: int,
    step: int,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
) -> np.ndarray:
    if remaining_units <= 0 or not contract.is_exercise_step(step, grid.maturity_step):
        return np.array([0], dtype=int)
    upper = min(grid.max_action_units, remaining_units)
    if grid.min_action_units == 0:
        base_actions = list(range(0, upper + 1))
    else:
        base_actions = [0] + list(range(grid.min_action_units, upper + 1))

    if not contract.enforce_min_total_volume or grid.min_total_units == 0:
        return np.asarray(base_actions, dtype=int)

    future_dates = sum(
        1
        for future_step in range(step + 1, grid.maturity_step + 1)
        if contract.is_exercise_step(future_step, grid.maturity_step)
    )
    feasible = []
    for action in base_actions:
        after = remaining_units - action
        exercised_after = grid.max_total_units - after
        max_future_volume = min(after, future_dates * grid.max_action_units)
        if exercised_after + max_future_volume >= grid.min_total_units:
            feasible.append(action)
    return np.asarray(feasible or base_actions, dtype=int)


def _terminal_shortfall_penalty(
    remaining_units_after_action: int,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
) -> float:
    exercised = grid.max_total_units - remaining_units_after_action
    shortfall_units = max(grid.min_total_units - exercised, 0)
    return float(shortfall_units * contract.volume_step * contract.shortfall_penalty_per_unit)


def _forward_apply_swing_policy_model(
    prices: np.ndarray,
    variances: np.ndarray | None,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
    policy_model: SwingLSMCPolicy,
    matrices: PathMatrices | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_paths = prices.shape[0]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    path_ids = np.arange(n_paths, dtype=int) if matrices is None else matrices.path_ids
    remaining = np.full(n_paths, grid.max_total_units, dtype=int)
    exercised = np.zeros(n_paths, dtype=int)
    path_values = np.zeros(n_paths, dtype=float)
    rows: list[dict[str, Any]] = []
    exercise_steps = [
        step
        for step in range(contract.exercise_start_step, grid.maturity_step + 1)
        if contract.is_exercise_step(step, grid.maturity_step)
    ]

    for step in exercise_steps:
        step_time = _time_at_step(matrices, step)
        for path_index in range(n_paths):
            price = float(prices[path_index, step])
            variance = _variance_at_step(variances, path_index, step)
            before = int(remaining[path_index])
            action_units = policy_model.nominate_units(
                step=step,
                price=price,
                variance=variance,
                remaining_units=before,
            )
            action_units = _project_internal_action_units(action_units, before, step, contract, grid)
            remaining[path_index] -= action_units
            exercised[path_index] += action_units
            action_volume = action_units * contract.volume_step
            immediate = action_volume * float(contract.margin(np.asarray([price], dtype=float))[0])
            discounted_immediate = immediate * np.power(step_discount, step)
            path_values[path_index] += discounted_immediate
            rows.append(
                {
                    "path": path_ids[path_index],
                    "step": step,
                    "time": step_time,
                    "price": price,
                    "remaining_volume_before": before * contract.volume_step,
                    "action_volume": action_volume,
                    "remaining_volume_after": remaining[path_index] * contract.volume_step,
                    "period_margin": float(contract.margin(np.asarray([price], dtype=float))[0]),
                    "immediate_payoff": immediate,
                    "discounted_immediate_payoff": discounted_immediate,
                }
            )

    if contract.enforce_min_total_volume and contract.shortfall_penalty_per_unit != 0.0:
        total_volume = exercised * contract.volume_step
        shortfall = np.maximum(contract.min_total_volume - total_volume, 0.0)
        path_values -= shortfall * contract.shortfall_penalty_per_unit * np.power(
            step_discount,
            grid.maturity_step,
        )

    return pd.DataFrame(rows), path_values


def _project_internal_action_units(
    requested_units: int,
    remaining_units: int,
    step: int,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
) -> int:
    feasible = _feasible_action_units(
        remaining_units=remaining_units,
        step=step,
        contract=contract,
        grid=grid,
    )
    if int(requested_units) in set(feasible):
        return int(requested_units)
    return int(min(feasible, key=lambda candidate: (abs(int(candidate) - int(requested_units)), int(candidate))))


def _time_at_step(matrices: PathMatrices | None, step: int) -> pd.Timestamp | None:
    if matrices is None or matrices.times is None or step not in matrices.times.index:
        return None
    value = matrices.times.loc[step]
    if pd.isna(value):
        return None
    return pd.Timestamp(value)


def _variance_at_step(variances: np.ndarray | None, path_index: int, step: int) -> float | None:
    if variances is None:
        return None
    value = float(variances[path_index, step])
    if not np.isfinite(value) or value < 0.0:
        return None
    return value


def _swing_exercise_profile(policy: pd.DataFrame) -> pd.DataFrame:
    profile = (
        policy.groupby("step", sort=True)
        .agg(
            exercise_probability=("action_volume", lambda values: float(np.mean(np.asarray(values) > 0.0))),
            mean_volume=("action_volume", "mean"),
            mean_positive_volume=("action_volume", lambda values: float(np.mean([v for v in values if v > 0.0])) if (np.asarray(values) > 0.0).any() else 0.0),
            mean_margin=("period_margin", "mean"),
        )
        .reset_index()
    )
    return profile


def _volume_summary(policy: pd.DataFrame, path_values: np.ndarray) -> dict[str, float]:
    total = policy.groupby("path")["action_volume"].sum().to_numpy(dtype=float)
    return {
        "mean_total_volume": float(np.mean(total)),
        "q05_total_volume": float(np.quantile(total, 0.05)),
        "q50_total_volume": float(np.quantile(total, 0.50)),
        "q95_total_volume": float(np.quantile(total, 0.95)),
        "mean_positive_value": float(np.mean(path_values > 0.0)),
        "mean_path_value": float(np.mean(path_values)),
    }


def _volume_to_units(value: float, volume_step: float, name: str) -> int:
    units = int(round(float(value) / float(volume_step)))
    if not np.isclose(units * volume_step, value, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must be an integer multiple of volume_step")
    return units


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
