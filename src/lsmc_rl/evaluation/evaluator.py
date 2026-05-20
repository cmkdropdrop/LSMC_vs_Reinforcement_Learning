"""Evaluate frozen policies on common test paths without refitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lsmc_rl.evaluation.policies import (
    AmericanExercisePolicy,
    AmericanPolicyState,
    ImmediateIntrinsicExercisePolicy,
    NeverEarlyExercisePolicy,
    SwingNominationPolicy,
    SwingPolicyState,
    ValidationSelectedAmericanPolicy,
)
from lsmc_rl.valuation.american import AmericanLSMCPolicy, AmericanOptionContract, _forward_apply_lsmc_policy
from lsmc_rl.valuation.common import PathMatrices, paths_frame_to_matrices
from lsmc_rl.valuation.swing import SwingOptionContract, SwingContractGrid


@dataclass(frozen=True)
class AmericanPolicyEvaluation:
    """Pathwise evaluation output for one American exercise policy."""

    policy_name: str
    path_values: np.ndarray
    exercise_steps: np.ndarray
    exercise_payoffs: np.ndarray
    path_results: pd.DataFrame
    decision_trace: pd.DataFrame
    contract: AmericanOptionContract


@dataclass(frozen=True)
class SwingPolicyEvaluation:
    """Pathwise evaluation output for one swing nomination policy."""

    policy_name: str
    path_values: np.ndarray
    path_results: pd.DataFrame
    decision_trace: pd.DataFrame
    nomination_profile: pd.DataFrame
    contract: SwingOptionContract


def evaluate_american_policy(
    paths: pd.DataFrame | np.ndarray,
    contract: AmericanOptionContract,
    policy: AmericanExercisePolicy,
    variance_paths: np.ndarray | None = None,
) -> AmericanPolicyEvaluation:
    """Evaluate a frozen American policy path by path.

    The evaluator constructs state only from the current step and static
    contract metadata. It does not fit regressions or inspect later prices.
    """

    matrices = _coerce_paths(paths, variance_paths)
    prices = matrices.prices
    n_paths, n_steps = prices.shape
    maturity = contract.validate(n_steps)
    prices = prices[:, : maturity + 1]
    variances = None if matrices.variances is None else matrices.variances[:, : maturity + 1]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    path_ids = matrices.path_ids

    fast_result = _try_fast_american_policy_evaluation(
        policy=policy,
        prices=prices,
        variances=variances,
        contract=contract,
        maturity=maturity,
        step_discount=step_discount,
        path_ids=path_ids,
        min_fast_paths=128,
    )
    if fast_result is not None:
        return fast_result

    exercise_steps = np.full(n_paths, maturity, dtype=int)
    terminal_payoff = contract.payoff(prices[:, maturity])
    exercise_payoffs = terminal_payoff.copy()
    trace_rows: list[dict[str, Any]] = []

    for path_index in range(n_paths):
        for step in range(contract.exercise_start_step, maturity):
            if not contract.is_exercise_step(step, maturity):
                continue
            price = float(prices[path_index, step])
            intrinsic = float(contract.payoff(np.asarray([price]))[0])
            state = AmericanPolicyState(
                step=step,
                time=_time_at_step(matrices, step),
                current_price=price,
                variance=_variance_at_step(variances, path_index, step),
                volatility=_volatility_at_step(variances, path_index, step),
                maturity_step=maturity,
                remaining_steps=maturity - step,
                intrinsic_value=intrinsic,
                contract=contract,
            )
            exercised = bool(policy.decide_exercise(state))
            trace_rows.append(
                {
                    "path": path_ids[path_index],
                    "step": step,
                    "time": state.time,
                    "price": price,
                    "intrinsic_value": intrinsic,
                    "exercised": exercised,
                }
            )
            if exercised:
                exercise_steps[path_index] = step
                exercise_payoffs[path_index] = intrinsic
                break

    path_values = exercise_payoffs * np.power(step_discount, exercise_steps)
    path_results = pd.DataFrame(
        {
            "path": path_ids,
            "policy": _policy_name(policy),
            "value": path_values,
            "exercise_step": exercise_steps,
            "exercise_payoff": exercise_payoffs,
            "constraint_violations": np.zeros(n_paths, dtype=int),
            "costs": np.zeros(n_paths, dtype=float),
        }
    )
    decision_trace = pd.DataFrame(trace_rows)
    return AmericanPolicyEvaluation(
        policy_name=_policy_name(policy),
        path_values=path_values,
        exercise_steps=exercise_steps,
        exercise_payoffs=exercise_payoffs,
        path_results=path_results,
        decision_trace=decision_trace,
        contract=contract,
    )


def _try_fast_american_policy_evaluation(
    *,
    policy: AmericanExercisePolicy,
    prices: np.ndarray,
    variances: np.ndarray | None,
    contract: AmericanOptionContract,
    maturity: int,
    step_discount: float,
    path_ids: np.ndarray,
    min_fast_paths: int,
) -> AmericanPolicyEvaluation | None:
    if prices.shape[0] < min_fast_paths:
        return None

    effective_policy: Any = policy
    if isinstance(policy, ValidationSelectedAmericanPolicy):
        effective_policy = policy.candidate if policy.use_candidate else policy.fallback

    if isinstance(effective_policy, NeverEarlyExercisePolicy):
        exercise_steps = np.full(prices.shape[0], maturity, dtype=int)
        exercise_payoffs = contract.payoff(prices[:, maturity])
        path_values = exercise_payoffs * np.power(step_discount, exercise_steps)
    elif isinstance(effective_policy, ImmediateIntrinsicExercisePolicy):
        exercise_steps, exercise_payoffs, path_values = _forward_apply_immediate_policy(
            prices=prices,
            contract=contract,
            policy=effective_policy,
            maturity=maturity,
        )
    elif isinstance(effective_policy, AmericanLSMCPolicy):
        exercise_steps, exercise_payoffs, path_values = _forward_apply_lsmc_policy(
            prices=prices,
            variances=variances,
            contract=contract,
            policy=effective_policy,
            maturity=maturity,
        )
    elif _is_fitted_q_policy(effective_policy):
        exercise_steps, exercise_payoffs, path_values = _forward_apply_fitted_q_policy(
            prices=prices,
            variances=variances,
            contract=contract,
            policy=effective_policy,
            maturity=maturity,
        )
    elif _is_kernel_fitted_q_policy(effective_policy):
        exercise_steps, exercise_payoffs, path_values = _forward_apply_kernel_fitted_q_policy(
            prices=prices,
            variances=variances,
            contract=contract,
            policy=effective_policy,
            maturity=maturity,
        )
    else:
        return None

    policy_name = _policy_name(policy)
    path_results = pd.DataFrame(
        {
            "path": path_ids,
            "policy": policy_name,
            "value": path_values,
            "exercise_step": exercise_steps,
            "exercise_payoff": exercise_payoffs,
            "constraint_violations": np.zeros(prices.shape[0], dtype=int),
            "costs": np.zeros(prices.shape[0], dtype=float),
        }
    )
    decision_trace = pd.DataFrame(columns=["path", "step", "time", "price", "intrinsic_value", "exercised"])
    return AmericanPolicyEvaluation(
        policy_name=policy_name,
        path_values=path_values,
        exercise_steps=exercise_steps,
        exercise_payoffs=exercise_payoffs,
        path_results=path_results,
        decision_trace=decision_trace,
        contract=contract,
    )


def _forward_apply_immediate_policy(
    *,
    prices: np.ndarray,
    contract: AmericanOptionContract,
    policy: ImmediateIntrinsicExercisePolicy,
    maturity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_paths = prices.shape[0]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    exercise_steps = np.full(n_paths, maturity, dtype=int)
    exercise_payoffs = contract.payoff(prices[:, maturity])
    alive = np.ones(n_paths, dtype=bool)
    for step in range(contract.exercise_start_step, maturity):
        if not contract.is_exercise_step(step, maturity):
            continue
        alive_index = np.flatnonzero(alive)
        if alive_index.size == 0:
            break
        intrinsic = contract.payoff(prices[alive_index, step])
        exercise_now = intrinsic > policy.tolerance
        if exercise_now.any():
            selected = alive_index[exercise_now]
            exercise_steps[selected] = step
            exercise_payoffs[selected] = intrinsic[exercise_now]
            alive[selected] = False
    path_values = exercise_payoffs * np.power(step_discount, exercise_steps)
    return exercise_steps, exercise_payoffs, path_values


def _is_fitted_q_policy(policy: Any) -> bool:
    return policy.__class__.__name__ == "AmericanFittedQPolicy"


def _is_kernel_fitted_q_policy(policy: Any) -> bool:
    return policy.__class__.__name__ == "AmericanKernelFittedQPolicy"


def _forward_apply_fitted_q_policy(
    *,
    prices: np.ndarray,
    variances: np.ndarray | None,
    contract: AmericanOptionContract,
    policy: Any,
    maturity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lsmc_rl.rl.fitted_q import american_fitted_q_features

    n_paths = prices.shape[0]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    exercise_steps = np.full(n_paths, maturity, dtype=int)
    exercise_payoffs = contract.payoff(prices[:, maturity])
    alive = np.ones(n_paths, dtype=bool)

    for step in range(contract.exercise_start_step, maturity):
        if not contract.is_exercise_step(step, maturity):
            continue
        alive_index = np.flatnonzero(alive)
        if alive_index.size == 0:
            break
        intrinsic = contract.payoff(prices[alive_index, step])
        continuation = np.full(
            alive_index.size,
            float(policy.fallback_continuation.get(step, 0.0)),
            dtype=float,
        )
        model = policy.continuation_models.get(step)
        if model is not None:
            step_variance = None if variances is None else np.asarray(variances[alive_index, step], dtype=float)
            features, _ = american_fitted_q_features(
                prices=prices[alive_index, step],
                variances=step_variance,
                steps=np.full(alive_index.size, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=policy.config,
            )
            continuation = model.predict(features)
        if policy.config.clip_negative_continuation:
            continuation = np.maximum(continuation, 0.0)
        candidates = np.ones(alive_index.size, dtype=bool)
        if policy.config.exercise_only_itm:
            candidates &= intrinsic > policy.config.exercise_tolerance
        exercise_now = candidates & (intrinsic > continuation + policy.config.exercise_tolerance)
        if exercise_now.any():
            selected = alive_index[exercise_now]
            exercise_steps[selected] = step
            exercise_payoffs[selected] = intrinsic[exercise_now]
            alive[selected] = False

    path_values = exercise_payoffs * np.power(step_discount, exercise_steps)
    return exercise_steps, exercise_payoffs, path_values


def _forward_apply_kernel_fitted_q_policy(
    *,
    prices: np.ndarray,
    variances: np.ndarray | None,
    contract: AmericanOptionContract,
    policy: Any,
    maturity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lsmc_rl.rl.kernel_fitted_q import american_kernel_base_features

    n_paths = prices.shape[0]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    exercise_steps = np.full(n_paths, maturity, dtype=int)
    exercise_payoffs = contract.payoff(prices[:, maturity])
    alive = np.ones(n_paths, dtype=bool)

    for step in range(contract.exercise_start_step, maturity):
        if not contract.is_exercise_step(step, maturity):
            continue
        alive_index = np.flatnonzero(alive)
        if alive_index.size == 0:
            break
        intrinsic = contract.payoff(prices[alive_index, step])
        continuation = np.full(
            alive_index.size,
            float(policy.fallback_continuation.get(step, 0.0)),
            dtype=float,
        )
        model = policy.continuation_models.get(step)
        if model is not None:
            step_variance = None if variances is None else np.asarray(variances[alive_index, step], dtype=float)
            base = american_kernel_base_features(
                prices=prices[alive_index, step],
                variances=step_variance,
                steps=np.full(alive_index.size, step, dtype=int),
                maturity_step=maturity,
                contract=contract,
                config=policy.config,
            )
            continuation = model.predict(policy.feature_map.transform(base))
        if policy.config.clip_negative_continuation:
            continuation = np.maximum(continuation, 0.0)
        candidates = np.ones(alive_index.size, dtype=bool)
        if policy.config.exercise_only_itm:
            candidates &= intrinsic > policy.config.exercise_tolerance
        exercise_now = candidates & (intrinsic > continuation + policy.config.exercise_tolerance)
        if exercise_now.any():
            selected = alive_index[exercise_now]
            exercise_steps[selected] = step
            exercise_payoffs[selected] = intrinsic[exercise_now]
            alive[selected] = False

    path_values = exercise_payoffs * np.power(step_discount, exercise_steps)
    return exercise_steps, exercise_payoffs, path_values


def evaluate_swing_policy(
    paths: pd.DataFrame | np.ndarray,
    contract: SwingOptionContract,
    policy: SwingNominationPolicy,
    variance_paths: np.ndarray | None = None,
) -> SwingPolicyEvaluation:
    """Evaluate a frozen swing nomination policy on fixed test paths."""

    matrices = _coerce_paths(paths, variance_paths)
    prices = matrices.prices
    n_paths, n_steps = prices.shape
    grid = contract.validate(n_steps)
    maturity = grid.maturity_step
    prices = prices[:, : maturity + 1]
    variances = None if matrices.variances is None else matrices.variances[:, : maturity + 1]
    step_discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    path_ids = matrices.path_ids
    exercise_steps = _exercise_steps(contract, maturity)

    values = np.zeros(n_paths, dtype=float)
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for path_index in range(n_paths):
        remaining_units = grid.max_total_units
        exercised_units = 0
        period_violations = 0
        discounted_payoff = 0.0

        for step in exercise_steps:
            price = float(prices[path_index, step])
            margin = float(contract.margin(np.asarray([price]))[0])
            state = SwingPolicyState(
                step=step,
                time=_time_at_step(matrices, step),
                current_price=price,
                variance=_variance_at_step(variances, path_index, step),
                volatility=_volatility_at_step(variances, path_index, step),
                maturity_step=maturity,
                remaining_steps=maturity - step,
                remaining_exercise_dates=sum(later >= step for later in exercise_steps),
                remaining_volume=remaining_units * contract.volume_step,
                exercised_volume=exercised_units * contract.volume_step,
                margin=margin,
                contract=contract,
            )
            requested = float(policy.nominate(state))
            action_units, violated = _project_period_action_units(requested, remaining_units, contract, grid)
            action_volume = action_units * contract.volume_step
            period_violations += int(violated)
            before_units = remaining_units
            remaining_units -= action_units
            exercised_units += action_units
            immediate = action_volume * margin
            discounted_payoff += immediate * np.power(step_discount, step)
            trace_rows.append(
                {
                    "path": path_ids[path_index],
                    "step": step,
                    "time": state.time,
                    "price": price,
                    "margin": margin,
                    "requested_volume": requested,
                    "action_volume": action_volume,
                    "remaining_volume_before": before_units * contract.volume_step,
                    "remaining_volume_after": remaining_units * contract.volume_step,
                    "period_constraint_violation": int(violated),
                    "immediate_payoff": immediate,
                    "discounted_immediate_payoff": immediate * np.power(step_discount, step),
                }
            )

        total_volume = exercised_units * contract.volume_step
        remaining_volume = remaining_units * contract.volume_step
        shortfall_volume = (
            max(contract.min_total_volume - total_volume, 0.0)
            if contract.enforce_min_total_volume
            else 0.0
        )
        excess_volume = max(total_volume - contract.max_total_volume, 0.0)
        shortfall_penalty = shortfall_volume * contract.shortfall_penalty_per_unit
        discounted_penalty = shortfall_penalty * np.power(step_discount, maturity)
        path_value = discounted_payoff - discounted_penalty
        values[path_index] = path_value
        total_violations = period_violations + int(shortfall_volume > 1e-9) + int(excess_volume > 1e-9)
        result_rows.append(
            {
                "path": path_ids[path_index],
                "policy": _policy_name(policy),
                "value": path_value,
                "total_volume": total_volume,
                "remaining_volume": remaining_volume,
                "shortfall_volume": shortfall_volume,
                "excess_volume": excess_volume,
                "period_constraint_violations": period_violations,
                "constraint_violations": total_violations,
                "shortfall_penalty": shortfall_penalty,
                "discounted_shortfall_penalty": discounted_penalty,
                "costs": discounted_penalty,
            }
        )

    decision_trace = pd.DataFrame(trace_rows)
    path_results = pd.DataFrame(result_rows)
    return SwingPolicyEvaluation(
        policy_name=_policy_name(policy),
        path_values=values,
        path_results=path_results,
        decision_trace=decision_trace,
        nomination_profile=_nomination_profile(decision_trace),
        contract=contract,
    )


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


def _policy_name(policy: Any) -> str:
    return str(getattr(policy, "name", policy.__class__.__name__))


def _time_at_step(matrices: PathMatrices, step: int) -> pd.Timestamp | None:
    if matrices.times is None or step not in matrices.times.index:
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


def _volatility_at_step(variances: np.ndarray | None, path_index: int, step: int) -> float | None:
    variance = _variance_at_step(variances, path_index, step)
    return None if variance is None else float(np.sqrt(variance))


def _exercise_steps(contract: SwingOptionContract, maturity: int) -> list[int]:
    return [
        step
        for step in range(contract.exercise_start_step, maturity + 1)
        if contract.is_exercise_step(step, maturity)
    ]


def _project_period_action_units(
    requested_volume: float,
    remaining_units: int,
    contract: SwingOptionContract,
    grid: SwingContractGrid,
) -> tuple[int, bool]:
    feasible = _period_feasible_action_units(remaining_units, grid)
    if not np.isfinite(requested_volume):
        return 0, True
    raw_units = requested_volume / contract.volume_step
    requested_units = int(round(raw_units))
    grid_violation = not np.isclose(requested_units, raw_units, rtol=0.0, atol=1e-9)
    bounds_violation = requested_units < 0 or requested_units > grid.max_action_units or requested_units > remaining_units
    min_lot_violation = 0 < requested_units < grid.min_action_units
    violated = bool(grid_violation or bounds_violation or min_lot_violation or requested_units not in set(feasible))
    if requested_units in set(feasible):
        return requested_units, violated
    action_units = min(feasible, key=lambda candidate: (abs(candidate - requested_units), candidate))
    return int(action_units), violated


def _period_feasible_action_units(remaining_units: int, grid: SwingContractGrid) -> list[int]:
    if remaining_units <= 0:
        return [0]
    upper = min(grid.max_action_units, remaining_units)
    if grid.min_action_units == 0:
        return list(range(0, upper + 1))
    return [0] + list(range(grid.min_action_units, upper + 1))


def _nomination_profile(decision_trace: pd.DataFrame) -> pd.DataFrame:
    if decision_trace.empty:
        return pd.DataFrame(columns=["step", "exercise_probability", "mean_volume", "mean_margin"])
    return (
        decision_trace.groupby("step", sort=True)
        .agg(
            exercise_probability=("action_volume", lambda values: float(np.mean(np.asarray(values) > 0.0))),
            mean_volume=("action_volume", "mean"),
            mean_margin=("margin", "mean"),
        )
        .reset_index()
    )
