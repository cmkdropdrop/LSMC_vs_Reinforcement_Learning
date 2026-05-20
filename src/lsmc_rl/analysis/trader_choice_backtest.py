"""Historical trader-choice backtest for LSMC versus RL exercise policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc_rl.data import load_market_data
from lsmc_rl.evaluation import (
    ImmediateIntrinsicExercisePolicy,
    NeverEarlyExercisePolicy,
    evaluate_american_policy,
    paired_policy_metrics,
    path_value_summary,
    select_american_policy_by_validation,
)
from lsmc_rl.rl import (
    FittedQConfig,
    KernelFittedQConfig,
    train_american_fitted_q,
    train_american_kernel_fitted_q,
)
from lsmc_rl.valuation import AmericanOptionContract, RegressionConfig, value_american_option_lsmc
from lsmc_rl.valuation.common import json_ready
from lsmc_rl.volatility import GJRGARCHModel


@dataclass(frozen=True)
class TraderChoiceBacktestConfig:
    """Controls for the historical trader-choice backtest."""

    database_path: Path = Path("ttf_klines_5m_from_1m.sqlite")
    symbol: str = "FRONT"
    interval: str = "5m"
    output_dir: Path = Path("outputs/trader_choice_backtest_front")
    option_type: str = "put"
    strike_moneyness: float = 1.0
    risk_free_rate: float = 0.03
    exercise_fee_per_unit: float = 0.0
    slippage_bps: float = 0.0
    n_backtest_days: int = 12
    min_day_observations: int = 30
    training_lookback_returns: int = 1500
    min_training_returns: int = 300
    n_training_paths: int = 300
    n_validation_paths: int = 300
    seed: int = 20260523
    validation_seed_offset: int = 100_000
    bootstrap_seed: int = 20260525
    n_bootstrap: int = 1000
    deployment_min_ci_low: float = 0.0
    deployment_min_cvar_5_delta: float | None = 0.0
    kernel_rff_features: int = 64
    kernel_length_scale: float = 1.0
    kernel_feature_seed: int = 20260524
    cost_stress_slippage_bps: tuple[float, ...] = (0.0, 1.0, 5.0, 25.0)
    cost_stress_exercise_fee_per_unit: tuple[float, ...] = (0.0, 0.10, 0.50)


def run_backtest(config: TraderChoiceBacktestConfig) -> dict[str, Any]:
    """Run the historical daily exercise-policy backtest."""

    _ensure_allowed_artifact_dir(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    market, quality = load_market_data(config.database_path, config.symbol, config.interval)
    eligible_days = _eligible_trade_days(market, config)
    if not eligible_days:
        raise ValueError("No eligible backtest days found for the configured filters")

    policy_rows: list[dict[str, Any]] = []
    skipped_days: list[dict[str, Any]] = []
    for day_index, trade_date in enumerate(eligible_days):
        day_frame = _day_frame(market, trade_date)
        first_time = day_frame["open_datetime"].iloc[0]
        training_returns = (
            market.loc[market["open_datetime"] < first_time, "log_return"]
            .dropna()
            .tail(config.training_lookback_returns)
        )
        if len(training_returns) < config.min_training_returns:
            skipped_days.append({"date": str(trade_date), "reason": "insufficient_training_returns"})
            continue

        try:
            rows = _run_one_trade_day(
                day_index=day_index,
                trade_date=trade_date,
                day_frame=day_frame,
                training_returns=training_returns,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - retained in report instead of aborting a long backtest
            skipped_days.append({"date": str(trade_date), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        policy_rows.extend(rows)

    if not policy_rows:
        raise ValueError(f"All candidate days failed: {skipped_days}")

    results = pd.DataFrame(policy_rows)
    summary = _summarize_results(results)
    paired_diagnostics = _historical_paired_diagnostics(results, config)
    validation_audit = _validation_gate_audit(results)
    model_price_diagnostics = _model_price_diagnostics(results)
    cost_stress = _cost_stress_summary(results, config)
    cost_stress_frontier = _cost_stress_frontier(cost_stress)
    regime_summary = _market_regime_summary(results)
    worst_days = _worst_day_diagnostics(results)
    plots = _create_plots(config.output_dir, results)

    results_path = config.output_dir / "daily_policy_results.csv"
    summary_path = config.output_dir / "summary.csv"
    paired_path = config.output_dir / "historical_paired_diagnostics.csv"
    validation_audit_path = config.output_dir / "validation_gate_audit.csv"
    model_price_path = config.output_dir / "model_price_diagnostics.csv"
    cost_stress_path = config.output_dir / "cost_stress_summary.csv"
    cost_stress_frontier_path = config.output_dir / "cost_stress_frontier.csv"
    regime_summary_path = config.output_dir / "market_regime_summary.csv"
    worst_days_path = config.output_dir / "worst_day_diagnostics.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired_diagnostics.to_csv(paired_path, index=False)
    validation_audit.to_csv(validation_audit_path, index=False)
    model_price_diagnostics.to_csv(model_price_path, index=False)
    cost_stress.to_csv(cost_stress_path, index=False)
    cost_stress_frontier.to_csv(cost_stress_frontier_path, index=False)
    regime_summary.to_csv(regime_summary_path, index=False)
    worst_days.to_csv(worst_days_path, index=False)

    metrics: dict[str, Any] = {
        "scenario": _scenario_text(config),
        "config": _config_dict(config),
        "data_quality": quality.to_dict(),
        "backtest_period": {
            "first_trade_date": str(min(results["date"])),
            "last_trade_date": str(max(results["date"])),
            "trade_days": int(results["date"].nunique()),
            "skipped_days": skipped_days,
        },
        "summary": summary.to_dict(orient="records"),
        "valuation_assessment": _valuation_assessment(
            summary,
            paired_diagnostics,
            validation_audit,
            model_price_diagnostics,
            cost_stress_frontier,
        ),
        "historical_paired_diagnostics": paired_diagnostics.to_dict(orient="records"),
        "validation_gate_audit": validation_audit.to_dict(orient="records"),
        "model_price_diagnostics": model_price_diagnostics.to_dict(orient="records"),
        "cost_stress_frontier": cost_stress_frontier.to_dict(orient="records"),
        "market_regime_summary": regime_summary.to_dict(orient="records"),
        "worst_day_diagnostics": worst_days.to_dict(orient="records"),
        "path_value_summary": {
            method: path_value_summary(group["gross_pnl"].to_numpy(dtype=float))
            for method, group in results.groupby("method", sort=True)
        },
        "plots": {name: str(path) for name, path in plots.items()},
        "artifacts": {
            "daily_policy_results": str(results_path),
            "summary": str(summary_path),
            "historical_paired_diagnostics": str(paired_path),
            "validation_gate_audit": str(validation_audit_path),
            "model_price_diagnostics": str(model_price_path),
            "cost_stress_summary": str(cost_stress_path),
            "cost_stress_frontier": str(cost_stress_frontier_path),
            "market_regime_summary": str(regime_summary_path),
            "worst_day_diagnostics": str(worst_days_path),
        },
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    (config.output_dir / "README.md").write_text(_render_report(metrics), encoding="utf-8")
    return metrics


def _run_one_trade_day(
    day_index: int,
    trade_date: Any,
    day_frame: pd.DataFrame,
    training_returns: pd.Series,
    config: TraderChoiceBacktestConfig,
) -> list[dict[str, Any]]:
    actual_prices = day_frame["close"].to_numpy(dtype=float)
    if actual_prices.size < 3:
        raise ValueError("trade day must contain at least three prices")

    start_price = float(actual_prices[0])
    maturity = int(actual_prices.size - 1)
    contract = AmericanOptionContract(
        strike=start_price * config.strike_moneyness,
        option_type=config.option_type,  # type: ignore[arg-type]
        risk_free_rate=config.risk_free_rate,
        time_step_years=_infer_step_years(day_frame),
        maturity_step=maturity,
        exercise_start_step=1,
        exercise_step_interval=1,
    )
    model = GJRGARCHModel().fit(training_returns)
    train_prices, _, train_variances = model.simulate(
        start_price=start_price,
        horizon_steps=maturity,
        n_paths=config.n_training_paths,
        seed=config.seed + day_index,
    )
    validation_prices, _, validation_variances = model.simulate(
        start_price=start_price,
        horizon_steps=maturity,
        n_paths=config.n_validation_paths,
        seed=config.seed + config.validation_seed_offset + day_index,
    )

    lsmc = value_american_option_lsmc(
        train_prices,
        contract,
        RegressionConfig(
            degree=3,
            ridge_alpha=1e-5,
            itm_only=True,
            min_regression_paths=20,
            include_log_moneyness=True,
            include_intrinsic=True,
            include_variance=True,
        ),
        variance_paths=train_variances,
    )
    fitted_q = train_american_fitted_q(
        train_prices,
        contract,
        FittedQConfig(
            degree=3,
            ridge_alpha=1e-5,
            min_regression_paths=20,
            fit_itm_only=False,
            include_log_moneyness=True,
            include_intrinsic=True,
            include_variance=True,
            include_time_features=True,
        ),
        variance_paths=train_variances,
    )
    kernel_q = train_american_kernel_fitted_q(
        train_prices,
        contract,
        KernelFittedQConfig(
            n_rff_features=config.kernel_rff_features,
            length_scale=config.kernel_length_scale,
            ridge_alpha=1e-5,
            min_regression_paths=30,
            fit_itm_only=False,
            include_linear_features=True,
            include_variance=True,
            feature_seed=config.kernel_feature_seed,
        ),
        variance_paths=train_variances,
    )
    never_policy = NeverEarlyExercisePolicy()
    selection_kwargs = {
        "variance_paths": validation_variances,
        "baseline_policy": never_policy,
        "bootstrap_seed": config.bootstrap_seed + day_index,
        "n_bootstrap": config.n_bootstrap,
        "min_ci_low": config.deployment_min_ci_low,
        "min_cvar_5_delta": config.deployment_min_cvar_5_delta,
    }
    lsmc_selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        lsmc.policy,
        selected_name="american_lsmc_validation_selected_deployment",
        **selection_kwargs,
    )
    fitted_q_selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        fitted_q.policy,
        selected_name="american_fitted_q_validation_selected_deployment",
        **selection_kwargs,
    )
    kernel_q_selection = select_american_policy_by_validation(
        validation_prices,
        contract,
        kernel_q.policy,
        selected_name="american_kernel_fitted_q_validation_selected_deployment",
        **selection_kwargs,
    )

    actual_path = actual_prices.reshape(1, -1)
    actual_variances = _realized_garch_variance_path(model, actual_prices).reshape(1, -1)
    never = evaluate_american_policy(actual_path, contract, never_policy, variance_paths=actual_variances)
    immediate = evaluate_american_policy(
        actual_path,
        contract,
        ImmediateIntrinsicExercisePolicy(),
        variance_paths=actual_variances,
    )
    lsmc_eval = evaluate_american_policy(actual_path, contract, lsmc.policy, variance_paths=actual_variances)
    lsmc_selected_eval = evaluate_american_policy(
        actual_path,
        contract,
        lsmc_selection.policy,
        variance_paths=actual_variances,
    )
    fitted_q_eval = evaluate_american_policy(actual_path, contract, fitted_q.policy, variance_paths=actual_variances)
    fitted_q_selected_eval = evaluate_american_policy(
        actual_path,
        contract,
        fitted_q_selection.policy,
        variance_paths=actual_variances,
    )
    kernel_q_eval = evaluate_american_policy(actual_path, contract, kernel_q.policy, variance_paths=actual_variances)
    kernel_q_selected_eval = evaluate_american_policy(
        actual_path,
        contract,
        kernel_q_selection.policy,
        variance_paths=actual_variances,
    )
    oracle = _oracle_american_exercise(actual_prices, contract)

    day_meta = {
        "date": str(trade_date),
        "start_time": day_frame["open_datetime"].iloc[0].isoformat(),
        "end_time": day_frame["open_datetime"].iloc[-1].isoformat(),
        "observations": int(len(day_frame)),
        "start_price": start_price,
        "end_price": float(actual_prices[-1]),
        "strike": float(contract.strike),
        "min_price": float(np.min(actual_prices)),
        "max_price": float(np.max(actual_prices)),
        "realized_log_return": float(np.log(actual_prices[-1] / start_price)),
        "terminal_return_pct": float(actual_prices[-1] / start_price - 1.0),
        "downside_excursion_pct": float(max(start_price - np.min(actual_prices), 0.0) / start_price),
        "upside_excursion_pct": float(max(np.max(actual_prices) - start_price, 0.0) / start_price),
        "intraday_range_pct": float((np.max(actual_prices) - np.min(actual_prices)) / start_price),
        "terminal_intrinsic": float(contract.payoff(np.asarray([actual_prices[-1]], dtype=float))[0]),
        "max_intrinsic": float(np.max(contract.payoff(actual_prices))),
        "training_return_count": int(len(training_returns)),
        "exercise_fee_per_unit": float(config.exercise_fee_per_unit),
        "slippage_bps": float(config.slippage_bps),
        "training_paths": int(config.n_training_paths),
        "validation_paths": int(config.n_validation_paths),
        "deployment_min_ci_low": float(config.deployment_min_ci_low),
        "deployment_min_cvar_5_delta": config.deployment_min_cvar_5_delta,
        "garch_persistence": float(model.params.persistence) if model.params is not None else float("nan"),
    }
    maturity_value = float(never.path_values[0])
    oracle_value = float(oracle["gross_pnl"])
    oracle_exercise_step = int(oracle["exercise_step"])
    oracle_exercise_price = float(actual_prices[oracle_exercise_step])
    oracle_exercise_payoff = float(oracle["exercise_payoff"])
    oracle_execution_cost = _exercise_execution_cost(oracle_exercise_payoff, oracle_exercise_price, day_meta)

    rows = [
        _policy_row(day_meta, "Maturity", never, None, maturity_value, oracle_value, day_frame),
        _policy_row(day_meta, "Immediate intrinsic", immediate, None, maturity_value, oracle_value, day_frame),
        _policy_row(day_meta, "Raw LSMC", lsmc_eval, lsmc.price, maturity_value, oracle_value, day_frame),
        _policy_row(
            day_meta,
            "Selected LSMC",
            lsmc_selected_eval,
            lsmc.price if lsmc_selection.used_candidate else None,
            maturity_value,
            oracle_value,
            day_frame,
            validation_selection=lsmc_selection.validation_metrics,
        ),
        _policy_row(
            day_meta,
            "Raw Linear Fitted-Q",
            fitted_q_eval,
            fitted_q.price,
            maturity_value,
            oracle_value,
            day_frame,
        ),
        _policy_row(
            day_meta,
            "Selected Linear Fitted-Q",
            fitted_q_selected_eval,
            fitted_q.price if fitted_q_selection.used_candidate else None,
            maturity_value,
            oracle_value,
            day_frame,
            validation_selection=fitted_q_selection.validation_metrics,
        ),
        _policy_row(
            day_meta,
            "Raw Kernel Fitted-Q",
            kernel_q_eval,
            kernel_q.price,
            maturity_value,
            oracle_value,
            day_frame,
        ),
        _policy_row(
            day_meta,
            "Selected Kernel Fitted-Q",
            kernel_q_selected_eval,
            kernel_q.price if kernel_q_selection.used_candidate else None,
            maturity_value,
            oracle_value,
            day_frame,
            validation_selection=kernel_q_selection.validation_metrics,
        ),
        {
            **day_meta,
            "method": "Oracle",
            "exercise_step": oracle_exercise_step,
            "exercise_time": _time_for_step(day_frame, oracle_exercise_step),
            "exercise_price": oracle_exercise_price,
            "gross_pnl": oracle_value,
            "exercise_payoff": oracle_exercise_payoff,
            "execution_cost": oracle_execution_cost,
            "cost_adjusted_pnl": oracle_value - oracle_execution_cost,
            "model_price": np.nan,
            "model_net_pnl": np.nan,
            "pnl_vs_maturity": oracle_value - maturity_value,
            "regret_vs_oracle": 0.0,
            "exercised_early": bool(int(oracle["exercise_step"]) < maturity),
            "validation_used_candidate": np.nan,
            "validation_selected_policy_name": "",
            "validation_mean_delta": np.nan,
            "validation_ci_low": np.nan,
            "validation_ci_high": np.nan,
            "validation_cvar_5_delta": np.nan,
        },
    ]
    return rows


def _policy_row(
    day_meta: dict[str, Any],
    method: str,
    evaluation: Any,
    model_price: float | None,
    maturity_value: float,
    oracle_value: float,
    day_frame: pd.DataFrame,
    validation_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exercise_step = int(evaluation.exercise_steps[0])
    gross_pnl = float(evaluation.path_values[0])
    exercise_price = float(day_frame["close"].iloc[exercise_step])
    exercise_payoff = float(evaluation.exercise_payoffs[0])
    execution_cost = _exercise_execution_cost(exercise_payoff, exercise_price, day_meta)
    model_value = np.nan if model_price is None else float(model_price)
    ci_low, ci_high = (
        validation_selection.get("bootstrap_mean_delta_ci95", [np.nan, np.nan])
        if validation_selection is not None
        else [np.nan, np.nan]
    )
    return {
        **day_meta,
        "method": method,
        "exercise_step": exercise_step,
        "exercise_time": _time_for_step(day_frame, exercise_step),
        "exercise_price": exercise_price,
        "gross_pnl": gross_pnl,
        "exercise_payoff": exercise_payoff,
        "execution_cost": execution_cost,
        "cost_adjusted_pnl": gross_pnl - execution_cost,
        "model_price": model_value,
        "model_net_pnl": np.nan if model_price is None else gross_pnl - model_value,
        "pnl_vs_maturity": gross_pnl - maturity_value,
        "regret_vs_oracle": oracle_value - gross_pnl,
        "exercised_early": bool(exercise_step < day_meta["observations"] - 1),
        "validation_used_candidate": (
            bool(validation_selection["used_candidate"]) if validation_selection is not None else np.nan
        ),
        "validation_selected_policy_name": (
            str(validation_selection["selected_policy_name"]) if validation_selection is not None else ""
        ),
        "validation_mean_delta": (
            float(validation_selection["mean_delta"]) if validation_selection is not None else np.nan
        ),
        "validation_ci_low": float(ci_low),
        "validation_ci_high": float(ci_high),
        "validation_cvar_5_delta": (
            float(validation_selection["cvar_5_delta"]) if validation_selection is not None else np.nan
        ),
    }


def _exercise_execution_cost(exercise_payoff: float, exercise_price: float, day_meta: dict[str, Any]) -> float:
    if exercise_payoff <= 0.0:
        return 0.0
    fee = float(day_meta.get("exercise_fee_per_unit", 0.0))
    slippage = float(day_meta.get("slippage_bps", 0.0)) / 10_000.0 * float(exercise_price)
    return max(fee, 0.0) + max(slippage, 0.0)


def _oracle_american_exercise(prices: np.ndarray, contract: AmericanOptionContract) -> dict[str, float | int]:
    maturity = contract.validate(len(prices))
    discount = float(np.exp(-contract.risk_free_rate * contract.time_step_years))
    payoffs = contract.payoff(np.asarray(prices, dtype=float))
    values = np.full(maturity + 1, -np.inf, dtype=float)
    for step in range(contract.exercise_start_step, maturity + 1):
        if contract.is_exercise_step(step, maturity):
            values[step] = payoffs[step] * np.power(discount, step)
    exercise_step = int(np.argmax(values))
    return {
        "exercise_step": exercise_step,
        "exercise_payoff": float(payoffs[exercise_step]),
        "gross_pnl": float(values[exercise_step]),
    }


def _realized_garch_variance_path(model: GJRGARCHModel, prices: np.ndarray) -> np.ndarray:
    """Build a no-lookahead conditional variance path for realized prices."""

    if model.params is None or model.last_residual is None or model.last_variance is None:
        raise RuntimeError("GJR-GARCH model must be fitted before building realized variance features")
    price_array = np.asarray(prices, dtype=float)
    if price_array.ndim != 1 or price_array.size < 2:
        raise ValueError("prices must be a one-dimensional path with at least two observations")
    if not np.isfinite(price_array).all() or (price_array <= 0.0).any():
        raise ValueError("realized prices must be finite and positive")

    variances = np.full(price_array.shape, np.nan, dtype=float)
    residual = float(model.last_residual)
    variance = float(model.last_variance)
    params = model.params
    for step in range(1, price_array.size):
        leverage = 1.0 if residual < 0.0 else 0.0
        variance = (
            params.omega
            + params.alpha * residual**2
            + params.gamma * leverage * residual**2
            + params.beta * variance
        )
        variance = max(float(variance), model.min_variance)
        variances[step] = variance / params.return_scale**2
        scaled_return = np.log(price_array[step] / price_array[step - 1]) * params.return_scale
        residual = float(scaled_return - params.mu)
    return variances


def _eligible_trade_days(market: pd.DataFrame, config: TraderChoiceBacktestConfig) -> list[Any]:
    dates = []
    with_dates = market.assign(trade_date=market["open_datetime"].dt.date)
    for trade_date, day_frame in with_dates.groupby("trade_date", sort=True):
        if len(day_frame) < config.min_day_observations:
            continue
        first_time = day_frame["open_datetime"].iloc[0]
        available_returns = market.loc[market["open_datetime"] < first_time, "log_return"].dropna()
        if len(available_returns) >= config.min_training_returns:
            dates.append(trade_date)
    return dates[-config.n_backtest_days :]


def _day_frame(market: pd.DataFrame, trade_date: Any) -> pd.DataFrame:
    mask = market["open_datetime"].dt.date == trade_date
    return market.loc[mask].sort_values("open_datetime", kind="mergesort").reset_index(drop=True)


def _infer_step_years(day_frame: pd.DataFrame) -> float:
    times = day_frame["open_datetime"]
    if len(times) < 2:
        return 5.0 / (365.0 * 24.0 * 60.0)
    seconds = np.diff(times.astype("int64").to_numpy()) / 1_000_000_000
    seconds = seconds[np.isfinite(seconds) & (seconds > 0.0)]
    if seconds.size == 0:
        return 5.0 / (365.0 * 24.0 * 60.0)
    return float(np.median(seconds) / (365.0 * 24.0 * 60.0 * 60.0))


def _time_for_step(day_frame: pd.DataFrame, step: int) -> str:
    return pd.Timestamp(day_frame["open_datetime"].iloc[int(step)]).isoformat()


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in results.groupby("method", sort=True):
        ordered = group.sort_values("date", kind="mergesort")
        gross = ordered["gross_pnl"].to_numpy(dtype=float)
        cost_adjusted_series = ordered.get("cost_adjusted_pnl", ordered["gross_pnl"])
        execution_cost_series = ordered.get("execution_cost", pd.Series(0.0, index=ordered.index))
        cost_adjusted = cost_adjusted_series.to_numpy(dtype=float)
        net = ordered["model_net_pnl"].dropna().to_numpy(dtype=float)
        validation_used = ordered["validation_used_candidate"].dropna()
        rows.append(
            {
                "method": method,
                "trade_days": int(len(ordered)),
                "total_gross_pnl": float(np.sum(gross)),
                "mean_gross_pnl": float(np.mean(gross)),
                "median_gross_pnl": float(np.median(gross)),
                "total_execution_cost": float(execution_cost_series.sum()),
                "total_cost_adjusted_pnl": float(np.sum(cost_adjusted)),
                "mean_cost_adjusted_pnl": float(np.mean(cost_adjusted)),
                "win_rate": float(np.mean(gross > 0.0)),
                "total_pnl_vs_maturity": float(ordered["pnl_vs_maturity"].sum()),
                "mean_pnl_vs_maturity": float(ordered["pnl_vs_maturity"].mean()),
                "share_days_beating_maturity": float(np.mean(ordered["pnl_vs_maturity"] > 0.0)),
                "mean_regret_vs_oracle": float(ordered["regret_vs_oracle"].mean()),
                "q95_regret_vs_oracle": float(np.quantile(ordered["regret_vs_oracle"], 0.95)),
                "early_exercise_rate": float(ordered["exercised_early"].mean()),
                "max_drawdown": _max_drawdown(np.cumsum(gross)),
                "worst_day_gross_pnl": float(np.min(gross)),
                "mean_model_price": float(ordered["model_price"].dropna().mean()) if ordered["model_price"].notna().any() else np.nan,
                "total_model_net_pnl": float(np.sum(net)) if net.size else np.nan,
                "mean_model_net_pnl": float(np.mean(net)) if net.size else np.nan,
                "validation_deploy_rate": float(validation_used.mean()) if len(validation_used) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["total_gross_pnl", "method"], ascending=[False, True]).reset_index(drop=True)


def _historical_paired_diagnostics(results: pd.DataFrame, config: TraderChoiceBacktestConfig) -> pd.DataFrame:
    """Summarize realized paired deltas versus the maturity fallback."""

    columns = [
        "method",
        "trade_days",
        "total_gross_pnl",
        "total_delta_vs_maturity",
        "mean_delta_vs_maturity",
        "median_delta_vs_maturity",
        "ci95_low_delta_vs_maturity",
        "ci95_high_delta_vs_maturity",
        "share_days_beating_maturity",
        "q05_delta_vs_maturity",
        "q95_delta_vs_maturity",
        "cvar_5_delta_vs_maturity",
        "total_regret_vs_oracle",
        "mean_regret_vs_oracle",
        "q95_regret_vs_oracle",
        "oracle_capture_rate",
        "cumulative_delta_max_drawdown",
    ]
    pivot = results.pivot(index="date", columns="method", values="gross_pnl").sort_index()
    if "Maturity" not in pivot.columns or "Oracle" not in pivot.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    baseline = pivot["Maturity"]
    oracle = pivot["Oracle"]
    for method in sorted(pivot.columns):
        aligned = pd.concat(
            [
                pivot[method].rename("method_value"),
                baseline.rename("maturity_value"),
                oracle.rename("oracle_value"),
            ],
            axis=1,
        ).dropna()
        if aligned.empty:
            continue
        metrics = paired_policy_metrics(
            aligned["method_value"].to_numpy(dtype=float),
            aligned["maturity_value"].to_numpy(dtype=float),
            name_a=method,
            name_b="Maturity",
            bootstrap_seed=config.bootstrap_seed,
            n_bootstrap=config.n_bootstrap,
        )
        ci_low, ci_high = metrics["bootstrap_mean_delta_ci95"]
        delta = aligned["method_value"].to_numpy(dtype=float) - aligned["maturity_value"].to_numpy(dtype=float)
        regret = aligned["oracle_value"].to_numpy(dtype=float) - aligned["method_value"].to_numpy(dtype=float)
        oracle_total = float(aligned["oracle_value"].sum())
        rows.append(
            {
                "method": method,
                "trade_days": int(len(aligned)),
                "total_gross_pnl": float(aligned["method_value"].sum()),
                "total_delta_vs_maturity": float(delta.sum()),
                "mean_delta_vs_maturity": float(metrics["mean_delta"]),
                "median_delta_vs_maturity": float(metrics["median_delta"]),
                "ci95_low_delta_vs_maturity": float(ci_low),
                "ci95_high_delta_vs_maturity": float(ci_high),
                "share_days_beating_maturity": float(metrics["share_delta_positive"]),
                "q05_delta_vs_maturity": float(metrics["q05_delta"]),
                "q95_delta_vs_maturity": float(metrics["q95_delta"]),
                "cvar_5_delta_vs_maturity": float(metrics["cvar_5_delta"]),
                "total_regret_vs_oracle": float(regret.sum()),
                "mean_regret_vs_oracle": float(np.mean(regret)),
                "q95_regret_vs_oracle": float(np.quantile(regret, 0.95)),
                "oracle_capture_rate": float(aligned["method_value"].sum() / oracle_total) if oracle_total else np.nan,
                "cumulative_delta_max_drawdown": _max_drawdown(np.cumsum(delta)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["total_delta_vs_maturity", "method"], ascending=[False, True]
    ).reset_index(drop=True)


def _validation_gate_audit(results: pd.DataFrame) -> pd.DataFrame:
    """Compare validation gate decisions with realized candidate deltas."""

    columns = [
        "candidate_method",
        "selected_method",
        "trade_days",
        "deployed_days",
        "deploy_rate",
        "raw_total_delta_vs_maturity",
        "selected_total_delta_vs_maturity",
        "validation_saved_pnl_vs_raw",
        "raw_days_beating_maturity",
        "rejected_good_days",
        "rejected_bad_days",
        "deployed_bad_days",
        "screen_accuracy",
        "mean_validation_delta",
        "mean_validation_ci_low",
        "min_validation_ci_low",
        "mean_validation_cvar_5_delta",
        "realized_delta_when_rejected",
        "realized_delta_when_deployed",
    ]
    method_pairs = {
        "Raw LSMC": "Selected LSMC",
        "Raw Linear Fitted-Q": "Selected Linear Fitted-Q",
        "Raw Kernel Fitted-Q": "Selected Kernel Fitted-Q",
    }
    rows: list[dict[str, Any]] = []
    for raw_method, selected_method in method_pairs.items():
        raw = results.loc[results["method"] == raw_method, ["date", "pnl_vs_maturity"]].rename(
            columns={"pnl_vs_maturity": "raw_delta_vs_maturity"}
        )
        selected = results.loc[
            results["method"] == selected_method,
            [
                "date",
                "pnl_vs_maturity",
                "validation_used_candidate",
                "validation_mean_delta",
                "validation_ci_low",
                "validation_cvar_5_delta",
            ],
        ].rename(columns={"pnl_vs_maturity": "selected_delta_vs_maturity"})
        merged = raw.merge(selected, on="date", how="inner")
        if merged.empty:
            continue
        deployed = merged["validation_used_candidate"].map(lambda value: bool(value) if pd.notna(value) else False)
        raw_delta = merged["raw_delta_vs_maturity"].to_numpy(dtype=float)
        selected_delta = merged["selected_delta_vs_maturity"].to_numpy(dtype=float)
        raw_good = raw_delta > 0.0
        screen_correct = (deployed.to_numpy() & raw_good) | (~deployed.to_numpy() & ~raw_good)
        rows.append(
            {
                "candidate_method": raw_method,
                "selected_method": selected_method,
                "trade_days": int(len(merged)),
                "deployed_days": int(deployed.sum()),
                "deploy_rate": float(deployed.mean()),
                "raw_total_delta_vs_maturity": float(np.sum(raw_delta)),
                "selected_total_delta_vs_maturity": float(np.sum(selected_delta)),
                "validation_saved_pnl_vs_raw": float(np.sum(selected_delta) - np.sum(raw_delta)),
                "raw_days_beating_maturity": int(np.sum(raw_good)),
                "rejected_good_days": int(np.sum((~deployed.to_numpy()) & raw_good)),
                "rejected_bad_days": int(np.sum((~deployed.to_numpy()) & ~raw_good)),
                "deployed_bad_days": int(np.sum(deployed.to_numpy() & ~raw_good)),
                "screen_accuracy": float(np.mean(screen_correct)),
                "mean_validation_delta": float(merged["validation_mean_delta"].mean()),
                "mean_validation_ci_low": float(merged["validation_ci_low"].mean()),
                "min_validation_ci_low": float(merged["validation_ci_low"].min()),
                "mean_validation_cvar_5_delta": float(merged["validation_cvar_5_delta"].mean()),
                "realized_delta_when_rejected": float(raw_delta[~deployed.to_numpy()].sum())
                if np.any(~deployed.to_numpy())
                else np.nan,
                "realized_delta_when_deployed": float(raw_delta[deployed.to_numpy()].sum())
                if np.any(deployed.to_numpy())
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _model_price_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    """Treat each raw model value as a hypothetical premium and audit realized cashflows."""

    columns = [
        "method",
        "trade_days",
        "total_model_price",
        "total_realized_gross_pnl",
        "total_model_net_pnl",
        "mean_model_price",
        "mean_realized_gross_pnl",
        "mean_model_error_realized_minus_model",
        "median_model_error_realized_minus_model",
        "overvalued_day_rate",
        "total_model_shortfall",
        "total_model_surplus",
        "model_price_to_realized_pnl_ratio",
        "model_price_to_maturity_pnl_ratio",
    ]
    frame = results.loc[results["model_price"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    maturity = results.loc[results["method"] == "Maturity", ["date", "gross_pnl"]].rename(
        columns={"gross_pnl": "maturity_gross_pnl"}
    )
    frame = frame.merge(maturity, on="date", how="left")
    rows: list[dict[str, Any]] = []
    for method, group in frame.groupby("method", sort=True):
        model_price = group["model_price"].to_numpy(dtype=float)
        realized = group["gross_pnl"].to_numpy(dtype=float)
        maturity_value = group["maturity_gross_pnl"].to_numpy(dtype=float)
        error = realized - model_price
        shortfall = np.maximum(model_price - realized, 0.0)
        surplus = np.maximum(realized - model_price, 0.0)
        realized_total = float(np.sum(realized))
        maturity_total = float(np.sum(maturity_value))
        rows.append(
            {
                "method": method,
                "trade_days": int(len(group)),
                "total_model_price": float(np.sum(model_price)),
                "total_realized_gross_pnl": realized_total,
                "total_model_net_pnl": float(np.sum(error)),
                "mean_model_price": float(np.mean(model_price)),
                "mean_realized_gross_pnl": float(np.mean(realized)),
                "mean_model_error_realized_minus_model": float(np.mean(error)),
                "median_model_error_realized_minus_model": float(np.median(error)),
                "overvalued_day_rate": float(np.mean(model_price > realized)),
                "total_model_shortfall": float(np.sum(shortfall)),
                "total_model_surplus": float(np.sum(surplus)),
                "model_price_to_realized_pnl_ratio": float(np.sum(model_price) / realized_total)
                if realized_total
                else np.nan,
                "model_price_to_maturity_pnl_ratio": float(np.sum(model_price) / maturity_total)
                if maturity_total
                else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["total_model_net_pnl", "method"], ascending=[False, True]
    ).reset_index(drop=True)


def _cost_stress_summary(results: pd.DataFrame, config: TraderChoiceBacktestConfig) -> pd.DataFrame:
    """Revalue realized exercise cashflows under simple fee/slippage stress assumptions."""

    columns = [
        "scenario",
        "exercise_fee_per_unit",
        "slippage_bps",
        "method",
        "trade_days",
        "total_execution_cost",
        "total_cost_adjusted_pnl",
        "mean_cost_adjusted_pnl",
        "total_pnl_vs_maturity",
        "share_days_beating_maturity",
        "worst_day_cost_adjusted_pnl",
    ]
    scenario_rows: list[dict[str, Any]] = []
    for fee, slippage_bps in _cost_stress_grid(config):
        data = results.copy()
        data["stress_execution_cost"] = _stress_execution_cost(
            data["exercise_payoff"].to_numpy(dtype=float),
            data["exercise_price"].to_numpy(dtype=float),
            fee,
            slippage_bps,
        )
        data["stress_cost_adjusted_pnl"] = data["gross_pnl"] - data["stress_execution_cost"]
        maturity = data.loc[data["method"] == "Maturity", ["date", "stress_cost_adjusted_pnl"]].rename(
            columns={"stress_cost_adjusted_pnl": "maturity_stress_cost_adjusted_pnl"}
        )
        data = data.merge(maturity, on="date", how="left")
        data["stress_pnl_vs_maturity"] = (
            data["stress_cost_adjusted_pnl"] - data["maturity_stress_cost_adjusted_pnl"]
        )
        scenario = _cost_stress_name(fee, slippage_bps, config)
        for method, group in data.groupby("method", sort=True):
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "exercise_fee_per_unit": float(fee),
                    "slippage_bps": float(slippage_bps),
                    "method": method,
                    "trade_days": int(len(group)),
                    "total_execution_cost": float(group["stress_execution_cost"].sum()),
                    "total_cost_adjusted_pnl": float(group["stress_cost_adjusted_pnl"].sum()),
                    "mean_cost_adjusted_pnl": float(group["stress_cost_adjusted_pnl"].mean()),
                    "total_pnl_vs_maturity": float(group["stress_pnl_vs_maturity"].sum()),
                    "share_days_beating_maturity": float(np.mean(group["stress_pnl_vs_maturity"] > 0.0)),
                    "worst_day_cost_adjusted_pnl": float(group["stress_cost_adjusted_pnl"].min()),
                }
            )
    return pd.DataFrame(scenario_rows, columns=columns)


def _cost_stress_frontier(cost_stress: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "exercise_fee_per_unit",
        "slippage_bps",
        "best_non_oracle_method",
        "best_non_oracle_cost_adjusted_pnl",
        "best_non_oracle_delta_vs_maturity",
        "best_raw_method",
        "best_raw_delta_vs_maturity",
        "maturity_cost_adjusted_pnl",
    ]
    if cost_stress.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for scenario, group in cost_stress.groupby("scenario", sort=False):
        non_oracle = group.loc[group["method"] != "Oracle"].sort_values(
            ["total_cost_adjusted_pnl", "method"], ascending=[False, True]
        )
        raw = group.loc[group["method"].str.startswith("Raw ")].sort_values(
            ["total_cost_adjusted_pnl", "method"], ascending=[False, True]
        )
        maturity = group.loc[group["method"] == "Maturity"]
        if non_oracle.empty or maturity.empty:
            continue
        best = non_oracle.iloc[0]
        best_raw = raw.iloc[0] if not raw.empty else None
        rows.append(
            {
                "scenario": scenario,
                "exercise_fee_per_unit": float(best["exercise_fee_per_unit"]),
                "slippage_bps": float(best["slippage_bps"]),
                "best_non_oracle_method": str(best["method"]),
                "best_non_oracle_cost_adjusted_pnl": float(best["total_cost_adjusted_pnl"]),
                "best_non_oracle_delta_vs_maturity": float(best["total_pnl_vs_maturity"]),
                "best_raw_method": str(best_raw["method"]) if best_raw is not None else "",
                "best_raw_delta_vs_maturity": float(best_raw["total_pnl_vs_maturity"])
                if best_raw is not None
                else np.nan,
                "maturity_cost_adjusted_pnl": float(maturity.iloc[0]["total_cost_adjusted_pnl"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _market_regime_summary(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "terminal_regime",
        "trade_days",
        "mean_terminal_return_pct",
        "mean_downside_excursion_pct",
        "total_gross_pnl",
        "total_pnl_vs_maturity",
        "mean_regret_vs_oracle",
        "early_exercise_rate",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    data = results.copy()
    data["terminal_regime"] = np.where(data["terminal_return_pct"] < 0.0, "down close", "up or flat close")
    rows: list[dict[str, Any]] = []
    for (method, regime), group in data.groupby(["method", "terminal_regime"], sort=True):
        rows.append(
            {
                "method": method,
                "terminal_regime": regime,
                "trade_days": int(len(group)),
                "mean_terminal_return_pct": float(group["terminal_return_pct"].mean()),
                "mean_downside_excursion_pct": float(group["downside_excursion_pct"].mean()),
                "total_gross_pnl": float(group["gross_pnl"].sum()),
                "total_pnl_vs_maturity": float(group["pnl_vs_maturity"].sum()),
                "mean_regret_vs_oracle": float(group["regret_vs_oracle"].mean()),
                "early_exercise_rate": float(group["exercised_early"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _worst_day_diagnostics(results: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    columns = [
        "date",
        "method",
        "gross_pnl",
        "pnl_vs_maturity",
        "regret_vs_oracle",
        "exercise_step",
        "exercise_time",
        "exercise_price",
        "start_price",
        "end_price",
        "min_price",
        "max_price",
        "terminal_return_pct",
        "downside_excursion_pct",
        "validation_mean_delta",
        "validation_ci_low",
        "validation_cvar_5_delta",
    ]
    frame = results.loc[results["method"] != "Oracle", columns].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.sort_values(
        ["pnl_vs_maturity", "regret_vs_oracle", "method"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(top_n).reset_index(drop=True)


def _valuation_assessment(
    summary: pd.DataFrame,
    paired_diagnostics: pd.DataFrame,
    validation_audit: pd.DataFrame,
    model_price_diagnostics: pd.DataFrame,
    cost_stress_frontier: pd.DataFrame,
) -> dict[str, Any]:
    non_oracle = summary.loc[summary["method"] != "Oracle"].sort_values(
        ["total_cost_adjusted_pnl", "method"], ascending=[False, True]
    )
    raw = summary.loc[summary["method"].str.startswith("Raw ")].sort_values(
        ["total_cost_adjusted_pnl", "method"], ascending=[False, True]
    )
    best_non_oracle = non_oracle.iloc[0].to_dict() if not non_oracle.empty else {}
    best_raw = raw.iloc[0].to_dict() if not raw.empty else {}
    raw_positive = paired_diagnostics.loc[
        paired_diagnostics["method"].str.startswith("Raw ")
        & (paired_diagnostics["total_delta_vs_maturity"] > 0.0)
    ]
    selected = summary.loc[summary["method"].str.startswith("Selected ")]
    deploy_rates = {
        str(row["method"]): row["validation_deploy_rate"]
        for _, row in selected.iterrows()
        if pd.notna(row["validation_deploy_rate"])
    }
    return {
        "best_non_oracle_method": best_non_oracle.get("method", ""),
        "best_non_oracle_cost_adjusted_pnl": best_non_oracle.get("total_cost_adjusted_pnl", np.nan),
        "best_raw_method": best_raw.get("method", ""),
        "best_raw_delta_vs_maturity": best_raw.get("total_pnl_vs_maturity", np.nan),
        "raw_methods_with_positive_total_delta_vs_maturity": raw_positive["method"].tolist(),
        "validation_total_saved_pnl_vs_raw": float(validation_audit["validation_saved_pnl_vs_raw"].sum())
        if not validation_audit.empty
        else np.nan,
        "selected_deploy_rates": deploy_rates,
        "best_model_net_method": model_price_diagnostics.iloc[0]["method"]
        if not model_price_diagnostics.empty
        else "",
        "best_model_net_pnl": float(model_price_diagnostics.iloc[0]["total_model_net_pnl"])
        if not model_price_diagnostics.empty
        else np.nan,
        "cost_stress_best_methods": cost_stress_frontier[
            ["scenario", "best_non_oracle_method", "best_raw_method", "best_raw_delta_vs_maturity"]
        ].to_dict(orient="records")
        if not cost_stress_frontier.empty
        else [],
    }


def _cost_stress_grid(config: TraderChoiceBacktestConfig) -> list[tuple[float, float]]:
    fees = {max(float(value), 0.0) for value in config.cost_stress_exercise_fee_per_unit}
    slippages = {max(float(value), 0.0) for value in config.cost_stress_slippage_bps}
    fees.add(max(float(config.exercise_fee_per_unit), 0.0))
    slippages.add(max(float(config.slippage_bps), 0.0))
    return [(fee, slippage) for fee in sorted(fees) for slippage in sorted(slippages)]


def _cost_stress_name(fee: float, slippage_bps: float, config: TraderChoiceBacktestConfig) -> str:
    if np.isclose(fee, config.exercise_fee_per_unit) and np.isclose(slippage_bps, config.slippage_bps):
        return "configured"
    return f"fee_{fee:.4g}_slippage_{slippage_bps:.4g}bps"


def _stress_execution_cost(
    exercise_payoff: np.ndarray,
    exercise_price: np.ndarray,
    exercise_fee_per_unit: float,
    slippage_bps: float,
) -> np.ndarray:
    payoff = np.asarray(exercise_payoff, dtype=float)
    price = np.asarray(exercise_price, dtype=float)
    if payoff.shape != price.shape:
        raise ValueError("exercise payoff and price arrays must have the same shape")
    fee = max(float(exercise_fee_per_unit), 0.0)
    slippage = max(float(slippage_bps), 0.0) / 10_000.0 * price
    return np.where(payoff > 0.0, fee + slippage, 0.0)


def _max_drawdown(cumulative: np.ndarray) -> float:
    if cumulative.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - running_max))


def _create_plots(output_dir: Path, results: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    pivot = results.pivot(index="date", columns="method", values="gross_pnl").sort_index()
    cumulative = pivot.cumsum()
    ax = cumulative.plot(figsize=(10.5, 5.2), linewidth=1.5)
    ax.set_title("Trader-choice backtest cumulative gross PnL")
    ax.set_xlabel("Trade date")
    ax.set_ylabel("Cumulative gross PnL per 1 unit notional")
    ax.legend(fontsize=8)
    plt.tight_layout()
    paths["cumulative_gross_pnl"] = output_dir / "cumulative_gross_pnl.png"
    plt.savefig(paths["cumulative_gross_pnl"], dpi=140)
    plt.close()

    methods = [
        "Selected LSMC",
        "Selected Linear Fitted-Q",
        "Selected Kernel Fitted-Q",
        "Raw LSMC",
        "Raw Linear Fitted-Q",
        "Raw Kernel Fitted-Q",
        "Maturity",
        "Oracle",
    ]
    available = [method for method in methods if method in pivot.columns]
    ax = pivot[available].plot(kind="bar", figsize=(11.5, 5.5))
    ax.set_title("Daily realized gross PnL by policy")
    ax.set_xlabel("Trade date")
    ax.set_ylabel("Gross PnL per 1 unit notional")
    ax.legend(fontsize=8)
    plt.tight_layout()
    paths["daily_gross_pnl"] = output_dir / "daily_gross_pnl.png"
    plt.savefig(paths["daily_gross_pnl"], dpi=140)
    plt.close()

    if "Maturity" in pivot.columns:
        delta_methods = [
            method
            for method in [
                "Oracle",
                "Selected LSMC",
                "Selected Linear Fitted-Q",
                "Selected Kernel Fitted-Q",
                "Raw LSMC",
                "Raw Linear Fitted-Q",
                "Raw Kernel Fitted-Q",
                "Immediate intrinsic",
            ]
            if method in pivot.columns
        ]
        if delta_methods:
            cumulative_delta = pivot[delta_methods].subtract(pivot["Maturity"], axis=0).cumsum()
            ax = cumulative_delta.plot(figsize=(10.5, 5.2), linewidth=1.5)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title("Cumulative realized PnL delta versus maturity exercise")
            ax.set_xlabel("Trade date")
            ax.set_ylabel("Cumulative delta per 1 unit notional")
            ax.legend(fontsize=8)
            plt.tight_layout()
            paths["cumulative_delta_vs_maturity"] = output_dir / "cumulative_delta_vs_maturity.png"
            plt.savefig(paths["cumulative_delta_vs_maturity"], dpi=140)
            plt.close()

    model_rows = results.loc[results["model_price"].notna()].copy()
    if not model_rows.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        for method, group in model_rows.groupby("method", sort=True):
            ax.scatter(group["model_price"], group["gross_pnl"], label=method, s=38, alpha=0.8)
        low = float(min(model_rows["model_price"].min(), model_rows["gross_pnl"].min(), 0.0))
        high = float(max(model_rows["model_price"].max(), model_rows["gross_pnl"].max()))
        ax.plot([low, high], [low, high], color="black", linewidth=0.9, linestyle="--")
        ax.set_title("Model value versus realized exercise cashflow")
        ax.set_xlabel("Model value per 1 unit notional")
        ax.set_ylabel("Realized gross PnL per 1 unit notional")
        ax.legend(fontsize=8)
        plt.tight_layout()
        paths["model_price_vs_realized"] = output_dir / "model_price_vs_realized.png"
        plt.savefig(paths["model_price_vs_realized"], dpi=140)
        plt.close()

    return paths


def _scenario_text(config: TraderChoiceBacktestConfig) -> str:
    return (
        "A trader owns one daily ATM American put exercise right on the TTF FRONT series. "
        "At the first observed print of each selected trade day, strike is set to the current close. "
        "Only historical returns before that print are used to fit GJR-GARCH and simulate training paths. "
        "LSMC, linear Fitted-Q, and kernel Fitted-Q policies are frozen, validated on independent "
        "same-day simulation paths, then replayed on the realized intraday path as raw candidates and "
        "validation-selected deployments. Gross PnL is the realized option exercise cashflow per 1 unit notional. "
        "Cost-adjusted PnL subtracts the configured exercise fee and slippage assumptions. "
        "Model-net PnL subtracts the method's own training-path value and is only a paper model-price diagnostic, "
        "because the database contains no traded option premium."
    )


def _render_report(metrics: dict[str, Any]) -> str:
    summary = metrics["summary"]
    plots = metrics["plots"]
    paired = metrics.get("historical_paired_diagnostics", [])
    validation_audit = metrics.get("validation_gate_audit", [])
    model_price = metrics.get("model_price_diagnostics", [])
    cost_stress = metrics.get("cost_stress_frontier", [])
    regime = metrics.get("market_regime_summary", [])
    worst_days = metrics.get("worst_day_diagnostics", [])
    assessment = metrics.get("valuation_assessment", {})
    paired_methods = [
        "Oracle",
        "Maturity",
        "Selected LSMC",
        "Selected Linear Fitted-Q",
        "Selected Kernel Fitted-Q",
        "Raw LSMC",
        "Raw Linear Fitted-Q",
        "Raw Kernel Fitted-Q",
        "Immediate intrinsic",
    ]
    paired_lookup = {row["method"]: row for row in paired}
    paired_rows = [paired_lookup[method] for method in paired_methods if method in paired_lookup]
    regime_methods = {
        "Maturity",
        "Selected LSMC",
        "Selected Linear Fitted-Q",
        "Selected Kernel Fitted-Q",
        "Raw LSMC",
        "Raw Linear Fitted-Q",
        "Raw Kernel Fitted-Q",
    }
    lines = [
        f"# Trader Choice Backtest: {metrics['config']['symbol']}",
        "",
        "## Scenario",
        "",
        metrics["scenario"],
        "",
        "## Setup",
        "",
        f"- Data source: `{metrics['config']['database_path']}`",
        f"- Trade period: `{metrics['backtest_period']['first_trade_date']}` to `{metrics['backtest_period']['last_trade_date']}`",
        f"- Trade days: `{metrics['backtest_period']['trade_days']}`",
        f"- Training paths per day: `{metrics['config']['n_training_paths']}`",
        f"- Validation paths per day: `{metrics['config']['n_validation_paths']}`",
        f"- Training lookback returns: `{metrics['config']['training_lookback_returns']}`",
        f"- Exercise fee per unit: `{metrics['config']['exercise_fee_per_unit']}`",
        f"- Slippage: `{metrics['config']['slippage_bps']}` bps",
        f"- Kernel RFF features: `{metrics['config']['kernel_rff_features']}`",
        (
            f"- Deployment gate: validation CI-low > `{metrics['config']['deployment_min_ci_low']}`"
            f"{_deployment_cvar_text(metrics['config']['deployment_min_cvar_5_delta'])}"
        ),
        "",
        "## Results",
        "",
        _markdown_table(
            [
                "method",
                "total gross PnL",
                "cost-adjusted PnL",
                "mean/day",
                "vs maturity",
                "days > maturity",
                "regret vs oracle/day",
                "p95 regret",
                "early exercise",
                "deploy rate",
                "model net PnL",
            ],
            [
                [
                    row["method"],
                    _fmt(row["total_gross_pnl"]),
                    _fmt(row["total_cost_adjusted_pnl"]),
                    _fmt(row["mean_gross_pnl"]),
                    _fmt(row["total_pnl_vs_maturity"]),
                    _fmt(row["share_days_beating_maturity"]),
                    _fmt(row["mean_regret_vs_oracle"]),
                    _fmt(row["q95_regret_vs_oracle"]),
                    _fmt(row["early_exercise_rate"]),
                    _fmt(row.get("validation_deploy_rate")),
                    _fmt(row.get("total_model_net_pnl")),
                ]
                for row in summary
            ],
        ),
        "",
        (
            "Gross PnL is the realized exercise cashflow before any option premium. "
            "Selected methods are global deployment wrappers chosen before the realized path is replayed. "
            "Model-net PnL subtracts each raw method's own model value and should not be read as a market-observed premium PnL."
        ),
        "",
        "## Valuation Read",
        "",
        (
            f"Best non-oracle realized strategy: `{assessment.get('best_non_oracle_method', 'n/a')}` "
            f"with cost-adjusted PnL `{_fmt(assessment.get('best_non_oracle_cost_adjusted_pnl'))}`. "
            f"Best raw learned candidate: `{assessment.get('best_raw_method', 'n/a')}` with total delta "
            f"versus maturity `{_fmt(assessment.get('best_raw_delta_vs_maturity'))}`."
        ),
        "",
        (
            "Raw learned methods with positive total edge versus maturity: "
            f"`{', '.join(assessment.get('raw_methods_with_positive_total_delta_vs_maturity', [])) or 'none'}`. "
            f"The validation gate changed realized PnL versus always deploying raw candidates by "
            f"`{_fmt(assessment.get('validation_total_saved_pnl_vs_raw'))}` across the audited candidate set."
        ),
        "",
        (
            f"Best paper model-net diagnostic: `{assessment.get('best_model_net_method', 'n/a')}` "
            f"with realized cashflow minus own model value `{_fmt(assessment.get('best_model_net_pnl'))}`. "
            "Because no option quotes are present, this is a valuation sanity check, not live trade PnL."
        ),
        "",
        "## Paired Historical Edge",
        "",
        _markdown_table(
            [
                "method",
                "total delta vs maturity",
                "mean delta/day",
                "mean CI95",
                "days > maturity",
                "CVaR5 delta",
                "oracle capture",
                "delta drawdown",
            ],
            [
                [
                    row["method"],
                    _fmt(row["total_delta_vs_maturity"]),
                    _fmt(row["mean_delta_vs_maturity"]),
                    f"[{_fmt(row['ci95_low_delta_vs_maturity'])}, {_fmt(row['ci95_high_delta_vs_maturity'])}]",
                    _fmt(row["share_days_beating_maturity"]),
                    _fmt(row["cvar_5_delta_vs_maturity"]),
                    _fmt(row["oracle_capture_rate"]),
                    _fmt(row["cumulative_delta_max_drawdown"]),
                ]
                for row in paired_rows
            ],
        ),
        "",
        (
            "This table is the main valuation-strategy read: it treats maturity exercise as the deployable "
            "fallback and measures realized paired edge, downside tail, and oracle capture on the same days."
        ),
        "",
        "## Validation Gate Audit",
        "",
        _markdown_table(
            [
                "candidate",
                "deployed days",
                "raw delta",
                "selected delta",
                "saved vs raw",
                "rejected good",
                "rejected bad",
                "screen accuracy",
                "mean CI-low",
            ],
            [
                [
                    row["candidate_method"],
                    f"{int(row['deployed_days'])}/{int(row['trade_days'])}",
                    _fmt(row["raw_total_delta_vs_maturity"]),
                    _fmt(row["selected_total_delta_vs_maturity"]),
                    _fmt(row["validation_saved_pnl_vs_raw"]),
                    str(int(row["rejected_good_days"])),
                    str(int(row["rejected_bad_days"])),
                    _fmt(row["screen_accuracy"]),
                    _fmt(row["mean_validation_ci_low"]),
                ]
                for row in validation_audit
            ],
        ),
        "",
        "## Model Price Diagnostics",
        "",
        _markdown_table(
            [
                "method",
                "model value",
                "realized cashflow",
                "realized - model",
                "overvalued days",
                "shortfall",
                "model/realized",
            ],
            [
                [
                    row["method"],
                    _fmt(row["total_model_price"]),
                    _fmt(row["total_realized_gross_pnl"]),
                    _fmt(row["total_model_net_pnl"]),
                    _fmt(row["overvalued_day_rate"]),
                    _fmt(row["total_model_shortfall"]),
                    _fmt(row["model_price_to_realized_pnl_ratio"]),
                ]
                for row in model_price
            ],
        ),
        "",
        "## Cost Stress Frontier",
        "",
        _markdown_table(
            [
                "scenario",
                "fee",
                "slippage bps",
                "best non-oracle",
                "best PnL",
                "best raw",
                "best raw delta",
                "maturity PnL",
            ],
            [
                [
                    row["scenario"],
                    _fmt(row["exercise_fee_per_unit"]),
                    _fmt(row["slippage_bps"]),
                    row["best_non_oracle_method"],
                    _fmt(row["best_non_oracle_cost_adjusted_pnl"]),
                    row["best_raw_method"],
                    _fmt(row["best_raw_delta_vs_maturity"]),
                    _fmt(row["maturity_cost_adjusted_pnl"]),
                ]
                for row in cost_stress
            ],
        ),
        "",
        "## Market Regime Attribution",
        "",
        _markdown_table(
            [
                "method",
                "regime",
                "days",
                "mean close return",
                "mean downside excursion",
                "total vs maturity",
                "regret/day",
                "early exercise",
            ],
            [
                [
                    row["method"],
                    row["terminal_regime"],
                    str(int(row["trade_days"])),
                    _fmt(row["mean_terminal_return_pct"]),
                    _fmt(row["mean_downside_excursion_pct"]),
                    _fmt(row["total_pnl_vs_maturity"]),
                    _fmt(row["mean_regret_vs_oracle"]),
                    _fmt(row["early_exercise_rate"]),
                ]
                for row in regime
                if row["method"] in regime_methods
            ],
        ),
        "",
        "## Worst Decision Days",
        "",
        _markdown_table(
            [
                "date",
                "method",
                "vs maturity",
                "regret",
                "exercise time",
                "exercise price",
                "end price",
                "downside excursion",
                "validation CI-low",
            ],
            [
                [
                    row["date"],
                    row["method"],
                    _fmt(row["pnl_vs_maturity"]),
                    _fmt(row["regret_vs_oracle"]),
                    row["exercise_time"],
                    _fmt(row["exercise_price"]),
                    _fmt(row["end_price"]),
                    _fmt(row["downside_excursion_pct"]),
                    _fmt(row["validation_ci_low"]),
                ]
                for row in worst_days[:8]
            ],
        ),
        "",
        f"![Cumulative gross PnL]({Path(plots['cumulative_gross_pnl']).name})",
        "",
        f"![Daily gross PnL]({Path(plots['daily_gross_pnl']).name})",
        "",
        *(
            [
                f"![Cumulative delta versus maturity]({Path(plots['cumulative_delta_vs_maturity']).name})",
                "",
            ]
            if "cumulative_delta_vs_maturity" in plots
            else []
        ),
        *(
            [
                f"![Model value versus realized cashflow]({Path(plots['model_price_vs_realized']).name})",
                "",
            ]
            if "model_price_vs_realized" in plots
            else []
        ),
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python -m lsmc_rl.analysis.trader_choice_backtest --output-dir outputs/trader_choice_backtest_front",
        "```",
        "",
        (
            "Detailed daily rows are stored in `daily_policy_results.csv`; aggregate metrics are stored in "
            "`metrics.json`. Additional diagnostics are written to `historical_paired_diagnostics.csv`, "
            "`validation_gate_audit.csv`, `model_price_diagnostics.csv`, `cost_stress_summary.csv`, "
            "`market_regime_summary.csv`, and `worst_day_diagnostics.csv`."
        ),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    if abs(numeric) >= 1000 or (abs(numeric) < 0.001 and numeric != 0.0):
        return f"{numeric:.6e}"
    return f"{numeric:.6f}"


def _deployment_cvar_text(value: Any) -> str:
    if value is None:
        return ""
    return f" and validation CVaR5 >= `{_fmt(value)}`"


def _config_dict(config: TraderChoiceBacktestConfig) -> dict[str, Any]:
    data = dict(config.__dict__)
    data["database_path"] = str(config.database_path)
    data["output_dir"] = str(config.output_dir)
    return data


def _ensure_allowed_artifact_dir(path: Path) -> None:
    normalized = path.resolve()
    allowed_roots = [(Path.cwd() / "outputs").resolve(), (Path.cwd() / "runs").resolve()]
    if not any(normalized == root or root in normalized.parents for root in allowed_roots):
        raise ValueError("Backtest artifacts must be written under outputs/ or runs/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a historical trader-choice backtest for LSMC and RL policies.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--output-dir", default="outputs/trader_choice_backtest_front")
    parser.add_argument("--exercise-fee-per-unit", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--n-backtest-days", type=int, default=12)
    parser.add_argument("--min-day-observations", type=int, default=30)
    parser.add_argument("--training-lookback-returns", type=int, default=1500)
    parser.add_argument("--min-training-returns", type=int, default=300)
    parser.add_argument("--n-training-paths", type=int, default=300)
    parser.add_argument("--n-validation-paths", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--validation-seed-offset", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260525)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--deployment-min-ci-low", type=float, default=0.0)
    parser.add_argument("--deployment-min-cvar-5-delta", type=float, default=0.0)
    parser.add_argument("--kernel-rff-features", type=int, default=64)
    parser.add_argument("--cost-stress-slippage-bps", type=float, nargs="*", default=None)
    parser.add_argument("--cost-stress-exercise-fee-per-unit", type=float, nargs="*", default=None)
    args = parser.parse_args(argv)

    config = TraderChoiceBacktestConfig(
        database_path=Path(args.database_path).resolve(),
        symbol=args.symbol,
        interval=args.interval,
        output_dir=Path(args.output_dir).resolve(),
        exercise_fee_per_unit=args.exercise_fee_per_unit,
        slippage_bps=args.slippage_bps,
        n_backtest_days=args.n_backtest_days,
        min_day_observations=args.min_day_observations,
        training_lookback_returns=args.training_lookback_returns,
        min_training_returns=args.min_training_returns,
        n_training_paths=args.n_training_paths,
        n_validation_paths=args.n_validation_paths,
        seed=args.seed,
        validation_seed_offset=args.validation_seed_offset,
        bootstrap_seed=args.bootstrap_seed,
        n_bootstrap=args.n_bootstrap,
        deployment_min_ci_low=args.deployment_min_ci_low,
        deployment_min_cvar_5_delta=args.deployment_min_cvar_5_delta,
        kernel_rff_features=args.kernel_rff_features,
        cost_stress_slippage_bps=(
            tuple(args.cost_stress_slippage_bps)
            if args.cost_stress_slippage_bps is not None
            else TraderChoiceBacktestConfig.cost_stress_slippage_bps
        ),
        cost_stress_exercise_fee_per_unit=(
            tuple(args.cost_stress_exercise_fee_per_unit)
            if args.cost_stress_exercise_fee_per_unit is not None
            else TraderChoiceBacktestConfig.cost_stress_exercise_fee_per_unit
        ),
    )
    metrics = run_backtest(config)
    print(f"Wrote report: {config.output_dir / 'README.md'}")
    print(f"Wrote metrics: {config.output_dir / 'metrics.json'}")
    for row in metrics["summary"]:
        print(
            f"{row['method']}: total_gross_pnl={_fmt(row['total_gross_pnl'])}, "
            f"vs_maturity={_fmt(row['total_pnl_vs_maturity'])}, "
            f"model_net={_fmt(row.get('total_model_net_pnl'))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
