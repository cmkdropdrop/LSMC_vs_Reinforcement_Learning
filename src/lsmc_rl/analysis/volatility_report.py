"""Generate diagnostics for volatility models and Monte-Carlo paths.

The report is intentionally descriptive. It estimates how well the first
GJR-GARCH and HAR-RV building blocks reproduce selected historical features,
without claiming that either model is a complete market model.
"""

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
from scipy import stats

from lsmc_rl.data import DataQualityReport, load_market_data
from lsmc_rl.volatility import GJRGARCHModel, HARRVModel


@dataclass(frozen=True)
class ReturnSplit:
    train_returns: pd.Series
    test_returns: pd.Series
    train_market: pd.DataFrame
    test_market: pd.DataFrame
    split_time: pd.Timestamp
    start_price: float


@dataclass(frozen=True)
class VolatilityReportConfig:
    database_path: Path
    symbol: str = "FRONT"
    interval: str = "5m"
    train_fraction: float = 0.70
    n_paths: int = 1000
    seed: int = 20260519
    output_dir: Path = Path("outputs/volatility_report_front")
    steps_per_day: int = 288


def temporal_return_split(market: pd.DataFrame, train_fraction: float) -> ReturnSplit:
    """Split close-to-close returns by time, preserving the price anchor."""

    if not 0.2 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.2 and 0.9")
    returns = market["log_return"].dropna()
    split_count = int(len(returns) * train_fraction)
    if split_count < 30 or len(returns) - split_count < 10:
        raise ValueError("Not enough returns for the requested train/test split")

    train_returns = returns.iloc[:split_count]
    test_returns = returns.iloc[split_count:]
    split_pos = int(train_returns.index[-1])
    train_market = market.iloc[: split_pos + 1].copy()
    test_market = market.loc[test_returns.index].copy()

    return ReturnSplit(
        train_returns=train_returns,
        test_returns=test_returns,
        train_market=train_market,
        test_market=test_market,
        split_time=pd.Timestamp(market.loc[split_pos, "open_datetime"]),
        start_price=float(market.loc[split_pos, "close"]),
    )


def gjr_one_step_variance_forecast(
    model: GJRGARCHModel,
    test_returns: pd.Series,
) -> pd.Series:
    """One-step-ahead GJR variance forecasts, updated with realized returns."""

    if model.params is None or model.last_residual is None or model.last_variance is None:
        raise RuntimeError("GJR model must be fitted before OOS forecasting")

    params = model.params
    residual = float(model.last_residual)
    variance = float(model.last_variance)
    forecasts: list[float] = []
    for raw_return in test_returns:
        leverage = 1.0 if residual < 0.0 else 0.0
        variance = (
            params.omega
            + params.alpha * residual**2
            + params.gamma * leverage * residual**2
            + params.beta * variance
        )
        variance = max(float(variance), model.min_variance)
        forecasts.append(variance / params.return_scale**2)
        residual = float(raw_return) * params.return_scale - params.mu
    return pd.Series(forecasts, index=test_returns.index, name="gjr_variance_forecast")


def har_one_step_daily_forecast(
    model: HARRVModel,
    train_daily_rv: pd.DataFrame,
    test_daily_rv: pd.DataFrame,
) -> pd.DataFrame:
    """One-step-ahead HAR-RV forecasts updated with prior observed daily RV."""

    if model.params is None:
        raise RuntimeError("HAR-RV model must be fitted before OOS forecasting")

    history = list(
        train_daily_rv.sort_values("date")["realized_variance"]
        .astype(float)
        .clip(lower=model.params.min_variance)
        .to_numpy()
    )
    rows: list[dict[str, Any]] = []
    for row in test_daily_rv.sort_values("date").itertuples(index=False):
        recent = np.asarray(history, dtype=float)
        rv_daily = float(recent[-1])
        rv_weekly = float(np.mean(recent[-model.weekly_window :]))
        rv_monthly = float(np.mean(recent[-model.monthly_window :]))
        forecast = (
            model.params.intercept
            + model.params.beta_daily * rv_daily
            + model.params.beta_weekly * rv_weekly
            + model.params.beta_monthly * rv_monthly
        )
        forecast = max(float(forecast), model.params.min_variance)
        actual = float(row.realized_variance)
        rows.append(
            {
                "date": row.date,
                "actual_rv": actual,
                "forecast_rv": forecast,
                "naive_persistence_rv": rv_daily,
            }
        )
        history.append(max(actual, model.params.min_variance))
    return pd.DataFrame(rows)


def variance_metrics(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> dict[str, float]:
    """Evaluate positive variance forecasts against realized variance proxies."""

    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    mask = np.isfinite(actual_array) & np.isfinite(forecast_array) & (forecast_array > 0.0)
    actual_array = actual_array[mask]
    forecast_array = forecast_array[mask]
    if actual_array.size == 0:
        raise ValueError("No finite actual/forecast pairs available")

    error = forecast_array - actual_array
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    qlike = float(np.mean(np.log(forecast_array) + actual_array / forecast_array))
    correlation = _safe_corr(actual_array, forecast_array)
    baseline_mean = float(np.mean(actual_array))
    baseline_mse = float(np.mean((actual_array - baseline_mean) ** 2))
    r2_vs_mean = float(1.0 - mse / baseline_mse) if baseline_mse > 0.0 else float("nan")
    return {
        "observations": float(actual_array.size),
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "qlike": qlike,
        "correlation": correlation,
        "r2_vs_realized_mean": r2_vs_mean,
    }


def distribution_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "skew": float(stats.skew(array, bias=False)) if array.size > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(array, fisher=True, bias=False)) if array.size > 3 else float("nan"),
    }


def daily_close_returns(market: pd.DataFrame) -> pd.Series:
    daily = (
        market[["open_datetime", "close"]]
        .assign(date=lambda frame: pd.to_datetime(frame["open_datetime"], utc=True).dt.floor("D"))
        .groupby("date", sort=True)["close"]
        .last()
    )
    return np.log(daily).diff().dropna()


def run_report(config: VolatilityReportConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    market, data_quality = load_market_data(config.database_path, config.symbol, config.interval)
    split = temporal_return_split(market, config.train_fraction)

    gjr_model = GJRGARCHModel().fit(split.train_returns)
    gjr_variance = gjr_one_step_variance_forecast(gjr_model, split.test_returns)
    gjr_actual_variance = split.test_returns**2
    gjr_metrics = variance_metrics(gjr_actual_variance, gjr_variance)
    gjr_metrics.update(gjr_var_coverage_metrics(gjr_model, split.test_returns, gjr_variance))

    daily_rv = HARRVModel.daily_realized_variance(market)
    split_daily_count = int(len(daily_rv) * config.train_fraction)
    train_daily_rv = daily_rv.iloc[:split_daily_count].copy()
    test_daily_rv = daily_rv.iloc[split_daily_count:].copy()
    har_model = HARRVModel().fit(train_daily_rv, step_mean_return=float(split.train_returns.mean()))
    har_oos = har_one_step_daily_forecast(har_model, train_daily_rv, test_daily_rv)
    har_metrics = variance_metrics(har_oos["actual_rv"], har_oos["forecast_rv"])
    har_naive_metrics = variance_metrics(har_oos["actual_rv"], har_oos["naive_persistence_rv"])
    har_metrics["mse_skill_vs_persistence"] = _skill_score(har_metrics["mse"], har_naive_metrics["mse"])
    har_metrics["qlike_delta_vs_persistence"] = float(har_metrics["qlike"] - har_naive_metrics["qlike"])

    horizon_steps = len(split.test_returns)
    gjr_prices, _, _ = gjr_model.simulate(
        start_price=split.start_price,
        horizon_steps=horizon_steps,
        n_paths=config.n_paths,
        seed=config.seed,
    )
    har_prices, _, _ = har_model.simulate(
        start_price=split.start_price,
        horizon_steps=horizon_steps,
        n_paths=config.n_paths,
        steps_per_day=config.steps_per_day,
        seed=config.seed,
    )

    actual_price_path = np.concatenate([[split.start_price], split.test_market["close"].to_numpy(dtype=float)])
    actual_final_price = float(actual_price_path[-1])
    gjr_final_prices = gjr_prices[:, -1]
    har_final_prices = har_prices[:, -1]

    test_daily_returns = daily_close_returns(split.test_market)
    gjr_daily_returns = np.log(gjr_prices[:, min(config.steps_per_day, horizon_steps)] / gjr_prices[:, 0])
    har_daily_returns = np.log(har_prices[:, min(config.steps_per_day, horizon_steps)] / har_prices[:, 0])

    path_metrics = {
        "actual_final_price": actual_final_price,
        "gjr_final_price_percentile": _empirical_percentile(gjr_final_prices, actual_final_price),
        "har_final_price_percentile": _empirical_percentile(har_final_prices, actual_final_price),
        "gjr_final_price_distribution": distribution_summary(gjr_final_prices),
        "har_final_price_distribution": distribution_summary(har_final_prices),
        "historical_test_daily_return_distribution": distribution_summary(test_daily_returns),
        "gjr_one_day_return_distribution": distribution_summary(gjr_daily_returns),
        "har_one_day_return_distribution": distribution_summary(har_daily_returns),
        "gjr_daily_return_ks_stat": _ks_statistic(test_daily_returns, gjr_daily_returns),
        "har_daily_return_ks_stat": _ks_statistic(test_daily_returns, har_daily_returns),
    }

    plot_paths = create_plots(
        output_dir=config.output_dir,
        market=market,
        split=split,
        gjr_variance=gjr_variance,
        har_oos=har_oos,
        gjr_prices=gjr_prices,
        har_prices=har_prices,
        actual_price_path=actual_price_path,
        test_daily_returns=test_daily_returns,
        gjr_daily_returns=gjr_daily_returns,
        har_daily_returns=har_daily_returns,
    )

    metrics: dict[str, Any] = {
        "config": {
            "database_path": str(config.database_path),
            "symbol": config.symbol,
            "interval": config.interval,
            "train_fraction": config.train_fraction,
            "n_paths": config.n_paths,
            "seed": config.seed,
            "steps_per_day": config.steps_per_day,
        },
        "data_quality": data_quality.to_dict(),
        "split": {
            "train_start": split.train_market["open_datetime"].iloc[0].isoformat(),
            "train_end": split.split_time.isoformat(),
            "test_start": split.test_market["open_datetime"].iloc[0].isoformat(),
            "test_end": split.test_market["open_datetime"].iloc[-1].isoformat(),
            "train_return_observations": int(len(split.train_returns)),
            "test_return_observations": int(len(split.test_returns)),
            "train_daily_rv_observations": int(len(train_daily_rv)),
            "test_daily_rv_observations": int(len(test_daily_rv)),
        },
        "gjr_garch": {
            "params": {
                "mu": gjr_model.params.mu if gjr_model.params is not None else None,
                "omega": gjr_model.params.omega if gjr_model.params is not None else None,
                "alpha": gjr_model.params.alpha if gjr_model.params is not None else None,
                "gamma": gjr_model.params.gamma if gjr_model.params is not None else None,
                "beta": gjr_model.params.beta if gjr_model.params is not None else None,
                "persistence": gjr_model.params.persistence if gjr_model.params is not None else None,
                "return_scale": gjr_model.params.return_scale if gjr_model.params is not None else None,
            },
            "oos_variance_metrics": gjr_metrics,
        },
        "har_rv": {
            "params": {
                "intercept": har_model.params.intercept if har_model.params is not None else None,
                "beta_daily": har_model.params.beta_daily if har_model.params is not None else None,
                "beta_weekly": har_model.params.beta_weekly if har_model.params is not None else None,
                "beta_monthly": har_model.params.beta_monthly if har_model.params is not None else None,
                "min_variance": har_model.params.min_variance if har_model.params is not None else None,
            },
            "oos_variance_metrics": har_metrics,
            "naive_persistence_metrics": har_naive_metrics,
        },
        "path_realism": path_metrics,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    report_path = config.output_dir / "README.md"
    report_path.write_text(render_report(metrics), encoding="utf-8")
    return metrics


def gjr_var_coverage_metrics(
    model: GJRGARCHModel,
    test_returns: pd.Series,
    forecast_variance: pd.Series,
) -> dict[str, float]:
    if model.params is None:
        raise RuntimeError("GJR model must be fitted")
    mean_return = model.params.mu / model.params.return_scale
    std = np.sqrt(np.asarray(forecast_variance, dtype=float))
    actual = np.asarray(test_returns, dtype=float)
    cutoff_5 = mean_return + std * stats.norm.ppf(0.05)
    cutoff_1 = mean_return + std * stats.norm.ppf(0.01)
    standardized = (actual - mean_return) / std
    return {
        "var_5pct_hit_rate": float(np.mean(actual < cutoff_5)),
        "var_1pct_hit_rate": float(np.mean(actual < cutoff_1)),
        "standardized_residual_mean": float(np.mean(standardized)),
        "standardized_residual_std": float(np.std(standardized, ddof=1)),
        "standardized_residual_skew": float(stats.skew(standardized, bias=False)),
        "standardized_residual_excess_kurtosis": float(stats.kurtosis(standardized, fisher=True, bias=False)),
    }


def create_plots(
    output_dir: Path,
    market: pd.DataFrame,
    split: ReturnSplit,
    gjr_variance: pd.Series,
    har_oos: pd.DataFrame,
    gjr_prices: np.ndarray,
    har_prices: np.ndarray,
    actual_price_path: np.ndarray,
    test_daily_returns: pd.Series,
    gjr_daily_returns: np.ndarray,
    har_daily_returns: np.ndarray,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    plt.figure(figsize=(11, 4.8))
    plt.plot(market["open_datetime"], market["close"], linewidth=1.0)
    plt.axvline(split.split_time, color="black", linestyle="--", linewidth=1.0, label="train/test split")
    plt.title("FRONT close price with temporal train/test split")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    paths["price_split"] = output_dir / "price_split.png"
    plt.savefig(paths["price_split"], dpi=140)
    plt.close()

    gjr_plot = pd.DataFrame(
        {
            "time": split.test_market["open_datetime"].to_numpy(),
            "squared_return": split.test_returns.to_numpy() ** 2,
            "forecast_variance": gjr_variance.to_numpy(),
        }
    )
    window = min(288, max(12, len(gjr_plot) // 20))
    gjr_plot["squared_return_roll"] = gjr_plot["squared_return"].rolling(window, min_periods=1).mean()
    gjr_plot["forecast_variance_roll"] = gjr_plot["forecast_variance"].rolling(window, min_periods=1).mean()
    plt.figure(figsize=(11, 4.8))
    plt.plot(gjr_plot["time"], gjr_plot["squared_return_roll"], label="realized squared return, rolling", linewidth=1.0)
    plt.plot(gjr_plot["time"], gjr_plot["forecast_variance_roll"], label="GJR one-step variance, rolling", linewidth=1.0)
    plt.title("GJR-GARCH out-of-sample variance diagnostics")
    plt.ylabel("5m variance")
    plt.legend()
    plt.tight_layout()
    paths["gjr_variance"] = output_dir / "gjr_variance_forecast.png"
    plt.savefig(paths["gjr_variance"], dpi=140)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.plot(har_oos["date"], har_oos["actual_rv"], label="actual daily RV", linewidth=1.0)
    plt.plot(har_oos["date"], har_oos["forecast_rv"], label="HAR-RV one-step forecast", linewidth=1.0)
    plt.plot(har_oos["date"], har_oos["naive_persistence_rv"], label="persistence baseline", linewidth=0.9, alpha=0.75)
    plt.title("HAR-RV out-of-sample daily realized variance")
    plt.ylabel("Daily realized variance")
    plt.legend()
    plt.tight_layout()
    paths["har_rv"] = output_dir / "har_rv_forecast.png"
    plt.savefig(paths["har_rv"], dpi=140)
    plt.close()

    max_steps = min(gjr_prices.shape[1], har_prices.shape[1], len(actual_price_path))
    plot_steps = np.arange(max_steps)
    plt.figure(figsize=(11, 5.2))
    _plot_fan(plot_steps, gjr_prices[:, :max_steps], "GJR-GARCH", "#1f77b4")
    _plot_fan(plot_steps, har_prices[:, :max_steps], "HAR-RV", "#2ca02c")
    plt.plot(plot_steps, actual_price_path[:max_steps], color="black", linewidth=1.3, label="actual test path")
    plt.title("Monte-Carlo price fan versus actual test path")
    plt.xlabel("Test step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    paths["price_fan"] = output_dir / "mc_price_fan.png"
    plt.savefig(paths["price_fan"], dpi=140)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    bins = 40
    plt.hist(test_daily_returns, bins=bins, density=True, alpha=0.45, label="historical test daily returns")
    plt.hist(gjr_daily_returns, bins=bins, density=True, histtype="step", linewidth=1.6, label="GJR simulated one-day returns")
    plt.hist(har_daily_returns, bins=bins, density=True, histtype="step", linewidth=1.6, label="HAR simulated one-day returns")
    plt.title("Daily return distribution: historical test vs simulated")
    plt.xlabel("Daily log return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    paths["daily_return_distribution"] = output_dir / "daily_return_distribution.png"
    plt.savefig(paths["daily_return_distribution"], dpi=140)
    plt.close()

    return paths


def render_report(metrics: dict[str, Any]) -> str:
    split = metrics["split"]
    dq = metrics["data_quality"]
    gjr = metrics["gjr_garch"]
    har = metrics["har_rv"]
    path = metrics["path_realism"]
    plots = metrics["plots"]

    lines = [
        f"# Volatility Model Analysis: {metrics['config']['symbol']}",
        "",
        "This report is generated from the repository SQLite database opened through the read-only data loader.",
        "It evaluates the first GJR-GARCH and HAR-RV building blocks as simulation inputs, not as final trading or valuation models.",
        "",
        "## Data And Split",
        "",
        f"- Symbol: `{metrics['config']['symbol']}`",
        f"- Interval: `{metrics['config']['interval']}`",
        f"- Train period: `{split['train_start']}` to `{split['train_end']}`",
        f"- Test period: `{split['test_start']}` to `{split['test_end']}`",
        f"- Train 5m returns: `{split['train_return_observations']}`",
        f"- Test 5m returns: `{split['test_return_observations']}`",
        f"- Train daily RV observations: `{split['train_daily_rv_observations']}`",
        f"- Test daily RV observations: `{split['test_daily_rv_observations']}`",
        "",
        "Data quality checks:",
        "",
        f"- Missing values by column: `{dq['missing_by_column']}`",
        f"- Duplicate open timestamps: `{dq['duplicate_open_time_count']}`",
        f"- Invalid OHLC rows: `{dq['invalid_ohlc_count']}`",
        f"- Non-positive price rows: `{dq['non_positive_price_count']}`",
        f"- Calendar 5m gaps: `{dq['gap_count']}`",
        f"- Maximum gap in seconds: `{dq['max_gap_seconds']}`",
        "",
        f"![Close price split]({Path(plots['price_split']).name})",
        "",
        "## GJR-GARCH Diagnostics",
        "",
        "Fitted equation: `sigma_t^2 = omega + alpha eps_{t-1}^2 + gamma I(eps_{t-1}<0) eps_{t-1}^2 + beta sigma_{t-1}^2`.",
        "",
        _markdown_table(
            ["parameter", "estimate"],
            [
                ["mu", _fmt(gjr["params"]["mu"])],
                ["omega", _fmt(gjr["params"]["omega"])],
                ["alpha", _fmt(gjr["params"]["alpha"])],
                ["gamma", _fmt(gjr["params"]["gamma"])],
                ["beta", _fmt(gjr["params"]["beta"])],
                ["persistence", _fmt(gjr["params"]["persistence"])],
            ],
        ),
        "",
        "Out-of-sample one-step variance metrics against squared 5m returns:",
        "",
        _metrics_table(gjr["oos_variance_metrics"], [
            "observations",
            "mae",
            "rmse",
            "qlike",
            "correlation",
            "r2_vs_realized_mean",
            "var_5pct_hit_rate",
            "var_1pct_hit_rate",
            "standardized_residual_std",
            "standardized_residual_excess_kurtosis",
        ]),
        "",
        f"![GJR variance forecast]({Path(plots['gjr_variance']).name})",
        "",
        "Interpretation: GJR-GARCH is useful here mainly as a volatility-clustering generator. The variance target is noisy at 5-minute frequency, so low correlation or negative out-of-sample R2 is not by itself a failure. VaR hit rates near 5% and 1%, standardized residual standard deviation near 1, and reasonable final-price percentiles are stronger sanity checks for this stage.",
        "",
        "## HAR-RV Diagnostics",
        "",
        "HAR-RV forecasts daily realized variance from lagged daily, weekly, and monthly RV features. The test forecast uses only prior observed days.",
        "",
        _markdown_table(
            ["parameter", "estimate"],
            [
                ["intercept", _fmt(har["params"]["intercept"])],
                ["beta_daily", _fmt(har["params"]["beta_daily"])],
                ["beta_weekly", _fmt(har["params"]["beta_weekly"])],
                ["beta_monthly", _fmt(har["params"]["beta_monthly"])],
                ["variance_floor", _fmt(har["params"]["min_variance"])],
            ],
        ),
        "",
        "Out-of-sample daily RV metrics:",
        "",
        _metrics_table(har["oos_variance_metrics"], [
            "observations",
            "mae",
            "rmse",
            "qlike",
            "correlation",
            "r2_vs_realized_mean",
            "mse_skill_vs_persistence",
            "qlike_delta_vs_persistence",
        ]),
        "",
        "Persistence baseline metrics:",
        "",
        _metrics_table(har["naive_persistence_metrics"], [
            "mae",
            "rmse",
            "qlike",
            "correlation",
            "r2_vs_realized_mean",
        ]),
        "",
        f"![HAR-RV forecast]({Path(plots['har_rv']).name})",
        "",
        "Interpretation: HAR-RV is evaluated at the same aggregation level it models: daily realized variance. Positive MSE skill versus the persistence baseline means the simple multi-horizon features add information for average squared error on this split. A positive QLIKE delta is worse than persistence and flags under-forecasted high-volatility days or too many forecasts near the variance floor.",
        "",
        "## Monte-Carlo Path Realism",
        "",
        f"- Actual test final price: `{_fmt(path['actual_final_price'])}`",
        f"- Actual final price percentile under GJR-GARCH paths: `{_fmt(path['gjr_final_price_percentile'])}`",
        f"- Actual final price percentile under HAR-RV paths: `{_fmt(path['har_final_price_percentile'])}`",
        f"- GJR daily-return KS statistic versus test daily returns: `{_fmt(path['gjr_daily_return_ks_stat']['statistic'])}`",
        f"- HAR daily-return KS statistic versus test daily returns: `{_fmt(path['har_daily_return_ks_stat']['statistic'])}`",
        "",
        f"![Price fan]({Path(plots['price_fan']).name})",
        "",
        f"![Daily return distribution]({Path(plots['daily_return_distribution']).name})",
        "",
        "Model distribution summaries:",
        "",
        _markdown_table(
            ["distribution", "mean", "std", "q05", "q50", "q95", "excess_kurtosis"],
            [
                _dist_row("historical test daily returns", path["historical_test_daily_return_distribution"]),
                _dist_row("GJR simulated one-day returns", path["gjr_one_day_return_distribution"]),
                _dist_row("HAR simulated one-day returns", path["har_one_day_return_distribution"]),
                _dist_row("GJR final prices", path["gjr_final_price_distribution"]),
                _dist_row("HAR final prices", path["har_final_price_distribution"]),
            ],
        ),
        "",
        "## Bottom Line",
        "",
        _bottom_line(metrics),
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python -m lsmc_rl.analysis.volatility_report --output-dir outputs/volatility_report_front",
        "```",
        "",
        "The detailed numeric output is also stored in `metrics.json` next to this report.",
        "",
    ]
    return "\n".join(lines)


def _plot_fan(steps: np.ndarray, prices: np.ndarray, label: str, color: str) -> None:
    q05 = np.quantile(prices, 0.05, axis=0)
    q50 = np.quantile(prices, 0.50, axis=0)
    q95 = np.quantile(prices, 0.95, axis=0)
    plt.fill_between(steps, q05, q95, color=color, alpha=0.15)
    plt.plot(steps, q50, color=color, linewidth=1.0, label=f"{label} median / 5-95 band")


def _metrics_table(metrics: dict[str, Any], keys: list[str]) -> str:
    return _markdown_table(["metric", "value"], [[key, _fmt(metrics.get(key))] for key in keys])


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _dist_row(name: str, summary: dict[str, float]) -> list[str]:
    return [
        name,
        _fmt(summary["mean"]),
        _fmt(summary["std"]),
        _fmt(summary["q05"]),
        _fmt(summary["q50"]),
        _fmt(summary["q95"]),
        _fmt(summary["excess_kurtosis"]),
    ]


def _bottom_line(metrics: dict[str, Any]) -> str:
    gjr = metrics["gjr_garch"]["oos_variance_metrics"]
    har = metrics["har_rv"]["oos_variance_metrics"]
    path = metrics["path_realism"]
    statements = [
        "Both models are useful as controlled baseline path generators, but the current diagnostics do not justify treating them as realistic market models yet.",
        f"GJR-GARCH produced a 5% VaR hit rate of {_fmt(gjr['var_5pct_hit_rate'])} and a 1% hit rate of {_fmt(gjr['var_1pct_hit_rate'])}. Values above the nominal levels show that normal innovations materially understate tail risk on this split.",
        f"HAR-RV achieved an MSE skill versus persistence of {_fmt(har['mse_skill_vs_persistence'])}; positive values indicate improvement over yesterday's RV on this split. Its QLIKE delta versus persistence is {_fmt(har['qlike_delta_vs_persistence'])}, where lower is better.",
        f"The actual test final price falls at percentile {_fmt(path['gjr_final_price_percentile'])} under GJR paths and {_fmt(path['har_final_price_percentile'])} under HAR-RV paths.",
        "Before using these paths for option valuation, the next useful checks are rolling-origin validation, Student-t or bootstrap innovations, intraday seasonality, and sensitivity of option values to the volatility generator.",
    ]
    return "\n\n".join(statements)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _skill_score(model_score: float, baseline_score: float) -> float:
    if not np.isfinite(baseline_score) or baseline_score == 0.0:
        return float("nan")
    return float(1.0 - model_score / baseline_score)


def _empirical_percentile(samples: np.ndarray, observed: float) -> float:
    samples = np.asarray(samples, dtype=float)
    return float(np.mean(samples <= observed))


def _ks_statistic(observed: pd.Series | np.ndarray, simulated: pd.Series | np.ndarray) -> dict[str, float]:
    observed_array = np.asarray(observed, dtype=float)
    simulated_array = np.asarray(simulated, dtype=float)
    observed_array = observed_array[np.isfinite(observed_array)]
    simulated_array = simulated_array[np.isfinite(simulated_array)]
    result = stats.ks_2samp(observed_array, simulated_array)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1000 or (abs(numeric) < 0.001 and numeric != 0.0):
        return f"{numeric:.6e}"
    return f"{numeric:.6f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate volatility model diagnostics and plots.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--n-paths", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--output-dir", default="outputs/volatility_report_front")
    parser.add_argument("--steps-per-day", type=int, default=288)
    args = parser.parse_args(argv)

    config = VolatilityReportConfig(
        database_path=Path(args.database_path).resolve(),
        symbol=args.symbol,
        interval=args.interval,
        train_fraction=args.train_fraction,
        n_paths=args.n_paths,
        seed=args.seed,
        output_dir=Path(args.output_dir).resolve(),
        steps_per_day=args.steps_per_day,
    )
    metrics = run_report(config)
    print(f"Wrote report: {config.output_dir / 'README.md'}")
    print(f"Wrote metrics: {config.output_dir / 'metrics.json'}")
    print(f"GJR 5pct VaR hit rate: {_fmt(metrics['gjr_garch']['oos_variance_metrics']['var_5pct_hit_rate'])}")
    print(f"HAR MSE skill vs persistence: {_fmt(metrics['har_rv']['oos_variance_metrics']['mse_skill_vs_persistence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
