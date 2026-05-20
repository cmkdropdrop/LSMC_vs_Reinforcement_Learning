"""Distributional path-quality diagnostics for volatility path generators."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lsmc_rl.analysis.garch_refit_report import daily_forecast_frame, static_gjr_forecast
from lsmc_rl.analysis.volatility_report import har_one_step_daily_forecast, temporal_return_split
from lsmc_rl.data import load_market_data
from lsmc_rl.volatility import GJRGARCHModel, HARRVModel


@dataclass(frozen=True)
class PathQualityReportConfig:
    database_path: Path
    symbol: str = "FRONT"
    interval: str = "5m"
    train_fraction: float = 0.70
    n_paths: int = 1200
    steps_per_day: int = 288
    seed: int = 20260519
    output_dir: Path = Path("outputs/path_quality_report_front")
    report_path: Path = Path("docs/path_quality_analysis.md")


def qlike_loss(actual_variance: np.ndarray, forecast_variance: np.ndarray) -> float:
    """Mean QLIKE loss for positive variance forecasts."""

    actual, forecast = _positive_pairs(actual_variance, forecast_variance)
    return float(np.mean(np.log(forecast) + actual / forecast))


def normal_nll(actual_return: np.ndarray, forecast_mean: np.ndarray, forecast_variance: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood for return forecasts."""

    returns = np.asarray(actual_return, dtype=float)
    means = np.asarray(forecast_mean, dtype=float)
    variances = np.asarray(forecast_variance, dtype=float)
    mask = np.isfinite(returns) & np.isfinite(means) & np.isfinite(variances) & (variances > 0.0)
    if not mask.any():
        raise ValueError("No finite return/mean/variance triples available")
    residual = returns[mask] - means[mask]
    variance = variances[mask]
    return float(0.5 * np.mean(np.log(2.0 * np.pi * variance) + residual**2 / variance))


def energy_score(observations: np.ndarray, ensemble: np.ndarray) -> float:
    """Average multivariate Energy Score for an ensemble distribution."""

    obs = np.asarray(observations, dtype=float)
    ens = np.asarray(ensemble, dtype=float)
    if obs.ndim != 2 or ens.ndim != 2 or obs.shape[1] != ens.shape[1]:
        raise ValueError("observations and ensemble must be 2D arrays with matching feature counts")
    if obs.shape[0] == 0 or ens.shape[0] == 0:
        raise ValueError("energy score requires non-empty inputs")

    obs_scaled, ens_scaled = standardize_pair(obs, ens)
    ens_obs_dist = _euclidean_distances(ens_scaled, obs_scaled)
    ens_pair_dist = _euclidean_distances(ens_scaled, ens_scaled)
    return float(np.mean(ens_obs_dist, axis=0).mean() - 0.5 * np.mean(ens_pair_dist))


def multiband_mmd(
    observed_bands: dict[str, np.ndarray],
    simulated_bands: dict[str, np.ndarray],
    bandwidth_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Biased RBF MMD averaged across feature bands and bandwidths."""

    rows: dict[str, Any] = {}
    for band, observed in observed_bands.items():
        if band not in simulated_bands:
            raise ValueError(f"Missing simulated feature band: {band}")
        obs_scaled, sim_scaled = standardize_pair(observed, simulated_bands[band])
        base_bandwidth = _median_pairwise_distance(np.vstack([obs_scaled, sim_scaled]))
        scores = [
            _rbf_mmd_biased(obs_scaled, sim_scaled, base_bandwidth * multiplier)
            for multiplier in bandwidth_multipliers
        ]
        rows[band] = {
            "base_bandwidth": float(base_bandwidth),
            "band_mmd2": float(np.mean(scores)),
            "bandwidth_scores": {
                str(multiplier): float(score)
                for multiplier, score in zip(bandwidth_multipliers, scores)
            },
        }
    rows["aggregate_mmd2"] = float(np.mean([value["band_mmd2"] for value in rows.values()]))
    rows["bandwidth_multipliers"] = list(bandwidth_multipliers)
    return rows


def run_report(config: PathQualityReportConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)

    market, data_quality = load_market_data(config.database_path, config.symbol, config.interval)
    split = temporal_return_split(market, config.train_fraction)
    first_scoring_date = pd.Timestamp(split.split_time).floor("D") + pd.Timedelta(days=1)
    pre_scoring_market = market.loc[market["open_datetime"] < first_scoring_date].copy()
    scoring_market = market.loc[market["open_datetime"] >= first_scoring_date].copy()

    gjr_model = GJRGARCHModel().fit(split.train_returns)
    gjr_steps, _ = static_gjr_forecast(market, split.train_returns, split.test_returns)
    gjr_daily = daily_forecast_frame(gjr_steps)

    train_daily_rv = HARRVModel.daily_realized_variance(pre_scoring_market)
    test_daily_rv = HARRVModel.daily_realized_variance(scoring_market)
    pre_scoring_returns = pre_scoring_market["log_return"].dropna()
    har_model = HARRVModel().fit(train_daily_rv, step_mean_return=float(pre_scoring_returns.mean()))
    har_daily = har_one_step_daily_forecast(har_model, train_daily_rv, test_daily_rv)

    daily_returns = daily_return_frame(market)
    har_daily = har_daily.merge(daily_returns, on="date", how="inner")
    train_daily_returns = daily_returns.loc[daily_returns["date"] < first_scoring_date, "actual_return"]
    har_daily["forecast_mean"] = float(train_daily_returns.mean()) if len(train_daily_returns) else 0.0
    har_daily = har_daily.rename(columns={"forecast_rv": "forecast_variance", "actual_rv": "actual_variance"})
    gjr_daily = gjr_daily.rename(columns={"forecast_rv": "forecast_variance", "actual_rv": "actual_variance"})
    gjr_daily = gjr_daily.loc[gjr_daily["date"] >= first_scoring_date].copy()

    scoring_dates = sorted(set(gjr_daily["date"]).intersection(set(har_daily["date"])))
    gjr_scoring = gjr_daily.loc[gjr_daily["date"].isin(scoring_dates)].sort_values("date").reset_index(drop=True)
    har_scoring = har_daily.loc[har_daily["date"].isin(scoring_dates)].sort_values("date").reset_index(drop=True)

    gjr_prices, _, _ = gjr_model.simulate(
        start_price=split.start_price,
        horizon_steps=config.steps_per_day,
        n_paths=config.n_paths,
        seed=config.seed,
    )
    har_prices, _, _ = har_model.simulate(
        start_price=split.start_price,
        horizon_steps=config.steps_per_day,
        n_paths=config.n_paths,
        steps_per_day=config.steps_per_day,
        seed=config.seed,
    )

    observed_bands = observed_daily_feature_bands(scoring_market)
    gjr_bands = simulated_daily_feature_bands(gjr_prices)
    har_bands = simulated_daily_feature_bands(har_prices)
    observed_all = combine_bands(observed_bands)
    gjr_all = combine_bands(gjr_bands)
    har_all = combine_bands(har_bands)

    metrics: dict[str, Any] = {
        "config": {
            "symbol": config.symbol,
            "interval": config.interval,
            "train_fraction": config.train_fraction,
            "n_paths": config.n_paths,
            "steps_per_day": config.steps_per_day,
            "seed": config.seed,
            "output_dir": str(config.output_dir),
            "report_path": str(config.report_path),
        },
        "data_quality": data_quality.to_dict(),
        "split": {
            "train_start": split.train_market["open_datetime"].iloc[0].isoformat(),
            "train_end": split.split_time.isoformat(),
            "test_start": split.test_market["open_datetime"].iloc[0].isoformat(),
            "test_end": split.test_market["open_datetime"].iloc[-1].isoformat(),
            "first_scoring_date": first_scoring_date.isoformat(),
            "scoring_days": int(len(scoring_dates)),
            "observed_feature_days": int(next(iter(observed_bands.values())).shape[0]),
        },
        "models": {
            "gjr_garch": {
                "qlike": qlike_loss(gjr_scoring["actual_variance"], gjr_scoring["forecast_variance"]),
                "normal_nll": normal_nll(
                    gjr_scoring["actual_return"],
                    gjr_scoring["forecast_mean"],
                    gjr_scoring["forecast_variance"],
                ),
                "energy_score": energy_score(observed_all, gjr_all),
                "multiband_mmd": multiband_mmd(observed_bands, gjr_bands),
            },
            "har_rv": {
                "qlike": qlike_loss(har_scoring["actual_variance"], har_scoring["forecast_variance"]),
                "normal_nll": normal_nll(
                    har_scoring["actual_return"],
                    har_scoring["forecast_mean"],
                    har_scoring["forecast_variance"],
                ),
                "energy_score": energy_score(observed_all, har_all),
                "multiband_mmd": multiband_mmd(observed_bands, har_bands),
            },
        },
        "feature_bands": {
            band: list(frame.columns) if isinstance(frame, pd.DataFrame) else []
            for band, frame in observed_daily_feature_band_frames(scoring_market).items()
        },
    }
    metrics["ranking"] = rank_models(metrics["models"])

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2), encoding="utf-8")
    report = render_report(metrics, config.report_path)
    (config.output_dir / "README.md").write_text(report, encoding="utf-8")
    config.report_path.write_text(report, encoding="utf-8")
    return metrics


def daily_return_frame(market: pd.DataFrame) -> pd.DataFrame:
    returns = market["log_return"].dropna()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(market.loc[returns.index, "open_datetime"], utc=True).dt.floor("D"),
            "actual_return": returns.to_numpy(dtype=float),
        }
    )
    return frame.groupby("date", sort=True)["actual_return"].sum().reset_index()


def observed_daily_feature_band_frames(market: pd.DataFrame, min_returns: int = 12) -> dict[str, pd.DataFrame]:
    frame = market[["open_datetime", "log_return"]].dropna().copy()
    frame["date"] = pd.to_datetime(frame["open_datetime"], utc=True).dt.floor("D")
    rows: list[dict[str, float]] = []
    for _, group in frame.groupby("date", sort=True):
        returns = group["log_return"].to_numpy(dtype=float)
        if len(returns) >= min_returns:
            rows.append(_feature_row_from_returns(returns))
    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No observed daily feature rows available")
    return _split_feature_bands(features)


def observed_daily_feature_bands(market: pd.DataFrame, min_returns: int = 12) -> dict[str, np.ndarray]:
    return {name: frame.to_numpy(dtype=float) for name, frame in observed_daily_feature_band_frames(market, min_returns).items()}


def simulated_daily_feature_bands(price_paths: np.ndarray) -> dict[str, np.ndarray]:
    prices = np.asarray(price_paths, dtype=float)
    if prices.ndim != 2 or prices.shape[1] < 2:
        raise ValueError("price_paths must be a 2D array with at least two steps")
    returns = np.diff(np.log(prices), axis=1)
    rows = [_feature_row_from_returns(row) for row in returns]
    features = pd.DataFrame(rows)
    return {name: frame.to_numpy(dtype=float) for name, frame in _split_feature_bands(features).items()}


def combine_bands(bands: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([bands[name] for name in sorted(bands)])


def standardize_pair(observed: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    if obs.ndim != 2 or sim.ndim != 2 or obs.shape[1] != sim.shape[1]:
        raise ValueError("observed and simulated arrays must be 2D with matching columns")
    combined = np.vstack([obs, sim])
    center = np.nanmedian(combined, axis=0)
    q25, q75 = np.nanquantile(combined, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback_scale = np.nanstd(combined, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback_scale)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (obs - center) / scale, (sim - center) / scale


def rank_models(models: dict[str, dict[str, Any]]) -> dict[str, str]:
    keys = {
        "qlike": lambda data: data["qlike"],
        "normal_nll": lambda data: data["normal_nll"],
        "energy_score": lambda data: data["energy_score"],
        "aggregate_mmd2": lambda data: data["multiband_mmd"]["aggregate_mmd2"],
    }
    return {
        metric: min(models, key=lambda model: extractor(models[model]))
        for metric, extractor in keys.items()
    }


def render_report(metrics: dict[str, Any], report_path: Path) -> str:
    models = metrics["models"]
    lines = [
        "# Path Quality Diagnostics",
        "",
        "This report evaluates the GJR-GARCH and HAR-RV path generators with distributional scores rather than point-forecast metrics. Lower values are better for QLIKE, Gaussian NLL, Energy Score, and multiband MMD. The path-quality diagnostic is intentionally table-only; plots are not needed for the model decision at this stage.",
        "",
        "## Scores",
        "",
        _markdown_table(
            ["model", "QLIKE", "Gaussian NLL", "Energy Score", "multiband MMD^2"],
            [
                [
                    model,
                    _fmt(data["qlike"]),
                    _fmt(data["normal_nll"]),
                    _fmt(data["energy_score"]),
                    _fmt(data["multiband_mmd"]["aggregate_mmd2"]),
                ]
                for model, data in models.items()
            ],
        ),
        "",
        "Ranking by score:",
        "",
        _markdown_table(
            ["score", "best model"],
            [[score, model] for score, model in metrics["ranking"].items()],
        ),
        "",
        "QLIKE and Gaussian NLL are computed on aligned out-of-sample daily return and variance forecasts. Energy Score and multiband MMD are computed on robustly standardized daily path-feature vectors built from returns, realized volatility, downside/upside variation, drawdown, intraday range, and multi-horizon return and variance bands.",
        "",
        "## MMD By Feature Band",
        "",
        _markdown_table(
            ["model", "core", "extremes", "multi_horizon", "aggregate"],
            [
                [
                    model,
                    _fmt(data["multiband_mmd"]["core"]["band_mmd2"]),
                    _fmt(data["multiband_mmd"]["extremes"]["band_mmd2"]),
                    _fmt(data["multiband_mmd"]["multi_horizon"]["band_mmd2"]),
                    _fmt(data["multiband_mmd"]["aggregate_mmd2"]),
                ]
                for model, data in models.items()
            ],
        ),
        "",
        "## Interpretation",
        "",
        _interpretation(metrics),
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python -m lsmc_rl.analysis.path_quality_report --output-dir outputs/path_quality_report_front --report-path docs/path_quality_analysis.md",
        "```",
        "",
        "The full numeric output is stored in `outputs/path_quality_report_front/metrics.json`.",
        "",
    ]
    return "\n".join(lines)


def _feature_row_from_returns(returns: np.ndarray) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("returns must be a non-empty 1D array")
    cumulative = np.cumsum(values)
    price_rel = np.exp(cumulative)
    running_max = np.maximum.accumulate(price_rel)
    drawdown = 1.0 - price_rel / running_max
    positive = np.maximum(values, 0.0)
    negative = np.minimum(values, 0.0)
    row: dict[str, float] = {
        "daily_return": float(np.sum(values)),
        "realized_volatility": float(np.sqrt(np.sum(values**2))),
        "downside_volatility": float(np.sqrt(np.sum(negative**2))),
        "upside_volatility": float(np.sqrt(np.sum(positive**2))),
        "max_drawdown": float(np.max(drawdown)),
        "intraday_log_range": float(np.max(cumulative) - np.min(cumulative)),
        "min_step_return": float(np.min(values)),
        "max_step_return": float(np.max(values)),
    }
    for label, fraction in (("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q100", 1.00)):
        end = max(1, int(np.ceil(values.size * fraction)))
        prefix = values[:end]
        row[f"return_{label}"] = float(np.sum(prefix))
        row[f"rv_{label}"] = float(np.sum(prefix**2))
    return row


def _split_feature_bands(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "core": features[["daily_return", "realized_volatility", "downside_volatility", "upside_volatility"]],
        "extremes": features[["max_drawdown", "intraday_log_range", "min_step_return", "max_step_return"]],
        "multi_horizon": features[["return_q25", "return_q50", "return_q75", "return_q100", "rv_q25", "rv_q50", "rv_q100"]],
    }


def _positive_pairs(actual_variance: np.ndarray, forecast_variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual_variance, dtype=float)
    forecast = np.asarray(forecast_variance, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(forecast) & (actual >= 0.0) & (forecast > 0.0)
    if not mask.any():
        raise ValueError("No finite variance pairs available")
    return actual[mask], forecast[mask]


def _euclidean_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_sq = np.sum(x**2, axis=1)[:, None]
    y_sq = np.sum(y**2, axis=1)[None, :]
    squared = np.maximum(x_sq + y_sq - 2.0 * x @ y.T, 0.0)
    return np.sqrt(squared)


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_sq = np.sum(x**2, axis=1)[:, None]
    y_sq = np.sum(y**2, axis=1)[None, :]
    return np.maximum(x_sq + y_sq - 2.0 * x @ y.T, 0.0)


def _median_pairwise_distance(x: np.ndarray) -> float:
    distances = _euclidean_distances(x, x)
    values = distances[np.triu_indices_from(distances, k=1)]
    values = values[np.isfinite(values) & (values > 1e-12)]
    if values.size == 0:
        return 1.0
    return float(np.median(values))


def _rbf_mmd_biased(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    bandwidth = max(float(bandwidth), 1e-12)
    gamma = 1.0 / (2.0 * bandwidth**2)
    k_xx = np.exp(-gamma * _squared_distances(x, x)).mean()
    k_yy = np.exp(-gamma * _squared_distances(y, y)).mean()
    k_xy = np.exp(-gamma * _squared_distances(x, y)).mean()
    return float(max(k_xx + k_yy - 2.0 * k_xy, 0.0))


def _interpretation(metrics: dict[str, Any]) -> str:
    ranking = metrics["ranking"]
    energy_winner = ranking["energy_score"]
    mmd_winner = ranking["aggregate_mmd2"]
    realistic = (
        f"`{energy_winner}` currently produces the more realistic paths."
        if energy_winner == mmd_winner
        else "The distributional scores disagree, so neither method should be treated as clearly realistic yet."
    )
    lines = [
        f"- QLIKE winner: `{ranking['qlike']}`.",
        f"- Gaussian NLL winner: `{ranking['normal_nll']}`.",
        f"- Energy Score winner: `{ranking['energy_score']}`.",
        f"- Multiband MMD winner: `{ranking['aggregate_mmd2']}`.",
        f"- Practical assessment: {realistic}",
        "",
        "A model is more credible for option work only if it performs well on both forecast likelihood scores and distributional path scores. QLIKE/NLL evaluate the one-day return and variance forecasts; Energy Score and multiband MMD test whether simulated daily path features look like observed daily path features across several bands. Disagreement between these scores is a warning that a model may forecast variance acceptably while still generating unrealistic path shapes.",
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
        return "nan"
    if abs(numeric) >= 1000 or (abs(numeric) < 0.001 and numeric != 0.0):
        return f"{numeric:.6e}"
    return f"{numeric:.6f}"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate distributional path-quality diagnostics.")
    parser.add_argument("--database-path", default="ttf_klines_5m_from_1m.sqlite")
    parser.add_argument("--symbol", default="FRONT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--n-paths", type=int, default=1200)
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--output-dir", default="outputs/path_quality_report_front")
    parser.add_argument("--report-path", default="docs/path_quality_analysis.md")
    args = parser.parse_args(argv)

    config = PathQualityReportConfig(
        database_path=Path(args.database_path).resolve(),
        symbol=args.symbol,
        interval=args.interval,
        train_fraction=args.train_fraction,
        n_paths=args.n_paths,
        steps_per_day=args.steps_per_day,
        seed=args.seed,
        output_dir=Path(args.output_dir).resolve(),
        report_path=Path(args.report_path).resolve(),
    )
    metrics = run_report(config)
    print(f"Wrote report: {config.report_path}")
    print(f"Wrote metrics: {config.output_dir / 'metrics.json'}")
    for model, data in metrics["models"].items():
        print(
            f"{model}: qlike={_fmt(data['qlike'])}, nll={_fmt(data['normal_nll'])}, "
            f"energy={_fmt(data['energy_score'])}, mmd={_fmt(data['multiband_mmd']['aggregate_mmd2'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
