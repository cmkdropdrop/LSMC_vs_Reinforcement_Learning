"""CLI and reusable functions for Monte-Carlo price path generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lsmc_rl.data import DataQualityReport, load_market_data
from lsmc_rl.volatility import GJRGARCHModel, HARRVModel


@dataclass(frozen=True)
class PathSimulationConfig:
    database_path: Path
    symbol: str = "FRONT"
    interval: str = "5m"
    model_type: str = "gjr_garch"
    horizon_steps: int = 288
    n_paths: int = 100
    seed: int | None = 1234
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    start_price: float | None = None
    time_step: str = "5min"
    steps_per_day: int = 288
    output_path: Path | None = None


@dataclass(frozen=True)
class PathSimulationResult:
    paths: pd.DataFrame
    summary: dict[str, Any]
    data_quality: DataQualityReport


def load_config(config_path: str | Path) -> PathSimulationConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_dir = path.parent
    database_path = _resolve_path(raw.get("database_path", "ttf_klines_5m_from_1m.sqlite"), base_dir)
    output_raw = raw.get("output_path")
    output_path = _resolve_path(output_raw, base_dir) if output_raw else None

    return PathSimulationConfig(
        database_path=database_path,
        symbol=str(raw.get("symbol", "FRONT")),
        interval=str(raw.get("interval", "5m")),
        model_type=str(raw.get("model_type", "gjr_garch")),
        horizon_steps=int(raw.get("horizon_steps", 288)),
        n_paths=int(raw.get("n_paths", 100)),
        seed=None if raw.get("seed") is None else int(raw.get("seed")),
        start=raw.get("start"),
        end=raw.get("end"),
        start_price=None if raw.get("start_price") is None else float(raw.get("start_price")),
        time_step=str(raw.get("time_step", "5min")),
        steps_per_day=int(raw.get("steps_per_day", 288)),
        output_path=output_path,
    )


def run_simulation(config: PathSimulationConfig) -> PathSimulationResult:
    market, report = load_market_data(
        db_path=config.database_path,
        symbol=config.symbol,
        interval=config.interval,
        start=config.start,
        end=config.end,
    )
    start_price = float(config.start_price if config.start_price is not None else market["close"].iloc[-1])
    if start_price <= 0:
        raise ValueError("Configured or inferred start_price must be positive")

    returns = market["log_return"].dropna()
    model_type = config.model_type.lower()
    if model_type == "gjr_garch":
        model = GJRGARCHModel().fit(returns)
        prices, simulated_returns, variances = model.simulate(
            start_price=start_price,
            horizon_steps=config.horizon_steps,
            n_paths=config.n_paths,
            seed=config.seed,
        )
        model_details = {
            "persistence": model.params.persistence if model.params is not None else None,
        }
    elif model_type == "har_rv":
        daily_rv = HARRVModel.daily_realized_variance(market)
        model = HARRVModel().fit(daily_rv=daily_rv, step_mean_return=float(returns.mean()))
        prices, simulated_returns, variances = model.simulate(
            start_price=start_price,
            horizon_steps=config.horizon_steps,
            n_paths=config.n_paths,
            steps_per_day=config.steps_per_day,
            seed=config.seed,
        )
        model_details = {
            "daily_rv_observations": int(len(daily_rv)),
        }
    else:
        raise ValueError("model_type must be one of: gjr_garch, har_rv")

    time_grid = _future_time_grid(market["open_datetime"].iloc[-1], config.time_step, config.horizon_steps)
    paths = paths_to_frame(
        prices=prices,
        returns=simulated_returns,
        variances=variances,
        times=time_grid,
        model=model_type,
    )
    summary = build_summary(config, market, report, paths, model_details)
    return PathSimulationResult(paths=paths, summary=summary, data_quality=report)


def paths_to_frame(
    prices: np.ndarray,
    returns: np.ndarray,
    variances: np.ndarray,
    times: pd.DatetimeIndex,
    model: str,
) -> pd.DataFrame:
    """Convert simulation arrays to the common long-form path interface."""

    if prices.shape != returns.shape or prices.shape != variances.shape:
        raise ValueError("prices, returns, and variances must have identical shapes")
    n_paths, n_steps = prices.shape
    if len(times) != n_steps:
        raise ValueError("time grid length must match the number of simulated steps")

    path_ids = np.repeat(np.arange(n_paths), n_steps)
    step_ids = np.tile(np.arange(n_steps), n_paths)
    frame = pd.DataFrame(
        {
            "path": path_ids,
            "step": step_ids,
            "time": np.tile(times.to_numpy(), n_paths),
            "price": prices.reshape(-1),
            "return": returns.reshape(-1),
            "variance": variances.reshape(-1),
            "model": model,
        }
    )
    frame["volatility"] = np.sqrt(frame["variance"])
    return frame


def build_summary(
    config: PathSimulationConfig,
    market: pd.DataFrame,
    report: DataQualityReport,
    paths: pd.DataFrame,
    model_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_prices = paths.loc[paths["step"] == config.horizon_steps, "price"]
    summary: dict[str, Any] = {
        "symbol": config.symbol,
        "historical_start": market["open_datetime"].iloc[0].isoformat(),
        "historical_end": market["open_datetime"].iloc[-1].isoformat(),
        "historical_observations": int(len(market)),
        "model_type": config.model_type,
        "n_paths": config.n_paths,
        "horizon_steps": config.horizon_steps,
        "seed": config.seed,
        "requested_start": None if config.start is None else str(config.start),
        "requested_end": None if config.end is None else str(config.end),
        "data_quality": report.to_dict(),
        "final_price_min": float(final_prices.min()),
        "final_price_mean": float(final_prices.mean()),
        "final_price_median": float(final_prices.median()),
        "final_price_max": float(final_prices.max()),
        "final_price_std": float(final_prices.std(ddof=1)) if len(final_prices) > 1 else 0.0,
        "final_price_q05": float(final_prices.quantile(0.05)),
        "final_price_q95": float(final_prices.quantile(0.95)),
    }
    if model_details:
        summary["model_details"] = model_details
    return summary


def write_paths(paths: pd.DataFrame, output_path: str | Path) -> Path:
    """Write generated paths to an allowed artifact directory."""

    path = Path(output_path)
    _ensure_allowed_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit volatility model and simulate Monte-Carlo price paths.")
    parser.add_argument("--config", required=True, help="YAML configuration path.")
    parser.add_argument("--model-type", choices=["gjr_garch", "har_rv"], help="Override model_type from config.")
    parser.add_argument("--start", help="Optional inclusive historical calibration start timestamp.")
    parser.add_argument("--end", help="Optional inclusive historical calibration end timestamp.")
    parser.add_argument("--output-path", help="Override output_path from config.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.model_type or args.start or args.end or args.output_path:
        config = PathSimulationConfig(
            database_path=config.database_path,
            symbol=config.symbol,
            interval=config.interval,
            model_type=args.model_type or config.model_type,
            horizon_steps=config.horizon_steps,
            n_paths=config.n_paths,
            seed=config.seed,
            start=args.start if args.start is not None else config.start,
            end=args.end if args.end is not None else config.end,
            start_price=config.start_price,
            time_step=config.time_step,
            steps_per_day=config.steps_per_day,
            output_path=Path(args.output_path) if args.output_path else config.output_path,
        )

    result = run_simulation(config)
    if config.output_path is not None:
        output_path = write_paths(result.paths, config.output_path)
        result.summary["output_path"] = str(output_path)

    print(_format_summary(result.summary))
    return 0


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"outputs", "runs"}:
        return (Path.cwd() / path).resolve()
    config_relative = (base_dir / path).resolve()
    if config_relative.exists():
        return config_relative
    return (Path.cwd() / path).resolve()


def _future_time_grid(last_time: pd.Timestamp, time_step: str, horizon_steps: int) -> pd.DatetimeIndex:
    delta = pd.Timedelta(time_step)
    start = pd.Timestamp(last_time)
    return pd.date_range(start=start, periods=horizon_steps + 1, freq=delta, tz="UTC")


def _ensure_allowed_artifact_path(path: Path) -> None:
    normalized = path.resolve()
    allowed_roots = [(Path.cwd() / "outputs").resolve(), (Path.cwd() / "runs").resolve()]
    if not any(normalized == root or root in normalized.parents for root in allowed_roots):
        raise ValueError("Generated path artifacts must be written under outputs/ or runs/")


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Monte-Carlo path simulation summary",
        f"symbol: {summary['symbol']}",
        f"historical_period: {summary['historical_start']} -> {summary['historical_end']}",
        f"historical_observations: {summary['historical_observations']}",
        f"model_type: {summary['model_type']}",
        f"n_paths: {summary['n_paths']}",
        f"horizon_steps: {summary['horizon_steps']}",
        f"seed: {summary['seed']}",
        "simulated_final_prices:",
        f"  min: {summary['final_price_min']:.6f}",
        f"  mean: {summary['final_price_mean']:.6f}",
        f"  median: {summary['final_price_median']:.6f}",
        f"  max: {summary['final_price_max']:.6f}",
        f"  std: {summary['final_price_std']:.6f}",
        f"  q05: {summary['final_price_q05']:.6f}",
        f"  q95: {summary['final_price_q95']:.6f}",
        "data_quality:",
        f"  missing_by_column: {summary['data_quality']['missing_by_column']}",
        f"  duplicate_open_time_count: {summary['data_quality']['duplicate_open_time_count']}",
        f"  invalid_ohlc_count: {summary['data_quality']['invalid_ohlc_count']}",
        f"  non_positive_price_count: {summary['data_quality']['non_positive_price_count']}",
        f"  gap_count: {summary['data_quality']['gap_count']}",
        f"  max_gap_seconds: {summary['data_quality']['max_gap_seconds']}",
    ]
    if "output_path" in summary:
        lines.append(f"output_path: {summary['output_path']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
