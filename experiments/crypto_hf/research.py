"""Walk-forward parameter search with an untouched 2026 holdout."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_panel
from strategy import StrategyConfig, calculate_metrics, prepare_signal_data, run_backtest

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT",
]
FOLDS = [
    ("F1", "2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"),
    ("F2", "2024-07-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    ("F3", "2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z"),
    ("F4", "2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z"),
]
HOLDOUT = ("2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z")
BASE_COST_BPS = 10.0
DISPERSION_WINDOW_BARS = 30 * 24 * 12


def build_grid() -> list[StrategyConfig]:
    """Predeclared grid; the holdout is not involved in its construction."""
    configs: list[StrategyConfig] = []
    for mode, lookback, hold, vol, quantile in itertools.product(
        ["momentum", "reversal"],
        [3, 6, 12, 24, 48, 96],
        [6, 12],
        [0, 288],
        [0.0, 0.5, 0.7, 0.8],
    ):
        configs.append(StrategyConfig(
            mode=mode,
            lookback_bars=lookback,
            hold_bars=hold,
            volatility_window_bars=vol,
            top_k=2,
            roundtrip_cost_bps=BASE_COST_BPS,
            dispersion_quantile=quantile,
            dispersion_window_bars=DISPERSION_WINDOW_BARS,
        ))
    return configs


def slice_result(
    cycles: pd.DataFrame,
    trades: pd.DataFrame,
    config: StrategyConfig,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cycle_mask = (
        (pd.to_datetime(cycles["entry_time"], utc=True) >= start_ts)
        & (pd.to_datetime(cycles["exit_time"], utc=True) <= end_ts)
    )
    trade_mask = (
        (pd.to_datetime(trades["entry_time"], utc=True) >= start_ts)
        & (pd.to_datetime(trades["exit_time"], utc=True) <= end_ts)
    )
    fold_cycles = cycles.loc[cycle_mask].reset_index(drop=True)
    fold_trades = trades.loc[trade_mask].reset_index(drop=True)
    return fold_cycles, fold_trades, calculate_metrics(fold_cycles, fold_trades, config)


def utility(metrics: dict[str, float]) -> float:
    return float(
        metrics["sharpe_ratio"]
        + 0.04 * metrics["total_return_pct"]
        + 0.025 * metrics["maximum_drawdown_pct"]
        + 0.50 * (metrics["positive_month_fraction"] - 0.50)
    )


def summarize(fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for config_id, group in fold_rows.groupby("config_id", sort=False):
        utilities = group["fold_utility"].astype(float)
        complete = len(group) == len(FOLDS)
        frequency_ok = bool((group["minimum_monthly_trades"] >= 1000.0).all())
        robust = float(utilities.median() + 0.5 * utilities.min() - 0.25 * utilities.std(ddof=0))
        first = group.iloc[0]
        rows.append({
            "config_id": config_id,
            "complete_four_folds": complete,
            "frequency_constraint_met": frequency_ok,
            "robust_score": robust,
            "median_fold_utility": float(utilities.median()),
            "minimum_fold_utility": float(utilities.min()),
            "std_fold_utility": float(utilities.std(ddof=0)),
            "median_return_pct": float(group["total_return_pct"].median()),
            "minimum_return_pct": float(group["total_return_pct"].min()),
            "median_sharpe": float(group["sharpe_ratio"].median()),
            "minimum_sharpe": float(group["sharpe_ratio"].min()),
            "median_drawdown_pct": float(group["maximum_drawdown_pct"].median()),
            "minimum_monthly_trades": float(group["minimum_monthly_trades"].min()),
            "mode": first["mode"],
            "lookback_bars": int(first["lookback_bars"]),
            "hold_bars": int(first["hold_bars"]),
            "volatility_window_bars": int(first["volatility_window_bars"]),
            "top_k": int(first["top_k"]),
            "dispersion_quantile": float(first["dispersion_quantile"]),
            "dispersion_window_bars": int(first["dispersion_window_bars"]),
            "roundtrip_cost_bps": float(first["roundtrip_cost_bps"]),
        })
    return pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True)


def monthly_table(cycles: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    cycle_month = pd.to_datetime(cycles["entry_time"], utc=True).dt.to_period("M")
    trade_month = pd.to_datetime(trades["entry_time"], utc=True).dt.to_period("M")
    returns = cycles.groupby(cycle_month, observed=True)["net_return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0) * 100.0
    )
    gross = cycles.groupby(cycle_month, observed=True)["gross_return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0) * 100.0
    )
    counts = trades.groupby(trade_month, observed=True).size()
    result = pd.DataFrame({
        "net_return_pct": returns,
        "gross_return_pct": gross,
        "round_trip_trades": counts,
    }).reset_index(names="month")
    result["month"] = result["month"].astype(str)
    return result


def signal_inputs(
    closes: pd.DataFrame,
    config: StrategyConfig,
    score_cache: dict[tuple[int, int], pd.DataFrame],
    dispersion_cache: dict[tuple[int, int], pd.Series],
    threshold_cache: dict[tuple[int, int, float], pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    score_key = (config.lookback_bars, config.volatility_window_bars)
    if score_key not in score_cache:
        ungated = StrategyConfig(
            mode=config.mode,
            lookback_bars=config.lookback_bars,
            hold_bars=config.hold_bars,
            volatility_window_bars=config.volatility_window_bars,
            top_k=config.top_k,
            roundtrip_cost_bps=config.roundtrip_cost_bps,
            dispersion_quantile=0.0,
            dispersion_window_bars=config.dispersion_window_bars,
        )
        scores, _ = prepare_signal_data(closes, ungated)
        score_cache[score_key] = scores
        dispersion_cache[score_key] = (
            scores.max(axis=1, skipna=True) - scores.min(axis=1, skipna=True)
        )
    threshold_key = (*score_key, config.dispersion_quantile)
    if threshold_key not in threshold_cache:
        if config.dispersion_quantile <= 0.0:
            threshold = pd.Series(-np.inf, index=closes.index, name="dispersion_threshold")
        else:
            threshold = (
                dispersion_cache[score_key]
                .rolling(
                    config.dispersion_window_bars,
                    min_periods=config.dispersion_window_bars,
                )
                .quantile(config.dispersion_quantile)
                .shift(1)
                .rename("dispersion_threshold")
            )
        threshold_cache[threshold_key] = threshold
    return score_cache[score_key], threshold_cache[threshold_key]


def main() -> None:
    output = Path("outputs/crypto_hf")
    cache = Path(".cache/crypto_hf")
    output.mkdir(parents=True, exist_ok=True)
    print("Loading verified official Binance archives", flush=True)
    opens, closes, checksums = load_panel(
        SYMBOLS, "5m", "2023-12", "2026-07", cache
    )
    panel_meta = {
        "symbols": SYMBOLS,
        "rows": int(len(opens)),
        "start": opens.index.min().isoformat(),
        "end": opens.index.max().isoformat(),
        "archive_count": len(checksums),
        "checksums": checksums,
    }
    (output / "data_manifest.json").write_text(
        json.dumps(panel_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    fold_rows: list[dict[str, object]] = []
    config_map: dict[str, StrategyConfig] = {}
    score_cache: dict[tuple[int, int], pd.DataFrame] = {}
    dispersion_cache: dict[tuple[int, int], pd.Series] = {}
    threshold_cache: dict[tuple[int, int, float], pd.Series] = {}
    configs = build_grid()
    for number, config in enumerate(configs, start=1):
        config_map[config.config_id] = config
        scores, threshold = signal_inputs(
            closes, config, score_cache, dispersion_cache, threshold_cache
        )
        cycles, trades, _ = run_backtest(
            opens,
            closes,
            config,
            FOLDS[0][1],
            FOLDS[-1][2],
            include_trade_details=False,
            prepared_scores=scores,
            prepared_threshold=threshold,
        )
        for fold_id, start, end in FOLDS:
            _, _, metrics = slice_result(cycles, trades, config, start, end)
            fold_rows.append({
                **config.to_dict(),
                "fold_id": fold_id,
                "start": start,
                "end": end,
                **metrics,
                "fold_utility": utility(metrics),
            })
        if number % 16 == 0:
            print(f"Completed {number}/{len(configs)} configurations", flush=True)

    fold_frame = pd.DataFrame(fold_rows)
    summary = summarize(fold_frame)
    eligible = summary[
        summary["complete_four_folds"] & summary["frequency_constraint_met"]
    ]
    if eligible.empty:
        raise RuntimeError("No configuration met the four-fold trade-frequency constraint")
    selected_row = eligible.iloc[0]
    selected = config_map[str(selected_row["config_id"])]
    selected_scores, selected_threshold = signal_inputs(
        closes, selected, score_cache, dispersion_cache, threshold_cache
    )

    holdout_cycles, holdout_trades, holdout_metrics = run_backtest(
        opens,
        closes,
        selected,
        HOLDOUT[0],
        HOLDOUT[1],
        prepared_scores=selected_scores,
        prepared_threshold=selected_threshold,
    )
    monthly = monthly_table(holdout_cycles, holdout_trades)

    sensitivity_rows: list[dict[str, object]] = []
    for cost in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
        cost_config = StrategyConfig(
            mode=selected.mode,
            lookback_bars=selected.lookback_bars,
            hold_bars=selected.hold_bars,
            volatility_window_bars=selected.volatility_window_bars,
            top_k=selected.top_k,
            roundtrip_cost_bps=cost,
            dispersion_quantile=selected.dispersion_quantile,
            dispersion_window_bars=selected.dispersion_window_bars,
        )
        _, _, metrics = run_backtest(
            opens,
            closes,
            cost_config,
            HOLDOUT[0],
            HOLDOUT[1],
            include_trade_details=False,
            prepared_scores=selected_scores,
            prepared_threshold=selected_threshold,
        )
        sensitivity_rows.append({"roundtrip_cost_bps": cost, **metrics})

    fold_frame.to_csv(output / "development_fold_results.csv", index=False)
    summary.to_csv(output / "configuration_ranking.csv", index=False)
    holdout_cycles.to_csv(output / "holdout_cycles.csv", index=False)
    holdout_trades.to_csv(output / "holdout_trades.csv", index=False)
    monthly.to_csv(output / "holdout_monthly.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output / "cost_sensitivity.csv", index=False)

    selection = {
        "selected_config": selected.to_dict(),
        "selection_method": "Four pre-holdout six-month folds with a minimum of 1,000 round trips in every month",
        "development_robust_score": float(selected_row["robust_score"]),
        "holdout_start": HOLDOUT[0],
        "holdout_end": HOLDOUT[1],
        "holdout_metrics": holdout_metrics,
        "minimum_holdout_monthly_trades": int(monthly["round_trip_trades"].min()),
        "historical_target_met": bool(
            holdout_metrics["total_return_pct"] > 0.0
            and int(monthly["round_trip_trades"].min()) > 1000
        ),
        "funding_note": "Funding payments are not in the kline files; the short holding period reduces but does not eliminate this unmodelled component.",
    }
    (output / "selected_strategy.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nTOP CONFIGURATIONS")
    print(summary.head(15).to_string(index=False))
    print("\nSELECTED STRATEGY")
    print(json.dumps(selection, indent=2, sort_keys=True))
    print("\nHOLDOUT MONTHLY")
    print(monthly.to_string(index=False))
    print("\nCOST SENSITIVITY")
    print(pd.DataFrame(sensitivity_rows).to_string(index=False))


if __name__ == "__main__":
    main()
