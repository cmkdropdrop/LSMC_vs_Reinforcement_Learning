"""Leakage-safe cross-sectional crypto futures strategy and backtest."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    mode: str
    lookback_bars: int
    hold_bars: int
    volatility_window_bars: int
    top_k: int
    roundtrip_cost_bps: float
    bar_minutes: int = 5

    @property
    def config_id(self) -> str:
        vol = f"v{self.volatility_window_bars}" if self.volatility_window_bars else "raw"
        return (
            f"{self.mode}_l{self.lookback_bars}_h{self.hold_bars}_"
            f"{vol}_k{self.top_k}_c{self.roundtrip_cost_bps:g}"
        )

    def to_dict(self) -> dict[str, object]:
        return {"config_id": self.config_id, **asdict(self)}


def _scores(closes: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    log_close = np.log(closes)
    trailing = log_close.diff(config.lookback_bars)
    if config.volatility_window_bars > 0:
        one_bar = log_close.diff()
        scale = one_bar.rolling(
            config.volatility_window_bars,
            min_periods=config.volatility_window_bars,
        ).std(ddof=1) * np.sqrt(config.lookback_bars)
        trailing = trailing / scale.replace(0.0, np.nan)
    return trailing.replace([np.inf, -np.inf], np.nan)


def _utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def run_backtest(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    config: StrategyConfig,
    start: str,
    end: str,
    *,
    include_trade_details: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run non-overlapping round trips with next-bar-open execution.

    Signals use closes through bar t. Entry occurs at the open of t+1 and exit
    occurs exactly ``hold_bars`` later at the open. Each long or short position
    opened and closed is counted as one round-trip trade. The implementation is
    vectorized but follows the same ordering and timing rules as a bar loop.
    """
    if config.mode not in {"momentum", "reversal"}:
        raise ValueError(f"Unsupported mode: {config.mode}")
    if config.top_k < 1 or config.top_k * 2 > len(opens.columns):
        raise ValueError("top_k is incompatible with the number of symbols")
    if not opens.index.equals(closes.index) or list(opens.columns) != list(closes.columns):
        raise ValueError("Open and close panels must be aligned")

    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)
    interval = pd.Timedelta(minutes=config.bar_minutes)
    hold_delta = interval * config.hold_bars
    index_ns = opens.index.asi8

    epoch_bar = index_ns // interval.value
    entry_positions = np.flatnonzero((epoch_bar % config.hold_bars) == 0)
    signal_positions = entry_positions - 1
    exit_positions = entry_positions + config.hold_bars
    valid = (signal_positions >= 0) & (exit_positions < len(opens))
    entry_positions = entry_positions[valid]
    signal_positions = signal_positions[valid]
    exit_positions = exit_positions[valid]

    valid = (
        (index_ns[entry_positions] >= start_ts.value)
        & (index_ns[exit_positions] <= end_ts.value)
        & ((index_ns[entry_positions] - index_ns[signal_positions]) == interval.value)
        & ((index_ns[exit_positions] - index_ns[entry_positions]) == hold_delta.value)
    )
    entry_positions = entry_positions[valid]
    signal_positions = signal_positions[valid]
    exit_positions = exit_positions[valid]

    if len(entry_positions) == 0:
        empty = pd.DataFrame()
        return empty, empty, calculate_metrics(empty, empty, config)

    score_rows = _scores(closes, config).to_numpy(dtype=float)[signal_positions]
    finite_scores = np.isfinite(score_rows)
    valid = finite_scores.sum(axis=1) >= config.top_k * 2
    entry_positions = entry_positions[valid]
    signal_positions = signal_positions[valid]
    exit_positions = exit_positions[valid]
    score_rows = score_rows[valid]
    finite_scores = finite_scores[valid]

    if len(entry_positions) == 0:
        empty = pd.DataFrame()
        return empty, empty, calculate_metrics(empty, empty, config)

    sortable = np.where(finite_scores, score_rows, np.inf)
    sorted_indices = np.argsort(sortable, axis=1, kind="stable")
    low_indices = sorted_indices[:, : config.top_k]
    high_indices = sorted_indices[:, -config.top_k :]
    if config.mode == "momentum":
        long_indices, short_indices = high_indices, low_indices
    else:
        long_indices, short_indices = low_indices, high_indices

    open_values = opens.to_numpy(dtype=float)
    entry_rows = open_values[entry_positions]
    exit_rows = open_values[exit_positions]
    long_entry = np.take_along_axis(entry_rows, long_indices, axis=1)
    long_exit = np.take_along_axis(exit_rows, long_indices, axis=1)
    short_entry = np.take_along_axis(entry_rows, short_indices, axis=1)
    short_exit = np.take_along_axis(exit_rows, short_indices, axis=1)

    valid = (
        np.isfinite(long_entry).all(axis=1)
        & np.isfinite(long_exit).all(axis=1)
        & np.isfinite(short_entry).all(axis=1)
        & np.isfinite(short_exit).all(axis=1)
        & (long_entry > 0.0).all(axis=1)
        & (short_entry > 0.0).all(axis=1)
    )
    entry_positions = entry_positions[valid]
    signal_positions = signal_positions[valid]
    exit_positions = exit_positions[valid]
    long_indices = long_indices[valid]
    short_indices = short_indices[valid]
    long_entry = long_entry[valid]
    long_exit = long_exit[valid]
    short_entry = short_entry[valid]
    short_exit = short_exit[valid]

    long_returns = long_exit / long_entry - 1.0
    short_returns = 1.0 - short_exit / short_entry
    selected_returns = np.concatenate([long_returns, short_returns], axis=1)
    gross_cycles = selected_returns.mean(axis=1)
    cost = config.roundtrip_cost_bps / 10000.0
    net_cycles = gross_cycles - cost

    signal_times = opens.index.take(signal_positions) + interval
    entry_times = opens.index.take(entry_positions)
    exit_times = opens.index.take(exit_positions)
    cycle_data: dict[str, object] = {
        "signal_time": signal_times,
        "entry_time": entry_times,
        "exit_time": exit_times,
        "gross_return": gross_cycles,
        "net_return": net_cycles,
    }
    symbols = np.asarray(opens.columns, dtype=object)
    if include_trade_details:
        cycle_data["long_symbols"] = [
            ",".join(symbols[row].tolist()) for row in long_indices
        ]
        cycle_data["short_symbols"] = [
            ",".join(symbols[row].tolist()) for row in short_indices
        ]
    cycle_frame = pd.DataFrame(cycle_data)

    selected_indices = np.concatenate([long_indices, short_indices], axis=1)
    selected_entries = np.concatenate([long_entry, short_entry], axis=1)
    selected_exits = np.concatenate([long_exit, short_exit], axis=1)
    legs_per_cycle = config.top_k * 2
    sides = np.asarray(["LONG"] * config.top_k + ["SHORT"] * config.top_k, dtype=object)
    trade_data: dict[str, object] = {
        "signal_time": np.repeat(signal_times.to_numpy(), legs_per_cycle),
        "entry_time": np.repeat(entry_times.to_numpy(), legs_per_cycle),
        "exit_time": np.repeat(exit_times.to_numpy(), legs_per_cycle),
        "gross_return": selected_returns.reshape(-1),
        "net_return": (selected_returns - cost).reshape(-1),
        "roundtrip_cost_bps": np.full(len(selected_returns) * legs_per_cycle, config.roundtrip_cost_bps),
    }
    if include_trade_details:
        trade_data.update({
            "symbol": symbols[selected_indices].reshape(-1),
            "side": np.tile(sides, len(selected_returns)),
            "entry_price": selected_entries.reshape(-1),
            "exit_price": selected_exits.reshape(-1),
        })
    trade_frame = pd.DataFrame(trade_data)

    metrics = calculate_metrics(cycle_frame, trade_frame, config)
    return cycle_frame, trade_frame, metrics


def calculate_metrics(
    cycles: pd.DataFrame,
    trades: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, float]:
    if cycles.empty:
        return {
            "total_return_pct": -100.0,
            "gross_total_return_pct": -100.0,
            "sharpe_ratio": -999.0,
            "maximum_drawdown_pct": -100.0,
            "round_trip_trades": 0.0,
            "average_monthly_trades": 0.0,
            "minimum_monthly_trades": 0.0,
            "positive_month_fraction": 0.0,
            "average_net_cycle_bps": 0.0,
            "trade_win_rate_pct": 0.0,
        }

    net = cycles["net_return"].astype(float)
    gross = cycles["gross_return"].astype(float)
    equity = (1.0 + net).cumprod()
    gross_equity = (1.0 + gross).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    cycles_per_year = 365.0 * 24.0 * 60.0 / (config.hold_bars * config.bar_minutes)
    sharpe = 0.0
    if len(net) > 1 and float(net.std(ddof=1)) > 0.0:
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(cycles_per_year))

    entry_month = pd.to_datetime(trades["entry_time"], utc=True).dt.to_period("M")
    monthly_trades = trades.groupby(entry_month, observed=True).size()
    cycle_month = pd.to_datetime(cycles["entry_time"], utc=True).dt.to_period("M")
    monthly_return = cycles.groupby(cycle_month, observed=True)["net_return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    return {
        "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "gross_total_return_pct": float((gross_equity.iloc[-1] - 1.0) * 100.0),
        "sharpe_ratio": sharpe,
        "maximum_drawdown_pct": float(drawdown.min() * 100.0),
        "round_trip_trades": float(len(trades)),
        "average_monthly_trades": float(monthly_trades.mean()),
        "minimum_monthly_trades": float(monthly_trades.min()),
        "positive_month_fraction": float((monthly_return > 0.0).mean()),
        "average_net_cycle_bps": float(net.mean() * 10000.0),
        "trade_win_rate_pct": float((trades["net_return"] > 0.0).mean() * 100.0),
    }
