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


def run_backtest(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    config: StrategyConfig,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run non-overlapping round trips with next-bar-open execution.

    Signals use closes through bar t. Entry occurs at the open of t+1 and exit
    occurs exactly ``hold_bars`` later at the open. Each long or short position
    opened and closed is counted as one round-trip trade.
    """
    if config.mode not in {"momentum", "reversal"}:
        raise ValueError(f"Unsupported mode: {config.mode}")
    if config.top_k < 1 or config.top_k * 2 > len(opens.columns):
        raise ValueError("top_k is incompatible with the number of symbols")
    if not opens.index.equals(closes.index) or list(opens.columns) != list(closes.columns):
        raise ValueError("Open and close panels must be aligned")

    score = _scores(closes, config)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    interval = pd.Timedelta(minutes=config.bar_minutes)
    hold_delta = interval * config.hold_bars
    epoch_bar = opens.index.view("int64") // interval.value
    entry_mask = (epoch_bar % config.hold_bars) == 0
    candidate_entries = np.flatnonzero(entry_mask)

    cycles: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    cost = config.roundtrip_cost_bps / 10000.0

    for entry_pos in candidate_entries:
        signal_pos = entry_pos - 1
        exit_pos = entry_pos + config.hold_bars
        if signal_pos < 0 or exit_pos >= len(opens):
            continue
        entry_time = opens.index[entry_pos]
        signal_time = opens.index[signal_pos] + interval
        exit_time = opens.index[exit_pos]
        if entry_time < start_ts or exit_time > end_ts:
            continue
        if entry_time - opens.index[signal_pos] != interval:
            continue
        if exit_time - entry_time != hold_delta:
            continue

        row = score.iloc[signal_pos].dropna().sort_values()
        if len(row) < config.top_k * 2:
            continue
        low = row.index[: config.top_k].tolist()
        high = row.index[-config.top_k :].tolist()
        if config.mode == "momentum":
            long_symbols, short_symbols = high, low
        else:
            long_symbols, short_symbols = low, high

        entry_prices = opens.iloc[entry_pos]
        exit_prices = opens.iloc[exit_pos]
        if not np.isfinite(entry_prices[long_symbols + short_symbols]).all():
            continue
        if not np.isfinite(exit_prices[long_symbols + short_symbols]).all():
            continue

        long_returns = exit_prices[long_symbols] / entry_prices[long_symbols] - 1.0
        short_returns = 1.0 - exit_prices[short_symbols] / entry_prices[short_symbols]
        leg_returns = pd.concat([long_returns, short_returns])
        gross_cycle = float(leg_returns.mean())
        net_cycle = gross_cycle - cost
        cycles.append({
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "gross_return": gross_cycle,
            "net_return": net_cycle,
            "long_symbols": ",".join(long_symbols),
            "short_symbols": ",".join(short_symbols),
        })

        for side, symbols, returns in (
            ("LONG", long_symbols, long_returns),
            ("SHORT", short_symbols, short_returns),
        ):
            for symbol in symbols:
                leg_net = float(returns[symbol]) - cost
                trades.append({
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": float(entry_prices[symbol]),
                    "exit_price": float(exit_prices[symbol]),
                    "gross_return": float(returns[symbol]),
                    "net_return": leg_net,
                    "roundtrip_cost_bps": config.roundtrip_cost_bps,
                })

    cycle_frame = pd.DataFrame(cycles)
    trade_frame = pd.DataFrame(trades)
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
