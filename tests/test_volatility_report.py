from __future__ import annotations

import numpy as np
import pandas as pd

from lsmc_rl.analysis.volatility_report import temporal_return_split, variance_metrics


def test_temporal_return_split_preserves_order_and_anchor() -> None:
    frame = pd.DataFrame(
        {
            "open_datetime": pd.date_range("2024-01-01", periods=101, freq="5min", tz="UTC"),
            "close": np.linspace(10.0, 11.0, 101),
            "log_return": [np.nan] + [0.01] * 100,
        }
    )

    split = temporal_return_split(frame, train_fraction=0.6)

    assert len(split.train_returns) == 60
    assert len(split.test_returns) == 40
    assert split.split_time == frame["open_datetime"].iloc[60]
    assert split.start_price == frame["close"].iloc[60]
    assert split.test_market["open_datetime"].is_monotonic_increasing


def test_variance_metrics_identifies_better_than_mean_forecast() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    forecast = np.array([1.1, 1.9, 3.1, 3.9])

    metrics = variance_metrics(actual, forecast)

    assert metrics["observations"] == 4.0
    assert metrics["mae"] < 0.11
    assert metrics["r2_vs_realized_mean"] > 0.9
