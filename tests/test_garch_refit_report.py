from __future__ import annotations

import numpy as np
import pandas as pd

from lsmc_rl.analysis.garch_refit_report import daily_coverage_metrics, daily_forecast_frame


def test_daily_forecast_frame_aggregates_intraday_variances() -> None:
    step_frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="12h", tz="UTC"),
            "actual_return": [0.01, -0.02, 0.03, 0.04],
            "forecast_mean": [0.001, 0.001, 0.002, 0.002],
            "forecast_variance": [0.1, 0.2, 0.3, 0.4],
        }
    )

    daily = daily_forecast_frame(step_frame)

    assert len(daily) == 2
    np.testing.assert_allclose(daily["actual_rv"].iloc[0], 0.01**2 + (-0.02) ** 2)
    np.testing.assert_allclose(daily["forecast_rv"].iloc[0], 0.3)
    np.testing.assert_allclose(daily["forecast_mean"].iloc[1], 0.004)


def test_daily_coverage_metrics_returns_expected_keys() -> None:
    daily = pd.DataFrame(
        {
            "standardized_residual": [-2.0, -0.5, 0.0, 0.5, 2.0],
        }
    )

    metrics = daily_coverage_metrics(daily)

    assert metrics["observations"] == 5.0
    assert "var_5pct_hit_rate" in metrics
    assert "standardized_residual_std" in metrics
