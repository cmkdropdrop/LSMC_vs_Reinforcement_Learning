from __future__ import annotations

import numpy as np
import pandas as pd

from lsmc_rl.volatility import GJRGARCHModel, HARRVModel


def test_gjr_garch_fit_and_simulation_are_deterministic() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(0.0001 + 0.01 * rng.normal(size=300))

    model = GJRGARCHModel().fit(returns)
    prices_a, returns_a, variances_a = model.simulate(20.0, horizon_steps=8, n_paths=4, seed=7)
    prices_b, returns_b, variances_b = model.simulate(20.0, horizon_steps=8, n_paths=4, seed=7)

    assert prices_a.shape == (4, 9)
    assert returns_a.shape == (4, 9)
    assert variances_a.shape == (4, 9)
    assert np.all(prices_a > 0.0)
    np.testing.assert_allclose(prices_a, prices_b)
    np.testing.assert_allclose(returns_a, returns_b)
    np.testing.assert_allclose(variances_a[:, 1:], variances_b[:, 1:])
    assert model.params is not None
    assert model.params.persistence < 1.0


def test_har_rv_features_use_only_past_realized_variance() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=7, tz="UTC"),
            "realized_variance": np.arange(1.0, 8.0),
        }
    )
    model = HARRVModel()
    features = model.make_features(daily)
    row = features.loc[features["date"] == pd.Timestamp("2024-01-06", tz="UTC")].iloc[0]

    assert row["target_rv"] == 6.0
    assert row["rv_daily"] == 5.0
    assert row["rv_weekly"] == np.mean([1.0, 2.0, 3.0, 4.0, 5.0])
    assert row["rv_monthly"] == np.mean([1.0, 2.0, 3.0, 4.0, 5.0])


def test_har_rv_fit_forecast_and_simulation_are_deterministic() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, tz="UTC"),
            "realized_variance": np.linspace(0.0001, 0.0004, 40),
        }
    )
    model = HARRVModel().fit(daily, step_mean_return=0.0)

    forecast = model.forecast_daily_variance(3)
    prices_a, _, variances_a = model.simulate(30.0, horizon_steps=6, n_paths=3, steps_per_day=3, seed=11)
    prices_b, _, variances_b = model.simulate(30.0, horizon_steps=6, n_paths=3, steps_per_day=3, seed=11)

    assert forecast.shape == (3,)
    assert np.all(forecast > 0.0)
    assert prices_a.shape == (3, 7)
    assert np.all(prices_a > 0.0)
    np.testing.assert_allclose(prices_a, prices_b)
    np.testing.assert_allclose(variances_a[:, 1:], variances_b[:, 1:])
