from __future__ import annotations

import numpy as np
import pandas as pd

from lsmc_rl.analysis.path_quality_report import (
    energy_score,
    multiband_mmd,
    normal_nll,
    observed_daily_feature_bands,
    qlike_loss,
    simulated_daily_feature_bands,
    standardize_pair,
)


def test_qlike_and_normal_nll_are_finite_for_positive_variances() -> None:
    actual_variance = np.array([0.2, 0.3, 0.4])
    forecast_variance = np.array([0.25, 0.35, 0.45])
    actual_return = np.array([0.01, -0.02, 0.03])
    forecast_mean = np.zeros(3)

    assert np.isfinite(qlike_loss(actual_variance, forecast_variance))
    assert np.isfinite(normal_nll(actual_return, forecast_mean, forecast_variance))


def test_energy_score_prefers_matching_ensemble() -> None:
    observations = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]])
    close_ensemble = observations.copy()
    far_ensemble = observations + 10.0

    assert energy_score(observations, close_ensemble) < energy_score(observations, far_ensemble)


def test_multiband_mmd_prefers_matching_samples() -> None:
    observed = {"core": np.array([[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]])}
    matching = {"core": observed["core"].copy()}
    shifted = {"core": observed["core"] + 5.0}

    assert multiband_mmd(observed, matching)["aggregate_mmd2"] < multiband_mmd(observed, shifted)["aggregate_mmd2"]


def test_feature_band_builders_return_expected_bands() -> None:
    returns = np.array([0.01, -0.02, 0.005, 0.003] * 4)
    frame = pd.DataFrame(
        {
            "open_datetime": pd.date_range("2024-01-01", periods=len(returns), freq="5min", tz="UTC"),
            "log_return": returns,
        }
    )
    observed = observed_daily_feature_bands(frame, min_returns=4)

    prices = np.column_stack(
        [
            np.full(3, 100.0),
            100.0 * np.exp(np.cumsum(np.tile(returns, (3, 1)), axis=1)),
        ]
    )
    simulated = simulated_daily_feature_bands(prices)

    assert set(observed) == {"core", "extremes", "multi_horizon"}
    assert set(simulated) == set(observed)
    assert observed["core"].shape[1] == simulated["core"].shape[1]


def test_standardize_pair_handles_constant_columns() -> None:
    observed = np.ones((3, 2))
    simulated = np.ones((4, 2)) * 2.0

    obs_scaled, sim_scaled = standardize_pair(observed, simulated)

    assert np.isfinite(obs_scaled).all()
    assert np.isfinite(sim_scaled).all()
