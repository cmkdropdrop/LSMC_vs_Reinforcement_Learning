from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lsmc_rl.simulation.paths import PathSimulationConfig, paths_to_frame, run_simulation

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "ttf_klines_5m_from_1m.sqlite"


def test_paths_to_frame_common_interface() -> None:
    prices = np.array([[10.0, 10.1, 10.2], [10.0, 9.9, 10.3]])
    returns = np.array([[0.0, 0.01, 0.00985], [0.0, -0.01, 0.03961]])
    variances = np.array([[np.nan, 0.001, 0.001], [np.nan, 0.001, 0.001]])
    times = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")

    frame = paths_to_frame(prices, returns, variances, times, model="test")

    assert list(frame.columns) == ["path", "step", "time", "price", "return", "variance", "model", "volatility"]
    assert len(frame) == 6
    assert set(frame["path"]) == {0, 1}
    assert frame["step"].max() == 2
    assert (frame["price"] > 0.0).all()


def test_run_simulation_smoke_with_har_rv_repo_data() -> None:
    config = PathSimulationConfig(
        database_path=DB_PATH,
        symbol="FRONT",
        interval="5m",
        model_type="har_rv",
        horizon_steps=4,
        n_paths=2,
        seed=123,
        output_path=None,
    )

    result_a = run_simulation(config)
    result_b = run_simulation(config)

    assert result_a.paths.shape[0] == 2 * (4 + 1)
    assert (result_a.paths["price"] > 0.0).all()
    pd.testing.assert_frame_equal(result_a.paths, result_b.paths)
    assert result_a.summary["model_type"] == "har_rv"
