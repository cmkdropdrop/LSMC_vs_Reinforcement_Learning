from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lsmc_rl.data import EXPECTED_KLINES_COLUMNS, add_log_returns, load_klines, validate_klines

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "ttf_klines_5m_from_1m.sqlite"


def test_load_klines_schema_and_sorting() -> None:
    frame = load_klines(DB_PATH, symbol="FRONT", interval="5m")

    for column in EXPECTED_KLINES_COLUMNS:
        assert column in frame.columns
    assert frame["open_time"].is_monotonic_increasing
    assert str(frame["open_datetime"].dt.tz) == "UTC"


def test_validate_klines_reports_expected_fields() -> None:
    frame = load_klines(DB_PATH, symbol="FRONT", interval="5m")
    report = validate_klines(frame)

    assert report.rows == len(frame)
    assert report.symbol == "FRONT"
    assert report.interval == "5m"
    assert report.duplicate_open_time_count >= 0
    assert report.gap_count >= 0


def test_add_log_returns_uses_only_previous_close() -> None:
    frame = pd.DataFrame(
        {
            "open_time": [1, 2, 3],
            "close": [10.0, 11.0, 12.0],
        }
    )
    result = add_log_returns(frame)

    assert np.isnan(result["log_return"].iloc[0])
    np.testing.assert_allclose(result["log_return"].iloc[1], np.log(11.0 / 10.0))
    np.testing.assert_allclose(result["log_return"].iloc[2], np.log(12.0 / 11.0))
