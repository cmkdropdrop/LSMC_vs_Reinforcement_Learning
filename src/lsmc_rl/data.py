"""Read-only market data loading and validation utilities.

The curated source database is treated as immutable input data. All SQLite
connections created here use ``mode=ro``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_KLINES_COLUMNS = (
    "symbol",
    "interval",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True)
class DataQualityReport:
    """Compact diagnostics for one loaded OHLCV time series."""

    rows: int
    symbol: str | None
    interval: str | None
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None
    missing_by_column: dict[str, int]
    duplicate_open_time_count: int
    invalid_ohlc_count: int
    non_positive_price_count: int
    gap_count: int
    max_gap_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat() if self.start_time is not None else None
        data["end_time"] = self.end_time.isoformat() if self.end_time is not None else None
        return data


def connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""

    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_klines_schema(db_path: str | Path) -> list[str]:
    """Return the column names for the ``klines`` table."""

    with connect_read_only(db_path) as conn:
        rows = conn.execute("PRAGMA table_info(klines)").fetchall()
    return [row[1] for row in rows]


def _to_unix_millis(value: str | pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def load_klines(
    db_path: str | Path,
    symbol: str = "FRONT",
    interval: str = "5m",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load one OHLCV series from the curated SQLite database.

    Timestamps are converted from Unix milliseconds to timezone-aware UTC
    datetimes. Rows are always returned sorted by ``open_time``.
    """

    schema = set(get_klines_schema(db_path))
    missing = sorted(set(EXPECTED_KLINES_COLUMNS) - schema)
    if missing:
        raise ValueError(f"klines schema is missing expected columns: {missing}")

    clauses = ["symbol = ?", "interval = ?"]
    params: list[Any] = [symbol, interval]
    start_ms = _to_unix_millis(start)
    end_ms = _to_unix_millis(end)
    if start_ms is not None:
        clauses.append("open_time >= ?")
        params.append(start_ms)
    if end_ms is not None:
        clauses.append("open_time <= ?")
        params.append(end_ms)

    query = f"""
        SELECT {", ".join(EXPECTED_KLINES_COLUMNS)}
        FROM klines
        WHERE {" AND ".join(clauses)}
        ORDER BY open_time ASC
    """
    with connect_read_only(db_path) as conn:
        frame = pd.read_sql_query(query, conn, params=params)

    if frame.empty:
        raise ValueError(f"No klines found for symbol={symbol!r}, interval={interval!r}")

    frame = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    frame["open_datetime"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_datetime"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    return frame


def validate_klines(frame: pd.DataFrame, expected_interval: str = "5m") -> DataQualityReport:
    """Check missing values, duplicates, OHLC consistency, and timestamp gaps."""

    missing_by_column = {
        column: int(frame[column].isna().sum())
        for column in EXPECTED_KLINES_COLUMNS
        if column in frame.columns
    }

    duplicate_open_time_count = int(frame.duplicated(subset=["open_time"]).sum())
    price_columns = ["open", "high", "low", "close"]
    prices = frame[price_columns]
    non_positive_price_count = int((prices <= 0).any(axis=1).sum())

    invalid_ohlc_mask = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())

    expected_delta_ms = int(pd.Timedelta(expected_interval).total_seconds() * 1000)
    unique_times = np.sort(frame["open_time"].dropna().unique())
    if len(unique_times) > 1:
        diffs = np.diff(unique_times)
        gap_mask = diffs > expected_delta_ms
        gap_count = int(gap_mask.sum())
        max_gap_seconds = float(diffs.max() / 1000.0)
    else:
        gap_count = 0
        max_gap_seconds = None

    symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame and not frame.empty else None
    interval = str(frame["interval"].iloc[0]) if "interval" in frame and not frame.empty else None
    start_time = frame["open_datetime"].iloc[0] if "open_datetime" in frame and not frame.empty else None
    end_time = frame["open_datetime"].iloc[-1] if "open_datetime" in frame and not frame.empty else None

    return DataQualityReport(
        rows=int(len(frame)),
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        missing_by_column=missing_by_column,
        duplicate_open_time_count=duplicate_open_time_count,
        invalid_ohlc_count=invalid_ohlc_count,
        non_positive_price_count=non_positive_price_count,
        gap_count=gap_count,
        max_gap_seconds=max_gap_seconds,
    )


def add_log_returns(
    frame: pd.DataFrame,
    price_col: str = "close",
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Add close-to-close log returns without lookahead."""

    if (frame[price_col] <= 0).any():
        raise ValueError(f"Cannot compute log returns with non-positive {price_col} values")
    result = frame.sort_values("open_time", kind="mergesort").reset_index(drop=True).copy()
    result[return_col] = np.log(result[price_col]).diff()
    return result


def load_market_data(
    db_path: str | Path,
    symbol: str = "FRONT",
    interval: str = "5m",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load, validate, and enrich one symbol with log returns."""

    frame = load_klines(db_path=db_path, symbol=symbol, interval=interval, start=start, end=end)
    report = validate_klines(frame, expected_interval=interval)
    return add_log_returns(frame), report
