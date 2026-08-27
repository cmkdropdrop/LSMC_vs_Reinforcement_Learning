"""Official Binance USD-M monthly kline loader with checksum verification."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]


def month_range(start: str, end: str) -> list[str]:
    """Return inclusive YYYY-MM labels."""
    periods = pd.period_range(start=start, end=end, freq="M")
    return [str(period) for period in periods]


def _request(url: str, timeout: int = 90) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_month(symbol: str, interval: str, month: str, cache: Path) -> Path:
    """Download one official archive and verify its published SHA256 hash."""
    cache.mkdir(parents=True, exist_ok=True)
    name = f"{symbol}-{interval}-{month}.zip"
    path = cache / name
    checksum_path = cache / f"{name}.CHECKSUM"
    url = f"{BASE}/{symbol}/{interval}/{name}"

    if not path.exists():
        path.write_bytes(_request(url))
    if not checksum_path.exists():
        checksum_path.write_bytes(_request(f"{url}.CHECKSUM"))

    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {expected}")
    return path


def read_month(path: Path, symbol: str) -> pd.DataFrame:
    """Read only timestamp, open, and close from one kline archive."""
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one CSV in {path}, found {members}")
        frame = pd.read_csv(
            archive.open(members[0]),
            header=None,
            names=KLINE_COLUMNS,
            usecols=["open_time", "open", "close"],
        )
    frame["symbol"] = symbol
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    frame["open"] = pd.to_numeric(frame["open"], errors="raise").astype(float)
    frame["close"] = pd.to_numeric(frame["close"], errors="raise").astype(float)
    return frame


def load_panel(
    symbols: list[str],
    interval: str,
    first_month: str,
    last_month: str,
    cache: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load synchronized open and close panels for all symbols."""
    frames: list[pd.DataFrame] = []
    checksums: dict[str, str] = {}
    for symbol in symbols:
        for month in month_range(first_month, last_month):
            path = download_month(symbol, interval, month, cache)
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames.append(read_month(path, symbol))

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(["symbol", "open_time"], keep="last")
    raw["timestamp"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    opens = raw.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = raw.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    common = opens.dropna().index.intersection(closes.dropna().index)
    opens = opens.loc[common, symbols]
    closes = closes.loc[common, symbols]
    if opens.empty or closes.empty:
        raise RuntimeError("No synchronized market panel was constructed")
    return opens, closes, checksums
