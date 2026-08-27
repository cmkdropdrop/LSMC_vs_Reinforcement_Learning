"""Inspect the read-only 5-minute SQLite input."""

import sqlite3
from pathlib import Path

DB = Path("ttf_klines_5m_from_1m.sqlite")
uri = f"file:{DB.resolve().as_posix()}?mode=ro"
with sqlite3.connect(uri, uri=True) as conn:
    rows = conn.execute(
        "SELECT symbol, interval, COUNT(*), MIN(open_time), MAX(open_time), AVG(volume) "
        "FROM klines GROUP BY symbol, interval ORDER BY symbol"
    ).fetchall()
    months = conn.execute(
        "SELECT symbol, strftime('%Y-%m', open_time / 1000, 'unixepoch') AS month, COUNT(*) "
        "FROM klines GROUP BY symbol, month ORDER BY month, symbol"
    ).fetchall()
print('SERIES')
for row in rows:
    print(row)
print('MONTHS')
for row in months:
    print(row)
