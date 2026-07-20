"""
Lightweight monitoring: every query's stage latencies, retrieved chunk ids,
cache hit/miss, and rewritten query are appended as one JSON line to a log
file. The Streamlit "Monitoring" tab reads this file back to plot
latency-over-time and cache hit-rate, no external observability stack
required.
"""
import json
import time
from contextlib import contextmanager
from typing import Any, Dict

import pandas as pd

from . import config


@contextmanager
def timer():
    start = time.perf_counter()
    box = {}
    yield box
    box["elapsed_ms"] = (time.perf_counter() - start) * 1000


def log_event(event: Dict[str, Any]) -> None:
    event["timestamp"] = time.time()
    with open(config.MONITOR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_events() -> pd.DataFrame:
    if not config.MONITOR_LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(config.MONITOR_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df
