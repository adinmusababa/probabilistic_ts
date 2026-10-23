"""Fitur waktu dari kolom date (polars), diskalakan ke [-0.5, 0.5]."""
from __future__ import annotations

import numpy as np
import polars as pl

NUM_TIME_FEATURES = 4  # jam, hari dalam minggu, hari dalam bulan, bulan


def time_features(df: pl.DataFrame, col: str = "date") -> np.ndarray:
    """Kembalikan array float32 (N, 4): hour, day-of-week, day-of-month, month."""
    out = df.select(
        (pl.col(col).dt.hour().cast(pl.Float64) / 23.0 - 0.5).alias("hour"),
        ((pl.col(col).dt.weekday().cast(pl.Float64) - 1.0) / 6.0 - 0.5).alias("dow"),
        ((pl.col(col).dt.day().cast(pl.Float64) - 1.0) / 30.0 - 0.5).alias("dom"),
        ((pl.col(col).dt.month().cast(pl.Float64) - 1.0) / 11.0 - 0.5).alias("month"),
    )
    return out.to_numpy().astype(np.float32)