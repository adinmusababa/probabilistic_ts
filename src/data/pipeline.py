"""Menyatukan: muat -> potong 20 bulan -> split -> scaler (fit train) -> fitur waktu -> window."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.data.scaler import StandardScaler
from src.data.splits import ett_segments, total_rows, window_origins
from src.data.time_features import time_features
from src.data.windows import WindowDataset


@dataclass
class PreparedData:
    datasets: dict[str, WindowDataset]
    scaler: StandardScaler
    values: np.ndarray                    # (N, D) terstandardisasi (float32)
    time_feats: np.ndarray                # (N, F)
    segments: dict[str, tuple[int, int]]
    columns: list[str]
    meta: dict                            # untuk manifest run (FR-D9)


def prepare_data_from_frame(
    df: pl.DataFrame,
    name: str,
    past_length: int,
    horizon: int,
    train_stride: int = 1,
) -> PreparedData:
    n_total = total_rows(name)
    df = df.sort("date")
    if df.height < n_total:
        raise ValueError(f"{name}: butuh {n_total} baris, hanya ada {df.height}.")
    df = df.head(n_total)

    columns = [c for c in df.columns if c != "date"]
    raw = df.select(columns).to_numpy().astype(np.float64)

    segments = ett_segments(name)
    train_end = segments["train"][1]
    scaler = StandardScaler().fit(raw[:train_end])  # hanya train
    values = scaler.transform(raw)
    tf = time_features(df)

    datasets = {}
    for split, seg in segments.items():
        stride = train_stride if split == "train" else 1
        origins = window_origins(seg, past_length, horizon, stride)
        datasets[split] = WindowDataset(values, tf, origins, past_length, horizon)

    meta = {
        "dataset": name,
        "n_rows": int(df.height),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "columns": columns,
        "segments": {k: list(v) for k, v in segments.items()},
        "data_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        "past_length": past_length,
        "horizon": horizon,
    }
    return PreparedData(datasets, scaler, values, tf, segments, columns, meta)


def prepare_data(name: str, past_length: int, horizon: int, train_stride: int = 1) -> PreparedData:
    """Versi yang memuat dari Supabase lewat loader."""
    from src.data.loader import load_ett  # impor lokal agar uji tidak butuh koneksi

    return prepare_data_from_frame(load_ett(name), name, past_length, horizon, train_stride)