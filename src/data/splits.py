"""Pembagian kronologis ETT mengikuti konvensi Autoformer/Informer."""
from __future__ import annotations

import numpy as np

_HOURLY = ("ETTh1", "ETTh2")
_QUARTER = ("ETTm1", "ETTm2")
_MONTH_HOURS = 30 * 24  # 720 jam


def _multiplier(name: str) -> int:
    if name in _HOURLY:
        return 1
    if name in _QUARTER:
        return 4  # 15 menit = 4x lebih banyak titik per jam
    raise ValueError(f"Dataset tidak dikenal: {name}")


def total_rows(name: str) -> int:
    return 20 * _MONTH_HOURS * _multiplier(name)


def ett_segments(name: str) -> dict[str, tuple[int, int]]:
    m = _multiplier(name)
    train_end = 12 * _MONTH_HOURS * m
    val_end = train_end + 4 * _MONTH_HOURS * m
    test_end = val_end + 4 * _MONTH_HOURS * m
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}


def window_origins(
    segment: tuple[int, int], past_length: int, horizon: int, stride: int = 1
) -> np.ndarray:
    """Indeks t (langkah pertama masa depan) yang valid untuk sebuah segmen.

    Syarat: target t..t+horizon-1 berada di dalam segmen, dan riwayat t-past_length..t-1
    tersedia. Riwayat boleh menjangkau ke segmen sebelumnya (itu masa lalu, bukan kebocoran).
    """
    start, end = segment
    first = max(start, past_length)
    last = end - horizon
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, stride, dtype=np.int64)