"""Baseline deterministik (numpy murni): seasonal naive dan last value."""
from __future__ import annotations

import numpy as np


def season_length(dataset_name: str) -> int:
    """Periode musiman harian: 24 titik untuk ETTh (per jam), 96 untuk ETTm (15 menit)."""
    if dataset_name.startswith("ETTh"):
        return 24
    if dataset_name.startswith("ETTm"):
        return 96
    raise ValueError(f"Dataset tidak dikenal: {dataset_name}")


def seasonal_naive(past_values: np.ndarray, horizon: int, m: int) -> np.ndarray:
    """Prediksi langkah h = nilai satu musim sebelumnya. past (B, T, D) -> (B, H, D)."""
    past_values = np.asarray(past_values)
    if past_values.shape[1] < m:
        raise ValueError("Riwayat lebih pendek dari periode musiman.")
    last = past_values[:, -m:, :]
    idx = np.arange(horizon) % m
    return last[:, idx, :]


def last_value(past_values: np.ndarray, horizon: int) -> np.ndarray:
    """Persistensi: ulangi nilai terakhir. past (B, T, D) -> (B, H, D)."""
    past_values = np.asarray(past_values)
    return np.repeat(past_values[:, -1:, :], horizon, axis=1)