"""Metrik probabilistik berbasis sampel (numpy murni)."""
from __future__ import annotations

import numpy as np


def crps_samples(samples: np.ndarray, y: np.ndarray, fair: bool = True) -> np.ndarray:
    """CRPS per elemen dari sampel prediksi.

    samples: (S, ...), y: (...) -> hasil (...).
    Estimator: E|X - y| - 0.5 * E|X - X'|. Suku kedua dihitung lewat pengurutan
    (O(S log S)): sum_{i,j}|xi - xj| = 2 * sum_i (2i - S - 1) * x_(i), i mulai dari 1.
    fair=True memakai pembagi S(S-1) (tak bias), fair=False memakai S^2.
    Untuk S == 1 hasilnya |x - y| (CRPS prediksi deterministik = MAE).
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = samples.shape[0]
    term1 = np.abs(samples - y[None]).mean(axis=0)
    if s == 1:
        return term1
    xs = np.sort(samples, axis=0)
    i = np.arange(1, s + 1, dtype=np.float64).reshape((s,) + (1,) * (xs.ndim - 1))
    pair_sum = 2.0 * ((2.0 * i - s - 1.0) * xs).sum(axis=0)
    denom = s * (s - 1) if fair else s * s
    return term1 - 0.5 * pair_sum / denom


def central_interval(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Batas bawah dan atas interval sentral dengan probabilitas `level`."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [a, 1.0 - a], axis=0)
    return lo, hi


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, level: float) -> np.ndarray:
    """Interval score (Gneiting & Raftery): lebar + penalti jika y di luar interval."""
    alpha = 1.0 - level
    return (
        (upper - lower)
        + (2.0 / alpha) * np.maximum(lower - y, 0.0)
        + (2.0 / alpha) * np.maximum(y - upper, 0.0)
    )