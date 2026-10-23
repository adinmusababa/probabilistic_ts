"""Metrik kalibrasi (numpy murni)."""
from __future__ import annotations

import numpy as np


def coverage_indicator(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((y >= lower) & (y <= upper)).astype(np.float64)


def pit_values(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """PIT empiris: fraksi sampel <= y. samples (S, ...), y (...)."""
    return (samples <= y[None]).mean(axis=0)


def pit_histogram(pit: np.ndarray, bins: int = 10) -> np.ndarray:
    counts, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    return counts