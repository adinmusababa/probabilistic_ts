"""Antarmuka learner untuk protokol online (tanpa torch, agar mudah diuji)."""
from __future__ import annotations

import numpy as np

from src.online.ledger import Ledger


class OnlineLearner:
    """Siklus per waktu t: update(t) memakai label < t, lalu predict(t) meramal t..t+H-1."""

    name = "base"

    def update(self, t: int, ledger: Ledger) -> None:
        """Perbarui state memakai label yang sudah terlihat (indeks < t). Baku: tidak melakukan apa-apa."""

    def predict(self, t: int, ledger: Ledger, n_samples: int, seed: int) -> np.ndarray:
        """Sampel prediksi berbentuk (S, 1, H, D) untuk origin t."""
        raise NotImplementedError

    def nll(self, t: int, ledger: Ledger, future_values: np.ndarray) -> float | None:
        """NLL prediksi origin t terhadap nilai nyata (hanya untuk penilaian, tidak boleh mengubah state)."""
        return None

    def diagnostics(self) -> dict:
        return {}