"""Akumulator metrik per batch, agregasi per horizon dan per kanal, konversi skala."""
from __future__ import annotations

import numpy as np

from src.metrics.calibration import coverage_indicator, pit_histogram, pit_values
from src.metrics.probabilistic import central_interval, crps_samples, interval_score

DEFAULT_LEVELS = (0.5, 0.8, 0.95)


class MetricAccumulator:
    """Mengakumulasi jumlah metrik per (horizon, kanal) agar sampel tidak perlu disimpan semua.

    update(samples, y): samples (S, B, H, D), y (B, H, D). S == 1 dipakai untuk prediksi
    deterministik (baseline): metrik interval dan PIT dilewati, CRPS menjadi MAE.
    """

    def __init__(self, levels=DEFAULT_LEVELS, pit_bins: int = 10, fair: bool = True):
        self.levels = tuple(levels)
        self.pit_bins = pit_bins
        self.fair = fair
        self.n = 0
        self._sums: dict[str, np.ndarray] = {}
        self._pit = np.zeros(pit_bins, dtype=np.int64)
        self._n_samples: int | None = None

    def _add(self, key: str, arr: np.ndarray) -> None:
        total = arr.sum(axis=0, dtype=np.float64)  # jumlahkan atas batch -> (H, D)
        self._sums[key] = self._sums.get(key, 0.0) + total

    def update(self, samples: np.ndarray, y: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        s = samples.shape[0]
        if self._n_samples is None:
            self._n_samples = s
        elif s != self._n_samples:
            raise ValueError("Jumlah sampel harus sama di semua batch.")

        mean = samples.mean(axis=0)
        median = np.median(samples, axis=0)
        self._add("mse_mean", (mean - y) ** 2)
        self._add("mae_mean", np.abs(mean - y))
        self._add("mse_median", (median - y) ** 2)
        self._add("mae_median", np.abs(median - y))
        self._add("crps", crps_samples(samples, y, fair=self.fair))

        if s > 1:
            for lv in self.levels:
                tag = int(round(lv * 100))
                lo, hi = central_interval(samples, lv)
                self._add(f"coverage_{tag}", coverage_indicator(lo, hi, y))
                self._add(f"width_{tag}", hi - lo)
                self._add(f"interval_score_{tag}", interval_score(lo, hi, y, lv))
            self._pit += pit_histogram(pit_values(samples, y), self.pit_bins)
        self.n += y.shape[0]

    def finalize(self) -> dict[str, np.ndarray]:
        """Rata-rata per (H, D) untuk tiap metrik."""
        if self.n == 0:
            raise RuntimeError("Belum ada data.")
        return {k: v / self.n for k, v in self._sums.items()}

    def pit_proportions(self) -> np.ndarray:
        """Proporsi PIT per bin (ideal: 1/bins di tiap bin). Kosong untuk prediksi deterministik."""
        total = self._pit.sum()
        return self._pit / total if total > 0 else self._pit.astype(np.float64)


def summarize(metrics: dict[str, np.ndarray]) -> dict[str, dict]:
    """Ringkas tiap metrik (H, D) menjadi nilai global, per horizon, dan per kanal."""
    return {
        k: {
            "global": float(v.mean()),
            "per_horizon": v.mean(axis=1).tolist(),
            "per_channel": v.mean(axis=0).tolist(),
        }
        for k, v in metrics.items()
    }


def _scale_power(key: str) -> int | None:
    if key.startswith("mse"):
        return 2
    if key.startswith(("mae", "crps", "width", "interval_score")):
        return 1
    return None  # coverage dan sejenisnya tidak bergantung skala


def to_original_scale(metrics: dict[str, np.ndarray], std: np.ndarray) -> dict[str, np.ndarray]:
    """Kembalikan metrik dari ruang terstandardisasi ke skala asli per kanal.

    MSE dikali std^2; MAE, CRPS, lebar interval, dan interval score dikali std.
    Hanya bermakna per kanal (kanal punya satuan berbeda).
    """
    std = np.asarray(std, dtype=np.float64)
    out = {}
    for k, v in metrics.items():
        p = _scale_power(k)
        out[k] = v if p is None else v * std ** p
    return out