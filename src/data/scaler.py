from __future__ import annotations

from pathlib import Path

import numpy as np


class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), self.eps)
        return self

    def _check(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler belum di-fit.")

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check()
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def save(self, path: str | Path) -> None:
        self._check()
        np.savez(path, mean=self.mean, std=self.std, eps=self.eps)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        z = np.load(path)
        s = cls(eps=float(z["eps"]))
        s.mean, s.std = z["mean"], z["std"]
        return s