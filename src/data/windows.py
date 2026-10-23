"""Dataset window untuk model (format batch sesuai ProbabilisticTransformer)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Satu item = satu origin t: riwayat t-P..t-1 dan masa depan t..t+H-1."""

    def __init__(
        self,
        values: np.ndarray,       # (N, D) sudah distandardisasi
        time_feats: np.ndarray,   # (N, F)
        origins: np.ndarray,      # (M,) indeks t
        past_length: int,
        horizon: int,
    ):
        assert len(values) == len(time_feats)
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.time_feats = np.ascontiguousarray(time_feats, dtype=np.float32)
        self.origins = np.asarray(origins, dtype=np.int64)
        self.past_length = past_length
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int) -> dict:
        t = int(self.origins[i])
        p, h = self.past_length, self.horizon
        past = self.values[t - p : t].copy()
        fut = self.values[t : t + h].copy()
        return {
            "past_values": torch.from_numpy(past),
            "past_time_features": torch.from_numpy(self.time_feats[t - p : t].copy()),
            "past_observed_mask": torch.ones_like(torch.from_numpy(past)),
            "future_values": torch.from_numpy(fut),
            "future_time_features": torch.from_numpy(self.time_feats[t : t + h].copy()),
            "future_observed_mask": torch.ones_like(torch.from_numpy(fut)),
        }