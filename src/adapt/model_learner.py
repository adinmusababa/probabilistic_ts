"""Learner berbasis model probabilistik: prediksi dan penilaian NLL lewat ledger."""
from __future__ import annotations

import copy

import numpy as np
import torch

from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.online.learner import OnlineLearner
from src.online.ledger import Ledger


class ModelLearner(OnlineLearner):
    """Memegang salinan model sendiri (deepcopy), sehingga model asli tidak pernah berubah."""

    name = "model"

    def __init__(self, model: ProbabilisticTransformer, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = copy.deepcopy(model).to(self.device).eval()
        self.past_length = model.past_length
        self.horizon = model.prediction_length

    @staticmethod
    def _f32(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).copy())

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _batch(self, t: int, ledger: Ledger, future_values: np.ndarray | None = None) -> dict:
        p, h = self.past_length, self.horizon
        past = self._f32(ledger.past_window(t, p))[None]
        batch = {
            "past_values": past,
            "past_time_features": self._f32(ledger.time_features(t - p, t))[None],
            "past_observed_mask": torch.ones_like(past),
            "future_time_features": self._f32(ledger.time_features(t, t + h))[None],
        }
        if future_values is not None:
            fut = self._f32(future_values)[None]
            batch["future_values"] = fut
            batch["future_observed_mask"] = torch.ones_like(fut)
        return self._to_device(batch)

    def _window_batch(self, ledger: Ledger, origins: list[int]) -> dict:
        """Batch dari beberapa window yang labelnya sudah lengkap (untuk pembaruan)."""
        p, h = self.past_length, self.horizon
        wins = [ledger.window(s, p, h) for s in origins]
        past = self._f32(np.stack([w[0] for w in wins]))
        fut = self._f32(np.stack([w[1] for w in wins]))
        batch = {
            "past_values": past,
            "past_time_features": self._f32(np.stack([ledger.time_features(s - p, s) for s in origins])),
            "past_observed_mask": torch.ones_like(past),
            "future_values": fut,
            "future_time_features": self._f32(np.stack([ledger.time_features(s, s + h) for s in origins])),
            "future_observed_mask": torch.ones_like(fut),
        }
        return self._to_device(batch)

    @torch.no_grad()
    def predict(self, t: int, ledger: Ledger, n_samples: int, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.model.eval()
        samples = self.model.sample(self._batch(t, ledger), n_samples)  # (S, 1, H, D)
        return samples.cpu().numpy()

    @torch.no_grad()
    def nll(self, t: int, ledger: Ledger, future_values: np.ndarray) -> float:
        self.model.eval()
        return float(self.model.training_loss(self._batch(t, ledger, future_values)).item())