"""Evaluasi statis: model dan baseline pada window yang sama (ruang terstandardisasi)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics.aggregate import DEFAULT_LEVELS, MetricAccumulator
from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.training.trainer import evaluate_loss
from src.utils.seed import set_seed


def evaluate_model(
    model: ProbabilisticTransformer,
    dataset,
    n_samples: int = 100,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 0,
    levels=DEFAULT_LEVELS,
) -> dict:
    device = torch.device(device)
    set_seed(seed)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    acc = MetricAccumulator(levels=levels)
    for batch in loader:
        y = batch["future_values"].numpy()
        dev_batch = {k: v.to(device) for k, v in batch.items()}
        samples = model.sample(dev_batch, n_samples).cpu().numpy()  # (S, B, H, D)
        acc.update(samples, y)

    nll_total = evaluate_loss(model, loader, device)
    return {
        "metrics": acc.finalize(),
        "pit": acc.pit_proportions(),
        "nll_total": nll_total,
        # log-likelihood dijumlahkan antar kanal di Hugging Face (lihat test_nll_is_sum_over_channels)
        "nll_per_channel": nll_total / model.input_size,
        "n_windows": acc.n,
    }


def evaluate_baseline(
    predict_fn: Callable[[np.ndarray, int], np.ndarray],
    dataset,
    batch_size: int = 64,
) -> dict:
    """predict_fn(past_values (B,T,D), horizon) -> prediksi (B,H,D)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    acc = MetricAccumulator()
    for batch in loader:
        y = batch["future_values"].numpy()
        pred = predict_fn(batch["past_values"].numpy(), y.shape[1])
        acc.update(pred[None], y)  # S = 1: deterministik
    return {"metrics": acc.finalize(), "n_windows": acc.n}