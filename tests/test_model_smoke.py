"""Smoke test model dengan data sintetis (tanpa database)."""
# python -m pytest tests/test_model_smoke.py -q

import pathlib

import pytest
import torch
import yaml

from src.models.probabilistic_transformer import ProbabilisticTransformer

CFG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs" / "model" / "probabilistic_transformer.yaml"
)


def _cfg(input_size: int) -> dict:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg.update(
        input_size=input_size,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_parallel_samples=5,
    )
    return cfg


def _batch(model: ProbabilisticTransformer, B: int = 4) -> dict:
    T, H = model.past_length, model.prediction_length
    D, F = model.input_size, model.config.num_time_features
    return {
        "past_values": torch.randn(B, T, D),
        "past_time_features": torch.randn(B, T, F),
        "past_observed_mask": torch.ones(B, T, D),
        "future_values": torch.randn(B, H, D),
        "future_time_features": torch.randn(B, H, F),
        "future_observed_mask": torch.ones(B, H, D),
    }


@pytest.mark.parametrize("input_size", [7, 1])
def test_loss_finite_and_backward(input_size):
    model = ProbabilisticTransformer(_cfg(input_size))
    loss = model.training_loss(_batch(model))
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("input_size", [7, 1])
def test_sample_shape(input_size):
    model = ProbabilisticTransformer(_cfg(input_size)).eval()
    batch = _batch(model, B=3)
    samples = model.sample(batch, n_samples=5)
    assert samples.shape == (5, 3, model.prediction_length, input_size)
    assert torch.isfinite(samples).all()