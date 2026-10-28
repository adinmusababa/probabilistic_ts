"""Uji celah encode/decode (FR-M3, FR-M5): hasilnya harus identik dengan jalur Hugging Face asli."""
import pytest
import torch

from src.models.probabilistic_transformer import EncoderState, ProbabilisticTransformer
from tests.test_model_smoke import _batch, _cfg


def _model(input_size, scaling):
    cfg = _cfg(input_size)
    cfg["scaling"] = scaling
    torch.manual_seed(0)
    return ProbabilisticTransformer(cfg).eval()  # eval: tanpa dropout, hasil deterministik


GRID = [(7, False), (7, "std"), (1, False), (1, "std")]


@pytest.mark.parametrize("input_size,scaling", GRID)
def test_decode_loss_equals_training_loss(input_size, scaling):
    model = _model(input_size, scaling)
    batch = _batch(model, B=4)
    state = model.encode(batch)
    assert isinstance(state, EncoderState)
    assert state.z.shape == (4, model.config.context_length, model.config.d_model)
    torch.testing.assert_close(model.decode_loss(state, batch), model.training_loss(batch), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("input_size,scaling", GRID)
def test_sample_from_state_equals_sample(input_size, scaling):
    model = _model(input_size, scaling)
    batch = _batch(model, B=3)
    torch.manual_seed(123)
    ref = model.sample(batch, n_samples=5)
    torch.manual_seed(123)
    got = model.sample_from_state(model.encode(batch), batch, n_samples=5)
    assert got.shape == ref.shape == (5, 3, model.prediction_length, input_size)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_zero_delta_is_identity_and_shift_changes_output():
    model = _model(7, "std")
    batch = _batch(model, B=4)
    state = model.encode(batch)
    base = model.decode_loss(state, batch)
    torch.testing.assert_close(model.decode_loss(state.with_z(state.z + 0.0), batch), base)
    assert not torch.allclose(model.decode_loss(state.with_z(state.z + 1.0), batch), base)


def test_encode_is_detached_by_default_and_gradient_reaches_z():
    model = _model(7, "std")
    batch = _batch(model, B=4)
    state = model.encode(batch)
    assert not state.z.requires_grad

    z = state.z.clone().requires_grad_(True)
    model.decode_loss(state.with_z(z), batch).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0

    attached = model.encode(batch, detach=False)
    assert attached.z.requires_grad