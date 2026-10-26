"""Uji evaluasi statis dengan model kecil dan data sintetis (tanpa database)."""
import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from src.baselines.seasonal_naive import last_value, seasonal_naive
from src.data.pipeline import prepare_data_from_frame
from src.evaluation.static_eval import evaluate_baseline, evaluate_model
from src.models.probabilistic_transformer import ProbabilisticTransformer
from tests.test_data import synthetic_frame
from tests.test_model_smoke import _batch, _cfg
  

def _small_cfg(input_size=7):
    cfg = _cfg(input_size)
    cfg.update(context_length=24, prediction_length=12, lags_sequence=[1, 2, 3, 24])
    return cfg


def test_evaluate_model_and_baselines_run():
    model = ProbabilisticTransformer(_small_cfg())
    prep = prepare_data_from_frame(
        synthetic_frame(), "ETTh1", model.past_length, model.prediction_length
    )
    ds = Subset(prep.datasets["val"], range(40))

    res = evaluate_model(model, ds, n_samples=10, batch_size=16, device="cpu")
    m = res["metrics"]
    assert res["n_windows"] == 40
    assert m["crps"].shape == (model.prediction_length, model.input_size)
    assert all(np.isfinite(v).all() for v in m.values())
    assert 0.0 <= m["coverage_80"].mean() <= 1.0
    np.testing.assert_allclose(res["pit"].sum(), 1.0)
    np.testing.assert_allclose(res["nll_per_channel"], res["nll_total"] / model.input_size)

    base = evaluate_baseline(lambda p, h: seasonal_naive(p, h, 24), ds, batch_size=16)
    np.testing.assert_allclose(base["metrics"]["crps"], base["metrics"]["mae_mean"])
    last = evaluate_baseline(last_value, ds, batch_size=16)
    assert np.isfinite(last["metrics"]["mse_mean"]).all()


def test_nll_is_sum_over_channels():
    """Memverifikasi asumsi: loss HF = rata-rata dari log-likelihood yang dijumlahkan antar kanal."""
    model = ProbabilisticTransformer(_small_cfg(7))
    batch = _batch(model, B=4)
    out = model.model(
        **model._past_inputs(batch),
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
        future_observed_mask=batch["future_observed_mask"],
    )
    try:
        dist = model.model.output_distribution(out.params, loc=out.loc, scale=out.scale)
    except (AttributeError, TypeError):
        pytest.skip("API output_distribution berbeda di versi transformers ini")
    lp = dist.log_prob(batch["future_values"])
    assert lp.shape == (4, model.prediction_length)  # (B, H): kanal sudah dijumlahkan
    torch.testing.assert_close(-lp.mean(), out.loss, rtol=1e-4, atol=1e-5)