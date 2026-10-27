"""Uji learner online berbasis model (butuh torch). Data sintetis, model kecil."""
import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from src.adapt.noadapt import NoAdapt
from src.adapt.online_update import OnlineUpdate, select_params
from src.data.pipeline import prepare_data_from_frame
from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.online.loop import run_online
from src.training.trainer import evaluate_loss
from tests.test_data import synthetic_frame
from tests.test_model_smoke import _cfg

MAX = 12


def _model(input_size=7, decoder_layers=2):
    cfg = _cfg(input_size)
    cfg.update(context_length=24, prediction_length=12, lags_sequence=[1, 2, 3, 24], decoder_layers=decoder_layers)
    torch.manual_seed(0)
    return ProbabilisticTransformer(cfg).eval()


@pytest.fixture(scope="module")
def prep():
    m = _model()
    return prepare_data_from_frame(synthetic_frame(), "ETTh1", m.past_length, m.prediction_length)


def _run(learner, prep, model, values=None, **kw):
    return run_online(
        learner, prep.values if values is None else values, prep.time_feats, prep.segments["val"],
        model.past_length, model.prediction_length, n_samples=kw.pop("n_samples", 5),
        max_origins=kw.pop("max_origins", MAX), season_m=24, **kw,
    )


def test_param_groups_nested_and_nonempty():
    m = _model(decoder_layers=2)
    total = sum(p.numel() for p in m.parameters())
    sizes = {}
    for g in ("head", "last_layer", "all"):
        sizes[g] = sum(p.numel() for p in select_params(m, g))
    assert 0 < sizes["head"] < sizes["last_layer"] < sizes["all"] == total


def test_noadapt_nll_matches_static_on_same_windows(prep):
    model = _model()
    res = _run(NoAdapt(model), prep, model)
    ds = Subset(prep.datasets["val"], range(MAX))  # mulai di origin yang sama (awal segmen val)
    static = evaluate_loss(model, torch.utils.data.DataLoader(ds, batch_size=4), torch.device("cpu"))
    np.testing.assert_allclose(res.nll_total, static, rtol=1e-4, atol=1e-5)


def test_online_update_changes_only_selected_params_and_not_original_model(prep):
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    learner = OnlineUpdate(model, params="head", lr=1e-2)
    res = _run(learner, prep, model, max_origins=30)
    assert res.diagnostics["n_updates"] > 0
    after = learner.model.state_dict()
    for k, v in before.items():
        changed = not torch.equal(v, after[k])
        if k.startswith("model.parameter_projection"):
            continue  # boleh berubah
        assert not changed, f"parameter beku berubah: {k}"
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("model.parameter_projection"))
    # model asli tidak tersentuh
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_online_update_has_no_leakage(prep):
    """Mengubah label di indeks >= T tidak boleh mengubah prediksi pada origin <= T."""
    model = _model()
    first = prep.segments["val"][0]
    T = first + 5
    values2 = prep.values.copy()
    values2[T:] = np.random.default_rng(1).normal(size=values2[T:].shape) * 5

    a = _run(OnlineUpdate(model, params="last_layer", lr=1e-2), prep, model, keep_samples=True, max_origins=8)
    b = _run(OnlineUpdate(model, params="last_layer", lr=1e-2), prep, model, values=values2,
             keep_samples=True, max_origins=8)
    for t in range(first, T + 1):
        np.testing.assert_allclose(a.samples[t], b.samples[t], atol=1e-5)
    assert not np.allclose(a.samples[T + 1], b.samples[T + 1])