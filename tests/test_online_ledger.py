"""Uji ledger dan loop online dengan learner tiruan (numpy murni, tanpa torch)."""
import json

import numpy as np
import pytest

from src.baselines.seasonal_naive import seasonal_naive
from src.evaluation.compare_runs import load_runs, summary_table
from src.online.learner import OnlineLearner
from src.online.ledger import Ledger, LeakageError
from src.online.loop import run_online, to_eval_json

N, D, P, H, M = 600, 3, 48, 12, 24
SEG = (200, 400)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    values = (np.sin(2 * np.pi * t / 24)[:, None] + 0.3 * rng.normal(size=(N, D))).astype(np.float32)
    tf = rng.normal(size=(N, 4)).astype(np.float32)
    return values, tf


# ---------------- ledger ----------------
def test_ledger_blocks_future_labels_and_time_going_backwards():
    values, tf = _data()
    led = Ledger(values, tf)
    led.advance(100)
    assert led.labels(90, 100).shape == (10, D)
    with pytest.raises(LeakageError):
        led.labels(95, 101)
    with pytest.raises(LeakageError):
        led.past_window(101, 5)
    with pytest.raises(ValueError):
        led.advance(99)
    assert led.max_label_read == 99


def test_ledger_values_are_read_only():
    values, tf = _data()
    led = Ledger(values, tf)
    led.advance(50)
    with pytest.raises(ValueError):
        led.labels(0, 10)[0, 0] = 123.0


def test_completed_origins_and_window_alignment():
    values, tf = _data()
    led = Ledger(values, tf)
    led.advance(100)
    assert led.completed_origins(H, P, count=3) == [88, 87, 86]
    early = Ledger(values, tf)
    early.advance(50)  # 50 - 12 = 38, 38 - 48 < 0 -> belum ada window lengkap
    assert early.completed_origins(H, P) == []
    led.advance(120)
    s = led.completed_origins(H, P)[0]
    past, fut = led.window(s, P, H)
    np.testing.assert_array_equal(past, values[s - P : s])
    np.testing.assert_array_equal(fut, values[s : s + H])
    with pytest.raises(LeakageError):
        led.window(s + 1, P, H + 1)  # masa depan melewati now


def test_calendar_features_may_look_ahead():
    values, tf = _data()
    led = Ledger(values, tf)
    led.advance(10)
    np.testing.assert_array_equal(led.time_features(10, 30), tf[10:30])


# ---------------- learner tiruan ----------------
class SeasonalLearner(OnlineLearner):
    name = "fake_seasonal"

    def predict(self, t, ledger, n_samples, seed):
        pred = seasonal_naive(ledger.past_window(t, P)[None], H, M)  # (1,H,D)
        return np.repeat(pred[None], n_samples, axis=0)               # (S,1,H,D)


class LevelLearner(OnlineLearner):
    """Stateful: level bergerak dari label terbaru. Menguji kebocoran lewat state."""

    name = "fake_level"

    def __init__(self):
        self.level = np.zeros(D)

    def update(self, t, ledger):
        self.level = 0.9 * self.level + 0.1 * ledger.labels(t - 1, t)[0]

    def predict(self, t, ledger, n_samples, seed):
        noise = np.random.default_rng(seed).normal(size=(n_samples, 1, H, D)) * 0.01
        return self.level[None, None, None, :] + noise


class Cheater(OnlineLearner):
    name = "cheater"

    def predict(self, t, ledger, n_samples, seed):
        ledger.labels(t, t + 1)  # mengintip label masa depan
        return np.zeros((n_samples, 1, H, D))


def _run(learner, values=None, tf=None, **kw):
    v, f = _data()
    return run_online(learner, v if values is None else values, f if tf is None else tf,
                      SEG, P, H, season_m=M, n_samples=kw.pop("n_samples", 20), **kw)


# ---------------- loop ----------------
def test_loop_matches_baseline_when_learner_is_seasonal_naive():
    res = _run(SeasonalLearner())
    assert res.n_windows == 189  # 200..388
    np.testing.assert_allclose(res.metrics["mae_median"], res.naive_metrics["mae_median"])
    np.testing.assert_allclose(res.metrics["crps"], res.naive_metrics["crps"])
    assert res.metrics["mae_median"].shape == (H, D)
    assert res.nll_total is None


def test_eval_stride_and_max_origins():
    assert _run(SeasonalLearner(), eval_stride=4).n_windows == 48
    assert _run(SeasonalLearner(), max_origins=10).n_windows == 10


def test_cheating_learner_is_caught():
    with pytest.raises(LeakageError):
        _run(Cheater(), max_origins=3)


def test_no_leakage_via_state_perturbation():
    """Mengubah label di indeks >= T tidak boleh mengubah prediksi pada origin <= T."""
    values, tf = _data()
    first = SEG[0]
    T = first + 5
    perturbed = values.copy()
    perturbed[T:] = np.random.default_rng(99).normal(size=perturbed[T:].shape) * 5

    a = _run(LevelLearner(), values=values, keep_samples=True, max_origins=9)
    b = _run(LevelLearner(), values=perturbed, keep_samples=True, max_origins=9)
    for t in range(first, T + 1):
        np.testing.assert_array_equal(a.samples[t], b.samples[t])
    # kontrol sensitivitas: origin T+1 sudah boleh melihat label T, jadi harus berbeda
    assert not np.allclose(a.samples[T + 1], b.samples[T + 1])


def test_eval_json_is_compatible_with_compare_runs(tmp_path):
    res = _run(SeasonalLearner(), eval_stride=4)
    ev = to_eval_json(res, ["a", "b", "c"], "val", 20, 0, "seasonal_naive_24")
    run = tmp_path / "exps" / "base_online_fake"
    run.mkdir(parents=True)
    (run / "eval_val.json").write_text(json.dumps(ev))
    (run / "manifest.json").write_text(json.dumps({
        "model_cfg": {"scaling": "std", "d_model": 64, "encoder_layers": 2, "decoder_layers": 2,
                      "dropout": 0.1, "prediction_length": H},
        "train_cfg": {"lr": 1e-3, "weight_decay": 0.0, "seed": 0},
        "data": {"dataset": "ETTh1"}, "best_epoch": 1, "epochs_run": 1,
        "online": {"strategy": "fake"},
    }))
    df = summary_table(load_runs(tmp_path / "exps", "val"))
    assert df["strategy"].to_list() == ["fake"]
    assert df["n_windows"].to_list() == [48]
    assert df["mae"][0] is not None