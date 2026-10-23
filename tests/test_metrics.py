"""Uji metrik dan baseline (numpy murni, tanpa torch)."""
import math

import numpy as np

from src.baselines.seasonal_naive import last_value, season_length, seasonal_naive
from src.metrics.aggregate import MetricAccumulator, summarize, to_original_scale
from src.metrics.calibration import pit_histogram, pit_values
from src.metrics.probabilistic import central_interval, crps_samples, interval_score

_erf = np.vectorize(math.erf)


def _phi(z):
    return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


def crps_normal(mu, sigma, y):
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * _phi(z) - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def _crps_bruteforce(samples, y, fair):
    s = samples.shape[0]
    t1 = np.abs(samples - y).mean()
    diffs = np.abs(samples[:, None] - samples[None, :]).sum()
    return t1 - 0.5 * diffs / (s * (s - 1) if fair else s * s)


def test_crps_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=7)
    for fair in (True, False):
        got = crps_samples(x.reshape(7, 1), np.array([0.3]), fair=fair)[0]
        np.testing.assert_allclose(got, _crps_bruteforce(x, 0.3, fair), rtol=1e-10)


def test_crps_matches_closed_form_normal():
    rng = np.random.default_rng(1)
    mu, sigma = 0.7, 1.9
    s = rng.normal(mu, sigma, size=(20000, 3))
    y = np.array([-1.0, 0.5, 4.0])
    got = crps_samples(s, y, fair=True)
    np.testing.assert_allclose(got, crps_normal(mu, sigma, y), rtol=0.02)


def test_crps_single_sample_is_abs_error():
    got = crps_samples(np.array([[2.0, 5.0]]), np.array([1.0, 7.0]))
    np.testing.assert_allclose(got, [1.0, 2.0])


def test_interval_score_hand_values():
    lo, hi = np.array(0.0), np.array(1.0)
    assert interval_score(lo, hi, np.array(0.5), 0.8) == 1.0
    np.testing.assert_allclose(interval_score(lo, hi, np.array(2.0), 0.8), 1 + 10 * 1)
    np.testing.assert_allclose(interval_score(lo, hi, np.array(-1.0), 0.8), 1 + 10 * 1)


def test_calibrated_forecast_has_nominal_coverage_and_flat_pit():
    rng = np.random.default_rng(2)
    n = 5000
    samples = rng.normal(size=(200, n, 1, 1))
    y = rng.normal(size=(n, 1, 1))
    acc = MetricAccumulator()
    acc.update(samples, y)
    m = acc.finalize()
    for tag, lv in ((50, 0.5), (80, 0.8), (95, 0.95)):
        assert abs(m[f"coverage_{tag}"].mean() - lv) < 0.03
    np.testing.assert_allclose(acc.pit_proportions(), 0.1, atol=0.03)


def test_overconfident_forecast_undercovers():
    rng = np.random.default_rng(3)
    n = 4000
    samples = rng.normal(scale=0.3, size=(200, n, 1, 1))  # terlalu sempit
    y = rng.normal(size=(n, 1, 1))
    acc = MetricAccumulator()
    acc.update(samples, y)
    assert acc.finalize()["coverage_80"].mean() < 0.5


def test_accumulator_batches_equal_one_shot():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(30, 12, 5, 3))
    y = rng.normal(size=(12, 5, 3))
    one = MetricAccumulator()
    one.update(samples, y)
    two = MetricAccumulator()
    two.update(samples[:, :7], y[:7])
    two.update(samples[:, 7:], y[7:])
    a, b = one.finalize(), two.finalize()
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-10)
    np.testing.assert_allclose(one.pit_proportions(), two.pit_proportions())


def test_deterministic_has_crps_equal_mae_and_no_interval_metrics():
    rng = np.random.default_rng(5)
    pred = rng.normal(size=(1, 10, 4, 2))
    y = rng.normal(size=(10, 4, 2))
    acc = MetricAccumulator()
    acc.update(pred, y)
    m = acc.finalize()
    np.testing.assert_allclose(m["crps"], m["mae_mean"])
    assert not any(k.startswith(("coverage", "width")) for k in m)


def test_original_scale_and_summary_shapes():
    m = {"mse_mean": np.ones((4, 3)), "mae_median": np.ones((4, 3)), "coverage_80": np.full((4, 3), 0.8)}
    std = np.array([1.0, 2.0, 3.0])
    o = to_original_scale(m, std)
    np.testing.assert_allclose(o["mse_mean"][0], std**2)
    np.testing.assert_allclose(o["mae_median"][0], std)
    np.testing.assert_allclose(o["coverage_80"], 0.8)
    s = summarize(m)
    assert len(s["mse_mean"]["per_horizon"]) == 4 and len(s["mse_mean"]["per_channel"]) == 3


def test_seasonal_naive_and_last_value():
    past = np.arange(10, dtype=float).reshape(1, 10, 1)  # 0..9
    out = seasonal_naive(past, horizon=7, m=3)  # musim terakhir: 7,8,9
    np.testing.assert_allclose(out[0, :, 0], [7, 8, 9, 7, 8, 9, 7])
    np.testing.assert_allclose(last_value(past, 4)[0, :, 0], [9, 9, 9, 9])
    assert season_length("ETTh1") == 24 and season_length("ETTm2") == 96