"""Uji lapisan data dengan data sintetis (tanpa database)."""
# python -m pytest .\tests\test_data.py -q  


from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.data.pipeline import prepare_data_from_frame
from src.data.scaler import StandardScaler
from src.data.splits import ett_segments, total_rows
from src.data.time_features import NUM_TIME_FEATURES, time_features

COLS = ["hufl", "hull", "mufl", "mull", "lufl", "lull", "ot"]
P, H = 48, 12


def synthetic_frame(name: str = "ETTh1", extra: int = 100) -> pl.DataFrame:
    n = total_rows(name) + extra
    step = "1h" if name.startswith("ETTh") else "15m"
    start = datetime(2016, 7, 1)
    delta = timedelta(hours=1) if step == "1h" else timedelta(minutes=15)
    dates = pl.datetime_range(start, start + delta * (n - 1), interval=step, eager=True)
    rng = np.random.default_rng(0)
    data = {c: np.cumsum(rng.normal(size=n)) + 10 * k for k, c in enumerate(COLS)}
    return pl.DataFrame({"date": dates, **data})


def test_segments_contiguous_and_scaled():
    for name, mult in [("ETTh1", 1), ("ETTm2", 4)]:
        seg = ett_segments(name)
        assert seg["train"][0] == 0
        assert seg["train"][1] == seg["val"][0] and seg["val"][1] == seg["test"][0]
        assert seg["test"][1] == total_rows(name) == 14400 * mult


def test_scaler_roundtrip_and_train_only():
    prep = prepare_data_from_frame(synthetic_frame(), "ETTh1", P, H)
    train_end = prep.segments["train"][1]
    np.testing.assert_allclose(prep.values[:train_end].mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(prep.values[:train_end].std(0), 1, atol=1e-3)
    raw_back = prep.scaler.inverse_transform(prep.values[:10])
    s2 = StandardScaler().fit(raw_back)  # hanya memastikan API berjalan
    assert s2.mean.shape == (len(COLS),)


def test_window_alignment_and_bounds():
    prep = prepare_data_from_frame(synthetic_frame(), "ETTh1", P, H)
    for split, ds in prep.datasets.items():
        start, end = prep.segments[split]
        assert len(ds) > 0
        assert ds.origins.min() >= max(start, P)
        assert ds.origins.max() + H <= end  # target tidak melewati batas segmen
        item = ds[0]
        t = int(ds.origins[0])
        assert item["past_values"].shape == (P, len(COLS))
        assert item["future_values"].shape == (H, len(COLS))
        np.testing.assert_allclose(np.asarray(item["past_values"])[-1], prep.values[t - 1])
        np.testing.assert_allclose(np.asarray(item["future_values"])[0], prep.values[t])
        assert np.asarray(item["past_time_features"]).shape == (P, NUM_TIME_FEATURES)


def test_time_features_range():
    tf = time_features(synthetic_frame())
    assert tf.shape[1] == NUM_TIME_FEATURES
    assert tf.min() >= -0.5 - 1e-6 and tf.max() <= 0.5 + 1e-6


def test_too_few_rows_raises():
    df = synthetic_frame(extra=0).head(1000)
    with pytest.raises(ValueError):
        prepare_data_from_frame(df, "ETTh1", P, H)