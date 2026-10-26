"""Uji perbandingan run dengan folder exps tiruan (tanpa torch, tanpa database)."""
import json

import numpy as np
import polars as pl
import pytest

from src.evaluation.compare_runs import (
    comparability_warnings, load_runs, naive_reference, summary_table,
)

CHANNELS = ["hufl", "hull", "mufl", "mull", "lufl", "lull", "ot"]
H = 24


def _metric(g, h=H, d=7):
    return {"global": g, "per_horizon": [g] * h, "per_channel": [g] * d}


def _eval(mae, mse, crps, cov, n_samples=100, pit_edge=0.15):
    model = {
        "mae_median": _metric(mae), "mae_mean": _metric(mae), "mse_mean": _metric(mse),
        "mse_median": _metric(mse), "crps": _metric(crps),
        "coverage_50": _metric(cov[0]), "coverage_80": _metric(cov[1]), "coverage_95": _metric(cov[2]),
        "width_80": _metric(1.0),
    }
    naive = {k: _metric(v) for k, v in
             {"mae_median": 0.448, "mae_mean": 0.448, "mse_mean": 0.511, "mse_median": 0.511, "crps": 0.448}.items()}
    pit = [pit_edge] + [(1 - 2 * pit_edge) / 8] * 8 + [pit_edge]
    return {
        "split": "val", "n_samples": n_samples, "seed": 0, "nll_total": 1.9, "nll_per_channel": 0.27,
        "pit": pit, "n_windows": 2857, "channels": CHANNELS,
        "standardized": {"model": model, "seasonal_naive_24": naive, "last_value": naive},
    }


def _write_run(root, name, ev, scaling=False, d_model=64, n_epochs=5):
    d = root / name
    d.mkdir(parents=True)
    (d / "eval_val.json").write_text(json.dumps(ev))
    (d / "manifest.json").write_text(json.dumps({
        "model_cfg": {"scaling": scaling, "d_model": d_model, "encoder_layers": 2, "decoder_layers": 2,
                      "dropout": 0.1, "prediction_length": 24},
        "train_cfg": {"lr": 6e-4, "weight_decay": 0.01, "seed": 42},
        "data": {"dataset": "ETTh1"}, "best_epoch": 3, "epochs_run": n_epochs,
    }))
    pl.DataFrame({"epoch": list(range(1, n_epochs + 1)),
                  "train_nll": list(np.linspace(5, 0, n_epochs)),
                  "val_nll": list(np.linspace(4, 2, n_epochs)),
                  "seconds": [1.0] * n_epochs}).write_csv(d / "history.csv")


@pytest.fixture()
def exps(tmp_path):
    root = tmp_path / "exps"
    _write_run(root, "base", _eval(0.516, 0.593, 0.391, (0.355, 0.608, 0.792)))
    _write_run(root, "std", _eval(0.492, 0.542, 0.365, (0.382, 0.647, 0.835)), scaling="std")
    (root / "optuna_study" / "trial_0").mkdir(parents=True)  # harus dilewati
    return root


def test_load_runs_skips_folders_without_eval(exps):
    runs = load_runs(exps, "val")
    assert set(runs) == {"base", "std"}


def test_summary_table_values_and_sorting(exps):
    df = summary_table(load_runs(exps, "val"))
    assert df["run"].to_list() == ["std", "base"]  # urut CRPS naik
    row = df.filter(pl.col("run") == "std").row(0, named=True)
    assert row["scaling"] == "std" and row["mae"] == 0.492
    np.testing.assert_allclose(row["mae_vs_naive_pct"], 100 * (0.492 / 0.448 - 1), atol=1e-3)
    np.testing.assert_allclose(row["pit_tails"], 0.30)
    assert row["lr"] == 6e-4  # tidak ikut dibulatkan


def test_comparability_warnings(exps):
    assert comparability_warnings(load_runs(exps, "val")) == []
    _write_run(exps, "s20", _eval(0.5, 0.6, 0.39, (0.3, 0.55, 0.71), n_samples=20))
    warns = comparability_warnings(load_runs(exps, "val"))
    assert any("n_samples" in w for w in warns)


def test_naive_reference(exps):
    ref = naive_reference(load_runs(exps, "val"))
    assert ref == {"mae": 0.448, "mse": 0.511}


def test_expected_coverage_matches_known_values():
    pytest.importorskip("matplotlib")
    from src.evaluation.plots import expected_coverage
    np.testing.assert_allclose(expected_coverage(0.8, 100), 0.784, atol=0.002)
    np.testing.assert_allclose(expected_coverage(0.95, 20), 0.86, atol=0.005)


def test_all_plots_and_save(exps, tmp_path):
    plt = pytest.importorskip("matplotlib")
    plt.use("Agg")
    from src.evaluation.plots import save_all

    runs = load_runs(exps, "val")
    df = summary_table(runs)
    paths = save_all(runs, df, tmp_path / "out", split="val")
    assert len(paths) == 7
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)