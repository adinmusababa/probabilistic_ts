"""Uji training dengan data sintetis (tanpa database)."""
import json
import pathlib

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.pipeline import prepare_data_from_frame
from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.training.trainer import evaluate_loss, load_checkpoint, train
from src.utils.seed import set_seed
from tests.test_data import synthetic_frame

CFG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs" / "model" / "probabilistic_transformer.yaml"
)


def _model_cfg() -> dict:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg.update(
        context_length=24, prediction_length=12, lags_sequence=[1, 2, 3, 24],
        d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
    )
    return cfg


def _train_cfg() -> dict:
    return dict(
        seed=0, batch_size=32, lr=1e-3, weight_decay=0.0, epochs=2, patience=5,
        min_delta=0.0, grad_clip=1.0, train_stride=24, num_workers=0, device="cpu",
    )


def _run(out_dir, seed=0):
    tcfg, mcfg = _train_cfg(), _model_cfg()
    tcfg["seed"] = seed
    set_seed(seed)
    model = ProbabilisticTransformer(mcfg)
    data = prepare_data_from_frame(
        synthetic_frame(), "ETTh1", model.past_length, model.prediction_length,
        train_stride=tcfg["train_stride"],
    )
    manifest = train(model, data, tcfg, out_dir, mcfg)
    return manifest, data


def test_train_writes_artifacts_and_checkpoint_reloads(tmp_path):
    manifest, data = _run(tmp_path / "run")
    out = tmp_path / "run"
    for name in ("best.pt", "history.csv", "manifest.json"):
        assert (out / name).exists()

    hist = pl.read_csv(out / "history.csv")
    assert 1 <= hist.height <= 2
    assert np.isfinite(hist["val_nll"].to_numpy()).all()
    assert json.loads((out / "manifest.json").read_text())["data"]["dataset"] == "ETTh1"

    # checkpoint terbaik memberi NLL validasi yang sama dengan yang tercatat
    model = load_checkpoint(out / "best.pt")
    val_loader = DataLoader(data.datasets["val"], batch_size=32, shuffle=False)
    val = evaluate_loss(model, val_loader, torch.device("cpu"))
    np.testing.assert_allclose(val, manifest["best_val_nll"], rtol=1e-4)


def test_same_seed_same_history(tmp_path):
    m1, _ = _run(tmp_path / "a", seed=1)
    m2, _ = _run(tmp_path / "b", seed=1)
    h1 = pl.read_csv(tmp_path / "a" / "history.csv")["val_nll"].to_numpy()
    h2 = pl.read_csv(tmp_path / "b" / "history.csv")["val_nll"].to_numpy()
    np.testing.assert_allclose(h1, h2, rtol=1e-4)