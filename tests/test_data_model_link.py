"""Menghubungkan lapisan data ke model: satu batch harus bisa masuk training_loss."""
# python -m pytest .\tests\test_data_model_link.py -q

import pathlib

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.pipeline import prepare_data_from_frame
from src.models.probabilistic_transformer import ProbabilisticTransformer
from tests.test_data import synthetic_frame

CFG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "configs" / "model" / "probabilistic_transformer.yaml"
)


def test_batch_goes_through_model():
    cfg = yaml.safe_load(CFG_PATH.read_text())
    cfg.update(
        context_length=24, prediction_length=12, lags_sequence=[1, 2, 3, 24],
        d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
    )
    model = ProbabilisticTransformer(cfg)
    prep = prepare_data_from_frame(
        synthetic_frame(), "ETTh1", model.past_length, model.prediction_length, train_stride=24
    )
    batch = next(iter(DataLoader(prep.datasets["train"], batch_size=8, shuffle=True)))
    loss = model.training_loss(batch)
    assert loss.dim() == 0 and torch.isfinite(loss)