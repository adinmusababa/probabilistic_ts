from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

_REQUIRED = (
    "context_length",
    "prediction_length",
    "input_size",
    "lags_sequence",
    "num_time_features",
)


def build_hf_config(cfg: dict) -> TimeSeriesTransformerConfig:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise KeyError(f"Config model kekurangan kunci: {missing}")

    return TimeSeriesTransformerConfig(
        prediction_length=cfg["prediction_length"],
        context_length=cfg["context_length"],
        input_size=cfg["input_size"],
        lags_sequence=list(cfg["lags_sequence"]),
        num_time_features=cfg["num_time_features"],
        # fitur statis dan dinamis lain tidak dipakai pada baseline
        num_static_categorical_features=0,
        num_static_real_features=0,
        num_dynamic_real_features=0,
        # False = tanpa scaling internal HF, karena data sudah distandardisasi
        # (scaler numpy, fit pada train). Hindari normalisasi ganda.
        scaling=cfg.get("scaling", False),
        distribution_output=cfg.get("distribution_output", "student_t"),
        loss="nll",
        d_model=cfg.get("d_model", 64),
        encoder_layers=cfg.get("encoder_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 2),
        encoder_attention_heads=cfg.get("encoder_attention_heads", 2),
        decoder_attention_heads=cfg.get("decoder_attention_heads", 2),
        encoder_ffn_dim=cfg.get("encoder_ffn_dim", 128),
        decoder_ffn_dim=cfg.get("decoder_ffn_dim", 128),
        dropout=cfg.get("dropout", 0.1),
        num_parallel_samples=cfg.get("num_parallel_samples", 100),
    )


class ProbabilisticTransformer(nn.Module):
    """Pembungkus tipis di atas TimeSeriesTransformerForPrediction."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.model = TimeSeriesTransformerForPrediction(build_hf_config(cfg))

    # properti bantu
    @property
    def config(self) -> TimeSeriesTransformerConfig:
        return self.model.config

    @property
    def input_size(self) -> int:
        return self.config.input_size

    @property
    def prediction_length(self) -> int:
        return self.config.prediction_length

    @property
    def past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    # util internal
    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        # HF mengharapkan (B, T) untuk input_size == 1 dan (B, T, D) untuk D > 1
        return x.squeeze(-1) if self.input_size == 1 else x

    def _past_inputs(self, batch: dict) -> dict:
        return {
            "past_values": self._squeeze(batch["past_values"]),
            "past_time_features": batch["past_time_features"],
            "past_observed_mask": self._squeeze(batch["past_observed_mask"]),
        }

    # API
    def training_loss(self, batch: dict) -> torch.Tensor:
        """NLL rata-rata (skalar) pada horizon prediksi."""
        future_mask: Optional[torch.Tensor] = batch.get("future_observed_mask")
        out = self.model(
            **self._past_inputs(batch),
            future_values=self._squeeze(batch["future_values"]),
            future_time_features=batch["future_time_features"],
            future_observed_mask=None if future_mask is None else self._squeeze(future_mask),
        )
        return out.loss

    @torch.no_grad()
    def sample(self, batch: dict, n_samples: Optional[int] = None) -> torch.Tensor:
        """Sampel prediksi, bentuk (n_samples, B, H, D). Panggil model.eval() dulu."""
        old = self.config.num_parallel_samples
        if n_samples is not None:
            self.config.num_parallel_samples = n_samples
        try:
            out = self.model.generate(
                **self._past_inputs(batch),
                future_time_features=batch["future_time_features"],
            )
        finally:
            self.config.num_parallel_samples = old

        seq = out.sequences  # (B, N, H) atau (B, N, H, D)
        if seq.dim() == 3:
            seq = seq.unsqueeze(-1)
        return seq.permute(1, 0, 2, 3).contiguous()