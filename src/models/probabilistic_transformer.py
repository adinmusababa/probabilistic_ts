"""Baseline probabilistic Transformer (Hugging Face TimeSeriesTransformer).

File ini hanya mendeklarasikan model. Data, training, dan metrik ada di modul lain.

Format batch (tensor float32 dari lapisan data):
    past_values            (B, past_length, D)
    past_time_features     (B, past_length, F)
    past_observed_mask     (B, past_length, D)
    future_values          (B, H, D)      # hanya saat training
    future_time_features   (B, H, F)
    future_observed_mask   (B, H, D)      # opsional
dengan past_length = context_length + max(lags_sequence),
D = input_size, F = num_time_features, H = prediction_length.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from transformers.modeling_outputs import BaseModelOutput

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


@dataclass(frozen=True)
class EncoderState:
    """Hasil encode: representasi z beserta konteks yang dibutuhkan decoder.

    z            (B, context_length, d_model)  keluaran encoder (ruang Z untuk ADAPT-Z)
    loc, scale   statistik normalisasi per window dari Hugging Face (nol dan satu bila scaling: false)
    static_feat  fitur statis turunan loc dan scale (dipakai decoder)
    """

    z: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    static_feat: torch.Tensor

    def with_z(self, z: torch.Tensor) -> "EncoderState":
        """Salinan dengan z diganti (mis. z + delta); loc, scale, dan static_feat tetap sama."""
        return replace(self, z=z)


class ProbabilisticTransformer(nn.Module):
    """Pembungkus tipis di atas TimeSeriesTransformerForPrediction."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.model = TimeSeriesTransformerForPrediction(build_hf_config(cfg))

    # ---- properti bantu -------------------------------------------------
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

    # ---- util internal --------------------------------------------------
    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        # HF mengharapkan (B, T) untuk input_size == 1 dan (B, T, D) untuk D > 1
        return x.squeeze(-1) if self.input_size == 1 else x

    def _past_inputs(self, batch: dict) -> dict:
        return {
            "past_values": self._squeeze(batch["past_values"]),
            "past_time_features": batch["past_time_features"],
            "past_observed_mask": self._squeeze(batch["past_observed_mask"]),
        }

    # ---- API ------------------------------------------------------------
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


    # ---- celah encode/decode untuk ADAPT-Z --------------------------------
    def encode(self, batch: dict, detach: bool = True) -> EncoderState:
        """Jalankan encoder sekali. z hanya bergantung pada konteks (bukan masa depan).

        detach=True (baku): tanpa graf gradien, cocok untuk encoder f yang dibekukan.
        """
        ctx = torch.no_grad() if detach else contextlib.nullcontext()
        with ctx:
            mdl = self.model.model  # TimeSeriesTransformerModel
            inputs, loc, scale, static_feat = mdl.create_network_inputs(**self._past_inputs(batch))
            z = mdl.encoder(inputs_embeds=inputs[:, : self.config.context_length, ...])[0]
        return EncoderState(z=z, loc=loc, scale=scale, static_feat=static_feat)

    def decode_loss(self, state: EncoderState, batch: dict) -> torch.Tensor:
        """NLL rata-rata (teacher forcing) dengan decoder membaca z dari state, tanpa menjalankan ulang encoder.

        Gradien mengalir ke state.z bila z memerlukan gradien.
        """
        future_mask: Optional[torch.Tensor] = batch.get("future_observed_mask")
        out = self.model(
            **self._past_inputs(batch),
            future_values=self._squeeze(batch["future_values"]),
            future_time_features=batch["future_time_features"],
            future_observed_mask=None if future_mask is None else self._squeeze(future_mask),
            encoder_outputs=BaseModelOutput(last_hidden_state=state.z),
        )
        return out.loss

    @torch.no_grad()
    def sample_from_state(
        self, state: EncoderState, batch: dict, n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Sampel autoregresif dari z di state, bentuk (n_samples, B, H, D). Panggil model.eval() dulu.

        Meniru loop `generate` Hugging Face (diverifikasi pada kode transformers 4.46 dan 5.21),
        tetapi memakai z dari state sehingga z yang sudah dikoreksi bisa dipakai.
        """
        hf = self.model  # TimeSeriesTransformerForPrediction
        mdl = hf.model
        s = n_samples if n_samples is not None else self.config.num_parallel_samples
        past_values = self._squeeze(batch["past_values"])
        future_tf = batch["future_time_features"]

        r_loc = state.loc.repeat_interleave(s, dim=0)
        r_scale = state.scale.repeat_interleave(s, dim=0)
        r_past = (past_values.repeat_interleave(s, dim=0) - r_loc) / r_scale
        static = state.static_feat.unsqueeze(1).expand(-1, future_tf.shape[1], -1)
        r_features = torch.cat((static, future_tf), dim=-1).repeat_interleave(s, dim=0)
        r_z = state.z.repeat_interleave(s, dim=0)

        decoder = mdl.get_decoder()
        samples = []
        for k in range(self.config.prediction_length):
            lagged = mdl.get_lagged_subsequences(sequence=r_past, subsequences_length=1 + k, shift=1)
            lagged = lagged.reshape(lagged.shape[0], lagged.shape[1], -1)
            dec_in = torch.cat((lagged, r_features[:, : k + 1]), dim=-1)
            hidden = decoder(inputs_embeds=dec_in, encoder_hidden_states=r_z).last_hidden_state
            params = hf.parameter_projection(hidden[:, -1:])
            nxt = hf.output_distribution(params, loc=r_loc, scale=r_scale).sample()
            r_past = torch.cat((r_past, (nxt - r_loc) / r_scale), dim=1)
            samples.append(nxt)

        seq = torch.cat(samples, dim=1).reshape((-1, s, self.config.prediction_length) + hf.target_shape)
        if seq.dim() == 3:
            seq = seq.unsqueeze(-1)
        return seq.permute(1, 0, 2, 3).contiguous()