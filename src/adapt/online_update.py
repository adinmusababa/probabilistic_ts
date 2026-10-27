"""Online update biasa: gradient descent pada window terbaru yang labelnya sudah lengkap."""
from __future__ import annotations

import numpy as np
import torch

from src.adapt.model_learner import ModelLearner
from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.online.ledger import Ledger

PARAM_GROUPS = ("head", "last_layer", "all")


def select_params(model: ProbabilisticTransformer, which: str) -> list[torch.nn.Parameter]:
    """Pilih parameter yang diperbarui dan bekukan sisanya (requires_grad=False).

    head: proyeksi parameter distribusi. last_layer: head + layer decoder terakhir. all: semua.
    Nama parameter mengikuti Hugging Face (model.parameter_projection, model.model.decoder.layers.N).
    """
    if which not in PARAM_GROUPS:
        raise ValueError(f"params harus salah satu dari {PARAM_GROUPS}")
    last = model.config.decoder_layers - 1
    prefixes = {
        "head": ["model.parameter_projection"],
        "last_layer": ["model.parameter_projection", f"model.model.decoder.layers.{last}."],
        "all": [""],
    }[which]
    selected = []
    for name, p in model.named_parameters():
        use = any(name.startswith(pre) for pre in prefixes)
        p.requires_grad_(use)
        if use:
            selected.append(p)
    if not selected:
        raise ValueError(f"Tidak ada parameter yang cocok untuk grup '{which}'; periksa nama parameter model.")
    return selected


class OnlineUpdate(ModelLearner):
    name = "online_update"

    def __init__(
        self,
        model: ProbabilisticTransformer,
        device: str | torch.device = "cpu",
        params: str = "last_layer",
        lr: float = 1e-4,
        n_steps: int = 1,
        buffer_size: int = 1,
        update_every: int = 1,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        grad_clip: float | None = 1.0,
    ):
        super().__init__(model, device)
        self.params_name = params
        self.n_steps = n_steps
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.grad_clip = grad_clip
        self.selected = select_params(self.model, params)
        self.n_trainable = int(sum(p.numel() for p in self.selected))
        self.n_total = int(sum(p.numel() for p in self.model.parameters()))
        opt_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
        self.opt = opt_cls(self.selected, lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.optimizer_name = optimizer
        self._calls = 0
        self.n_updates = 0
        self.losses: list[float] = []

    def update(self, t: int, ledger: Ledger) -> None:
        self._calls += 1
        if (self._calls - 1) % self.update_every:
            return
        origins = ledger.completed_origins(self.horizon, self.past_length, self.buffer_size)
        if not origins:
            return
        batch = self._window_batch(ledger, origins)
        self.model.eval()  # tanpa dropout: pembaruan deterministik
        loss = None
        for _ in range(self.n_steps):
            self.opt.zero_grad(set_to_none=True)
            loss = self.model.training_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss tidak finite pada pembaruan online (t={t}).")
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.selected, self.grad_clip)
            self.opt.step()
        self.n_updates += 1
        self.losses.append(float(loss.item()))

    def diagnostics(self) -> dict:
        recent = self.losses[-100:]
        return {
            "params": self.params_name,
            "lr": self.lr,
            "optimizer": self.optimizer_name,
            "n_steps": self.n_steps,
            "buffer_size": self.buffer_size,
            "update_every": self.update_every,
            "n_updates": self.n_updates,
            "n_trainable": self.n_trainable,
            "n_total": self.n_total,
            "mean_loss_last100": float(np.mean(recent)) if recent else None,
        }