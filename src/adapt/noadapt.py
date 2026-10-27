"""Strategi tanpa adaptasi: model beku."""
from __future__ import annotations

from src.adapt.model_learner import ModelLearner
from src.online.ledger import Ledger


class NoAdapt(ModelLearner):
    name = "noadapt"

    def update(self, t: int, ledger: Ledger) -> None:
        return None

    def diagnostics(self) -> dict:
        return {"n_updates": 0}