"""Jalankan training baseline dengan data Supabase.

Contoh:
    python -m src.training.run --dataset ETTh1 --pred-len 24
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.pipeline import prepare_data
from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.training.trainer import train
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--pred-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model-config", default="configs/model/probabilistic_transformer.yaml")
    ap.add_argument("--train-config", default="configs/training/default.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    train_cfg = yaml.safe_load(Path(args.train_config).read_text())
    if args.pred_len is not None:
        model_cfg["prediction_length"] = args.pred_len
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    set_seed(train_cfg["seed"])  # sebelum membuat model agar inisialisasi reprodusibel
    model = ProbabilisticTransformer(model_cfg)
    data = prepare_data(
        args.dataset, model.past_length, model.prediction_length,
        train_stride=train_cfg.get("train_stride", 1),
    )
    out_dir = args.out_dir or (
        f"exps/{args.dataset}_H{model.prediction_length}_seed{train_cfg['seed']}"
    )
    result = train(model, data, train_cfg, out_dir, model_cfg)
    print(f"selesai: best val NLL {result['best_val_nll']:.4f} (epoch {result['best_epoch']}) -> {out_dir}")


if __name__ == "__main__":
    main()