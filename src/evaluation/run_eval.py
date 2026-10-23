"""Evaluasi statis sebuah run training terhadap baseline seasonal naive dan last value.

Contoh:
    python -m src.evaluation.run_eval --exp-dir exps/ETTh1_H24_seed42 --split val
Catatan: pilih/perbaiki model dengan split val; split test dipakai untuk laporan akhir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.baselines.seasonal_naive import last_value, season_length, seasonal_naive
from src.data.pipeline import prepare_data
from src.evaluation.static_eval import evaluate_baseline, evaluate_model
from src.metrics.aggregate import summarize, to_original_scale
from src.training.trainer import load_checkpoint


def _row(name: str, metrics: dict) -> dict:
    g = lambda k: float(metrics[k].mean()) if k in metrics else None
    return {
        "metode": name,
        "mse": g("mse_mean"),
        "mae": g("mae_median"),
        "crps": g("crps"),
        "cov50": g("coverage_50"),
        "cov80": g("coverage_80"),
        "cov95": g("coverage_95"),
        "lebar80": g("width_80"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    manifest = json.loads((exp / "manifest.json").read_text())
    name = manifest["data"]["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_checkpoint(exp / "best.pt", device=device)
    data = prepare_data(name, model.past_length, model.prediction_length)
    if data.meta["data_sha256"] != manifest["data"]["data_sha256"]:
        print("PERINGATAN: isi data di database berbeda dari saat training (hash tidak cocok).")
    ds = data.datasets[args.split]
    print(f"{name} | split {args.split} | {len(ds)} window | device {device}")

    m = season_length(name)
    res = {
        "model": evaluate_model(model, ds, args.n_samples, args.batch_size, device, args.seed),
        f"seasonal_naive_{m}": evaluate_baseline(
            lambda p, h: seasonal_naive(p, h, m), ds, args.batch_size
        ),
        "last_value": evaluate_baseline(last_value, ds, args.batch_size),
    }

    rows = [_row(k, v["metrics"]) for k, v in res.items()]
    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=20, tbl_cols=20, float_precision=4):
        print(table)
    print(f"NLL model: total {res['model']['nll_total']:.4f} | per kanal {res['model']['nll_per_channel']:.4f}")
    print("PIT (proporsi per bin, ideal 0.1):", np.round(res["model"]["pit"], 3).tolist())

    std = data.scaler.std
    out = {
        "split": args.split,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "nll_total": res["model"]["nll_total"],
        "nll_per_channel": res["model"]["nll_per_channel"],
        "pit": res["model"]["pit"].tolist(),
        "n_windows": res["model"]["n_windows"],
        "channels": data.columns,
        "standardized": {k: summarize(v["metrics"]) for k, v in res.items()},
        "original_scale_per_channel": {
            k: summarize(to_original_scale(v["metrics"], std)) for k, v in res.items()
        },
    }
    path = exp / f"eval_{args.split}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"disimpan: {path}")


if __name__ == "__main__":
    main()