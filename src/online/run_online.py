"""Evaluasi online sebuah run training dengan strategi NoAdapt atau OnlineUpdate.

Contoh:
    python -m src.online.run_online --exp-dir exps/optuna_std --split val --strategy noadapt --eval-stride 4
    python -m src.online.run_online --exp-dir exps/optuna_std --split val --strategy online_update \\
        --params last_layer --lr 1e-4 --eval-stride 4

Hasil ditulis ke exps/<exp>_online_<strategy>[_tag]/ dengan eval_<split>.json yang kompatibel dengan
compare_runs. Gunakan eval-stride dan n-samples yang SAMA untuk semua strategi yang dibandingkan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch

from src.adapt.noadapt import NoAdapt
from src.adapt.online_update import PARAM_GROUPS, OnlineUpdate
from src.baselines.seasonal_naive import season_length
from src.data.pipeline import prepare_data
from src.online.loop import run_online, to_eval_json
from src.training.trainer import load_checkpoint


def _row(name: str, m: dict) -> dict:
    g = lambda k: float(m[k].mean()) if k in m else None
    return {
        "metode": name, "mse": g("mse_mean"), "mae": g("mae_median"), "crps": g("crps"),
        "cov50": g("coverage_50"), "cov80": g("coverage_80"), "cov95": g("coverage_95"),
        "lebar80": g("width_80"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="Folder run training (berisi best.pt dan manifest.json)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--strategy", default="noadapt", choices=["noadapt", "online_update"])
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--eval-stride", type=int, default=4, help="Evaluasi tiap k origin (update tetap tiap langkah)")
    ap.add_argument("--max-origins", type=int, default=None, help="Untuk uji cepat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="last_layer", choices=PARAM_GROUPS)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--buffer", type=int, default=1, help="Jumlah window lengkap terbaru per pembaruan")
    ap.add_argument("--update-every", type=int, default=1)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    manifest = json.loads((exp / "manifest.json").read_text())
    name = manifest["data"]["dataset"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_checkpoint(exp / "best.pt", device=device)
    data = prepare_data(name, model.past_length, model.prediction_length)
    if data.meta["data_sha256"] != manifest["data"]["data_sha256"]:
        print("PERINGATAN: isi data di database berbeda dari saat training (hash tidak cocok).")

    if args.strategy == "noadapt":
        learner = NoAdapt(model, device)
    else:
        learner = OnlineUpdate(
            model, device, params=args.params, lr=args.lr, n_steps=args.n_steps,
            buffer_size=args.buffer, update_every=args.update_every,
        )

    m = season_length(name)
    print(f"{name} | {args.split} | strategi {learner.name} | stride {args.eval_stride} | device {device}")
    res = run_online(
        learner, data.values, data.time_feats, data.segments[args.split],
        model.past_length, model.prediction_length,
        n_samples=args.n_samples, eval_stride=args.eval_stride, max_origins=args.max_origins,
        season_m=m, seed=args.seed,
    )

    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.exp_dir).parent / f"{exp.name}_online_{args.strategy}{tag}"
    out.mkdir(parents=True, exist_ok=True)

    ev = to_eval_json(res, data.columns, args.split, args.n_samples, args.seed, f"seasonal_naive_{m}")
    (out / f"eval_{args.split}.json").write_text(json.dumps(ev, indent=2))
    pl.DataFrame({k: v.tolist() for k, v in res.per_origin.items()}).write_csv(
        out / f"online_per_origin_{args.split}.csv"
    )
    man = dict(manifest)
    man["online"] = {
        "strategy": args.strategy, "base_run": exp.name, "split": args.split,
        "eval_stride": args.eval_stride, "n_samples": args.n_samples, "seed": args.seed,
        "learner": res.diagnostics,
        "seconds_update": round(res.seconds_update, 1), "seconds_predict": round(res.seconds_predict, 1),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str))

    table = pl.DataFrame([
        _row(args.strategy, res.metrics), _row(f"seasonal_naive_{m}", res.naive_metrics),
        _row("last_value", res.last_metrics),
    ])
    with pl.Config(tbl_rows=10, tbl_cols=20, float_precision=4):
        print(table)
    if res.nll_total is not None:
        print(f"NLL: total {res.nll_total:.4f} | per kanal {res.nll_total / len(data.columns):.4f}")
    print(f"{res.n_windows} window | update {res.seconds_update:.0f}s | prediksi {res.seconds_predict:.0f}s")
    print("diagnostik learner:", res.diagnostics)
    print(f"disimpan: {out}")


if __name__ == "__main__":
    main()