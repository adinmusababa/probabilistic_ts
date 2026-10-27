"""Membandingkan semua run di exps/ (tabel angka). Plot ada di plots.py.

Pemakaian di notebook:
    from src.evaluation.compare_runs import load_runs, summary_table, comparability_warnings
    runs = load_runs("exps", "val")
    df = summary_table(runs)

Pemakaian CLI (tabel + CSV + semua gambar):
    python -m src.evaluation.compare_runs --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

METRIC_COLS = [
    "nll_ch", "mse", "mae", "crps", "cov50", "cov80", "cov95", "width80",
    "mae_vs_naive_pct", "mse_vs_naive_pct", "pit_tails", "pit_skew",
]


def _naive_key(std: dict) -> str | None:
    return next((k for k in std if k.startswith("seasonal_naive")), None)


def load_runs(exps_dir: str | Path = "exps", split: str = "val", names=None) -> dict[str, dict]:
    """Muat semua run yang punya eval_<split>.json dan manifest.json.

    Folder tanpa evaluasi (mis. exps/optuna_study) dilewati otomatis.
    """
    runs: dict[str, dict] = {}
    for ev in sorted(Path(exps_dir).glob(f"*/eval_{split}.json")):
        exp = ev.parent
        if names is not None and exp.name not in names:
            continue
        man = exp / "manifest.json"
        if not man.exists():
            continue
        hist = exp / "history.csv"
        runs[exp.name] = {
            "eval": json.loads(ev.read_text()),
            "manifest": json.loads(man.read_text()),
            "history": pl.read_csv(hist) if hist.exists() else None,
        }
    return runs


def _g(d: dict | None, key: str):
    return d[key]["global"] if d and key in d else None


def summary_table(runs: dict[str, dict], sort_by: str = "crps", digits: int = 4) -> pl.DataFrame:
    """Satu baris per run: konfigurasi utama + metrik global (ruang terstandardisasi)."""
    rows = []
    for name, r in runs.items():
        e, m = r["eval"], r["manifest"]
        mc, tc = m["model_cfg"], m["train_cfg"]
        std = e["standardized"]
        s = std["model"]
        nk = _naive_key(std)
        naive = std[nk] if nk else None
        mae, mse = _g(s, "mae_median"), _g(s, "mse_mean")
        n_mae, n_mse = _g(naive, "mae_median"), _g(naive, "mse_mean")
        pit = e.get("pit") or []
        rows.append(
            {
                "run": name,
                "strategy": m.get("online", {}).get("strategy"),
                "scaling": str(mc.get("scaling")),
                "d_model": mc.get("d_model"),
                "layers": f"{mc.get('encoder_layers')}/{mc.get('decoder_layers')}",
                "dropout": mc.get("dropout"),
                "lr": tc.get("lr"),
                "wd": tc.get("weight_decay"),
                "seed": tc.get("seed"),
                "best_epoch": m.get("best_epoch"),
                "epochs": m.get("epochs_run"),
                "n_samples": e.get("n_samples"),
                "n_windows": e.get("n_windows"),
                "nll_ch": e.get("nll_per_channel"),
                "mse": mse,
                "mae": mae,
                "crps": _g(s, "crps"),
                "cov50": _g(s, "coverage_50"),
                "cov80": _g(s, "coverage_80"),
                "cov95": _g(s, "coverage_95"),
                "width80": _g(s, "width_80"),
                "mae_vs_naive_pct": 100 * (mae / n_mae - 1) if mae is not None and n_mae else None,
                "mse_vs_naive_pct": 100 * (mse / n_mse - 1) if mse is not None and n_mse else None,
                "pit_tails": (pit[0] + pit[-1]) if pit else None,
                "pit_skew": (pit[-1] - pit[0]) if pit else None,
            }
        )
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    # Round only numeric metric columns; skip any column that contains nulls (e.g., nll_ch when NLL not computed)
    numeric_cols = [c for c in METRIC_COLS if c in df.columns and df[c].dtype.is_numeric()]
    df = df.with_columns([pl.col(c).round(digits) for c in numeric_cols])
    if sort_by in df.columns:
        df = df.sort(sort_by, nulls_last=True)
    return df


def naive_reference(runs: dict[str, dict]) -> dict[str, float]:
    """Nilai seasonal naive (MAE dan MSE) dari run pertama, untuk garis acuan di plot."""
    first = next(iter(runs.values()))["eval"]["standardized"]
    nk = _naive_key(first)
    if nk is None:
        return {}
    return {"mae": first[nk]["mae_median"]["global"], "mse": first[nk]["mse_mean"]["global"]}


def comparability_warnings(runs: dict[str, dict]) -> list[str]:
    """Peringatan jika run tidak sebanding (jumlah sampel atau window berbeda)."""
    out = []
    ns = {r["eval"].get("n_samples") for r in runs.values()}
    if len(ns) > 1:
        out.append(
            f"n_samples berbeda antar run: {sorted(map(str, ns))}. Coverage, lebar interval, dan PIT "
            "tidak sebanding; evaluasi ulang dengan n_samples yang sama (disarankan 100)."
        )
    nw = {r["eval"].get("n_windows") for r in runs.values()}
    if len(nw) > 1:
        out.append(f"Jumlah window berbeda antar run: {sorted(map(str, nw))}.")
    ds = {(r["manifest"].get("data") or {}).get("dataset") for r in runs.values()}
    if len(ds) > 1:
        out.append(f"Dataset berbeda antar run: {sorted(map(str, ds))}.")
    hz = {r["manifest"]["model_cfg"].get("prediction_length") for r in runs.values()}
    if len(hz) > 1:
        out.append(f"Horizon berbeda antar run: {sorted(map(str, hz))}.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps-dir", default="exps")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out-dir", default=None, help="Default: <exps-dir>/_compare")
    ap.add_argument("--sort-by", default="crps")
    args = ap.parse_args()

    runs = load_runs(args.exps_dir, args.split)
    if not runs:
        print(f"Tidak ada run dengan eval_{args.split}.json di {args.exps_dir}.")
        return
    df = summary_table(runs, sort_by=args.sort_by)
    for w in comparability_warnings(runs):
        print("PERINGATAN:", w)
    with pl.Config(tbl_rows=100, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(df)

    from src.evaluation.plots import save_all  # impor lokal: matplotlib hanya dibutuhkan di sini

    out_dir = Path(args.out_dir or Path(args.exps_dir) / "_compare")
    paths = save_all(runs, df, out_dir, split=args.split)
    print("Disimpan:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()