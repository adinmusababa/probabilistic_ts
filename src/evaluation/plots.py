"""Plot perbandingan run (matplotlib). Semua fungsi plot mengembalikan Figure."""
from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

LEVELS = (0.5, 0.8, 0.95)


def expected_coverage(level: float, n_samples: int) -> float:
    """Perkiraan coverage prediksi yang SEMPURNA bila kuantil dihitung dari n_samples sampel.

    Memakai nilai harapan statistik terurut dengan interpolasi linear (np.quantile baku).
    Hanya pendekatan: berguna sebagai garis acuan, bukan nilai eksak.
    """
    a = (1.0 - level) / 2.0
    idx = a * (n_samples - 1)
    k = int(np.floor(idx))
    lo = (k + 1 + (idx - k)) / (n_samples + 1)
    return 1.0 - 2.0 * lo


def _names(runs: dict, names) -> list[str]:
    return list(runs) if names is None else [n for n in names if n in runs]


def _baseline_series(runs: dict, metric: str, field: str) -> dict[str, list[float]]:
    std = next(iter(runs.values()))["eval"]["standardized"]
    out = {}
    for key in std:
        if key.startswith("seasonal_naive") or key == "last_value":
            out[key] = std[key][metric][field]
    return out


def plot_metric_bars(df: pl.DataFrame, metrics=("mae", "mse", "crps", "nll_ch"), naive: dict | None = None):
    """Titik per run untuk tiap metrik (lebih kecil lebih baik). Garis putus = seasonal naive."""
    metrics = [m for m in metrics if m in df.columns]
    ncols = 2 if len(metrics) > 1 else 1
    nrows = ceil(len(metrics) / ncols)
    h = max(2.5, 0.38 * df.height + 1.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, h * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        d = df.select("run", m).drop_nulls().sort(m, descending=True)  # terbaik di atas
        runs, vals = d["run"].to_list(), d[m].to_list()
        ax.plot(vals, runs, "o")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(6, -3), fontsize=8)
        if naive and m in naive:
            ax.axvline(naive[m], ls="--", color="k", lw=1, label="seasonal naive")
            ax.legend(fontsize=8)
        ax.set_title(f"{m} (lebih kecil lebih baik)")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_per_horizon(runs: dict, metric: str = "mae_median", names=None, baselines: bool = True):
    """Metrik per langkah horizon; baseline putus-putus."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for n in _names(runs, names):
        ys = runs[n]["eval"]["standardized"]["model"][metric]["per_horizon"]
        ax.plot(range(1, len(ys) + 1), ys, marker="o", ms=3, label=n)
    if baselines:
        for key, ys in _baseline_series(runs, metric, "per_horizon").items():
            ls = "--" if key.startswith("seasonal") else ":"
            ax.plot(range(1, len(ys) + 1), ys, ls, color="k" if ls == "--" else "gray", label=key)
    ax.set_xlabel("langkah horizon")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per langkah horizon")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_per_channel(runs: dict, metric: str = "mae_median", names=None):
    """Metrik per kanal: batang berkelompok (run dan seasonal naive)."""
    first = next(iter(runs.values()))["eval"]
    chans = first["channels"]
    series = {n: runs[n]["eval"]["standardized"]["model"][metric]["per_channel"] for n in _names(runs, names)}
    for key, vals in _baseline_series(runs, metric, "per_channel").items():
        if key.startswith("seasonal"):
            series[key] = vals
    x = np.arange(len(chans))
    w = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (lab, vals) in enumerate(series.items()):
        ax.bar(x - 0.4 + w * (i + 0.5), vals, w, label=lab, color="k" if lab.startswith("seasonal") else None,
               alpha=0.6 if lab.startswith("seasonal") else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(chans)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per kanal")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_calibration(runs: dict, names=None):
    """Coverage empiris vs level nominal. Garis putus hitam = perkiraan nilai ideal untuk jumlah sampel run."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0.4, 1.0], [0.4, 1.0], "k-", lw=0.8, label="ideal (sampel tak hingga)")
    samples = {runs[n]["eval"].get("n_samples") for n in _names(runs, names)} - {None}
    for s in sorted(samples):
        ax.plot(LEVELS, [expected_coverage(lv, s) for lv in LEVELS], "k--", lw=1, label=f"ideal ≈ untuk {s} sampel")
    for n in _names(runs, names):
        m = runs[n]["eval"]["standardized"]["model"]
        ys = [m[f"coverage_{int(round(lv * 100))}"]["global"] for lv in LEVELS]
        ax.plot(LEVELS, ys, marker="o", label=f"{n} (S={runs[n]['eval'].get('n_samples')})")
    ax.set_xlabel("level nominal")
    ax.set_ylabel("coverage empiris")
    ax.set_title("Kalibrasi interval prediksi")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def plot_pit(runs: dict, names=None, ncols: int = 3):
    """Histogram PIT per run. Garis putus = nilai seragam (1/jumlah bin)."""
    ns = [n for n in _names(runs, names) if runs[n]["eval"].get("pit")]
    nrows = max(1, ceil(len(ns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, n in zip(axes.ravel(), ns):
        pit = runs[n]["eval"]["pit"]
        ax.bar(np.arange(len(pit)), pit, width=0.9)
        ax.axhline(1 / len(pit), ls="--", color="k", lw=1)
        ax.set_title(f"{n} (S={runs[n]['eval'].get('n_samples')})", fontsize=9)
        ax.set_xticks([])
    for ax in axes.ravel()[len(ns):]:
        ax.axis("off")
    fig.suptitle("PIT: bin pertama = nilai nyata di bawah semua sampel, bin terakhir = di atas", fontsize=10)
    fig.tight_layout()
    return fig


def plot_training_curves(runs: dict, names=None):
    """Kurva NLL train (kanan) dan val (kiri) dari history.csv."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for n in _names(runs, names):
        h = runs[n]["history"]
        if h is None:
            continue
        a1.plot(h["epoch"], h["val_nll"], marker="o", ms=3, label=n)
        a2.plot(h["epoch"], h["train_nll"], marker="o", ms=3, label=n)
    a1.set_title("NLL validasi (total antar kanal)")
    a2.set_title("NLL train (total antar kanal)")
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    a2.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def save_all(runs: dict, df: pl.DataFrame, out_dir: str | Path, split: str = "val") -> list[Path]:
    """Simpan tabel (CSV) dan semua gambar ke out_dir."""
    from src.evaluation.compare_runs import naive_reference

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    csv = out / f"summary_{split}.csv"
    df.write_csv(csv)
    paths.append(csv)
    figs = {
        "metrics": plot_metric_bars(df, naive=naive_reference(runs)),
        "mae_per_horizon": plot_per_horizon(runs, "mae_median"),
        "mae_per_channel": plot_per_channel(runs, "mae_median"),
        "calibration": plot_calibration(runs),
        "pit": plot_pit(runs),
        "training_curves": plot_training_curves(runs),
    }
    for name, fig in figs.items():
        p = out / f"{name}_{split}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths