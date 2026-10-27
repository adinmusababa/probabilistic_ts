"""Loop evaluasi online tanpa kebocoran informasi (torch-free)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.baselines.seasonal_naive import last_value, seasonal_naive
from src.data.splits import window_origins
from src.metrics.aggregate import MetricAccumulator, summarize
from src.metrics.calibration import coverage_indicator
from src.metrics.probabilistic import central_interval, crps_samples
from src.online.learner import OnlineLearner
from src.online.ledger import Ledger


@dataclass
class OnlineResult:
    learner_name: str
    metrics: dict[str, np.ndarray]
    naive_metrics: dict[str, np.ndarray]
    last_metrics: dict[str, np.ndarray]
    pit: np.ndarray
    n_windows: int
    nll_total: float | None
    per_origin: dict[str, np.ndarray]
    seconds_update: float
    seconds_predict: float
    diagnostics: dict
    samples: dict[int, np.ndarray] | None = field(default=None, repr=False)


def run_online(
    learner: OnlineLearner,
    values: np.ndarray,
    time_feats: np.ndarray,
    segment: tuple[int, int],
    past_length: int,
    horizon: int,
    n_samples: int = 100,
    eval_stride: int = 1,
    max_origins: int | None = None,
    season_m: int = 24,
    seed: int = 0,
    keep_samples: bool = False,
    progress_every: int = 100,
) -> OnlineResult:
    """Alur per waktu t (t berurutan, langkah 1):

    1. ledger.advance(t): label indeks < t menjadi terlihat.
    2. learner.update(t): memakai hanya label yang terlihat.
    3. Jika t termasuk origin evaluasi (tiap eval_stride): learner.predict(t), lalu skor terhadap
       nilai nyata t..t+H-1. Nilai nyata hanya dipegang loop ini, tidak diberikan ke learner
       kecuali untuk penilaian NLL (tanpa mengubah state).
    Baseline seasonal naive dan last value dihitung pada origin yang sama.
    """
    if eval_stride < 1:
        raise ValueError("eval_stride minimal 1.")
    truth = np.asarray(values, dtype=np.float64)
    ledger = Ledger(values, time_feats)
    origins = window_origins(segment, past_length, horizon, stride=1)
    if max_origins is not None:
        origins = origins[:max_origins]
    if len(origins) == 0:
        raise ValueError("Tidak ada origin yang valid pada segmen ini.")
    t_first = int(origins[0])
    n_eval_total = len(range(0, len(origins), eval_stride))

    acc, acc_naive, acc_last = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
    rows = {k: [] for k in ("t", "mae", "mse", "crps", "cov80", "nll")}
    stored: dict[int, np.ndarray] | None = {} if keep_samples else None
    nlls: list[float] = []
    sec_update = sec_predict = 0.0
    done = 0

    for t in map(int, origins):
        ledger.advance(t)
        t0 = time.perf_counter()
        learner.update(t, ledger)
        sec_update += time.perf_counter() - t0

        if (t - t_first) % eval_stride:
            continue

        t0 = time.perf_counter()
        samples = np.asarray(learner.predict(t, ledger, n_samples, seed + t), dtype=np.float64)  # (S,1,H,D)
        sec_predict += time.perf_counter() - t0

        y = truth[t : t + horizon]  # (H, D)
        acc.update(samples, y[None])
        past = np.asarray(ledger.past_window(t, past_length), dtype=np.float64)[None]
        acc_naive.update(seasonal_naive(past, horizon, season_m)[None], y[None])
        acc_last.update(last_value(past, horizon)[None], y[None])

        nll = learner.nll(t, ledger, y)
        if nll is not None:
            nlls.append(float(nll))

        s3 = samples[:, 0]
        lo, hi = central_interval(s3, 0.8)
        rows["t"].append(t)
        rows["mae"].append(float(np.abs(np.median(s3, axis=0) - y).mean()))
        rows["mse"].append(float(((s3.mean(axis=0) - y) ** 2).mean()))
        rows["crps"].append(float(crps_samples(s3, y).mean()))
        rows["cov80"].append(float(coverage_indicator(lo, hi, y).mean()))
        rows["nll"].append(float(nll) if nll is not None else np.nan)
        if stored is not None:
            stored[t] = samples

        done += 1
        if progress_every and done % progress_every == 0:
            print(f"  {done}/{n_eval_total} origin | update {sec_update:.0f}s | prediksi {sec_predict:.0f}s")

    return OnlineResult(
        learner_name=learner.name,
        metrics=acc.finalize(),
        naive_metrics=acc_naive.finalize(),
        last_metrics=acc_last.finalize(),
        pit=acc.pit_proportions(),
        n_windows=acc.n,
        nll_total=float(np.mean(nlls)) if nlls else None,
        per_origin={k: np.asarray(v) for k, v in rows.items()},
        seconds_update=sec_update,
        seconds_predict=sec_predict,
        diagnostics=learner.diagnostics(),
        samples=stored,
    )


def to_eval_json(
    res: OnlineResult, channels: list[str], split: str, n_samples: int, seed: int, naive_key: str
) -> dict:
    """Format sama dengan eval_<split>.json statis, sehingga compare_runs dan notebook bisa langsung memakainya."""
    d = len(channels)
    return {
        "split": split,
        "n_samples": n_samples,
        "seed": seed,
        "nll_total": res.nll_total,
        "nll_per_channel": None if res.nll_total is None else res.nll_total / d,
        "pit": res.pit.tolist(),
        "n_windows": res.n_windows,
        "channels": list(channels),
        "standardized": {
            "model": summarize(res.metrics),
            naive_key: summarize(res.naive_metrics),
            "last_value": summarize(res.last_metrics),
        },
    }