#!/usr/bin/env python
"""
Optuna hyperparameter search untuk Probabilistic Transformer (ETTh1).

- Tidak mengubah kode di src/ — hanya memakai fungsi train() & prepare_data() yang sudah ada.
- Mencari: dropout, d_model, lr, weight_decay, (opsional) encoder_layers, decoder_layers.
- Menyimpan best config ke configs/model/best_optuna.yaml & configs/training/best_optuna.yaml.
- Log tiap trial ke exps/optuna_study/trial_<number>/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import optuna
import torch

# ---- Pastikan src/ bisa di-import ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.pipeline import prepare_data
from src.models.probabilistic_transformer import ProbabilisticTransformer
from src.training.trainer import train


# ---------- Base config (tetap) ----------
BASE_MODEL_CFG = {
    "context_length": 96,
    "prediction_length": 24,
    "input_size": 7,
    "lags_sequence": [1, 2, 3, 4, 5, 6, 7, 24, 48, 168],
    "num_time_features": 4,
    "scaling": False,
    "distribution_output": "student_t",
    "encoder_attention_heads": 2,
    "decoder_attention_heads": 2,
    "num_parallel_samples": 100,
}

BASE_TRAIN_CFG = {
    "seed": 42,
    "batch_size": 64,
    "epochs": 30,
    "patience": 5,
    "min_delta": 0.0,
    "grad_clip": 1.0,
    "train_stride": 1,
    "num_workers": 0,
    "device": None,  # auto
}


# Module-level cache untuk PreparedData (trial berjalan sekual di proses yang sama)
_CACHED_DATA: dict = {"data": None, "past_length": None}


# ---------- Objective ----------
def objective(trial: optuna.Trial) -> float:
    # --- Search space ---
    dropout = trial.suggest_float("dropout", 0.05, 0.4, step=0.05)
    d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)

    # Opsional: depth
    enc_layers = trial.suggest_categorical("encoder_layers", [2, 3])
    dec_layers = trial.suggest_categorical("decoder_layers", [2, 3])

    # ffn_dim biasanya 2x atau 4x d_model
    ffn_mult = trial.suggest_categorical("ffn_mult", [2, 4])
    encoder_ffn_dim = d_model * ffn_mult
    decoder_ffn_dim = d_model * ffn_mult

    # --- Build configs ---
    model_cfg = BASE_MODEL_CFG.copy()
    model_cfg.update({
        "dropout": dropout,
        "d_model": d_model,
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "encoder_ffn_dim": encoder_ffn_dim,
        "decoder_ffn_dim": decoder_ffn_dim,
    })

    train_cfg = BASE_TRAIN_CFG.copy()
    train_cfg.update({
        "lr": lr,
        "weight_decay": weight_decay,
    })

    # --- Prepare data (cached in module-level dict) ---
    past_length = model_cfg["context_length"] + max(model_cfg["lags_sequence"])

    if _CACHED_DATA["data"] is None or _CACHED_DATA["past_length"] != past_length:
        print(f"[Trial {trial.number}] Memuat & mempersiapkan data ETTh1 (past_length={past_length})...")
        data = prepare_data("ETTh1", past_length, model_cfg["prediction_length"], train_stride=train_cfg["train_stride"])
        _CACHED_DATA["data"] = data
        _CACHED_DATA["past_length"] = past_length
    else:
        data = _CACHED_DATA["data"]

    # --- Output dir per trial ---
    trial_dir = Path(f"exps/optuna_study/trial_{trial.number}")
    trial_dir.mkdir(parents=True, exist_ok=True)

    # --- Build model & train ---
    model = ProbabilisticTransformer(model_cfg)

    try:
        manifest = train(model, data, train_cfg, trial_dir, model_cfg)
        val_nll = manifest["best_val_nll"]

        # Simpan config trial ke folder trial
        (trial_dir / "model_cfg.yaml").write_text(
            __import__("yaml").safe_dump(model_cfg, sort_keys=False)
        )
        (trial_dir / "train_cfg.yaml").write_text(
            __import__("yaml").safe_dump(train_cfg, sort_keys=False)
        )

        # Pruning: jika val_nll NaN/Inf, tarik trial
        if not torch.isfinite(torch.tensor(val_nll)):
            raise optuna.TrialPruned()

        return val_nll

    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
        # Hapus folder trial yang gagal biar bersih
        shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned()


# ---------- Callback simpan best config ----------
def save_best_config_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    if study.best_trial.number == trial.number:
        best_model_cfg = BASE_MODEL_CFG.copy()
        best_model_cfg.update({
            "dropout": trial.params["dropout"],
            "d_model": trial.params["d_model"],
            "encoder_layers": trial.params["encoder_layers"],
            "decoder_layers": trial.params["decoder_layers"],
            "encoder_ffn_dim": trial.params["d_model"] * trial.params["ffn_mult"],
            "decoder_ffn_dim": trial.params["d_model"] * trial.params["ffn_mult"],
        })
        best_train_cfg = BASE_TRAIN_CFG.copy()
        best_train_cfg.update({
            "lr": trial.params["lr"],
            "weight_decay": trial.params["weight_decay"],
        })

        # Simpan ke configs/
        model_out = Path("configs/model/best_optuna.yaml")
        train_out = Path("configs/training/best_optuna.yaml")
        model_out.parent.mkdir(parents=True, exist_ok=True)
        train_out.parent.mkdir(parents=True, exist_ok=True)

        import yaml
        model_out.write_text(yaml.safe_dump(best_model_cfg, sort_keys=False))
        train_out.write_text(yaml.safe_dump(best_train_cfg, sort_keys=False))

        print(f"\n✅ Best config updated (trial {trial.number}, val_nll={trial.value:.4f})")
        print(f"   Model config : {model_out}")
        print(f"   Train config : {train_out}")


# ---------- Main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna HPO untuk Probabilistic Transformer")
    ap.add_argument("--n-trials", type=int, default=30, help="Jumlah trial Optuna")
    ap.add_argument("--timeout", type=int, default=None, help="Timeout detik (opsional)")
    ap.add_argument("--study-name", default="prob_transformer_etth1", help="Nama study Optuna")
    ap.add_argument("--storage", default="sqlite:///exps/optuna_study/optuna.db", help="Storage URL Optuna")
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "random", "cmaes"], help="Sampler Optuna")
    ap.add_argument("--pruner", default="median", choices=["median", "none"], help="Pruner Optuna")
    args = ap.parse_args()

    # Sampler & Pruner
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.CmaEsSampler(seed=42)

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    else:
        pruner = optuna.pruners.NopPruner()

    # Buat/load study
    Path("exps/optuna_study").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"🚀 Memulai Optuna study: {args.study_name}")
    print(f"   Trials target : {args.n_trials}")
    print(f"   Storage       : {args.storage}")
    print(f"   Sampler       : {args.sampler}")
    print(f"   Pruner        : {args.pruner}\n")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[save_best_config_callback],
        show_progress_bar=True,
    )

    # Ringkasan akhir
    print("\n" + "=" * 60)
    print("🏁 OPTUNA SELESAI")
    print("=" * 60)
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best val NLL: {study.best_value:.4f}")
    print(f"Best params :")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    print(f"\nBest model config : configs/model/best_optuna.yaml")
    print(f"Best train config : configs/training/best_optuna.yaml")
    print(f"\nUntuk melanjutkan study lain waktu: jalankan script yang sama (load_if_exists=True).")


if __name__ == "__main__":
    main()