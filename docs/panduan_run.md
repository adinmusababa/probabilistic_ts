untuk menjalankan ekperimen:
1. python -m src.training.run --dataset ETTh1 --pred-len 24 --model-config configs/model/best_optuna.yaml --train-config configs/training/best_optuna.yaml --out-dir exps/ETTh1_H24_optuna_seed42_1

ganti output agar bisa dieval

2. python -m src.evaluation.run_eval --exp-dir exps/ETTh1_H24_optuna_seed42_1 --split val


untuk menjalankan optuna:
python scripts\optuna_search.py --n-trials 30
Opsi utama

┌──────────────┬───────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
│     Flag     │                Default                │                               Fungsi                                │
├──────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ --n-trials   │ 30                                    │ Jumlah trial Optuna                                                 │
├──────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ --timeout    │ None                                  │ Batasi waktu (detik), mis. --timeout 3600 = 1 jam                   │
├──────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ --study-name │ prob_transformer_etth1                │ Nama study (buat resume nanti)                                      │
├──────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ --storage    │ sqlite:///exps/optuna_study/optuna.db │ DB SQLite untuk persist study                                       │
├──────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ --sampler    │ tpe                                   │ tpe | random | cmaes                                                │
├──────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ --pruner     │ median                                │ median | none (median pruner: n_startup_trials=5, n_warmup_steps=3) │
└──────────────┴───────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

# Cepat: 10 trial, 10 menit max, sampler random
python scripts\optuna_search.py --n-trials 10 --timeout 600 --sampler random --pruner none

# Standar: 50 trial TPE + median pruner (default)
python scripts\optuna_search.py --n-trials 50

# Lanjutkan study yang sudah ada (load_if_exists=True)
python scripts\optuna_search.py --n-trials 20