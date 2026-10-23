buat beberapa variansi untuk menguji model dengan membuat file YAML. di mana file YAML ini akan dengan ketentuan:
1. dropout 0,1 vs 0,3,
2. d_model 64 vs 32 (dan mungkin 1 layer),
3. weight decay 0,01 vs 0,1,
4. learning rate 6e-4 vs 3e-4.
Empat sampai enam run sudah cukup. Penilaiannya memakai MAE, CRPS, dan coverage80 di val, bukan NLL saja, karena NLL val sangat sensitif pada distribusi yang terlalu sempit

python -m src.training.run --dataset ETTh1 --pred-len 24 --train-config configs/training/variant_A_dropout_0.3.yaml
python -m src.training.run --dataset ETTh1 --pred-len 24 --train-config configs/training/variant_B_weight_decay_0.05.yaml
python -m src.training.run --dataset ETTh1 --pred-len 24 --train-config configs/training/variant_C_lr_3e-4.yaml
python -m src.training.run --dataset ETTh1 --pred-len 24 --train-config configs/training/variant_A_dmodel_32.yaml
python -m src.training.run --dataset ETTh1 --pred-len 24 --train-config configs/training/variant_A_dropout_0.05.yaml
python -m src.training.run --dataset ETTh1 --pred-len 24 --train-config configs/training/variant_A1_dropout_0.1.yaml

python -m src.training.run `
  --dataset ETTh1 --pred-len 24 `
  --model-config configs/model/best_optuna.yaml `
  --train-config configs/training/best_optuna.yaml `
  --out-dir exps/final_best