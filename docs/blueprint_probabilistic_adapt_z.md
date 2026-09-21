# Blueprint Proyek: Online Probabilistic Time-Series Forecasting with ADAPT-Z

> **Status:** Master Blueprint / Implementation Guide  
> **Backbone:** Probabilistic Time-Series Transformer  
> **Adaptation:** ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space)  
> **Dataset:** ETT (ETTh1, ETTh2, ETTm1, ETTm2)  
> **Continual Learning:** Tidak menjadi komponen utama pada versi awal  
> **Primary Goal:** Membangun dan mengevaluasi probabilistic time-series forecasting model yang dapat melakukan feature adaptation secara online menggunakan ADAPT-Z.

---

# 1. Project Definition

## 1.1 Working Title

**Online Probabilistic Time-Series Forecasting with ADAPT-Z Feature Adaptation**

Judul ini merupakan working title dan tidak perlu dikunci sebagai judul penelitian final sebelum seluruh eksperimen selesai.

## 1.2 Problem Statement

Model time-series forecasting yang dilatih secara offline dapat mengalami penurunan performa ketika distribusi data masa depan berbeda dari distribusi data training.

Probabilistic forecasting memberikan informasi yang lebih lengkap daripada point forecasting karena model menghasilkan predictive distribution:

```text
p(y | x)
```

yang dapat digunakan untuk memperoleh:

- point forecast,
- uncertainty,
- prediction interval,
- probabilistic likelihood.

Proyek ini mengeksplorasi penggunaan ADAPT-Z sebagai mekanisme online feature adaptation pada probabilistic time-series Transformer.

Konsep utamanya:

```text
ETT Dataset
    ↓
Probabilistic Time-Series Transformer
    ↓
Hidden Representation
    ↓
ADAPT-Z
    ↓
Distribution Head
    ↓
Predictive Distribution
    ↓
Forecast + Uncertainty
```

---

# 2. Research Objective

Tujuan utama:

> Mengembangkan dan mengevaluasi probabilistic time-series forecasting model yang mampu melakukan adaptasi representasi secara online menggunakan ADAPT-Z ketika menghadapi perubahan distribusi data.

Tujuan turunan:

1. Membangun baseline probabilistic Transformer.
2. Membangun pipeline probabilistic forecasting yang dapat menghitung point dan probabilistic metrics.
3. Membuat simulasi online forecasting menggunakan ETT.
4. Mengimplementasikan mekanisme ADAPT-Z.
5. Mengintegrasikan ADAPT-Z dengan probabilistic Transformer.
6. Membandingkan model tanpa adaptation, online updating biasa, dan ADAPT-Z.
7. Menganalisis pengaruh adaptation terhadap forecast accuracy, predictive distribution, dan calibration.
8. Melakukan ablation study.
9. Menguji generalisasi pada beberapa dataset ETT.

---

# 3. Research Questions

## RQ1

Bagaimana performa probabilistic Transformer dalam melakukan time-series forecasting pada dataset ETT?

## RQ2

Apakah ADAPT-Z dapat meningkatkan akurasi forecasting dan kualitas predictive distribution ketika diterapkan pada probabilistic Transformer dalam skenario online forecasting?

## RQ3

Bagaimana pengaruh ADAPT-Z terhadap calibration dan uncertainty estimation selama online forecasting ketika terjadi distribution shift?

---

# 4. Hypotheses

## H1 — Point Forecasting

ADAPT-Z dapat mempertahankan atau meningkatkan point forecasting performance setelah terjadi distribution shift.

Metrics:

- MAE
- MSE
- RMSE

## H2 — Probabilistic Forecasting

ADAPT-Z dapat meningkatkan kualitas predictive distribution.

Metrics:

- NLL
- CRPS

## H3 — Calibration

ADAPT-Z dapat menghasilkan uncertainty estimation yang lebih sesuai terhadap observed outcomes.

Metrics:

- PICP
- MPIW
- coverage deviation

---

# 5. Scope

## 5.1 Included

```text
Dataset:
    ETTh1
    ETTh2
    ETTm1
    ETTm2

Backbone:
    Probabilistic Time-Series Transformer

Adaptation:
    ADAPT-Z

Evaluation:
    Point forecasting
    Probabilistic forecasting
    Calibration
    Online adaptation
    Distribution shift analysis
```

## 5.2 Not Included in Initial Version

```text
iTransformer
Continual Learning as a separate algorithm
Classification
Clustering
Multimodal data
Generative multimodal forecasting
```

Continual Learning hanya dipertimbangkan sebagai extension apabila eksperimen menunjukkan masalah retention/catastrophic forgetting yang memang relevan.

---

# 6. High-Level System Architecture

```text
                       ┌──────────────────┐
                       │   ETT Dataset    │
                       │                  │
                       │ ETTh1/2          │
                       │ ETTm1/2          │
                       └────────┬─────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Data Preprocessing  │
                     │                     │
                     │ Cleaning            │
                     │ Scaling             │
                     │ Time Features       │
                     │ Missing Mask        │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Window Construction │
                     │                     │
                     │ Context Window      │
                     │ Prediction Horizon  │
                     └──────────┬──────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │ Probabilistic Transformer   │
                 │                             │
                 │ Encoder                     │
                 │ Decoder                     │
                 │ Hidden Representation       │
                 └──────────────┬──────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    ADAPT-Z      │
                       │                 │
                       │ Persistent      │
                       │ Tracking        │
                       │ Feature         │
                       │ Adaptation      │
                       └────────┬────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Distribution Head   │
                     │                     │
                     │ Distribution Params │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Predictive          │
                     │ Distribution        │
                     │                     │
                     │ p(y | x)            │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
         Point Forecast      NLL/CRPS        Prediction
                                                Interval
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Evaluation &        │
                     │ Analysis            │
                     └─────────────────────┘
```

---

# 7. System Components

Proyek dibagi menjadi enam subsystem:

```text
1. Data Layer
2. Preprocessing Layer
3. Forecasting Model
4. Adaptation Layer
5. Evaluation Layer
6. Experiment & Reporting Layer
```

---

# 8. Data Layer

Dataset:

```text
ETT
├── ETTh1
├── ETTh2
├── ETTm1
└── ETTm2
```

Karakteristik:

- ETTh1 dan ETTh2: hourly
- ETTm1 dan ETTm2: 15-minute

Kolom utama:

```text
date
HUFL
HULL
MUFL
MULL
LUFL
LULL
OT
```

Target awal:

```text
OT
```

Development dimulai dengan:

```text
ETTh1
```

Setelah pipeline stabil:

```text
ETTh1
    ↓
ETTh2
    ↓
ETTm1
    ↓
ETTm2
```

---

# 9. Data Split

Gunakan chronological split.

Contoh awal:

```text
|----------------------|----------|-------------|
        Train             Val          Test
```

Misalnya:

```text
70% Train
10% Validation
20% Test
```

Atau gunakan split standar yang digunakan oleh benchmark/literature ETT apabila diperlukan untuk perbandingan.

### Critical Rule

Tidak boleh ada future information masuk ke training atau preprocessing training.

Scaler:

```text
fit(train)
    ↓
transform(train)
transform(validation)
transform(test)
```

Bukan:

```text
fit(all data)
```

---

# 10. Preprocessing Pipeline

```text
Raw CSV
   ↓
Parse datetime
   ↓
Sort chronologically
   ↓
Check missing values
   ↓
Train/Validation/Test split
   ↓
Fit scaler on train
   ↓
Transform datasets
   ↓
Generate time features
   ↓
Generate observed mask
   ↓
Window construction
```

Preprocessing harus dibuat reproducible.

---

# 11. Window Construction

Parameter awal:

```text
context_length = 96
prediction_length = 24
```

Contoh:

```text
Past:

t-95 ................. t
│                       │
└──── 96 observations ─┘

Future:

t+1 ........ t+24
│             │
└── 24 steps ┘
```

Satu sample:

```text
X = [t-95, ..., t]
Y = [t+1, ..., t+24]
```

Parameter ini dapat diubah melalui configuration file.

---

# 12. Probabilistic Transformer

Backbone awal:

**Hugging Face Time Series Transformer for Probabilistic Forecasting**

Konsep:

```text
Past Observations
       ↓
Input Embedding
       ↓
Transformer Encoder
       ↓
Temporal Representation
       ↓
Transformer Decoder
       ↓
Hidden Representation
       ↓
Distribution Projection
       ↓
Predictive Distribution
```

Model harus menghasilkan predictive distribution, bukan hanya point forecast.

Secara konseptual:

```text
θ = distribution parameters
```

Contoh distribusi:

```text
Y ~ Normal(μ, σ)
```

Jika menggunakan distribusi lain, pemilihan harus ditentukan berdasarkan karakteristik target dan implementasi backbone.

---

# 13. Probabilistic Output

Model menghasilkan:

```text
p(y | x)
```

Dari predictive distribution dapat diperoleh:

### Point forecast

```text
ŷ = E[Y | X]
```

atau parameter lokasi/mean yang sesuai dengan distribusi.

### Prediction interval

Contoh confidence/coverage level:

```text
95%
```

### Negative Log-Likelihood

```text
NLL = -log p(y | x)
```

---

# 14. ADAPT-Z

ADAPT-Z menjadi komponen utama penelitian.

Secara konseptual:

```text
Transformer Representation
          ↓
          z
          ↓
      ADAPT-Z
          ↓
 Persistent Tracking
          ↓
 Adaptive Delta
          ↓
       z + Δz
          ↓
 Distribution Head
```

### Important Implementation Rule

ADAPT-Z tidak boleh diimplementasikan hanya sebagai:

```python
z_adapted = z + delta
```

Kemudian langsung disebut ADAPT-Z.

Implementasi harus mengikuti mekanisme ADAPT-Z asli, termasuk mekanisme persistent tracking/historical gradient information yang relevan dengan paper sumber.

File khusus:

```text
src/models/adapt_z.py
```

---

# 15. Online Forecasting Protocol

ETT merupakan historical benchmark dataset, bukan native real-time stream.

Karena itu proyek menggunakan **streaming simulation**.

Protokol:

```text
Initial training
       ↓
t1 arrives
       ↓
Forecast
       ↓
Actual becomes available
       ↓
Calculate probabilistic loss
       ↓
ADAPT-Z update
       ↓
t2 arrives
       ↓
Forecast
       ↓
Actual becomes available
       ↓
Update
       ↓
...
```

### Critical Rule

Actual future value hanya boleh digunakan setelah forecast pada timestamp tersebut telah dibuat.

Tidak boleh:

```text
future actual
     ↓
forecast
```

Yang benar:

```text
forecast
     ↓
actual observed
     ↓
update
```

---

# 16. Online State

ADAPT-Z harus memiliki persistent state.

Contoh konseptual:

```python
adapt_state = {
    "delta": ...,
    "historical_gradient": ...,
    "step": ...,
}
```

State harus bertahan antar time step:

```text
t1
 ↓
update state
 ↓
t2
 ↓
update state
 ↓
t3
```

State tidak boleh diinisialisasi ulang setiap sample.

---

# 17. Offline Training vs Online Adaptation

Pisahkan dua fase.

## Phase A — Offline Pretraining

```text
Training Set
     ↓
Probabilistic Transformer
     ↓
General Forecasting Representation
```

## Phase B — Online Adaptation

```text
Test Stream
     ↓
Forecast
     ↓
Actual observed
     ↓
Calculate loss
     ↓
ADAPT-Z update
     ↓
Next forecast
```

---

# 18. Model Freezing Strategy

Eksperimen pertama menggunakan:

```text
Transformer
    │
    ├── frozen
    │
    ▼
ADAPT-Z
    │
    └── trainable online
```

Tujuannya adalah mengisolasi kontribusi ADAPT-Z.

Setelah baseline berhasil, optional experiment:

```text
Transformer
    │
    ├── partially trainable
    ▼
ADAPT-Z
```

Jangan memulai dari konfigurasi kompleks.

---

# 19. Experiment Matrix

Minimal tiga model.

## Experiment A — Offline Baseline

```text
Probabilistic Transformer
```

## Experiment B — Online Update Baseline

```text
Probabilistic Transformer
+
Standard Online Gradient Update
```

## Experiment C — Proposed Model

```text
Probabilistic Transformer
+
ADAPT-Z
```

Tujuan:

```text
Baseline
   vs
Generic Online Adaptation
   vs
ADAPT-Z
```

Ini memungkinkan kontribusi ADAPT-Z dianalisis secara lebih bersih.

---

# 20. Ablation Study

Setelah model utama berhasil:

```text
A. No adaptation
B. Standard online update
C. ADAPT-Z
D. ADAPT-Z without persistent tracking
E. ADAPT-Z without historical gradient
F. Different adaptation strength
G. Different adaptation frequency
```

Tidak semua ablation wajib dilakukan. Prioritaskan yang secara langsung menguji komponen inti ADAPT-Z.

---

# 21. Metrics

## 21.1 Point Forecasting

```text
MAE
MSE
RMSE
```

## 21.2 Probabilistic Forecasting

```text
NLL
CRPS
```

## 21.3 Calibration

```text
PICP
MPIW
Coverage Deviation
```

Optional:

```text
Winkler Score
```

---

# 22. Main Evaluation Table

Format hasil utama:

| Model | MAE | MSE | RMSE | NLL | CRPS | PICP | MPIW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probabilistic Transformer | | | | | | | |
| Transformer + Online Update | | | | | | | |
| Transformer + ADAPT-Z | | | | | | | |

---

# 23. Distribution Shift Analysis

ADAPT-Z harus dievaluasi dalam konteks perubahan distribusi.

Konsep:

```text
Stable Regime
      ↓
Distribution Shift
      ↓
Model Error Increases
      ↓
Online Adaptation
      ↓
Recovery
```

Visualisasi:

```text
Forecast Error
     │
     │             /\
     │            /  \
     │       ____/    \____
     │______/
     └──────────────────────── time
                ↑
          distribution shift
```

---

# 24. Distribution Analysis

Untuk analisis awal dapat menggunakan:

```text
Rolling Mean
Rolling Standard Deviation
```

Analisis lanjutan:

```text
KS Test
Wasserstein Distance
PSI
```

Tujuannya bukan membuat shift detector sebagai komponen utama model, tetapi memberikan evidence bahwa kondisi data memang berubah dan menguji respons model terhadap perubahan tersebut.

---

# 25. Adaptation Analysis

Selain forecasting metrics, analisis:

```text
Pre-shift error
Post-shift error
Recovery time
Adaptation magnitude
```

Tambahkan visualisasi:

```text
||Δz||
```

terhadap waktu untuk melihat apakah mekanisme ADAPT-Z melakukan penyesuaian ketika kondisi data berubah.

---

# 26. Software Architecture

Repository yang disarankan:

```text
probabilistic-adapt-z/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── etth1.yaml
│   ├── etth2.yaml
│   ├── ettm1.yaml
│   └── ettm2.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── scaler.py
│   │   └── window.py
│   │
│   ├── models/
│   │   ├── probabilistic_transformer.py
│   │   ├── distribution.py
│   │   └── adapt_z.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── optimizer.py
│   │
│   ├── online/
│   │   ├── simulator.py
│   │   ├── updater.py
│   │   └── state.py
│   │
│   ├── evaluation/
│   │   ├── point_metrics.py
│   │   ├── probabilistic_metrics.py
│   │   ├── calibration.py
│   │   └── evaluator.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── checkpoint.py
│
├── experiments/
│   ├── baseline.py
│   ├── online_baseline.py
│   ├── adapt_z.py
│   └── ablation.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_adapt_z.ipynb
│   └── 04_evaluation.ipynb
│
├── checkpoints/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── plots/
│   └── logs/
│
└── scripts/
    ├── train.py
    ├── evaluate.py
    └── run_experiment.py
```

---

# 27. Configuration

Parameter jangan di-hard-code dalam source code.

Contoh `configs/etth1.yaml`:

```yaml
dataset:
  name: ETTh1
  target: OT

window:
  context_length: 96
  prediction_length: 24

model:
  d_model: 64
  encoder_layers: 2
  decoder_layers: 2
  attention_heads: 4

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.0001

adapt_z:
  enabled: true
  learning_rate: 0.001
  adaptation_frequency: 1

evaluation:
  confidence_level: 0.95

reproducibility:
  seed: 42
```

Run:

```bash
python scripts/run_experiment.py --config configs/etth1.yaml
```

---

# 28. Logging

Setiap experiment menyimpan:

```text
dataset
model configuration
random seed
epoch
training loss
validation loss
online step
prediction
actual
NLL
CRPS
MAE
MSE
adaptation state
```

Contoh:

```text
results/
└── adapt_z/
    └── ETTh1/
        └── run_001/
            ├── config.yaml
            ├── metrics.json
            ├── predictions.csv
            ├── training.log
            └── checkpoint.pt
```

---

# 29. Reproducibility

Tetapkan:

```python
SEED = 42
```

Kontrol:

```text
Python seed
NumPy seed
PyTorch seed
CUDA seed
```

Perbandingan experiment harus menggunakan:

```text
same dataset
same split
same preprocessing
same seed
same context length
same prediction horizon
```

Yang berbeda hanya komponen adaptation/model yang memang sedang dibandingkan.

---

# 30. Program Development Plan

Implementasi tidak dilakukan sekaligus.

## Program 01 — Dataset Download

```text
download_dataset.py
```

Tujuan:

```text
ETT → local data
```

---

## Program 02 — Data Exploration

```text
explore_data.py
```

Pemeriksaan:

```text
shape
missing values
timestamp
statistics
distribution
```

---

## Program 03 — Preprocessing

```text
preprocess.py
```

Fungsi:

```text
scaling
time features
mask
chronological split
```

---

## Program 04 — Dataset Window

```text
dataset.py
```

Membentuk:

```text
context → future
```

---

## Program 05 — Probabilistic Transformer

```text
probabilistic_transformer.py
```

Membangun baseline tanpa ADAPT-Z.

---

## Program 06 — Baseline Training

```text
train_baseline.py
```

Target:

- training berhasil,
- validation loss dapat dihitung,
- model menghasilkan predictive distribution.

---

## Program 07 — Baseline Evaluation

```text
evaluate_baseline.py
```

Menghasilkan:

```text
MAE
MSE
RMSE
NLL
CRPS
PICP
MPIW
```

---

## Program 08 — ADAPT-Z

```text
adapt_z.py
```

Implementasi mekanisme ADAPT-Z sesuai algoritma sumber.

---

## Program 09 — Integrated Model

```text
adapted_model.py
```

Integrasi:

```text
Probabilistic Transformer
+
ADAPT-Z
```

---

## Program 10 — Online Simulator

```text
online_simulator.py
```

Alur:

```text
observe
→ forecast
→ reveal actual
→ calculate loss
→ update
→ next
```

---

## Program 11 — Main Experiment

```text
run_adapt_z.py
```

Eksperimen utama.

---

## Program 12 — Ablation

```text
ablation.py
```

Eksperimen variasi ADAPT-Z dan baseline online.

---

# 31. Visualization Plan

Minimal:

## Plot 1 — Actual vs Forecast

```text
Actual
Forecast
Prediction Interval
```

## Plot 2 — Forecast Error

```text
Error vs Timestamp
```

## Plot 3 — NLL

```text
NLL vs Timestamp
```

## Plot 4 — CRPS

```text
CRPS vs Timestamp
```

## Plot 5 — Prediction Interval

```text
Upper Bound
Actual
Median Forecast
Lower Bound
```

## Plot 6 — Adaptation Magnitude

```text
||Δz|| vs Timestamp
```

## Plot 7 — Distribution Shift

```text
Rolling Mean
Rolling Std
Shift Statistic
```

---

# 32. Experimental Protocol

## Experiment 1 — Backbone Validation

```text
ETTh1
+
Probabilistic Transformer
+
Offline Training
```

Tujuan:

> Memastikan backbone bekerja dengan benar.

---

## Experiment 2 — Online Baseline

```text
ETTh1
+
Probabilistic Transformer
+
Standard Online Update
```

Tujuan:

> Menentukan baseline adaptation.

---

## Experiment 3 — Proposed Method

```text
ETTh1
+
Probabilistic Transformer
+
ADAPT-Z
```

Tujuan:

> Mengevaluasi metode utama.

---

## Experiment 4 — Cross Dataset

```text
ETTh2
ETTm1
ETTm2
```

Tujuan:

> Menguji generalisasi.

---

## Experiment 5 — Ablation

```text
No adaptation
Standard online update
ADAPT-Z
ADAPT-Z variants
```

Tujuan:

> Mengidentifikasi kontribusi setiap mekanisme.

---

# 33. Statistical Analysis

Jika memungkinkan, hasil akhir tidak hanya membandingkan satu angka rata-rata.

Analisis:

```text
Mean
Median
Standard deviation
Per-window performance
Per-regime performance
```

Untuk perbandingan model, pertimbangkan statistical test yang sesuai dengan unit observasi dan dependency pada time-series.

Jangan menganggap setiap timestamp sebagai independent sample tanpa mempertimbangkan autocorrelation.

---

# 34. Research Workflow

```text
START
  │
  ▼
Literature Review
  │
  ▼
Define Research Gap
  │
  ▼
Dataset Preparation
  │
  ▼
Baseline Model
  │
  ▼
Baseline Evaluation
  │
  ▼
Implement ADAPT-Z
  │
  ▼
Integration Testing
  │
  ▼
Online Simulation
  │
  ▼
Main Experiment
  │
  ▼
Ablation Study
  │
  ▼
Cross-Dataset Test
  │
  ▼
Statistical Analysis
  │
  ▼
Results & Discussion
  │
  ▼
END
```

---

# 35. Milestones / Definition of Done

## Milestone 1 — Data

```text
[ ] ETT berhasil dibaca
[ ] Timestamp benar
[ ] Missing values diperiksa
[ ] Chronological split
[ ] Scaler benar
[ ] Window benar
```

## Milestone 2 — Backbone

```text
[ ] Transformer berhasil training
[ ] Loss dapat dihitung
[ ] Forecast masuk akal
[ ] Predictive distribution tersedia
[ ] Sampling berhasil
```

## Milestone 3 — Probabilistic Evaluation

```text
[ ] MAE
[ ] MSE
[ ] RMSE
[ ] NLL
[ ] CRPS
[ ] PICP
[ ] MPIW
```

## Milestone 4 — ADAPT-Z

```text
[ ] ADAPT-Z standalone
[ ] Persistent state
[ ] Update mechanism
[ ] Gradient tracking
[ ] Feature adaptation
[ ] Integration dengan Transformer
```

## Milestone 5 — Online Forecasting

```text
[ ] Sequential simulation
[ ] No future leakage
[ ] Actual hanya digunakan setelah forecast
[ ] ADAPT-Z state persist
```

## Milestone 6 — Research Experiments

```text
[ ] Offline baseline
[ ] Online baseline
[ ] ADAPT-Z
[ ] Ablation
[ ] ETTh1
[ ] ETTh2
[ ] ETTm1
[ ] ETTm2
[ ] Statistical analysis
```

---

# 36. Final Implementation Order

Gunakan urutan berikut sebagai roadmap utama:

```text
PHASE 0
Research Specification
        ↓
PHASE 1
ETT Data Pipeline
        ↓
PHASE 2
Hugging Face Probabilistic Transformer
        ↓
PHASE 3
Offline Probabilistic Forecasting
        ↓
PHASE 4
Evaluation Framework
        ↓
PHASE 5
Online Forecasting Simulator
        ↓
PHASE 6
Reproduce / Implement ADAPT-Z
        ↓
PHASE 7
Integrate ADAPT-Z
        ↓
PHASE 8
Online ADAPT-Z Adaptation
        ↓
PHASE 9
Baseline Comparison
        ↓
PHASE 10
Ablation Study
        ↓
PHASE 11
ETTh2 / ETTm1 / ETTm2
        ↓
PHASE 12
Statistical + Visualization Analysis
        ↓
PHASE 13
Research Results
```

---

# 37. Critical Research Principles

## Principle 1 — Jangan langsung membuat model kompleks

Urutan:

```text
Probabilistic Transformer
        ↓
Validate
        ↓
Online baseline
        ↓
Validate
        ↓
ADAPT-Z
        ↓
Validate
```

## Principle 2 — Pisahkan offline training dan online adaptation

```text
Offline:
Learn general representation

Online:
Adapt to incoming distribution
```

## Principle 3 — Jangan menggunakan future information

Forecast harus dibuat sebelum actual value digunakan untuk update.

## Principle 4 — ADAPT-Z harus diimplementasikan sesuai algoritma aslinya

Jangan mengganti ADAPT-Z dengan adapter sederhana lalu menyebutnya ADAPT-Z.

## Principle 5 — Jangan menambahkan Continual Learning tanpa kebutuhan eksperimen

CL hanya ditambahkan apabila ada research question yang jelas, misalnya retention terhadap recurring regimes atau catastrophic forgetting.

## Principle 6 — Probabilistic forecasting harus dievaluasi secara probabilistik

MSE/MAE saja tidak cukup.

Minimal:

```text
Point:
MAE / MSE / RMSE

Probabilistic:
NLL / CRPS

Calibration:
PICP / MPIW
```

---

# 38. Core Experimental Logic

Eksperimen utama harus menjawab:

```text
                    Distribution Shift
                           │
                           ▼
                  Forecast degradation
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       No adaptation               ADAPT-Z
             │                           │
             ▼                           ▼
        Error remains             Online adjustment
                                         │
                                         ▼
                                  New representation
                                         │
                                         ▼
                                  New distribution
                                         │
                                         ▼
                                    Forecast
```

Pertanyaan akhirnya bukan hanya:

> "Apakah ADAPT-Z menghasilkan MAE lebih kecil?"

tetapi:

> "Apakah feature adaptation berbasis ADAPT-Z dapat membantu probabilistic Transformer menyesuaikan predictive distribution terhadap perubahan distribusi data secara online, tanpa menggunakan informasi masa depan?"

---

# 39. Final Blueprint

```text
                    ┌─────────────────────────┐
                    │       ETT DATASET       │
                    │ ETTh1 / ETTh2 / ETTm1/2│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   DATA PREPROCESSING    │
                    │ Split / Scale / Window  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ PROBABILISTIC           │
                    │ TIME-SERIES TRANSFORMER │
                    │                         │
                    │ Encoder → Decoder       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                            Hidden z
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        ADAPT-Z          │
                    │                         │
                    │ Persistent Tracking     │
                    │ Historical Gradient     │
                    │ Feature Adaptation      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                           Adapted z
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   DISTRIBUTION HEAD     │
                    │                         │
                    │ Predictive Distribution │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                 ▼
          Point Forecast        NLL              CRPS
               │                                   │
               ▼                                   ▼
        MAE/MSE/RMSE                         Calibration
                                            PICP / MPIW
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ ONLINE STREAM SIMULATOR │
                    │                         │
                    │ Forecast → Actual →     │
                    │ Loss → ADAPT-Z Update   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   ABLATION + ANALYSIS   │
                    │                         │
                    │ Baseline                │
                    │ Online Update           │
                    │ ADAPT-Z                 │
                    │ Distribution Shift      │
                    │ Cross Dataset           │
                    └─────────────────────────┘
```

---

# 40. Immediate Next Step

Implementasi harus dimulai dari:

```text
PHASE 1
    ↓
ETTh1
    ↓
Data loader
    ↓
Chronological split
    ↓
Scaling
    ↓
Sliding window
    ↓
PyTorch Dataset/DataLoader
```

Setelah itu:

```text
PHASE 2
    ↓
Hugging Face TimeSeriesTransformerForPrediction
    ↓
Past values
    ↓
Time features
    ↓
Future values
    ↓
Predictive distribution
    ↓
NLL
```

**Jangan masuk ke ADAPT-Z sebelum dua tahap tersebut benar-benar berjalan.** Dengan cara ini, ketika nanti ada error setelah integrasi ADAPT-Z, kita dapat memastikan error berasal dari adaptation layer, bukan dari dataset atau probabilistic Transformer.
