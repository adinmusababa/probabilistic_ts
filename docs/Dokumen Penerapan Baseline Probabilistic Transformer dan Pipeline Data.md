# Dokumen Penerapan: Baseline Probabilistic Transformer dan Pipeline Data

Versi 1.0 · 23 Oktober 2026 · Mencakup Fase 0 sampai Fase 2 pada PRD (baseline, pipeline data, dan training)

## 1. Ringkasan

Dokumen ini mencatat apa yang sudah dibangun dan dijalankan untuk baseline probabilistic Transformer beserta pipeline datanya. Baseline memakai Time Series Transformer dari Hugging Face, data diambil dari Supabase lewat satu fungsi sumber, dan training berjalan end-to-end dengan hasil yang tersimpan di folder lokal. Seluruh uji otomatis lulus dan satu run training pada ETTh1 dengan horizon 24 sudah selesai.

Dokumen ini juga mencatat bagian PRD yang belum dikerjakan, supaya jelas apa yang menjadi pekerjaan berikutnya. Optimasi dan perbaikan model sengaja ditunda sampai dokumen ini selesai.

## 2. Status terhadap PRD

- **Fase 0 (Fondasi): sebagian besar selesai.** Struktur repositori, config YAML, seed global, dan manifest run sudah ada. Logging masih berupa `print`.
- **Fase 1 (Data): selesai.** Pemuatan dari Supabase, split, scaler, fitur waktu, dan windowing sudah berjalan dan lulus uji. Iterator streaming untuk evaluasi online (FR-D6) belum dibuat karena baru dibutuhkan di Fase 4.
- **Fase 2 (Baseline): training selesai, kriteria penerimaan belum lengkap.** Model dan training berjalan dan hasilnya dapat direproduksi. Dua kriteria Fase 2 belum terpenuhi: uji kesetaraan encode/decode (FR-M5) dan pembuktian bahwa model mengungguli baseline naif.
- **Fase 3 sampai Fase 7: belum dikerjakan.**

## 3. Struktur Kode

```
probabilistic_ts/
├── configs/
│   ├── model/probabilistic_transformer.yaml
│   └── training/default.yaml
├── src/
│   ├── data/
│   │   ├── data_supabase.py      # fungsi load_dataset (sudah ada sebelumnya)
│   │   ├── loader.py
│   │   ├── splits.py
│   │   ├── scaler.py
│   │   ├── time_features.py
│   │   ├── windows.py
│   │   └── pipeline.py
│   ├── models/probabilistic_transformer.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── run.py
│   └── utils/seed.py
├── tests/
│   ├── test_model_smoke.py
│   ├── test_data.py
│   ├── test_data_model_link.py
│   └── test_training.py
└── exps/                          # keluaran tiap run
```

Pembagian tanggung jawab:

- `loader.py`: satu-satunya pintu akses database. Memanggil `load_dataset` lalu mengubah hasilnya ke polars (`pl.from_pandas`).
- `splits.py`: batas split dan penentuan origin window.
- `scaler.py`: standardisasi per kanal dengan numpy.
- `time_features.py`: empat fitur waktu dari kolom `date`.
- `windows.py`: `WindowDataset`, yang menghasilkan batch sesuai format model.
- `pipeline.py`: menyatukan seluruh langkah data dan menghasilkan metadata untuk manifest.
- `probabilistic_transformer.py`: pembungkus model Hugging Face.
- `trainer.py` dan `run.py`: loop training dan pintu masuk command line.

## 4. Lapisan Data

### 4.1 Sumber data

Data ETTh1, ETTh2, ETTm1, dan ETTm2 dimuat dari Supabase lewat `load_dataset(dataset)` di `src/data/data_supabase.py`. Kolomnya adalah `date`, `hufl`, `hull`, `mufl`, `mull`, `lufl`, `lull`, dan `ot`. Tidak ada salinan CSV sumber di lokal; hanya hasil run yang disimpan lokal. Modul lain tidak membaca database langsung, hanya lewat `loader.py`.

### 4.2 Pembagian data

Mengikuti konvensi Autoformer, dipakai 20 bulan pertama dengan rasio 12/4/4 bulan:

- ETTh: 8640 baris train, 2880 val, 2880 test (total 14400).
- ETTm: empat kali lipat jumlah baris ETTh.

Jika data yang dimuat kurang dari jumlah itu, pipeline berhenti dengan galat yang jelas.

### 4.3 Normalisasi

Standardisasi per kanal dengan `StandardScaler` buatan sendiri (numpy), di-fit hanya pada split train. Scaling internal Hugging Face dimatikan (`scaling: false`) agar tidak terjadi normalisasi ganda. Scaler dapat disimpan dan dimuat ulang (`save`/`load`) dan mendukung `inverse_transform`.

### 4.4 Fitur waktu

Empat fitur dari kolom `date` dengan polars, diskalakan ke rentang -0,5 sampai 0,5: jam, hari dalam minggu, hari dalam bulan, dan bulan. Jumlahnya sesuai `num_time_features: 4` pada config model.

### 4.5 Windowing

- Panjang riwayat per sampel: `past_length = context_length + max(lags_sequence)`. Dengan config baku, 96 + 168 = 264 langkah.
- Satu sampel adalah satu origin t: riwayat t-past\_length sampai t-1 dan masa depan t sampai t+H-1.
- Target prediksi tidak boleh melewati batas segmennya. Riwayat val dan test boleh menjangkau segmen sebelumnya, karena itu masa lalu dan bukan kebocoran informasi.
- Window train pertama muncul di indeks `past_length`.
- Val dan test selalu memakai stride 1; train dapat memakai `train_stride` lebih besar untuk percobaan cepat.
- Kunci batch: `past_values`, `past_time_features`, `past_observed_mask`, `future_values`, `future_time_features`, dan `future_observed_mask`.

### 4.6 Metadata data untuk audit

`pipeline.py` menghasilkan metadata yang masuk ke manifest run: nama dataset, jumlah baris, rentang tanggal, nama kolom, batas segmen, panjang riwayat, horizon, dan hash SHA-256 isi data. Hash ini membuat run tetap dapat diaudit walau isi database kelak berubah.

## 5. Model

Model adalah pembungkus tipis `ProbabilisticTransformer` di atas `TimeSeriesTransformerForPrediction` dari Hugging Face, dengan konfigurasi dibaca dari YAML.

Konfigurasi baku (titik awal, belum dituning):

- context\_length 96, prediction\_length 24.
- input\_size 7 (multivariat, seluruh kanal ETT sekaligus). Nilai 1 memberi mode channel-independent dan sudah tercakup di uji.
- lags\_sequence: 1, 2, 3, 4, 5, 6, 7, 24, 48, 168 (untuk data per jam; ETTm perlu lag yang disesuaikan).
- num\_time\_features 4, scaling false, distribusi keluaran Student-t.
- d\_model 64, 2 layer encoder dan 2 layer decoder, 2 head per atensi, dimensi FFN 128, dropout 0,1, num\_parallel\_samples 100.
- Fitur statis dan dinamis tambahan tidak dipakai.

Antarmuka yang tersedia:

- `training_loss(batch)`: NLL rata-rata pada horizon prediksi.
- `sample(batch, n_samples)`: sampel prediksi berbentuk (n\_samples, B, H, D). Model harus dalam mode eval.
- Properti `past_length`, `prediction_length`, dan `input_size`.

Antarmuka `encode` dan `decode` (FR-M3) belum dibuat. Bagian ini sengaja ditunda karena memakai bagian dalam Hugging Face yang bergantung pada versi `transformers`, dan akan ditulis bersama uji kesetaraan numerik (FR-M5) sebelum ADAPT-Z dipasang.

## 6. Training

Parameter baku (`configs/training/default.yaml`):

- Optimizer AdamW, learning rate 6e-4, weight decay 0,01.
- Batch 64, maksimum 30 epoch, gradient clipping 1,0.
- Early stopping dengan patience 5 pada NLL validasi.
- Seed 42, `train_stride` 1, `num_workers` 0, device otomatis (CUDA bila tersedia).

Alur kerja:

- Seed diatur sebelum model dibuat, dan generator DataLoader diberi seed yang sama.
- Setiap epoch menghitung NLL train dan NLL validasi (dibobot jumlah sampel).
- Checkpoint terbaik (`best.pt`) disimpan setiap NLL validasi membaik. Isinya state model, config model, epoch, dan NLL validasi.
- Loss yang tidak finite menghentikan training dengan galat eksplisit.

Keluaran tiap run di `exps/<dataset>_H<horizon>_seed<seed>/`:

- `best.pt`: checkpoint terbaik.
- `history.csv`: NLL train dan validasi per epoch.
- `manifest.json`: config training dan model, metadata data, NLL validasi terbaik, jumlah epoch, durasi, device, versi paket, dan hash git.

Perintah:

```
python -m pytest tests -q
python -m src.training.run --dataset ETTh1 --pred-len 24
```

Checkpoint dapat dimuat ulang dengan `load_checkpoint(path)` di `trainer.py`.

## 7. Pengujian

Seluruh uji memakai data sintetis sehingga tidak membutuhkan koneksi database. Hasil di bawah adalah hasil yang dijalankan di lingkungan pengembangan.

- `test_model_smoke.py` (4 lulus): loss finite dan gradien mengalir, serta bentuk sampel benar, untuk input\_size 7 dan 1.
- `test_data.py` (5 lulus): batas split, scaler hanya di-fit pada train, keselarasan window, target tidak melewati segmen, rentang fitur waktu, dan galat saat data kurang.
- `test_data_model_link.py` (1 lulus): satu batch dari pipeline data dapat masuk ke model dan menghasilkan loss finite.
- `test_training.py` (2 lulus): artefak run terbentuk, checkpoint yang dimuat ulang memberi NLL validasi yang sama dengan yang tercatat, dan dua run dengan seed sama menghasilkan riwayat yang sama. Uji terakhir ini mengandalkan determinisme di CPU.

## 8. Hasil Run Pertama

Konfigurasi: ETTh1, horizon 24, seed 42, parameter baku. Hasil per epoch (NLL train, NLL validasi):

- Epoch 1: 5,1079 dan 3,6183
- Epoch 2: 1,8414 dan 2,7820
- Epoch 3: 1,1886 dan 2,3599
- Epoch 4: 0,8763 dan 2,3982
- Epoch 5: 0,6882 dan 2,2164 (terbaik)
- Epoch 6: 0,4967 dan 2,3894
- Epoch 7: 0,2986 dan 2,3614
- Epoch 8: 0,2235 dan 2,2269
- Epoch 9: 0,1671 dan 2,4416
- Epoch 10: 0,0568 dan 2,3436

Early stopping aktif di epoch 10 dengan checkpoint terbaik dari epoch 5 (NLL validasi 2,2164). Hasil tersimpan di `exps/ETTh1_H24_seed42`.

Pengamatan:

- NLL train turun terus sampai 0,06 sedangkan NLL validasi mendatar di sekitar 2,2 sejak epoch 3. Celah ini menunjukkan model mulai menghafal data train.
- Dugaan penyebab, belum diverifikasi: window train saling tumpang tindih sehingga sampel independen jauh lebih sedikit; NLL memberi imbalan untuk distribusi yang terlalu sempit; dan pergeseran distribusi antara periode train dan validasi.
- Skala angka NLL belum pasti. Untuk input multivariat, log-likelihood kemungkinan dijumlahkan antar kanal, sehingga angkanya mendekati NLL untuk seluruh 7 kanal sekaligus. Ini perlu dicek pada kode Hugging Face yang terpasang sebelum angka dibandingkan antar pengaturan.
- Angka NLL ini belum punya pembanding, sehingga belum dapat dinilai baik atau buruk.

## 9. Keterbatasan dan Penyimpangan dari PRD

- Encode/decode (FR-M3, FR-M5) belum diimplementasikan.
- Metrik evaluasi (MSE, MAE, CRPS, coverage, PIT) dan baseline seasonal naive belum ada, sehingga kriteria Fase 2 "mengungguli baseline naif" belum terbukti.
- Protokol online, ledger label, dan strategi adaptasi belum dibuat.
- Baru satu horizon (24), satu dataset (ETTh1), dan satu seed yang dijalankan. ETTm membutuhkan penyesuaian lag.
- Hiperparameter belum dituning.
- Logging hanya `print` ke layar; belum ada berkas log.
- Uji reprodusibilitas bergantung pada determinisme CPU; pada GPU hasil dapat berbeda tipis.

## 10. Langkah Berikutnya

1. Metrik dan evaluasi statis di data test, termasuk baseline seasonal naive dan NLL per kanal. Hipotesis awal yang diuji: model overconfident, yaitu coverage nyata lebih rendah dari level nominal.
2. Perbaikan baseline secukupnya (regularisasi, ukuran model, stride), dengan seleksi model yang lebih stabil dari NLL validasi saja, misalnya CRPS validasi.
3. Protokol online dengan NoAdapt dan OnlineUpdate.
4. Antarmuka encode/decode dengan uji kesetaraan numerik.
5. Integrasi ADAPT-Z dan eksperimen perbandingan.

## 11. Lampiran: Ketergantungan

- Model dan training: `torch`, `transformers`.
- Data: `polars`, `pyarrow` (untuk `pl.from_pandas`), `numpy`.
- Akses database (di `data_supabase.py`): `sqlalchemy`, `pandas`, `psycopg2`, `python-dotenv`.
- Konfigurasi dan uji: `pyyaml`, `pytest`.
- Tidak memakai scikit-learn.
