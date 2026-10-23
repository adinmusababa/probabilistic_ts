# PRD: Baseline Probabilistic Transformer dan Integrasi ADAPT-Z

Draf awal v0.1 · 23 Oktober 2026 · Panduan implementasi bertahap (belum untuk penulisan resmi)

## 1. Ringkasan

Proyek ini membangun, secara bertahap, sistem forecasting deret waktu probabilistik berbasis Transformer yang kemudian mampu mengadaptasi representasi internalnya secara online memakai ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space) ketika distribusi data berubah. Dokumen ini menjadi panduan untuk tiga hal: membangun baseline, membangun pipeline evaluasi, dan menjalankan eksperimen lanjutan.

Prinsip utama: baseline dan pipeline dibangun lebih dulu dan harus dapat berdiri sendiri. ADAPT-Z dipasang sesudahnya melalui antarmuka yang sudah disiapkan sejak awal, tanpa mengubah kode baseline.

## 2. Tujuan dan Non-Tujuan

### 2.1 Tujuan

- T1. Membangun baseline probabilistic Transformer.
- T2. Membangun pipeline forecasting probabilistik yang menghitung metrik titik dan metrik probabilistik.
- T3. Mengimplementasikan mekanisme ADAPT-Z sebagai modul mandiri.
- T4. Mengintegrasikan ADAPT-Z dengan model probabilistik.
- T5. Membandingkan tiga kondisi: tanpa adaptasi, online update biasa, dan ADAPT-Z.
- T6. Menganalisis pengaruh adaptasi terhadap akurasi prediksi, distribusi prediktif, dan kalibrasi.
- T7. Melakukan ablation study.
- T8. Menguji generalisasi pada beberapa dataset ETT.

### 2.2 Non-Tujuan (versi ini)

- Menulis makalah atau laporan resmi.
- Mengubah arsitektur dasar Transformer di luar yang dibutuhkan untuk antarmuka encoder, decoder, dan head.
- Baseline rekalibrasi conformal (kandidat pekerjaan lanjutan).
- Dataset di luar keluarga ETT (opsional setelah T8).
- Deployment atau serving model.

## 3. Prinsip Desain

- **Modular.** Data, model, metrik, protokol online, dan strategi adaptasi berada di modul terpisah dengan antarmuka eksplisit.
- **Dapat diaudit.** Setiap hasil dapat ditelusuri ke konfigurasi, seed, dan versi kode.
- **Tanpa kebocoran informasi.** Akses label dikendalikan satu komponen (ledger) dan diuji otomatis.
- **Ketergantungan minimal.** Pemrosesan data memakai polars dan perhitungan memakai numpy. Scaler, pembagi data, dan metrik ditulis sendiri tanpa library jadi seperti scikit-learn. PyTorch dan Hugging Face Transformers dipakai hanya untuk model. Pengecualian: lapisan akses Supabase (data\_supabase.py) memakai SQLAlchemy dan pandas, dan hasilnya segera dikonversi ke polars atau numpy. Hanya hasil run (checkpoint, prediksi, metrik, manifest) yang disimpan di lokal.
- **Berbasis konfigurasi.** Hiperparameter berada di file YAML, bukan di dalam kode.

## 4. Persyaratan Fungsional

### 4.1 Data (`src/data`)

- FR-D1. Seluruh data sumber dimuat hanya lewat fungsi load\_dataset(dataset) di src/data/data\_supabase.py (Supabase/PostgreSQL lewat SQLAlchemy), tanpa menyimpan salinan CSV sumber di lokal. Dataset yang didukung: ETTh1 (tabel ett\_h1), ETTh2 (ett\_h2), ETTm1 (ett\_m1), dan ETTm2 (ett\_m2), dengan kolom date, hufl, hull, mufl, mull, lufl, lull, dan ot, diurutkan menurut date.
- FR-D2. Pembagian kronologis train/val/test. Default mengikuti konvensi Autoformer (ETTh: 12/4/4 bulan; ETTm: jumlah titik 4 kali lipat). Indeks split disimpan sebagai artefak.
- FR-D3. Scaler standardisasi ditulis sendiri dengan numpy, di-fit hanya pada train, mendukung inverse transform. Model Hugging Face memiliki scaling internal per window, jadi satu strategi normalisasi harus dipilih dan didokumentasikan agar tidak terjadi normalisasi ganda.
- FR-D4. Fitur waktu (jam, hari dalam minggu, hari dalam bulan, bulan) dibuat dengan polars.
- FR-D5. Pembangun window dengan parameter context\_length, prediction\_length, dan stride. Output batch berisi past\_values, past\_time\_features, past\_observed\_mask, future\_time\_features, dan future\_values.
- FR-D6. Iterator streaming untuk evaluasi online yang pada waktu t hanya menyajikan data sampai t.

* FR-D7. Modul lain tidak boleh mengakses database langsung. Hasil load\_dataset (DataFrame pandas dari pd.read\_sql) segera dikonversi ke polars atau numpy di lapisan data, misalnya lewat pl.from\_pandas. Alternatifnya, pembacaan di data\_supabase.py diganti dengan pl.read\_database memakai engine SQLAlchemy yang sama.
* FR-D8. Kredensial dibaca dari environment variable (SUPABASE\_DB\_HOST, SUPABASE\_DB\_PORT, SUPABASE\_DB\_NAME, SUPABASE\_DB\_USER, SUPABASE\_DB\_PASSWORD) lewat berkas .env yang masuk .gitignore. Kredensial tidak boleh muncul di log, config, atau manifest.
* FR-D9. Karena tidak ada salinan lokal, setiap run memuat data satu kali per dataset dan menyimpannya di memori (bukan query per langkah online). Manifest mencatat nama tabel, jumlah baris, rentang date, dan hash isi data (misalnya SHA-256 dari array numpy) agar run tetap dapat diaudit bila isi database berubah.
* FR-D10. Uji unit memakai pemuat tiruan (data sintetis dengan kolom yang sama) sehingga pengujian tidak bergantung pada koneksi database.

### 4.2 Model (`src/models/probabilistic_transformer.py`)

- FR-M1. Dibangun di atas Time Series Transformer dari Hugging Face dan dideklarasikan di file ini. Training tidak berada di file model; modul training hanya memanggil class atau fungsinya.
- FR-M2. Konfigurasi dari YAML: context\_length, prediction\_length, lags\_sequence, d\_model, jumlah layer encoder dan decoder, jumlah head, dropout, distribution\_output (default Student-t, alternatif Normal), dan num\_parallel\_samples.
- FR-M3. Kontrak antarmuka `ForecastModel`:
  - `encode(batch) -> z`: representasi dari encoder.
  - `decode(z, batch) -> distribusi prediktif` (loc, scale, dan df bila Student-t).
  - `nll(distribusi, y, mask) -> loss per elemen`.
  - `sample(z, batch, n_samples) -> sampel` dengan bentuk (n\_samples, B, H, D).
  - Parameter dikelompokkan menjadi encoder, decoder, dan head agar strategi online update dapat memilih kelompok yang diperbarui.
- FR-M4. Antarmuka harus memungkinkan z yang sudah dimodifikasi disuntikkan ke tahap decoding tanpa mengubah bobot. Kemampuan ini diverifikasi di awal terhadap kode Hugging Face. Jika tidak didukung langsung, disiapkan pembungkus atau loop decoding sendiri.
- FR-M5. Uji kesetaraan numerik: encode lalu decode tanpa modifikasi z harus menghasilkan output identik dengan forward asli.

### 4.3 Training (`src/training`)

- FR-T1. Training offline pada split train, validasi pada split val, early stopping, dan penyimpanan checkpoint terbaik.
- FR-T2. Seed global dan pengaturan determinisme semampunya.
- FR-T3. Setiap run menyimpan konfigurasi terselesaikan, versi paket, hash git, dan kurva loss.
- FR-T4. Satu checkpoint per kombinasi (dataset, prediction\_length). Apakah satu model multi-horizon cukup masih menjadi pertanyaan terbuka.

### 4.4 Metrik (`src/metrics`, numpy murni)

Metrik titik:

- MSE dan MAE, dihitung dari median sampel dan dari mean sampel.

Metrik probabilistik:

- NLL dari parameter distribusi.
- CRPS dengan estimator berbasis sampel: E|X - y| dikurangi setengah E|X - X'|.
- Interval score atau weighted interval score.
- Coverage empiris pada level 50%, 80%, dan 95%.
- Lebar rerata interval (sharpness).
- PIT (histogram dan uji keseragaman sederhana).

Agregasi dan persyaratan:

- FR-E1. Metrik dapat diagregasi per langkah horizon, per variabel, dan global; dilaporkan pada ruang ternormalisasi dan skala asli dengan label yang jelas.
- FR-E2. Semua fungsi tervektorisasi, menerima array numpy, dan menangani mask.
- FR-E3. Diuji terhadap nilai analitik, misalnya CRPS distribusi Normal dengan bentuk tertutup.
- FR-E4. Baseline naif (misalnya seasonal naive, ditulis dengan numpy) tersedia sebagai pembanding sanity check.

### 4.5 Protokol Online (`src/online`)

- FR-O1. Ledger label mencatat label titik waktu mana yang sudah terlihat pada waktu t. Seluruh akses label melalui ledger.
- FR-O2. Alur per waktu t: (a) hasilkan prediksi untuk t+1 sampai t+H dari data sampai t; (b) simpan prediksi; (c) saat label t+h tiba, label yang sudah matang menjadi tersedia bagi learner. Pembaruan hanya boleh memakai pasangan prediksi-label yang sudah tersedia pada t, dan evaluasi memakai prediksi yang dibuat sebelum labelnya terlihat.
- FR-O3. Antarmuka learner: `predict(t) -> distribusi atau sampel` dan `observe(t, label)`.
- FR-O4. Uji kebocoran otomatis: mengubah label masa depan tidak boleh mengubah prediksi pada waktu t.
- FR-O5. Frekuensi adaptasi dapat diatur: setiap langkah atau setiap k langkah.

### 4.6 Strategi Adaptasi (`src/adapt`)

- FR-A1. **NoAdapt:** model beku.
- FR-A2. **OnlineUpdate:** pembaruan gradien online (OGD atau Adam) pada kelompok parameter terpilih: head saja, layer terakhir, atau opsional seluruh model. Learning rate dan buffer diatur lewat config.
- FR-A3. **ADAPT-Z:** modul mandiri sesuai Algorithm 1 pada makalah (arXiv 2509.03810). Adapter A (MLP) menerima z dan gradien historis yang dilacak, menghasilkan koreksi delta, lalu z' = z + delta, dan head g membentuk distribusi dari z'. Dikembangkan sebagai class terpisah yang menerima encoder f dan head g lewat antarmuka FR-M3, dan diuji dulu pada model mainan sebelum dipasang ke Transformer.
- FR-A4. Loss untuk gradien adalah NLL pada langkah yang labelnya sudah tersedia, sehingga label parsial atau tertunda tertangani.
- FR-A5. Opsi ADAPT-Z yang dapat diablasi: letak z, input adapter, panjang riwayat gradien, kapasitas adapter, parameter distribusi yang dikoreksi, inisialisasi adapter, dan learning rate.
- FR-A6. Ketiga strategi memakai antarmuka FR-O3 sehingga dapat ditukar lewat config tanpa mengubah kode lain.

### 4.7 Eksperimen dan Pelaporan (`src/experiments`)

- FR-X1. Runner: satu config sama dengan satu run. Menyimpan prediksi (npy atau parquet), metrik per langkah (parquet lewat polars), dan manifest run.
- FR-X2. Agregasi lintas seed dan tabel ringkasan dengan polars.
- FR-X3. Plot: reliability diagram, histogram PIT, deret waktu dengan band interval, dan besar delta terhadap waktu.

## 5. Struktur Repositori (Usulan)

```
project/
├── configs/
│   ├── data/            # ETTh1.yaml, ETTh2.yaml, ETTm1.yaml, ETTm2.yaml
│   ├── model/           # probabilistic_transformer.yaml
│   ├── training/
│   ├── online/          # protokol dan strategi: noadapt, online_update, adapt_z
│   └── experiments/
├── .env                 # kredensial Supabase (masuk .gitignore)
├── src/
│   ├── data/            # data_supabase.py (sumber), split, scaler, windowing, streaming
│   ├── models/
│   │   └── probabilistic_transformer.py
│   ├── training/
│   ├── metrics/         # point.py, probabilistic.py, calibration.py
│   ├── online/          # ledger.py, loop.py, protocol.py
│   ├── adapt/           # noadapt.py, online_update.py, adapt_z.py
│   ├── experiments/
│   └── utils/           # seed, io, logging
├── tests/
├── exps/                # keluaran tiap run
└── README.md
```

## 6. Persyaratan Non-Fungsional

- **Reprodusibilitas.** Seed, config terselesaikan, dan hash git tersimpan di manifest setiap run.
- **Auditabilitas.** Uji unit untuk scaler, windowing, metrik, ledger, dan kesetaraan model. Asersi anti-kebocoran aktif selama evaluasi online.
- **Kinerja.** Latensi per langkah dan pemakaian memori dicatat. Sampling autoregresif mahal, sehingga jumlah sampel dapat diatur.
- **Keterbacaan.** Type hints dan docstring yang menyebut bentuk tensor.
- **Portabilitas.** Berjalan di CPU dan satu GPU (Colab opsional).

## 7. Fase Pengembangan dan Kriteria Penerimaan

### Fase 0: Fondasi

Struktur repositori, pemuat config, seed, dan logging.

- Penerimaan: satu perintah menjalankan run kosong dan menulis manifest.

### Fase 1: Data

Pemuatan data lewat load\_dataset (Supabase) dengan konversi ke polars, split, scaler numpy, fitur waktu, windowing, dan iterator streaming.

- Penerimaan: tidak ada tumpang tindih antar split, scaler hanya di-fit pada train, uji windowing lulus, dan tidak ada salinan data sumber yang ditulis ke lokal.

### Fase 2: Baseline Probabilistic Transformer (T1)

Model, antarmuka FR-M3, dan training offline.

- Penerimaan: loss validasi menurun, hasil dengan seed sama dapat direproduksi dalam toleransi, uji FR-M5 lulus, dan model mengungguli baseline naif pada MAE.

### Fase 3: Pipeline Evaluasi Statis (T2)

Semua metrik titik dan probabilistik beserta agregasinya.

- Penerimaan: uji nilai analitik lulus dan coverage pada validasi mendekati level nominal dalam batas wajar.

### Fase 4: Protokol Online dan Baseline Adaptasi (T5 sebagian)

Ledger, loop online, NoAdapt, dan OnlineUpdate.

- Penerimaan: uji kebocoran lulus dan hasil NoAdapt pada loop online sama dengan evaluasi rolling statis.

### Fase 5: ADAPT-Z Mandiri (T3)

Modul adapter diuji pada model mainan dengan pergeseran distribusi buatan.

- Penerimaan: adapter menurunkan error setelah pergeseran dibanding tanpa adaptasi.

### Fase 6: Integrasi (T4)

ADAPT-Z dipasang pada Transformer probabilistik lewat antarmuka FR-M3.

- Penerimaan: dengan delta bernilai nol hasilnya identik dengan NoAdapt, tidak ada NaN sepanjang stream, dan latensi per langkah dilaporkan.

### Fase 7: Eksperimen (T5 sampai T8)

Perbandingan utama, analisis kalibrasi, ablasi, dan dataset ETT lain.

- Penerimaan: tabel dan plot dihasilkan otomatis dari manifest dengan minimal tiga seed.

## 8. Rencana Eksperimen (Setelah Fase 6)

- **E1. Perbandingan utama.** NoAdapt, OnlineUpdate (head saja dan layer terakhir), dan ADAPT-Z pada horizon 1, 24, dan 48. Mulai dari ETTh1.
- **E2. Distribusi dan kalibrasi.** Reliability diagram, PIT, lebar interval, dan perilaku sebelum, saat, dan sesudah titik pergeseran. Hipotesis kerja: adaptasi memperbaiki akurasi titik, tetapi kalibrasi belum tentu membaik karena koreksi pada z memengaruhi lokasi dan skala sekaligus.
- **E3. Ablasi.** Letak z (output encoder atau hidden decoder), input adapter (fitur saja, gradien saja, keduanya), panjang riwayat gradien, kapasitas adapter, parameter yang dikoreksi (lokasi saja atau lokasi dan skala), inisialisasi adapter (dilatih awal atau murni online), learning rate, dan kepala distribusi (Student-t atau Normal).
- **E4. Generalisasi.** ETTh1, ETTh2, ETTm1, dan ETTm2.

Aturan praktik:

- Variasikan satu faktor sambil menahan faktor lain pada nilai baku.
- Gunakan minimal tiga seed dan laporkan rerata dan simpangan baku.
- Pilih hiperparameter adaptasi pada stream validasi, bukan pada stream test.

## 9. Risiko dan Pertanyaan Terbuka

Keputusan di bawah ini dicatat dalam log keputusan sebelum fase terkait dimulai.

- Apakah kode Hugging Face mendukung penyuntikan z yang dimodifikasi pada tahap generate (FR-M4)?
- Strategi multivariat: channel-independent atau multivariat sungguhan (input\_size lebih dari satu)?
- Satu model multi-horizon atau satu model per horizon?
- Normalisasi ganda antara scaler data dan scaling internal Hugging Face.
- Biaya sampling autoregresif di dalam loop online.
- Stabilitas adapter: NLL berekor berat (Student-t) dapat membuat gradien noisy, sehingga gradient clipping perlu dipertimbangkan.
- Detail Algorithm 1 harus diverifikasi terhadap makalah dan kode resmi ADAPT-Z sebelum implementasi final.
- Adapter dilatih awal atau murni online.

* Ketersediaan dan latensi Supabase selama eksperimen panjang. Mitigasi: muat data sekali per run dan simpan di memori.
* Isi database dapat berubah antar run. Mitigasi: hash dan jumlah baris data dicatat di manifest (FR-D9).

## 10. Di Luar Cakupan Versi Ini

- Baseline rekalibrasi conformal online.
- Dataset non-ETT.
- Deployment, serving, dan antarmuka pengguna.

## 11. Referensi

- Huang et al., Online time series prediction using feature adjustment (ADAPT-Z), arXiv 2509.03810.
- Lau, Shao, Yeung, Fast and Slow Streams for Online Time Series Forecasting without Information Leakage (DSOF), ICLR 2025.
- Blog Hugging Face: time-series-transformers (Time Series Transformer untuk prediksi probabilistik).
