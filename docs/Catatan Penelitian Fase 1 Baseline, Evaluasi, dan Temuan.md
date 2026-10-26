# Catatan Penelitian Fase 1: Baseline Probabilistic Transformer, Evaluasi, dan Temuan

Versi 1.0 · 26 Oktober 2026 · Cakupan: dari pembangunan baseline sampai pembekuan baseline, sebelum protokol online dan ADAPT-Z

## 1. Tujuan dan Status Dokumen

Dokumen ini adalah catatan kerja untuk bahan penulisan: apa yang dibangun, bagaimana, keputusan yang diambil beserta alasannya, hasil yang diperoleh, dan batasan temuan. Dokumen ini tidak membahas protokol online maupun ADAPT-Z, yang menjadi fase berikutnya. Dokumen penerapan teknis sebelumnya (Dokumen Penerapan: Baseline Probabilistic Transformer dan Pipeline Data) mencatat fase 0 sampai 2; dokumen ini melanjutkannya sampai evaluasi, tuning, diagnosis, dan pembekuan baseline.

Status saat penulisan:

- Baseline utama dibekukan pada konfigurasi hasil Optuna dengan `scaling: std` (disetujui). Salinan config bernama `baseline_final` dan run seed tambahan belum dikonfirmasi selesai.
- Evaluasi dilakukan pada split validasi saja. Split test belum disentuh dan disimpan untuk laporan akhir.
- Seluruh angka berada di ruang terstandardisasi (statistik split train) kecuali disebutkan lain.

## 2. Posisi dalam Rancangan Riset

Tema riset adalah peramalan deret waktu probabilistik dengan Transformer yang dapat mengadaptasi representasinya secara online memakai ADAPT-Z ketika distribusi data berubah. Fase ini menjawab pertanyaan pendahulu: baseline probabilistik apa yang dipakai, seberapa baik, dan di mana kelemahannya, karena kelemahan itulah yang kelak diuji oleh adaptasi.

Rujukan utama:

- ADAPT-Z (Online time series prediction using feature adjustment, arXiv 2509.03810): koreksi aditif pada representasi fitur memakai adapter MLP yang menggabungkan fitur saat ini dengan gradien historis, untuk umpan balik tertunda pada prediksi multi-langkah.
- DSOF (Fast and Slow Streams for Online Time Series Forecasting without Information Leakage, ICLR 2025): menyoroti kebocoran informasi pada evaluasi peramalan online dan mengusulkan kerangka dua aliran.
- Time Series Transformer dari Hugging Face sebagai model dasar.

## 3. Data

**Sumber.** Data ETT dimuat dari Supabase lewat fungsi `load_dataset` (ETTh1, ETTh2, ETTm1, ETTm2) dengan kolom `date`, `hufl`, `hull`, `mufl`, `mull`, `lufl`, `lull`, dan `ot`. Seluruh akses database melalui satu modul (`loader.py`) dan tidak ada salinan CSV sumber di lokal. Hanya hasil run yang disimpan lokal.

**Pembagian.** Mengikuti konvensi Autoformer, dipakai 20 bulan pertama dengan rasio 12/4/4 bulan. Untuk ETTh1: 8640 baris train, 2880 val, 2880 test. ETTm memakai jumlah baris empat kali lipat.

**Normalisasi.** Standardisasi per kanal dengan scaler buatan sendiri (numpy), di-fit hanya pada split train.

**Fitur waktu.** Empat fitur dari kolom tanggal (jam, hari dalam minggu, hari dalam bulan, bulan), diskalakan ke rentang -0,5 sampai 0,5.

**Windowing.**

- Panjang konteks 96, lag 1 sampai 7, 24, 48, dan 168, sehingga riwayat per sampel 264 langkah. Horizon prediksi 24.
- Satu sampel adalah satu origin: riwayat sebelum origin dan 24 langkah sesudahnya.
- Target prediksi tidak boleh melewati batas segmennya. Riwayat window val dan test boleh menjangkau segmen sebelumnya karena itu masa lalu, bukan kebocoran.
- Jumlah window mengikuti aturan itu: 8353 untuk train (stride 1) dan 2857 untuk val.

**Audit data.** Manifest tiap run mencatat jumlah baris, rentang tanggal, batas segmen, dan hash SHA-256 isi data, sehingga run tetap dapat diaudit bila isi database berubah.

**Pergeseran distribusi (temuan data).** Rata-rata dan simpangan baku split val dalam satuan statistik train (train: rata-rata 0, simpangan baku 1):

- hufl: -0,25 dan 1,25
- hull: -0,24 dan 1,06
- mufl: -0,30 dan 1,31
- mull: -0,20 dan 1,04
- lufl: +0,58 dan 1,54 (pergeseran terbesar)
- lull: -0,19 dan 1,24
- ot: -0,17 dan 0,54 (variasi menyempit hampir separuh)

Semua kanal bergeser, dengan `lufl` paling menonjol. Pergeseran ini menjadi latar belakang hampir semua temuan di bawah.

## 4. Model dan Training

**Model.** Pembungkus tipis di atas `TimeSeriesTransformerForPrediction` dengan distribusi keluaran Student-t, input multivariat (7 kanal sekaligus), dan antarmuka `training_loss` serta `sample`. Antarmuka `encode` dan `decode` yang dibutuhkan ADAPT-Z sengaja ditunda ke fase integrasi karena memakai bagian dalam Hugging Face yang bergantung versi.

**NLL per kanal.** Log-likelihood di Hugging Face dijumlahkan antar kanal untuk input multivariat, sehingga NLL per kanal dihitung sebagai NLL total dibagi 7. Asumsi ini punya uji khusus (`test_nll_is_sum_over_channels`) yang hasilnya belum dikonfirmasi saat catatan ini ditulis.

**Konfigurasi awal.** d\_model 64, 2 layer encoder dan 2 layer decoder, 2 head, dimensi FFN 128, dropout 0,1, `scaling: false` (scaling internal Hugging Face dimatikan karena data sudah distandardisasi).

**Training.** AdamW (learning rate 6e-4, weight decay 0,01), batch 64, maksimum 30 epoch, gradient clipping 1,0, early stopping dengan patience 5 pada NLL validasi, checkpoint terbaik disimpan, seed 42. Tiap run menulis `best.pt`, `history.csv`, dan `manifest.json` (config, metadata data, versi paket, hash git).

**Struktur kode.** `src/data` (loader, split, scaler, fitur waktu, windowing, pipeline), `src/models`, `src/training`, `src/metrics`, `src/baselines`, `src/evaluation`, dan `src/utils`. Konfigurasi di `configs/`, hasil di `exps/`.

**Pengujian.** Seluruh uji memakai data sintetis tanpa database. Terkonfirmasi di mesin penulis: smoke test model (4 lulus), uji data (5), uji sambungan data-model (1), dan uji training (2). Notebook perbandingan run sudah berjalan. Belum dikonfirmasi di mesin penulis: `test_metrics.py`, `test_static_eval.py`, dan `test_compare_runs.py`.

## 5. Protokol Evaluasi Statis

- Window split val (2857 window), 100 sampel per window, dengan satu prediksi dihitung dari sampel itu.
- Metrik titik: MSE dari rata-rata sampel dan MAE dari median sampel.
- Metrik probabilistik: CRPS dengan estimator sampel tak bias (fair), NLL per kanal, coverage interval sentral 50, 80, dan 95 persen, lebar interval, dan histogram PIT dengan 10 bin.
- Pembanding: seasonal naive (periode 24 untuk data per jam) dan last value.

Catatan metodologis yang penting untuk penulisan:

- **Jumlah sampel memengaruhi coverage, lebar interval, dan PIT.** Kuantil dari sampel terbatas bersifat bias. Perkiraan coverage prediksi yang sempurna sekitar 0,49, 0,78, dan 0,93 pada 100 sampel, serta sekitar 0,45, 0,72, dan 0,86 pada 20 sampel (perkiraan dari statistik terurut, bukan pengukuran). Evaluasi dengan 20 sampel sempat membuat perbandingan menyesatkan; semua perbandingan akhir memakai 100 sampel. Pada 20 sampel PIT juga hanya punya 21 nilai berbeda sehingga bin terakhir membesar secara artifisial.
- **CRPS baseline deterministik sama dengan MAE-nya.** Pembandingan CRPS model probabilistik dengan baseline deterministik tidak adil karena baseline tidak mendapat kredit atas ketidakpastian. Pembanding probabilistik yang lebih adil (misalnya seasonal naive dengan pita residual) belum dibuat.
- **Seleksi memakai split val.** Early stopping dan tuning memakai NLL val, sehingga metrik val sedikit optimistis.

## 6. Kronologi Eksperimen dan Hasil

### 6.1 Baseline awal (config default, tanpa scaling per window, seed 42)

Training berhenti di epoch 10 dengan val NLL terbaik 2,2164 pada epoch 5, sementara NLL train turun sampai 0,06 (celah besar, tanda menghafal).

Evaluasi val: MSE 0,5933, MAE 0,5162, CRPS 0,3907, coverage 0,355 / 0,608 / 0,792, lebar interval 80 persen 0,986, NLL per kanal 0,3166. PIT: bin pertama 0,139 dan bin terakhir 0,246 (ideal sekitar 0,1).

Pembanding: seasonal naive MSE 0,5113 dan MAE 0,4476; last value MSE 1,2638 dan MAE 0,7252.

Temuan: model mengalahkan last value tetapi kalah dari seasonal naive pada akurasi titik. Interval terlalu sempit (coverage jauh di bawah nominal) dan PIT miring ke atas, tanda nilai nyata sering berada di atas seluruh sampel (prediksi meremehkan level).

### 6.2 Tuning hiperparameter dengan Optuna

- Objective: meminimalkan val NLL terbaik dari `train()` pada ETTh1 horizon 24, seed 42. Sampler TPE (seed 42). Jumlah trial tidak tercatat dalam catatan ini (nilai baku skrip 30).
- Ruang pencarian: dropout 0,05 sampai 0,4; d\_model 32, 64, 96, atau 128; learning rate 1e-4 sampai 1e-3 (log); weight decay 1e-4 sampai 0,1 (log); layer encoder dan decoder 2 atau 3; pengali FFN 2 atau 4.
- Konfigurasi terbaik: dropout 0,1, d\_model 128, encoder 2 layer, decoder 3 layer, FFN 512, learning rate sekitar 2,70e-4, weight decay sekitar 7,48e-4.

Run ulang dengan konfigurasi ini (seed 42): val NLL terbaik 1,8995 pada epoch 8. Evaluasi val: MSE 0,6023, MAE 0,5169, CRPS 0,3912, coverage 0,342 / 0,598 / 0,780, lebar interval 80 persen 0,968, NLL per kanal 0,2714.

Temuan: NLL membaik sekitar 14 persen (per kanal 0,3166 menjadi 0,2714), tetapi MAE, CRPS, dan coverage praktis tidak berubah. NLL tidak selaras dengan metrik lain sehingga kurang tepat sebagai kriteria seleksi tunggal.

Kekurangan skrip Optuna yang tercatat: inisialisasi model tidak di-seed sehingga hasil trial tidak persis dapat direproduksi, pruner tidak aktif karena `train()` tidak melaporkan nilai antara, callback penyimpan config terbaik berisiko error bila trial pertama gagal, dan error trial diubah diam-diam menjadi pruned.

### 6.3 Diagnosis: di mana model kalah

MAE per langkah horizon (config Optuna, tanpa scaling per window):

- Model lebih baik dari seasonal naive hanya sampai langkah 4. Di langkah 1 model sedikit di atas last value (0,282 vs 0,270).
- MAE model naik dari 0,282 di langkah 1 menjadi sekitar 0,55 di langkah 11 dan 0,576 di langkah 24, sedangkan seasonal naive datar di 0,448.

MAE per kanal (model vs seasonal naive): hufl 0,615 vs 0,538; hull 0,483 vs 0,519; mufl 0,564 vs 0,534; mull 0,481 vs 0,492; lufl 0,796 vs 0,440; lull 0,452 vs 0,394; ot 0,228 vs 0,215. Model lebih buruk di 5 dari 7 kanal. Kanal `lufl` menyumbang sekitar 73 persen dari total selisih, dan di enam kanal lain selisih rata-ratanya hanya sekitar 0,02.

Interpretasi: kekalahan terpusat pada kanal dengan pergeseran level terbesar, konsisten dengan data di bagian 3 dan dengan PIT yang miring. Seasonal naive menyalin hari sebelumnya sehingga otomatis mengikuti level baru, sedangkan model terikat pada statistik periode train.

### 6.4 Normalisasi per window (`scaling: std`)

Opsi `scaling: std` di Hugging Face menormalkan tiap window dengan rata-rata dan simpangan baku konteksnya (mirip RevIN). Config default dengan opsi ini dijalankan pada tiga seed (42, 1, 2):

- MAE: 0,4924 / 0,5086 / 0,5145 (rata-rata 0,505, simpangan baku sekitar 0,011)
- MSE: 0,5415 / 0,5672 / 0,5768 (rata-rata 0,562, simpangan baku sekitar 0,018)
- CRPS: 0,3649 / 0,3774 / 0,3809 (rata-rata 0,374, simpangan baku sekitar 0,008)
- NLL per kanal: 0,2423 / 0,2641 / 0,2580 (rata-rata 0,255, simpangan baku sekitar 0,011)
- Coverage 80 persen: 0,647 / 0,626 / 0,618

Skewness PIT (bin terakhir dikurangi bin pertama) pada ketiga seed adalah -0,002, -0,014, dan -0,079, sedangkan tanpa scaling per window +0,108 dan +0,143. Untuk seed 42, PIT menjadi hampir simetris (bin pertama 0,173 dan bin terakhir 0,171).

Temuan yang kokoh: normalisasi per window menghilangkan bias ke bawah pada prediksi secara konsisten di ketiga seed.

Koreksi atas pembacaan awal: hasil seed 42 sempat dibaca sebagai perbaikan nyata pada akurasi titik. Setelah tiga seed, seed 42 ternyata yang terbaik; rata-rata MAE hanya sekitar satu simpangan baku di bawah baseline satu seed (0,516), sehingga perbaikan MAE berada dalam kebisingan antar seed. Perbaikan MSE dan CRPS sekitar dua simpangan baku, menjanjikan tetapi belum meyakinkan karena baseline tanpa scaling baru satu seed.

### 6.5 Konfigurasi Optuna dengan `scaling: std` (seed 42)

MSE 0,5144, MAE 0,4893, CRPS 0,3619, NLL per kanal 0,2254, coverage 0,357 / 0,626 / 0,823, lebar interval 80 persen 1,030, skewness PIT -0,124. Dibanding seasonal naive: MAE 9,3 persen lebih tinggi dan MSE hanya 0,6 persen lebih tinggi. Ini run terbaik pada semua metrik, tetapi baru satu seed. Terhadap rata-rata tiga seed `std` default, MSE-nya lebih rendah sekitar 2,6 simpangan baku dan MAE sekitar 1,4 simpangan baku.

### 6.6 Ringkasan perbandingan (val, 100 sampel)

- Baseline awal (tanpa scaling, 1 seed): MAE 0,516, MSE 0,593, CRPS 0,391, NLL/kanal 0,317.
- Optuna tanpa scaling (1 seed): MAE 0,517, MSE 0,602, CRPS 0,391, NLL/kanal 0,271.
- `std` default (3 seed): MAE 0,505, MSE 0,562, CRPS 0,374, NLL/kanal 0,255.
- Optuna dengan `std` (1 seed): MAE 0,489, MSE 0,514, CRPS 0,362, NLL/kanal 0,225.
- Seasonal naive: MAE 0,448, MSE 0,511.

## 7. Temuan Utama

1. Terdapat pergeseran distribusi nyata antara split train dan val pada ETTh1, terbesar pada `lufl` (rata-rata +0,58 dan simpangan baku 1,54) dan penyempitan variasi pada `ot`.
2. Transformer probabilistik tanpa normalisasi per window cenderung meremehkan level pada periode bergeser (PIT miring) dan terlalu percaya diri (coverage 80 persen nominal hanya sekitar 0,6).
3. Normalisasi per window menghilangkan bias PIT secara konsisten, tetapi tidak menutup kekurangan coverage maupun kalah dari seasonal naive pada MAE.
4. NLL bukan kriteria seleksi yang andal: tuning terhadap NLL menurunkannya sekitar 14 persen tanpa memperbaiki MAE, CRPS, atau coverage.
5. Seasonal naive adalah pembanding yang kuat pada ETTh1 horizon 24. Semua model berada 9 sampai 15 persen di atasnya pada MAE.
6. Kebisingan antar seed (simpangan baku MAE sekitar 0,011 dengan tiga seed) sebesar efek yang dikejar tuning, sehingga evaluasi harus multi-seed.
7. Jumlah sampel evaluasi memengaruhi metrik interval dan PIT; nilainya harus dilaporkan dan dijaga sama antar perbandingan.

## 8. Keputusan dan Alasan

- **Baseline dibekukan pada konfigurasi Optuna dengan `scaling: std`.** Terbaik di semua metrik pada run yang ada, dan bias PIT-nya sudah hilang. Tuning lebih lanjut ditunda karena imbal hasilnya kecil dibanding kebisingan antar seed.
- **Pengaturan `scaling` dilaporkan sebagai dua varian baseline** (`false` dan `std`). Normalisasi per window sebagian menyerap pergeseran level, sehingga ruang perbaikan untuk ADAPT-Z bisa mengecil. Pertanyaan riset yang tepat adalah apakah ADAPT-Z masih membantu di atas baseline yang sudah memakai normalisasi per window.
- **Tidak membuat klaim bahwa `std` memperbaiki akurasi titik.** Klaim yang didukung data hanya penghilangan bias PIT.
- **Tidak mengklaim mengalahkan seasonal naive.** Seasonal naive dimasukkan sebagai pembanding pada eksperimen online.
- **Split test dijaga bersih** sampai laporan akhir.

## 9. Keterbatasan

- Satu dataset (ETTh1), satu horizon (24), dan jumlah seed kecil (tiga untuk `std` default, satu untuk konfigurasi Optuna).
- Baseline tanpa scaling per window baru satu seed sehingga perbandingan `false` vs `std` pada akurasi titik belum konklusif.
- Seleksi dan tuning memakai split val, sehingga metrik val sedikit optimistis.
- Pembanding probabilistik yang adil untuk seasonal naive belum dibuat.
- Reprodusibilitas: run dengan seed dan konfigurasi sama menghasilkan epoch terbaik berbeda (8 dan 11), kemungkinan karena perangkat berbeda (CPU dan CUDA). Perbandingan sebaiknya hanya antar run di mesin dan perangkat yang sama.
- Baris `ETTh1_H24_optuna_seed42` pada tabel perbandingan campuran: metriknya dari baseline awal (berkas evaluasi lama tidak tertimpa), konfigurasinya dari run Optuna yang menimpa folder. Baris itu tidak boleh dipakai.
- Distribusi Student-t dan input multivariat membuat NLL dijumlahkan antar kanal; asumsi pembagian 7 menunggu konfirmasi uji.

## 10. Pekerjaan Tertunda Sebelum Protokol Online

1. Hapus berkas evaluasi lama di folder `ETTh1_H24_optuna_seed42` (atau jalankan evaluasi ulang) agar tabel perbandingan bersih.
2. Jalankan seed 1 dan 2 untuk konfigurasi baseline beku, lalu evaluasi dengan 100 sampel.
3. Konfirmasi uji `test_static_eval.py` (termasuk uji NLL per kanal), `test_metrics.py`, dan `test_compare_runs.py` di mesin penulis.
4. Opsional: tiga seed untuk baseline tanpa scaling per window sebagai ablasi.
5. Perbarui dokumen penerapan teknis bila struktur kode berubah.

## 11. Rencana Fase Berikutnya

- Protokol online: ledger label, iterator streaming, strategi NoAdapt dan OnlineUpdate, serta uji kebocoran informasi.
- Antarmuka encode dan decode beserta uji kesetaraan numerik. Karena `scaling: std` dipakai, representasi z harus dikembalikan bersama statistik normalisasi (rata-rata dan simpangan baku window), dan decoding harus memakai statistik yang sama agar koreksi pada z tidak salah skala.
- Integrasi ADAPT-Z dan perbandingan tiga kondisi (tanpa adaptasi, online update biasa, ADAPT-Z), disertai analisis kalibrasi dan ablasi.

## 12. Lampiran: Perintah dan Artefak

Perintah yang dipakai:

```
python -m src.training.run --dataset ETTh1 --pred-len 24 --model-config <config model> --train-config <config training> --seed <seed> --out-dir exps/<nama>
python -m src.evaluation.run_eval --exp-dir exps/<nama> --split val --n-samples 100
python -m src.evaluation.compare_runs --split val
```

Artefak per run di `exps/<nama>/`: `best.pt`, `history.csv`, `manifest.json`, dan `eval_val.json`. Ringkasan lintas run disimpan di `exps/_compare/` (CSV dan gambar) oleh notebook `compare_runs.ipynb`.

Urutan run yang tercatat (semua split val, 100 sampel, 2857 window): baseline awal; Optuna tanpa scaling (`coba1`); `scaling_std` seed 42, 1, dan 2; Optuna dengan `std` (`optuna_std`, seed 42).
