# 🛰️ Prediksi Banjir dari Satelit Himawari-8 — KNN

**Universitas Telkom | Bandung, 2026**

---

## 📦 Isi Folder Ini

```
flood_prediction/
├── index.html                    ← Web App interaktif (buka di browser)
├── flood_prediction_knn.py       ← Script Python utama (Fase 1)
│
├── ── FASE 2 ──────────────────────────────────────────────────
├── himawari_downloader.py        ← Download data .nc dari FTP JAXA
├── himawari_parser.py            ← Baca & proses file NetCDF Himawari-8
├── pipeline_fase2.py             ← Pipeline otomatis Fase 2
├── requirements_fase2.txt        ← Library tambahan Fase 2
│
├── setup.bat                     ← Setup otomatis Windows
├── setup.sh                      ← Setup otomatis Mac/Linux
└── README.md                     ← File ini

Folder yang dibuat otomatis saat dijalankan:
├── data_himawari/                ← File .nc yang didownload dari JAXA
├── output_fase2/                 ← Hasil prediksi (gambar + log)
└── model_saved/                  ← Model KNN yang tersimpan
```

---

## ❓ FileZilla, Panoply, dan OpenGrADS — Masih Diperlukan?

Pertanyaan ini penting. Singkatnya:

| Tools dari Paper Asli | Fase 1 | Fase 2 |
|-----------------------|--------|--------|
| **FileZilla** (download data FTP) | ❌ Tidak perlu | ✅ **Digantikan** oleh `himawari_downloader.py` secara otomatis |
| **Panoply** (cek isi file .nc) | ❌ Tidak perlu | ⚠️ **Opsional** — hanya untuk inspeksi manual file .nc, bukan bagian pipeline |
| **OpenGrADS** (visualisasi data atmosfer) | ❌ Tidak perlu | ❌ **Tidak perlu** — digantikan oleh matplotlib Python |

**Kesimpulan: Fase 2 tidak memerlukan FileZilla, Panoply, maupun OpenGrADS.**

Semua fungsinya sudah digantikan oleh library Python:
- `ftplib` (built-in Python) → menggantikan **FileZilla**
- `netCDF4` + `numpy` → menggantikan **Panoply** untuk baca file .nc
- `matplotlib` → menggantikan **OpenGrADS** untuk visualisasi peta

Tools tersebut hanya relevan di paper asli karena penelitinya menggunakan workflow manual.
Fase 2 mengotomasi seluruh workflow itu menjadi satu perintah Python.

---

## 🚀 CARA MENJALANKAN DI LAPTOP SENDIRI

### ─── FASE 1 — Dashboard Web & Script Python (Data Simulasi) ───

Fase 1 **tidak butuh internet** dan **tidak butuh akun apapun**.

**Langkah 1 — Install Python** (skip jika sudah ada)
```
Download dari: https://www.python.org/downloads/
⚠️  Windows: Centang "Add Python to PATH" saat instalasi!
```

Cek berhasil:
```bash
python --version
# Harusnya muncul: Python 3.x.x
```

**Langkah 2 — Install library**
```bash
pip install numpy scikit-learn matplotlib scipy
```

**Langkah 3A — Jalankan Web App (paling mudah, cocok untuk presentasi)**
```
Cukup double-click file: index.html
Buka di browser Chrome, Firefox, atau Edge
Tidak perlu Python, tidak perlu internet!
```

Cara pakai web app:
- Atur nilai K, threshold banjir, ukuran training, dan mode peta
- Klik **▶ JALANKAN PREDIKSI**
- Hover di atas peta → tampil detail per titik (Lat, Lon, CTT, Curah Hujan, Risiko)

**Langkah 3B — Jalankan Script Python (menghasilkan gambar)**
```bash
python flood_prediction_knn.py
```
Output: `hasil_prediksi_banjir.png` (peta visualisasi 6 panel)

---

### ─── FASE 2 — Pipeline Otomatis dengan Data Asli Himawari-8 ───

Fase 2 punya **dua sub-mode**: tanpa akun JAXA (mode simulasi) dan dengan akun JAXA (data asli).

---

#### 📌 SUB-MODE A — Tanpa Akun JAXA (Mode Simulasi Otomatis)

Bisa langsung dicoba sekarang, cocok untuk demo dan pengembangan.

**Langkah 1 — Pastikan semua file Fase 2 ada di folder yang sama**
```
flood_prediction/
├── himawari_downloader.py
├── himawari_parser.py
├── pipeline_fase2.py
├── requirements_fase2.txt
└── (file-file Fase 1 lainnya)
```

**Langkah 2 — Install library tambahan Fase 2**
```bash
pip install netCDF4 scipy schedule
```

Jika `netCDF4` gagal install di Windows:
```bash
# Coba binary wheel:
pip install netCDF4 --only-binary :all:

# Atau pakai conda (direkomendasikan jika pakai Anaconda):
conda install -c conda-forge netcdf4
```

> ⚠️ Jika `netCDF4` tetap gagal: **tidak masalah!** Pipeline akan otomatis
> beralih ke mode simulasi. Semua fitur tetap berjalan.

**Langkah 3 — Jalankan pipeline**
```bash
# Buka terminal / command prompt di folder flood_prediction/
python pipeline_fase2.py --mode local
```

Yang terjadi secara otomatis:
1. Model KNN dibangun dan disimpan ke `model_saved/knn_model.pkl`
2. Data simulasi 7 frame (07:00–13:00 UTC) di-generate
3. Setiap frame diprediksi: CTT → Kelas Awan → Risiko Banjir
4. Hasil disimpan ke folder `output_fase2/` sebagai gambar .png
5. Log JSON disimpan ke `output_fase2/prediction_log.jsonl`
6. Status alert tampil di terminal (AMAN / WASPADA / BAHAYA)

Output yang dihasilkan:
```
output_fase2/
├── prediksi_20240415_0700.png   ← Peta 4-panel: CTT, Curah Hujan, Kelas Awan, Risiko Banjir
├── prediksi_20240415_0800.png
├── prediksi_20240415_0900.png
├── ...
└── prediction_log.jsonl         ← Log semua prediksi (bisa dibuka di Notepad/VS Code)

model_saved/
├── knn_model.pkl                ← Model KNN tersimpan (tidak perlu training ulang)
├── scaler.pkl                   ← StandardScaler tersimpan
└── model_meta.json              ← Info K, akurasi, tanggal training
```

---

#### 📌 SUB-MODE B — Dengan Akun JAXA (Data Asli Himawari-8)

Ini adalah Fase 2 sesungguhnya. Akun JAXA gratis, daftar sekali, berlaku selamanya.

**Langkah 1 — Daftar akun JAXA**
```
Buka: https://www.eorc.jaxa.jp/ptree/registration_top.html
Isi form pendaftaran → Submit
Tunggu email dari JAXA berisi username & password (1-2 hari kerja)
Format username dari JAXA biasanya: emailkamu@domain.com_jaxa
```

**Langkah 2 — Set kredensial di script**

Buka file `himawari_downloader.py`, cari baris:
```python
FTP_USER = os.environ.get("JAXA_USER", "your_email@example.com_jaxa")
FTP_PASS = os.environ.get("JAXA_PASS", "your_password")
```
Ganti dengan kredensial dari email JAXA:
```python
FTP_USER = "emailkamu@domain.com_jaxa"   # dari email JAXA
FTP_PASS = "password_dari_jaxa"
```

Atau lebih aman, set sebagai environment variable:

Windows (Command Prompt):
```cmd
set JAXA_USER=emailkamu@domain.com_jaxa
set JAXA_PASS=password_dari_jaxa
```

Mac/Linux (Terminal):
```bash
export JAXA_USER="emailkamu@domain.com_jaxa"
export JAXA_PASS="password_dari_jaxa"
```

**Langkah 3 — Pilih mode download dan jalankan**

```bash
# Download data 1 jam terakhir (6 file) lalu prediksi
python pipeline_fase2.py --mode live

# Download rentang waktu tertentu (format: YYYY-MM-DD HH:MM, waktu UTC)
# Contoh: data tanggal 15 April 2024, jam 14:00–20:00 WIB = 07:00–13:00 UTC
python pipeline_fase2.py --mode range \
    --start "2024-04-15 07:00" --end "2024-04-15 13:00"

# Jalankan terus-menerus: download + prediksi otomatis setiap 10 menit
# Cocok untuk deployment monitoring jangka panjang
# Tekan Ctrl+C untuk berhenti
python pipeline_fase2.py --mode scheduler
```

> 💡 **Tips waktu:** Himawari-8 pakai UTC. Jawa Barat = UTC+7.
> Contoh: mau data jam 14:00 WIB → masukkan jam 07:00 UTC.

**Langkah 4 — Cek hasilnya**

Hasil tersimpan otomatis di folder `output_fase2/`. Buka gambar .png
untuk melihat peta 4-panel (CTT, Curah Hujan, Kelas Awan, Risiko Banjir).

---

#### 📌 PERINTAH TAMBAHAN

```bash
# Lihat daftar file .nc yang sudah ada di lokal
python -c "from himawari_downloader import list_local_files; list_local_files()"

# Force training ulang model KNN dari awal
python pipeline_fase2.py --mode local --retrain

# Jalankan hanya downloader saja (tanpa prediksi)
python himawari_downloader.py

# Test parser pada satu file .nc tertentu
python -c "
from himawari_parser import parse_nc_file
from pathlib import Path
result = parse_nc_file(Path('data_himawari/H08_20240415_0700_R10_FLDK.02401_02401.nc'))
if result:
    print('CTT range:', result['ctt'].min(), '-', result['ctt'].max(), 'K')
    print('Timestamp:', result['timestamp'])
"
```

---

## 📊 Tentang Sistem

### Alur Kerja — Fase 1

```
Data Simulasi (generate_simulation_data)
    ↓
Preprocessing & Labeling suhu awan
    ↓
KNN Training (K = 5, 7, 9, 11) + Evaluasi Akurasi
    ↓
Prediksi seluruh grid 50×50
    ↓
Peta Risiko Banjir + Visualisasi 6 panel
```

### Alur Kerja — Fase 2 (Pipeline Otomatis)

```
FTP JAXA (ftp.ptree.jaxa.jp)
    ↓  [himawari_downloader.py]
Download file .nc setiap 10 menit
    ↓  [himawari_parser.py]
Parse NetCDF → Ekstrak Brightness Temperature (B13)
    ↓
Konversi BT → Cloud Top Temperature (CTT)
    ↓
Crop ke Jawa Barat (LAT -8°–-5°, LON 105°–109°)
    ↓
Resample ke grid 50×50
    ↓  [pipeline_fase2.py]
KNN Predict (model dari file .pkl)
    ↓
Estimasi curah hujan dari CTT (model empiris)
    ↓
Klasifikasi Risiko Banjir: AMAN / WASPADA / BAHAYA
    ↓
Simpan plot PNG + log JSON + alert otomatis
```

### Hasil Akurasi

| K  | Prediksi Hujan | Prediksi Banjir |
|----|---------------|-----------------|
| 5  | 98.33%        | 93.33%          |
| 7  | 98.33%        | 91.67%          |
| 9  | **100.00%**   | 91.67%          |
| 11 | 96.67%        | 91.67%          |

> **Catatan:** Akurasi ini dari data simulasi. Hasil dengan data asli Himawari-8 akan berbeda.

---

## ⚠️ Limitasi Sistem

### Fase 1 (Sudah Teratasi di Fase 2)
- ~~Data Simulasi~~ → Fase 2 mendukung data .nc asli
- ~~Offline~~ → Fase 2 ada scheduler + download otomatis
- ~~Tidak ada pipeline~~ → Fase 2 punya pipeline penuh

### Masih Dalam Pengembangan (Fase 3 & 4)
1. **Tanpa CCTV** — Belum ada integrasi kamera lapangan untuk verifikasi
2. **Tanpa Topografi** — Belum mempertimbangkan data ketinggian (DEM)
3. **Estimasi curah hujan** — Masih empiris (CTT-based), belum divalidasi dengan data BMKG aktual
4. **Alert hanya ke terminal** — Belum ada notifikasi WhatsApp/Telegram/email

---

## 🗺️ Roadmap Pengembangan

### FASE 1 — ✅ Selesai
- [x] Simulasi data Himawari-8 (CTT + curah hujan)
- [x] KNN classifier (3 kelas awan)
- [x] Ekstensi prediksi risiko banjir (3 level)
- [x] Dashboard web interaktif
- [x] Visualisasi peta 50×50 grid

### FASE 2 — ✅ Selesai

**File baru yang ditambahkan:**

| File | Fungsi |
|------|--------|
| `himawari_downloader.py` | Download data .nc dari FTP JAXA otomatis (menggantikan FileZilla) |
| `himawari_parser.py` | Baca file .nc, ekstrak CTT, crop ke Jawa Barat (menggantikan Panoply + OpenGrADS) |
| `pipeline_fase2.py` | Pipeline lengkap: download → parse → KNN → simpan → alert |
| `requirements_fase2.txt` | Library tambahan yang diperlukan |

**Yang sudah diimplementasikan:**
- [x] Download otomatis dari FTP JAXA dengan retry logic (tanpa FileZilla)
- [x] Parser NetCDF (.nc) robust untuk berbagai format Himawari-8 (tanpa Panoply)
- [x] Konversi Brightness Temperature → CTT + koreksi kalibrasi AHI
- [x] Crop & resample ke area Jawa Barat (tanpa OpenGrADS)
- [x] Model KNN persistent (tersimpan di .pkl, tidak perlu retrain setiap run)
- [x] Scheduler otomatis setiap 10 menit
- [x] Log prediksi dalam format JSON Lines
- [x] Alert otomatis: AMAN / WASPADA / BAHAYA
- [x] Fallback ke simulasi jika `netCDF4` belum terinstall

> Lihat bagian **"CARA MENJALANKAN DI LAPTOP SENDIRI"** di atas untuk panduan lengkap.

### FASE 3 — +6 Bulan
- [ ] Integrasi feed CCTV banjir (OpenCV + RTSP stream)
- [ ] Computer vision untuk deteksi genangan di frame video
- [ ] Alert system: notifikasi WhatsApp/Telegram jika bahaya terdeteksi
- [ ] API endpoint (FastAPI) untuk integrasi ke sistem lain

### FASE 4 — +12 Bulan
- [ ] Model hybrid: KNN + CNN/LSTM untuk sequence prediction
- [ ] Integrasi data BMKG (curah hujan aktual) sebagai ground truth
- [ ] Data topografi DEM (Digital Elevation Model) dari DEMNAS/SRTM
- [ ] Platform web real-time dengan peta Leaflet.js
- [ ] Dokumentasi dan publikasi ilmiah

---

## 📁 Referensi Struktur Data Himawari-8

Jika ingin inspeksi file .nc secara manual (opsional, tidak wajib):

```
Struktur FTP JAXA:
    Host   : ftp.ptree.jaxa.jp
    Folder : /jma/hsd/YYYYMM/DD/HHmm/
    File   : H08_YYYYMMDD_HHmm_R10_FLDK.02401_02401.nc
    Cadence: setiap 10 menit UTC

Tools inspeksi manual (opsional):
    Panoply  → https://www.giss.nasa.gov/tools/panoply/  (buka & lihat isi .nc)
    HDFView  → https://www.hdfgroup.org/downloads/hdfview/ (alternatif Panoply)
    FileZilla→ https://filezilla-project.org/ (browser FTP manual)

Kanal yang digunakan sistem ini:
    B13 (10.4 μm, Infrared Thermal) → Cloud Top Temperature (CTT)
```

> Fase 2 sudah mengotomasi seluruh proses download dan parsing.
> Tools di atas hanya diperlukan jika ingin inspeksi file secara manual.

---

## 🐛 Troubleshooting

**"Python tidak ditemukan"**
```
Reinstall Python dari python.org, centang "Add to PATH"
Restart komputer setelah instalasi
```

**"pip bukan perintah yang dikenal" (Windows)**
```bash
python -m pip install numpy scikit-learn matplotlib scipy
```

**"No module named 'sklearn'" saat jalankan .py**
```bash
pip install scikit-learn
```

**"No module named 'netCDF4'" saat jalankan Fase 2**
```bash
pip install netCDF4
# Jika gagal di Windows, coba versi binary:
pip install netCDF4 --only-binary :all:
# Atau install via conda:
conda install -c conda-forge netcdf4
```

**"Connection refused" saat download FTP JAXA**
```
- Pastikan username dan password dari email JAXA sudah benar
- Format username biasanya: email@domain.com_jaxa (ada _jaxa di akhir)
- Coba jalankan tanpa download dulu: python pipeline_fase2.py --mode local
```

**Web app tidak muncul / blank**
```
Pastikan JavaScript diaktifkan di browser
Coba browser lain (Chrome / Firefox)
Jangan buka dari network drive, buka dari folder lokal
```

**Gambar .png tidak muncul setelah jalankan Python**
```
Cek folder output_fase2/ untuk hasil Fase 2
Cek folder yang sama dengan script untuk Fase 1
```

---

## 📚 Referensi

- Nisya, H., et al. (2023). *Prediksi Curah Hujan Dari Data Satelit Himawari-8 Menggunakan Metode K-Nearest Neighbor (KNN)*. e-Proceeding of Engineering, Vol.10, No.1.
- Risyanto, G.N. (2019). *Identification of rainfall area in Indonesia using infrared channels of Himawari-8 AHI*. IOP Conference.
- Saadatfar, H., et al. (2020). *A New K-Nearest Neighbor Classifier for Big Data Based on Efficient Data Pruning*. MDPI Mathematics.
- JAXA Himawari Monitor: https://www.eorc.jaxa.jp/ptree/
- GSMaP (Global Satellite Mapping of Precipitation): https://sharaku.eorc.jaxa.jp/GSMaP/

---

*Dibuat untuk keperluan presentasi Tugas Akhir — Universitas Telkom × BRIN, 2024*
