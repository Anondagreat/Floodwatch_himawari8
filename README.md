# 🛰️ Prediksi Banjir dari Satelit Himawari-8 — KNN

**Universitas Telkom × BRIN | Bandung, 2024**

---

## 📦 Isi Folder Ini

```
flood_prediction/
├── index.html                    ← Web App interaktif (buka di browser)
├── flood_prediction_knn.py       ← Script Python utama
├── setup.bat                     ← Setup otomatis Windows
├── setup.sh                      ← Setup otomatis Mac/Linux
└── README.md                     ← File ini
```

---

## 🚀 CARA CEPAT — 3 Langkah

### Windows
```
1. Klik 2x → setup.bat
2. Tunggu instalasi selesai
3. Buka index.html di browser (Chrome/Firefox/Edge)
```

### Mac / Linux
```bash
chmod +x setup.sh
./setup.sh
open index.html     # Mac
xdg-open index.html # Linux
```

---

## 🔧 Instalasi Manual (jika setup.bat gagal)

### Langkah 1 — Install Python
Download dari: https://www.python.org/downloads/
> ⚠️ Centang **"Add Python to PATH"** saat instalasi di Windows!

Cek instalasi:
```bash
python --version
# Harusnya muncul: Python 3.x.x
```

### Langkah 2 — Install Library
Buka Terminal / Command Prompt, ketik:
```bash
pip install numpy scikit-learn matplotlib scipy
```

Jika error di Mac/Linux:
```bash
pip3 install numpy scikit-learn matplotlib scipy
# atau
python3 -m pip install numpy scikit-learn matplotlib scipy
```

### Langkah 3 — Jalankan Sistem

**Opsi A — Web App (Direkomendasikan untuk presentasi):**
```
Cukup double-click file: index.html
Buka di browser Chrome, Firefox, atau Edge
Tidak perlu Python, tidak perlu koneksi internet!
```

**Opsi B — Script Python:**
```bash
python flood_prediction_knn.py
# atau
python3 flood_prediction_knn.py
```
Output: file gambar `hasil_prediksi_banjir.png`

---

## 🖥️ Cara Menggunakan Web App

Setelah `index.html` terbuka di browser:

1. **Atur parameter** di panel kontrol:
   - **Nilai K** — tetangga terdekat (1–15, ganjil)
   - **Threshold Banjir** — batas curah hujan (mm/jam)
   - **Ukuran Training** — proporsi data training
   - **Mode Peta** — tampilan: Risiko Banjir / Kelas Awan / CTT / Curah Hujan

2. **Klik "▶ JALANKAN PREDIKSI"**

3. **Lihat hasil:**
   - Peta warna risiko banjir (Hijau=Aman, Kuning=Waspada, Merah=Bahaya)
   - Grafik akurasi untuk K=5, 7, 9, 11
   - Statistik area (% Aman, Waspada, Bahaya)
   - **Hover di atas peta** untuk melihat detail per titik (Lat, Lon, CTT, Curah Hujan, Risiko)

---

## 📊 Tentang Sistem

### Alur Kerja
```
Data Satelit Himawari-8 (NetCDF .nc)
    ↓
Ekstrak Cloud Top Temperature (CTT) dalam Kelvin
    ↓
Klasifikasi Suhu Awan → 3 Kelas:
    - Tinggi (270–300 K) → Tidak Hujan
    - Sedang (230–270 K) → Mendung
    - Rendah (180–230 K) → Hujan
    ↓
K-Nearest Neighbor (KNN) Training
    - 120 data training + 60 data validasi
    - Uji K = 5, 7, 9, 11
    - Jarak Euclidean: Dxy = √Σ(xi−yi)²
    ↓
[EKSTENSI] Prediksi Risiko Banjir:
    - Aman         → < 10 mm/jam
    - Waspada      → 10–20 mm/jam
    - Bahaya Banjir→ > 20 mm/jam (threshold BMKG)
    ↓
Peta Prediksi + Alert Otomatis
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

## ⚠️ Limitasi Sistem (Versi Saat Ini)

1. **Data Simulasi** — Belum menggunakan file .nc asli dari satelit Himawari-8
2. **Offline** — Belum terhubung ke live feed satelit secara real-time
3. **Tanpa CCTV** — Belum ada integrasi kamera lapangan untuk verifikasi
4. **Tanpa Topografi** — Belum mempertimbangkan data ketinggian (DEM)
5. **Validasi Manual** — Perbandingan prediksi vs kenyataan masih visual/kualitatif

---

## 🗺️ Roadmap Pengembangan

### FASE 1 — Sekarang ✅
- [x] Simulasi data Himawari-8 (CTT + curah hujan)
- [x] KNN classifier (3 kelas awan)
- [x] Ekstensi prediksi risiko banjir (3 level)
- [x] Dashboard web interaktif
- [x] Visualisasi peta 50×50 grid

### FASE 2 — +3 Bulan
- [ ] Download data .nc asli dari FTP JAXA
  ```
  URL: ftp://ftp.ptree.jaxa.jp
  Registrasi: https://www.eorc.jaxa.jp/ptree/registration_top.html
  Tool: FileZilla
  ```
- [ ] Parser NetCDF otomatis dengan `netCDF4`
- [ ] Pipeline: download → parse → prediksi → simpan hasil
- [ ] Jadwal otomatis setiap 10 menit (sesuai cadence Himawari-8)

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

## 📁 Cara Mendapatkan Data Himawari-8 Asli

1. **Daftar akun** di: https://www.eorc.jaxa.jp/ptree/registration_top.html
2. **Tunggu email** dengan kredensial FTP (1-2 hari kerja)
3. **Download FileZilla**: https://filezilla-project.org/
4. **Login FTP:**
   ```
   Host: ftp.ptree.jaxa.jp
   Username: [dari email JAXA]
   Password: [dari email JAXA]
   Port: 21
   ```
5. **Navigasi ke folder data:**
   ```
   /jma/hsd/[tahun]/[bulan]/[tanggal]/
   File format: HSF_<tanggal>_<jam>_R10_FLDK.02701_02701.nc
   ```
6. **Cek data dengan Panoply**: https://www.giss.nasa.gov/tools/panoply/

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

**Web app tidak muncul / blank**
```
Pastikan JavaScript diaktifkan di browser
Coba browser lain (Chrome / Firefox)
Jangan buka dari network drive, buka dari folder lokal
```

**Gambar .png tidak muncul setelah jalankan Python**
```
Cek folder yang sama dengan script .py
Pastikan tidak ada error di terminal
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
