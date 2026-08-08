"""
forecasting_fase3.py
====================
Major Changes Fase 3: Time Series Forecasting dari data Himawari-9

Alur:
    1. Download 1 hari data dari JAXA (atau baca dari folder lokal)
    2. Resample ke 3 interval: 10 menit, 30 menit, 60 menit
    3. Sliding window forecasting → prediksi CTT 3 jam ke depan
    4. KNN classify hasil prediksi → kelas awan + risiko banjir
    5. Bandingkan akurasi ketiga interval
    6. Output: PNG sequence + GIF animasi + grafik perbandingan

Cara pakai:
    # Download 1 hari kemarin lalu forecast
    python forecasting_fase3.py --mode auto

    # Pakai file .nc yang sudah ada di data_himawari/
    python forecasting_fase3.py --mode local

    # Download tanggal spesifik
    python forecasting_fase3.py --mode date --date "2026-05-20"

    # Simulasi (tanpa file .nc)
    python forecasting_fase3.py --mode simulate
"""

import re
import argparse
import json
import pickle
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom as sp_zoom
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
# Area studi: Kota Bandung dan Kabupaten Bandung, Jawa Barat
LAT_MIN, LAT_MAX = -7.0, -6.8
LON_MIN, LON_MAX = 107.5, 107.8
GRID_SIZE        = 50

# Interval yang dibandingkan (dalam menit)
INTERVALS = {
    "10 menit":  10,
    "30 menit":  30,
    "60 menit":  60,
}

# Target forecasting: 3 jam ke depan
FORECAST_HORIZON_MINUTES = 180

# Sliding window: berapa frame sebelumnya dipakai sebagai input
# Untuk tiap interval, window = 3 jam data historis
WINDOW_SIZE = {
    "10 menit":  18,   # 3 jam ÷ 10 menit = 18 frame
    "30 menit":   6,   # 3 jam ÷ 30 menit = 6 frame
    "60 menit":   3,   # 3 jam ÷ 60 menit = 3 frame
}

# Threshold kelas awan (Kelvin)
CTT_HIGH   = 270   # >= 270 K → Tidak Hujan
CTT_MID    = 230   # >= 230 K → Mendung
# < 230 K  → Hujan

# Threshold risiko banjir (mm/jam)
FLOOD_SAFE = 10
FLOOD_WARN = 20

# Output
OUTPUT_DIR   = Path("./KNN_forecast_output")
MODEL_DIR    = Path("./model_saved")
DATA_DIR     = Path("./data_himawari")
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 1: DOWNLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_one_day(target_date: datetime,
                     interval_minutes: int = 10) -> list[Path]:
    """
    Download semua file .nc untuk satu hari penuh dari JAXA.
    
    Himawari-9 beroperasi 00:00–23:50 UTC, interval 10 menit.
    Satu hari = 144 file × ~120 MB = bisa sangat berat.
    
    Untuk efisiensi, default download dengan interval 10 menit
    tapi hanya jam operasional yang relevan (00:00–23:50 UTC).
    
    Args:
        target_date: Tanggal yang mau didownload
        interval_minutes: Interval download (10, 30, atau 60)
    
    Returns:
        List path file yang berhasil didownload
    """
    import ftplib
    import os

    FTP_HOST = "ftp.ptree.jaxa.jp"
    FTP_USER = os.environ.get("JAXA_USER", "asmodes123_gmail.com")
    FTP_PASS = os.environ.get("JAXA_PASS", "SP+wari8")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate semua timestamp untuk hari itu
    timestamps = []
    current = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end     = target_date.replace(hour=23, minute=50, second=0, microsecond=0)
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD DATA HIMAWARI-9")
    print(f"  Tanggal : {target_date.strftime('%d %B %Y')}")
    print(f"  Interval: {interval_minutes} menit")
    print(f"  Target  : {len(timestamps)} file")
    print(f"{'='*60}")

    downloaded = []
    folder_cache = {}   # cache folder yang sudah dicek

    try:
        with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
            ftp.login(FTP_USER, FTP_PASS)
            print("  Login FTP OK")

            for i, ts in enumerate(timestamps, 1):
                folder = (f"/jma/netcdf/{ts.strftime('%Y%m')}/"
                          f"{ts.strftime('%d')}/")
                fname  = (f"NC_H09_{ts.strftime('%Y%m%d_%H%M')}"
                          f"_R21_FLDK.02801_02401.nc")
                local  = DATA_DIR / fname

                # Skip jika sudah ada dan valid
                if local.exists() and local.stat().st_size > 10_000:
                    downloaded.append(local)
                    if i % 10 == 0:
                        print(f"  [{i}/{len(timestamps)}] "
                              f"{ts.strftime('%H:%M')} UTC — SKIP (sudah ada)")
                    continue

                # Pindah folder jika perlu
                if folder not in folder_cache:
                    try:
                        ftp.cwd(folder)
                        available = ftp.nlst()
                        folder_cache[folder] = available
                    except ftplib.error_perm:
                        folder_cache[folder] = []
                        print(f"  [{i}/{len(timestamps)}] "
                              f"{ts.strftime('%H:%M')} UTC — folder tidak ada")
                        continue

                available = folder_cache[folder]

                # Cari file yang cocok (toleransi nama berbeda)
                target_ts = ts.strftime('%Y%m%d_%H%M')
                candidates = [f for f in available
                              if target_ts in f
                              and "FLDK" in f
                              and f.endswith(".nc")]

                if not candidates:
                    print(f"  [{i}/{len(timestamps)}] "
                          f"{ts.strftime('%H:%M')} UTC — file tidak ada")
                    continue

                actual_fname = candidates[0]
                actual_local = DATA_DIR / actual_fname

                try:
                    ftp.cwd(folder)
                    print(f"  [{i}/{len(timestamps)}] "
                          f"{ts.strftime('%H:%M')} UTC — downloading...",
                          end=" ", flush=True)
                    with open(actual_local, "wb") as f:
                        ftp.retrbinary(f"RETR {actual_fname}", f.write)
                    size_mb = actual_local.stat().st_size / 1024 / 1024
                    print(f"OK ({size_mb:.0f} MB)")
                    downloaded.append(actual_local)
                except Exception as e:
                    print(f"GAGAL — {e}")
                    if actual_local.exists():
                        actual_local.unlink()

    except Exception as e:
        print(f"  FTP Error: {e}")

    print(f"\n  ✓ Berhasil: {len(downloaded)}/{len(timestamps)} file")
    return downloaded


def load_local_files(target_date: datetime | None = None) -> list[Path]:
    """
    Muat file .nc dari folder lokal.
    Jika target_date diberikan, filter hanya file dari tanggal itu.
    """
    if not DATA_DIR.exists():
        return []

    all_files = sorted(DATA_DIR.glob("NC_H09_*.nc"))
    all_files = [f for f in all_files if f.stat().st_size > 10_000]

    if target_date is not None:
        date_str = target_date.strftime("%Y%m%d")
        all_files = [f for f in all_files if date_str in f.name]

    return all_files


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 2: PARSE FILE .NC → CTT GRID
# ══════════════════════════════════════════════════════════════════════════════

def parse_nc_to_ctt(nc_path: Path,
                    verbose: bool = False) -> tuple[np.ndarray, datetime] | None:
    """
    Baca file NC_H09 R21 Full Disk dan ekstrak CTT area Bandung.
    Coverage R21: LAT -60–60, LON 70–210
    Variabel CTT: tbb_14 (kanal 14, 11.2μm)
    """
    from scipy.ndimage import zoom as sp_zoom

    # Hanya proses file R21 (Full Disk)
    if "_R21_" not in nc_path.name and "_r21_" not in nc_path.name:
        return None

    m = re.search(r'NC_H09_(\d{8})_(\d{4})', nc_path.name)
    if not m:
        return None
    timestamp = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")

    try:
        import netCDF4 as nc_lib
    except ImportError:
        return _simulate_ctt(timestamp)

    try:
        with nc_lib.Dataset(nc_path, "r") as ds:

            # ── 1. Pilih variabel CTT ─────────────────────────────────────
            tbb_priority = ["tbb_14", "tbb_13", "tbb_15", "tbb_07"]
            bt_data  = None
            var_used = None

            for vname in tbb_priority:
                if vname in ds.variables:
                    bt_data  = ds.variables[vname][:]
                    var_used = vname
                    break

            if bt_data is None:
                for vname in ds.variables:
                    if "tbb" in vname.lower():
                        bt_data  = ds.variables[vname][:]
                        var_used = vname
                        break

            if bt_data is None:
                return None

            # ── 2. Konversi DN → Brightness Temperature ───────────────────
            bv     = ds.variables[var_used]
            scale  = float(getattr(bv, "scale_factor", 1.0))
            offset = float(getattr(bv, "add_offset",   0.0))
            fill   = getattr(bv, "_FillValue", None)

            if hasattr(bt_data, "filled"):
                fill_val = fill if fill is not None else -32768
                bt_data  = bt_data.filled(fill_value=fill_val)

            bt_arr = np.array(bt_data, dtype=np.float32)

            if fill is not None:
                invalid = (bt_arr == float(fill))
            else:
                invalid = np.zeros_like(bt_arr, dtype=bool)

            # Konversi dengan pengecekan
            bt_arr = bt_arr * scale + offset
            bt_arr[invalid] = np.nan

            while bt_arr.ndim > 2:
                bt_arr = bt_arr[0]

            # ── 3. Konversi BT → CTT ─────────────────────────────────────────────
            bt_arr = bt_arr.astype(np.float32)

            # Isi nilai invalid dengan 295K (cerah/tidak ada awan)
            bt_arr = np.nan_to_num(bt_arr, nan=295.0)

            # Konversi standar — nilai rendah (variasi kecil) normal untuk hari cerah
            ctt = np.clip(0.9991 * bt_arr + 0.3, 150, 330).astype(np.float32)

            # ── 4. Baca koordinat ─────────────────────────────────────────
            lat_1d = np.array(ds.variables["latitude"][:],  dtype=np.float32)
            lon_1d = np.array(ds.variables["longitude"][:], dtype=np.float32)

            # Pastikan lat menurun (utara ke selatan)
            if lat_1d[0] < lat_1d[-1]:
                lat_1d = lat_1d[::-1]
                ctt    = ctt[::-1, :]

            # ── 5. Crop ke Kota Bandung ───────────────────────────────────
            lat_mask = (lat_1d >= LAT_MIN) & (lat_1d <= LAT_MAX)
            lon_mask = (lon_1d >= LON_MIN) & (lon_1d <= LON_MAX)
            lat_idx  = np.where(lat_mask)[0]
            lon_idx  = np.where(lon_mask)[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                if verbose:
                    print(f"    Area Bandung tidak ditemukan di {nc_path.name}")
                    print(f"    LAT file: {lat_1d.min():.2f}–{lat_1d.max():.2f}")
                    print(f"    LON file: {lon_1d.min():.2f}–{lon_1d.max():.2f}")
                return None

            ctt_crop = ctt[lat_idx[0] : lat_idx[-1] + 1,
                           lon_idx[0] : lon_idx[-1] + 1]

            if ctt_crop.shape[0] < 2 or ctt_crop.shape[1] < 2:
                return None

            # ── 6. Resample ke GRID_SIZE × GRID_SIZE ─────────────────────
            zr      = GRID_SIZE / ctt_crop.shape[0]
            zc      = GRID_SIZE / ctt_crop.shape[1]
            ctt_out = sp_zoom(ctt_crop, (zr, zc), order=1).astype(np.float32)

            if verbose:
                print(f"    ✓ {nc_path.name}: {var_used} "
                      f"CTT={ctt_out.min():.1f}–{ctt_out.max():.1f}K "
                      f"crop={ctt_crop.shape}→{ctt_out.shape}")

            return ctt_out, timestamp

    except Exception as e:
        if verbose:
            print(f"    ERROR {nc_path.name}: {e}")
        return None


def _simulate_ctt(timestamp: datetime) -> tuple[np.ndarray, datetime]:
    """Generate CTT simulasi konsisten berdasarkan timestamp."""
    seed = int(timestamp.strftime("%Y%m%d%H%M")) % (2**31)
    rng  = np.random.default_rng(seed)
    G    = GRID_SIZE

    # Base CTT dengan variasi diurnal (siang lebih hangat, malam lebih dingin)
    hour_utc = timestamp.hour
    hour_wib = (hour_utc + 7) % 24
    # Konveksi Bandung biasanya siang–sore WIB (10:00–17:00)
    if 10 <= hour_wib <= 17:
        base_ctt = 245 + 15 * rng.standard_normal((G, G))
    else:
        base_ctt = 265 + 15 * rng.standard_normal((G, G))

    # Sel konvektif acak
    n_cells = rng.integers(2, 6)
    for _ in range(n_cells):
        cx = rng.integers(5, G-5)
        cy = rng.integers(5, G-5)
        r  = rng.integers(3, 9)
        intensity = rng.uniform(20, 50)
        for i in range(G):
            for j in range(G):
                d = np.sqrt((i-cx)**2 + (j-cy)**2)
                if d < r:
                    base_ctt[i, j] -= intensity * (1 - d/r)

    ctt = np.clip(base_ctt, 180, 300).astype(np.float32)
    return ctt, timestamp


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 3: BUILD TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def build_time_series(input_data,
                      interval_minutes: int) -> list[tuple[datetime, np.ndarray]]:
    """
    Terima input Path atau tuple, bersihkan duplikat,
    isi gap, lalu resample ke interval target.
    """
    if not input_data:
        return []

    first = input_data[0]

    # ── Parse semua file jadi (timestamp, ctt) ──────────────────────────
    if isinstance(first, tuple):
        raw_series = list(input_data)
    else:
        print(f"\n  Parsing {len(input_data)} file ...")
        raw_series = []
        for f in input_data:
            result = parse_nc_to_ctt(f)
            if result is not None:
                ctt, ts = result
                raw_series.append((ts, ctt))

    if not raw_series:
        return []

    # ── Urutkan berdasarkan timestamp ────────────────────────────────────
    raw_series.sort(key=lambda x: x[0])

    # ── Hapus duplikat timestamp — ambil yang pertama saja ───────────────
    seen_ts  = set()
    deduped  = []
    dup_count = 0
    for ts, ctt in raw_series:
        # Bulatkan ke 10 menit terdekat untuk normalisasi
        minute_norm = (ts.minute // 10) * 10
        ts_norm = ts.replace(minute=minute_norm, second=0, microsecond=0)
        if ts_norm not in seen_ts:
            seen_ts.add(ts_norm)
            deduped.append((ts_norm, ctt))
        else:
            dup_count += 1

    if dup_count > 0:
        print(f"  Duplikat dihapus: {dup_count} frame")

    deduped.sort(key=lambda x: x[0])
    print(f"  Frame unik setelah deduplikasi: {len(deduped)}")

    # ── Isi gap dengan interpolasi linear antar frame ────────────────────
    # Himawari interval asli 10 menit — isi gap yang <= 60 menit
    filled   = []
    for i in range(len(deduped)):
        filled.append(deduped[i])
        if i < len(deduped) - 1:
            ts_curr, ctt_curr = deduped[i]
            ts_next, ctt_next = deduped[i + 1]
            gap_min = (ts_next - ts_curr).total_seconds() / 60

            # Isi gap jika 10 < gap <= 60 menit
            if 10 < gap_min <= 60:
                n_missing = int(gap_min // 10) - 1
                for k in range(1, n_missing + 1):
                    alpha    = k / (n_missing + 1)
                    ts_interp  = ts_curr + timedelta(minutes=10 * k)
                    ctt_interp = ((1 - alpha) * ctt_curr +
                                  alpha       * ctt_next).astype(np.float32)
                    filled.append((ts_interp, ctt_interp))

    filled.sort(key=lambda x: x[0])
    n_interp = len(filled) - len(deduped)
    if n_interp > 0:
        print(f"  Frame interpolasi untuk isi gap: {n_interp}")
    print(f"  Total frame setelah fill gap: {len(filled)}")

    # ── Resample ke interval target ──────────────────────────────────────
    if interval_minutes == 10:
        print(f"  → {len(filled)} frame (interval 10 menit)")
        return filled

    resampled = []
    start_ts  = filled[0][0]
    end_ts    = filled[-1][0]
    current   = start_ts

    while current <= end_ts:
        closest  = min(filled,
                       key=lambda x: abs((x[0] - current).total_seconds()))
        diff_min = abs((closest[0] - current).total_seconds()) / 60
        if diff_min <= interval_minutes / 2:
            resampled.append(closest)
        current += timedelta(minutes=interval_minutes)

    # Hapus duplikat resample
    seen2, unique = set(), []
    for ts, ctt in resampled:
        if ts not in seen2:
            seen2.add(ts)
            unique.append((ts, ctt))

    print(f"  → {len(unique)} frame setelah resample ke {interval_minutes} menit")
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 4: FORECASTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.neighbors import KNeighborsRegressor

def knn_forecast(series: list[np.ndarray],
                 n_steps_ahead: int,
                 k: int = 5) -> np.ndarray:
    """
    Forecasting CTT menggunakan KNN Regression.
    
    Konsep: 
        - Setiap window historis = 1 sampel training
        - Fitur = rata-rata CTT per frame dalam window (flattened)
        - Target = CTT pada t + horizon
    
    Karena data per piksel terlalu besar untuk KNN langsung,
    kita reduksi dengan mengambil statistik per frame:
        [mean, std, min, max] per frame × window_size = feature vector
    """
    n = len(series)
    if n < n_steps_ahead + 2:
        return series[-1].copy()

    G = series[0].shape[0]

    # ── Build feature matrix dari seluruh history ───────────────────
    # Setiap baris = fitur dari satu window
    # Fitur per frame: [mean, std, min, max] → 4 × window_size fitur
    window = len(series)

    def extract_features(frames: list[np.ndarray]) -> np.ndarray:
        feats = []
        for frame in frames:
            feats.extend([
                float(frame.mean()),
                float(frame.std()),
                float(frame.min()),
                float(frame.max()),
            ])
        return np.array(feats)

    # Kita tidak punya cukup history untuk training KNN yang bermakna
    # dalam satu run, jadi gunakan KNN per piksel dengan time index
    # sebagai fitur (lebih sederhana dan tetap valid secara ML)
    
    stack    = np.stack(series, axis=0)   # (n, G, G)
    t_values = np.arange(n).reshape(-1, 1).astype(np.float32)
    t_pred   = np.array([[n - 1 + n_steps_ahead]], dtype=np.float32)

    # KNN Regression per piksel
    forecast = np.zeros((G, G), dtype=np.float32)

    knn_reg = KNeighborsRegressor(
        n_neighbors=min(k, n),
        metric="euclidean",
        weights="distance"   # frame lebih dekat ke t diberi bobot lebih
    )

    for i in range(G):
        for j in range(G):
            y = stack[:, i, j]
            knn_reg.fit(t_values, y)
            forecast[i, j] = knn_reg.predict(t_pred)[0]

    # Smoothing spasial
    from scipy.ndimage import gaussian_filter
    forecast = gaussian_filter(forecast, sigma=1.0)

    # Clip berdasarkan historis
    hist_min = float(stack.min())
    hist_max = float(stack.max())
    forecast = np.clip(forecast, hist_min - 5, hist_max + 5)

    return forecast.astype(np.float32)


def run_forecast_experiment(time_series: list[tuple[datetime, np.ndarray]],
                             interval_minutes: int,
                             knn: KNeighborsClassifier,
                             scaler: StandardScaler) -> dict:
    """
    Jalankan eksperimen forecasting untuk satu interval.
    
    Untuk setiap titik waktu t dalam time series:
        - Ambil window sebelum t sebagai input
        - Prediksi CTT pada t + 3 jam
        - Bandingkan dengan CTT aktual di t + 3 jam
    
    Returns dict berisi hasil, metrik akurasi, dan data untuk visualisasi.
    """
    window    = WINDOW_SIZE[f"{interval_minutes} menit"]
    horizon   = FORECAST_HORIZON_MINUTES // interval_minutes   # steps ke depan

    print(f"\n  [{interval_minutes} menit] "
          f"window={window} frame, horizon={horizon} step ({FORECAST_HORIZON_MINUTES} menit) ...")

    timestamps = [ts for ts, _ in time_series]
    ctt_frames = [ctt for _, ctt in time_series]
    n          = len(time_series)

    results = []
    mae_list     = []
    acc_list_ctt = []
    acc_list_flood = []

    for i in range(window, n - horizon):
        # Input: frame i-window sampai i
        input_frames  = ctt_frames[i - window: i]
        # Ground truth: frame i + horizon
        actual_ctt    = ctt_frames[i + horizon]
        actual_ts     = timestamps[i + horizon]
        input_ts      = timestamps[i]

        # Prediksi
        pred_ctt = knn_forecast(input_frames, n_steps_ahead=horizon, k=5)

        # Evaluasi per piksel
        mae = float(np.mean(np.abs(pred_ctt - actual_ctt)))
        mae_list.append(mae)

        # Klasifikasi kelas awan: actual vs predicted
        actual_class = classify_ctt_grid(actual_ctt)
        pred_class_threshold = classify_ctt_grid(pred_ctt)

        # Gunakan model KNN yang sudah dilatih untuk mengklasifikasikan CTT
        X_pred = pred_ctt.flatten().reshape(-1, 1)
        X_pred_sc = scaler.transform(X_pred)
        pred_class_knn = knn.predict(X_pred_sc).reshape(pred_ctt.shape)

        acc_ctt_threshold = float(np.mean(actual_class == pred_class_threshold) * 100)
        acc_ctt_knn = float(np.mean(actual_class == pred_class_knn) * 100)
        acc_list_ctt.append(acc_ctt_knn)

        # Klasifikasi risiko banjir
        actual_precip = ctt_to_precip(actual_ctt)
        pred_precip   = ctt_to_precip(pred_ctt)
        actual_flood  = classify_flood(actual_precip)
        pred_flood    = classify_flood(pred_precip)
        acc_flood     = float(np.mean(actual_flood == pred_flood) * 100)
        acc_list_flood.append(acc_flood)

        results.append({
            "input_ts":            input_ts,
            "actual_ts":           actual_ts,
            "input_ctt":           input_frames[-1],
            "pred_ctt":            pred_ctt,
            "actual_ctt":          actual_ctt,
            "pred_class_threshold": pred_class_threshold,
            "pred_class_knn":      pred_class_knn,
            "actual_class":        actual_class,
            "pred_flood":          pred_flood,
            "actual_flood":        actual_flood,
            "mae":                 mae,
            "acc_ctt_threshold":   acc_ctt_threshold,
            "acc_ctt_knn":         acc_ctt_knn,
            "acc_flood":           acc_flood,
        })

    metrics = {
        "interval":       interval_minutes,
        "n_predictions":  len(results),
        "mae_mean":       float(np.mean(mae_list)) if mae_list else 0,
        "mae_std":        float(np.std(mae_list)) if mae_list else 0,
        "acc_ctt_mean":   float(np.mean(acc_list_ctt)) if acc_list_ctt else 0,
        "acc_flood_mean": float(np.mean(acc_list_flood)) if acc_list_flood else 0,
        "results":        results,
    }

    print(f"    MAE CTT    : {metrics['mae_mean']:.2f} K ± {metrics['mae_std']:.2f}")
    print(f"    Akurasi Awan : {metrics['acc_ctt_mean']:.1f}%")
    print(f"    Akurasi Banjir: {metrics['acc_flood_mean']:.1f}%")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 5: HELPER KLASIFIKASI
# ══════════════════════════════════════════════════════════════════════════════

def classify_ctt_grid(ctt: np.ndarray) -> np.ndarray:
    """CTT grid → kelas awan (0=tidak hujan, 1=mendung, 2=hujan)."""
    c = np.zeros_like(ctt, dtype=int)
    c[ctt >= CTT_HIGH] = 0
    c[(ctt >= CTT_MID) & (ctt < CTT_HIGH)] = 1
    c[ctt < CTT_MID] = 2
    return c


def ctt_to_precip(ctt: np.ndarray) -> np.ndarray:
    """Estimasi curah hujan dari CTT (model empiris GPI)."""
    precip = np.zeros_like(ctt)
    mask   = ctt < 235
    precip[mask] = 3.0 * np.exp(-0.036 * (ctt[mask] - 235))
    return np.clip(precip, 0, 100)


def classify_flood(precip: np.ndarray) -> np.ndarray:
    """Curah hujan → risiko banjir (0=aman, 1=waspada, 2=bahaya)."""
    risk = np.zeros_like(precip, dtype=int)
    risk[(precip >= FLOOD_SAFE) & (precip < FLOOD_WARN)] = 1
    risk[precip >= FLOOD_WARN] = 2
    return risk


def build_knn_model() -> tuple[KNeighborsClassifier, StandardScaler]:
    """Load atau build model KNN."""
    model_path  = MODEL_DIR / "knn_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and scaler_path.exists():
        with open(model_path, "rb") as f:
            knn = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return knn, scaler

    # Build dari scratch
    rng = np.random.default_rng(42)
    n   = 600
    c0  = rng.uniform(270, 300, n//3)
    c1  = rng.uniform(230, 270, n//3)
    c2  = rng.uniform(180, 230, n//3)
    X   = np.concatenate([c0, c1, c2]).reshape(-1, 1)
    y   = np.array([0]*(n//3) + [1]*(n//3) + [2]*(n//3))
    X  += rng.normal(0, 1.5, X.shape)
    X   = np.clip(X, 180, 300)

    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)
    knn      = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_sc, y)

    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return knn, scaler


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 6: VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════

CITIES = {
    "Bandung (Pusat)": (-6.921, 107.607),
    "Cicendo":         (-6.902, 107.585),
    "Andir":           (-6.910, 107.578),
    "Coblong":         (-6.889, 107.614),
    "Sukasari":        (-6.878, 107.589),
    "Buahbatu":        (-6.948, 107.643),
    "Rancasari":       (-6.955, 107.672),
    "Gedebage":        (-6.960, 107.699),
    "Arcamanik":       (-6.928, 107.674),
    "Bojongloa":       (-6.937, 107.578),
}

LATS = np.linspace(LAT_MIN, LAT_MAX, GRID_SIZE)
LONS = np.linspace(LON_MIN, LON_MAX, GRID_SIZE)

FLOOD_CMAP  = mcolors.ListedColormap(["#22c55e", "#f59e0b", "#ef4444"])
FLOOD_NORM  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
CLOUD_CMAP  = mcolors.ListedColormap(["#FFD700", "#87CEEB", "#1E3A8A"])
CLOUD_NORM  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
ALERT_COLOR = {"AMAN": "#22c55e", "WASPADA": "#f59e0b", "BAHAYA": "#ef4444"}


def _add_cities(ax):
    """Tambah marker kota ke axes."""
    for name, (la, lo) in CITIES.items():
        if LAT_MIN <= la <= LAT_MAX and LON_MIN <= lo <= LON_MAX:
            ax.plot(lo, la, "w^", ms=5, zorder=5)
            ax.text(lo + 0.02, la + 0.02, name,
                    fontsize=6, color="white", fontweight="bold", zorder=6)


def _flood_alert(flood_grid: np.ndarray) -> str:
    """Tentukan level alert dari grid risiko banjir."""
    total = flood_grid.size
    pct_danger = (flood_grid == 2).sum() / total * 100
    pct_warn   = (flood_grid == 1).sum() / total * 100
    if pct_danger >= 15:
        return "BAHAYA"
    elif pct_danger >= 5 or pct_warn >= 30:
        return "WASPADA"
    return "AMAN"


FIG_WIDTH_PX  = 1920
FIG_HEIGHT_PX = 1000
FIG_DPI       = 100   # → figure size = 19.2 × 10 inch
FIG_W_INCH    = FIG_WIDTH_PX  / FIG_DPI
FIG_H_INCH    = FIG_HEIGHT_PX / FIG_DPI

CTT_VMIN_GLOBAL  = None   # diisi saat pertama kali plot
CTT_VMAX_GLOBAL  = None
ERROR_VMIN_GLOBAL = 0
ERROR_VMAX_GLOBAL = None  # diisi dari data aktual

def compute_global_color_range(all_metrics: dict) -> None:
    """
    Hitung range warna dari data aktual.
    Untuk data variasi rendah (< 5K), gunakan stretch kontras lokal
    agar tetap ada gradasi warna yang informatif.
    """
    global CTT_VMIN_GLOBAL, CTT_VMAX_GLOBAL, ERROR_VMAX_GLOBAL

    all_ctt_vals   = []
    all_error_vals = []

    for metrics in all_metrics.values():
        for r in metrics.get("results", []):
            all_ctt_vals.append(r["input_ctt"].flatten())
            all_ctt_vals.append(r["pred_ctt"].flatten())
            all_ctt_vals.append(r["actual_ctt"].flatten())
            err = np.abs(r["pred_ctt"] - r["actual_ctt"])
            all_error_vals.append(err.flatten())

    if all_ctt_vals:
        combined = np.concatenate(all_ctt_vals)
        p1  = float(np.percentile(combined, 1))
        p99 = float(np.percentile(combined, 99))
        mid = (p1 + p99) / 2
        rng = p99 - p1

        print(f"\n  Data CTT: {combined.min():.2f}–{combined.max():.2f} K")
        print(f"  Variasi (p1–p99): {p1:.2f}–{p99:.2f} K (rentang {rng:.2f} K)")

        if rng < 5.0:
            # Data sangat seragam — stretch kontras agar tetap ada gradasi
            # Ambil std deviation sebagai ukuran variasi
            std = float(np.std(combined))
            stretch = max(std * 3, 2.0)   # minimal ±2K dari tengah
            CTT_VMIN_GLOBAL = mid - stretch
            CTT_VMAX_GLOBAL = mid + stretch
            print(f"  ⚠️  Variasi rendah ({rng:.2f}K) — stretch kontras ke "
                  f"{CTT_VMIN_GLOBAL:.2f}–{CTT_VMAX_GLOBAL:.2f} K")
        else:
            CTT_VMIN_GLOBAL = p1
            CTT_VMAX_GLOBAL = p99

    if all_error_vals:
        combined_err      = np.concatenate(all_error_vals)
        p95               = float(np.percentile(combined_err, 95))
        ERROR_VMAX_GLOBAL = max(p95, 0.5)   # minimal 0.5K
        print(f"  Error range: 0–{ERROR_VMAX_GLOBAL:.2f} K")

def stretch_contrast(data: np.ndarray,
                     vmin: float,
                     vmax: float) -> np.ndarray:
    """
    Stretch kontras data ke range vmin–vmax untuk visualisasi.
    Nilai fisik tidak diubah — hanya untuk keperluan plotting.
    Hasilnya: data yang tadinya 274.5–276.3K akan terlihat
    seperti memiliki gradasi penuh dari biru ke merah.
    """
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < 0.01:
        # Benar-benar flat — kembalikan nilai tengah
        return np.full_like(data, (vmin + vmax) / 2)
    # Normalisasi ke 0–1 lalu scale ke vmin–vmax
    normalized = (data - data_min) / (data_max - data_min)
    return (normalized * (vmax - vmin) + vmin).astype(np.float32)

# Konstanta layout untuk gambar terpisah
FIG_WIDTH_PX  = 1920
FIG_HEIGHT_PX = 500    # separuh dari sebelumnya karena hanya 1 baris
FIG_DPI       = 100
FIG_W_INCH    = FIG_WIDTH_PX  / FIG_DPI
FIG_H_INCH    = FIG_HEIGHT_PX / FIG_DPI

def plot_single_frame(result: dict,
                      interval_label: str,
                      out_path: Path) -> tuple[Path, Path]:
    """
    Buat 2 file gambar terpisah per frame:
      - out_path_ctt   : 3 panel CTT (Input, Prediksi, Aktual)
      - out_path_class : 3 panel klasifikasi (Kelas Awan, Risiko Banjir, Error)
    
    Returns: (path_ctt, path_class)
    """
    vmin_ctt   = CTT_VMIN_GLOBAL  if CTT_VMIN_GLOBAL  is not None else 180.0
    vmax_ctt   = CTT_VMAX_GLOBAL  if CTT_VMAX_GLOBAL  is not None else 300.0
    vmax_error = ERROR_VMAX_GLOBAL if ERROR_VMAX_GLOBAL is not None else 10.0

    ctt_levels   = np.linspace(vmin_ctt,   vmax_ctt,   21)
    error_levels = np.linspace(0,          vmax_error,  16)

    input_ts_wib  = result["input_ts"]  + timedelta(hours=7)
    actual_ts_wib = result["actual_ts"] + timedelta(hours=7)
    alert         = _flood_alert(result["pred_flood"])
    alert_color   = {"AMAN": "#22c55e",
                     "WASPADA": "#f59e0b",
                     "BAHAYA":  "#ef4444"}[alert]

    # Path output — tambah suffix _ctt dan _class
    stem         = out_path.stem        # misal: forecast_20260613_0700
    out_ctt      = out_path.parent / f"{stem}_ctt.png"
    out_class    = out_path.parent / f"{stem}_class.png"

    # ── Layout helper ─────────────────────────────────────────────────────
    pad_l, pad_r = 0.04, 0.02
    pad_b, pad_t = 0.10, 0.14
    cbar_w       = 0.016
    cbar_gap     = 0.008
    col_gap      = 0.045

    total_w = 1 - pad_l - pad_r
    ax_w    = (total_w - 2*col_gap - 3*(cbar_w+cbar_gap)) / 3
    ax_h    = 1 - pad_b - pad_t

    def ax_pos(col):
        unit_w = ax_w + cbar_w + cbar_gap + col_gap
        left   = pad_l + col * unit_w
        return [left, pad_b, ax_w, ax_h]

    def cbar_pos(col):
        ap = ax_pos(col)
        return [ap[0]+ap[2]+cbar_gap, ap[1], cbar_w, ap[3]]

    def style_ax(ax, title, title_color="white"):
        ax.set_title(title, color=title_color, fontsize=10,
                     pad=5, fontweight="bold")
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        ax.set_xlabel("Lon", color="#64748b", fontsize=8)
        ax.set_ylabel("Lat", color="#64748b", fontsize=8)
        _add_cities(ax)

    def make_ctt_panel(fig, ax, cax, data, title):
        data_s = stretch_contrast(data, vmin_ctt, vmax_ctt)
        cf = ax.contourf(LONS, LATS, data_s,
                         levels=ctt_levels, cmap="RdYlBu_r",
                         vmin=vmin_ctt, vmax=vmax_ctt, extend="both")
        cb = fig.colorbar(cf, cax=cax)
        # Tampilkan nilai aktual di colorbar
        tick_v = np.linspace(vmin_ctt, vmax_ctt, 5)
        tick_a = np.linspace(data.min(), data.max(), 5)
        cb.set_ticks(tick_v)
        cb.set_ticklabels([f"{v:.1f}" for v in tick_a])
        cb.set_label("K (aktual)", color="#94a3b8", fontsize=7)
        cb.ax.tick_params(colors="#94a3b8", labelsize=6)
        ax.text(0.02, 0.03,
                f"{data.min():.2f}–{data.max():.2f} K",
                transform=ax.transAxes, fontsize=6.5,
                color="#fbbf24",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#0f172a", alpha=0.8))
        style_ax(ax, title)

    # ════════════════════════════════════════════════════════════════════
    # GAMBAR 1: 3 Panel CTT
    # ════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(FIG_W_INCH, FIG_H_INCH),
                      dpi=FIG_DPI, facecolor="#0f172a")

    fig1.text(0.5, 0.97,
              f"Cloud Top Temperature — Interval {interval_label}  |  "
              f"Kota Bandung",
              ha="center", va="top",
              fontsize=13, color="white", fontweight="bold",
              fontfamily="monospace")
    fig1.text(0.5, 0.91,
              f"Input: {input_ts_wib.strftime('%d %b %Y %H:%M WIB')}  →  "
              f"Prediksi: {actual_ts_wib.strftime('%H:%M WIB')}  |  "
              f"MAE: {result['mae']:.3f} K  |  "
              f"Akurasi KNN: {result['acc_ctt_knn']:.1f}%",
              ha="center", va="top",
              fontsize=9, color="#94a3b8")

    axes1  = [fig1.add_axes(ax_pos(c),   facecolor="#1e293b") for c in range(3)]
    cbars1 = [fig1.add_axes(cbar_pos(c))                      for c in range(3)]

    make_ctt_panel(fig1, axes1[0], cbars1[0], result["input_ctt"],
                   f"CTT Input  {input_ts_wib.strftime('%H:%M WIB')}")
    make_ctt_panel(fig1, axes1[1], cbars1[1], result["pred_ctt"],
                   f"CTT Prediksi  +3 jam = {actual_ts_wib.strftime('%H:%M WIB')}")
    make_ctt_panel(fig1, axes1[2], cbars1[2], result["actual_ctt"],
                   f"CTT Aktual  {actual_ts_wib.strftime('%H:%M WIB')}")

    plt.figure(fig1.number)
    plt.savefig(out_ctt, dpi=FIG_DPI, bbox_inches=None,
                facecolor="#0f172a")
    plt.close(fig1)

    # ════════════════════════════════════════════════════════════════════
    # GAMBAR 2: 3 Panel Klasifikasi
    # ════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(FIG_W_INCH, FIG_H_INCH),
                      dpi=FIG_DPI, facecolor="#0f172a")

    fig2.text(0.5, 0.97,
              f"Klasifikasi & Risiko Banjir — Interval {interval_label}  |  "
              f"Kota Bandung",
              ha="center", va="top",
              fontsize=13, color="white", fontweight="bold",
              fontfamily="monospace")
    fig2.text(0.5, 0.91,
              f"Input: {input_ts_wib.strftime('%d %b %Y %H:%M WIB')}  →  "
              f"Prediksi: {actual_ts_wib.strftime('%H:%M WIB')}  |  "
              f"Status: {alert}  |  "
              f"Akurasi Banjir: {result['acc_flood']:.1f}%",
              ha="center", va="top",
              fontsize=9, color=alert_color)

    axes2  = [fig2.add_axes(ax_pos(c),   facecolor="#1e293b") for c in range(3)]
    cbars2 = [fig2.add_axes(cbar_pos(c))                      for c in range(3)]

    # Panel 0: Kelas Awan Aktual
    axes2[0].pcolormesh(LONS, LATS, result["actual_class"],
                         cmap=CLOUD_CMAP, norm=CLOUD_NORM, shading="auto")
    sm0 = plt.cm.ScalarMappable(cmap=CLOUD_CMAP, norm=CLOUD_NORM)
    cb0 = fig2.colorbar(sm0, cax=cbars2[0])
    cb0.set_ticks([0, 1, 2])
    cb0.set_ticklabels(["Tdk Hujan", "Mendung", "Hujan"], fontsize=7)
    cb0.ax.tick_params(colors="#94a3b8", labelsize=7)
    style_ax(axes2[0], "Kelas Awan Aktual")

    # Panel 1: Prediksi Risiko Banjir
    axes2[1].pcolormesh(LONS, LATS, result["pred_flood"],
                         cmap=FLOOD_CMAP, norm=FLOOD_NORM, shading="auto")
    sm1 = plt.cm.ScalarMappable(cmap=FLOOD_CMAP, norm=FLOOD_NORM)
    cb1 = fig2.colorbar(sm1, cax=cbars2[1])
    cb1.set_ticks([0, 1, 2])
    cb1.set_ticklabels(["Aman", "Waspada", "Bahaya"], fontsize=7)
    cb1.ax.tick_params(
        colors=alert_color, labelsize=7)
    style_ax(axes2[1],
             f"★ Prediksi Risiko Banjir [{alert}]",
             title_color=alert_color)
    for sp in axes2[1].spines.values():
        sp.set_edgecolor(alert_color)

    # Panel 2: Error Map
    error           = np.abs(result["pred_ctt"] - result["actual_ctt"])
    error_stretched = stretch_contrast(error, 0, vmax_error)
    actual_err_max  = float(error.max())
    cf2 = axes2[2].contourf(
        LONS, LATS, error_stretched,
        levels=error_levels, cmap="hot_r",
        vmin=0, vmax=vmax_error, extend="max")
    cb2 = fig2.colorbar(cf2, cax=cbars2[2])
    tick_e      = np.linspace(0, vmax_error, 5)
    tick_e_act  = np.linspace(0, actual_err_max, 5)
    cb2.set_ticks(tick_e)
    cb2.set_ticklabels([f"{v:.3f}" for v in tick_e_act])
    cb2.set_label("K (aktual)", color="#94a3b8", fontsize=7)
    cb2.ax.tick_params(colors="#94a3b8", labelsize=6)
    style_ax(axes2[2],
             f"Error |Pred−Aktual|  MAE={result['mae']:.3f}K")

    plt.figure(fig2.number)
    plt.savefig(out_class, dpi=FIG_DPI, bbox_inches=None,
                facecolor="#0f172a")
    plt.close(fig2)

    return out_ctt, out_class


def plot_comparison_chart(all_metrics: dict, out_path: Path) -> None:
    """
    Plot grafik perbandingan akurasi ketiga interval.
    Panel 1: Bar chart akurasi kelas awan
    Panel 2: Bar chart akurasi risiko banjir
    Panel 3: Bar chart MAE
    Panel 4: Line chart akurasi per prediksi (overtime)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0f172a")
    fig.suptitle(
        "Perbandingan Interval Forecasting — Prediksi 3 Jam ke Depan\n"
        "Bandung & Sekitarnya | Data Himawari-9",
        fontsize=14, color="white", fontweight="bold", y=0.98
    )

    labels  = ["10 menit", "30 menit", "60 menit"]
    colors  = ["#0ea5e9", "#f59e0b", "#22c55e"]
    metrics = [all_metrics.get(lbl, {}) for lbl in labels]

    ax_kw = dict(facecolor="#1e293b")
    for ax in axes.flat:
        ax.set(**ax_kw)
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")

    # Panel 1: Akurasi Kelas Awan
    acc_ctt = [m.get("acc_ctt_mean", 0) for m in metrics]
    bars = axes[0, 0].bar(labels, acc_ctt, color=colors, alpha=0.85,
                           edgecolor="white", linewidth=0.5)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_ylabel("Akurasi (%)", color="#94a3b8")
    axes[0, 0].set_title("Akurasi Prediksi Kelas Awan", color="white", fontweight="bold")
    for bar, val in zip(bars, acc_ctt):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                         val + 1, f"{val:.1f}%",
                         ha="center", color="white", fontsize=11, fontweight="bold")

    # Panel 2: Akurasi Risiko Banjir
    acc_flood = [m.get("acc_flood_mean", 0) for m in metrics]
    bars2 = axes[0, 1].bar(labels, acc_flood, color=colors, alpha=0.85,
                             edgecolor="white", linewidth=0.5)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_ylabel("Akurasi (%)", color="#94a3b8")
    axes[0, 1].set_title("Akurasi Prediksi Risiko Banjir", color="white", fontweight="bold")
    for bar, val in zip(bars2, acc_flood):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2,
                         val + 1, f"{val:.1f}%",
                         ha="center", color="white", fontsize=11, fontweight="bold")

    # Panel 3: MAE
    mae_vals = [m.get("mae_mean", 0) for m in metrics]
    mae_std  = [m.get("mae_std", 0) for m in metrics]
    bars3 = axes[1, 0].bar(labels, mae_vals, color=colors, alpha=0.85,
                             edgecolor="white", linewidth=0.5,
                             yerr=mae_std, capsize=5,
                             error_kw=dict(ecolor="white", linewidth=1.5))
    axes[1, 0].set_ylabel("MAE (Kelvin)", color="#94a3b8")
    axes[1, 0].set_title("Mean Absolute Error CTT", color="white", fontweight="bold")
    axes[1, 0].invert_yaxis()   # MAE lebih kecil = lebih baik (di atas)
    for bar, val in zip(bars3, mae_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2,
                         val + 0.2, f"{val:.2f} K",
                         ha="center", color="white", fontsize=10, fontweight="bold")

    # Panel 4: Akurasi overtime per interval
    axes[1, 1].set_title("Akurasi Kelas Awan per Prediksi (Overtime)",
                           color="white", fontweight="bold")
    axes[1, 1].set_xlabel("Index Prediksi", color="#94a3b8")
    axes[1, 1].set_ylabel("Akurasi (%)", color="#94a3b8")

    for lbl, color, m in zip(labels, colors, metrics):
        if m and m.get("results"):
            accs = [r.get("acc_ctt_knn", r.get("acc_ctt_threshold", 0)) for r in m["results"]]
            # Smooth dengan rolling average
            window = max(1, len(accs) // 10)
            smoothed = np.convolve(accs,
                                   np.ones(window)/window,
                                   mode="valid")
            axes[1, 1].plot(smoothed, label=lbl, color=color,
                             linewidth=2, alpha=0.9)

    axes[1, 1].legend(facecolor="#1e293b", labelcolor="white", fontsize=10)
    axes[1, 1].set_ylim(0, 105)

    # Tambah rekomendasi teks
    if acc_ctt and any(v > 0 for v in acc_ctt):
        best_idx  = int(np.argmax(acc_ctt))
        best_lbl  = labels[best_idx]
        best_val  = acc_ctt[best_idx]
        best_mae  = mae_vals[best_idx]
        fig.text(0.5, 0.01,
                 f"★  Rekomendasi: Interval {best_lbl} memberikan akurasi terbaik "
                 f"({best_val:.1f}%) dengan MAE {best_mae:.2f} K untuk prediksi 3 jam ke depan",
                 ha="center", color="#fbbf24", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  ✓ Grafik perbandingan: {out_path.name}")


def create_gif(png_paths: list[Path],
               out_path: Path,
               fps: int = 2) -> None:
    """
    Buat GIF stabil — setiap frame diganti penuh, tidak ada ghosting.
    Layout tidak bergeser karena semua frame ukuran piksel identik.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  pip install Pillow untuk GIF")
        return

    valid = [p for p in sorted(png_paths) if p.exists()]
    if not valid:
        return

    frames   = []
    ref_size = None

    for p in valid:
        img = Image.open(p).convert("RGB")
        if ref_size is None:
            ref_size = img.size
        # Paksa ukuran sama jika ada perbedaan kecil
        if img.size != ref_size:
            img = img.resize(ref_size, Image.LANCZOS)
        # Konversi ke palette dengan dithering minimal
        frames.append(img.quantize(colors=256, method=2, dither=0))

    if not frames:
        return

    duration_ms = int(1000 / fps)

    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,   # jangan optimize — mencegah frame shift
        disposal=2,       # clear frame sebelumnya sebelum gambar berikutnya
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ✓ GIF: {out_path.name} "
          f"({len(frames)} frame, {fps} fps, {size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 7: MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fase 3 — Forecasting 3 Jam ke Depan dari Data Himawari-9"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "local", "date", "simulate"],
        default="simulate",
        help=(
            "auto     : download 1 hari kemarin otomatis\n"
            "local    : pakai file .nc yang sudah ada\n"
            "date     : download tanggal spesifik\n"
            "simulate : simulasi tanpa file .nc"
        )
    )
    parser.add_argument("--date", type=str,
                        help="Tanggal format YYYY-MM-DD (untuk mode date)")
    parser.add_argument("--interval", type=int, default=10,
                        choices=[10, 30, 60],
                        help="Interval download (menit), default 10")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  FASE 3 — FORECASTING 3 JAM KE DEPAN")
    print("  Bandung & Sekitarnya | Himawari-9 + KNN + Trend Forecasting")
    print("="*65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Dapatkan file .nc ──────────────────────────────────────
    print("\n[1] Mendapatkan data ...")
    nc_files = []

    if args.mode == "simulate":
        print("  Mode SIMULASI — generate data sintetis 1 hari")
        # Buat timestamp dummy untuk 1 hari
        base_date = datetime.now() - timedelta(days=1)
        base_date = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
        dummy_timestamps = [
            base_date + timedelta(minutes=10*i)
            for i in range(144)   # 144 frame = 1 hari × 10 menit
        ]
        # Langsung ke time series tanpa file
        raw_series = [_simulate_ctt(ts) for ts in dummy_timestamps]
        raw_series_paired = [(ts, ctt) for ctt, ts in raw_series]
        nc_files = None   # tandai bahwa pakai simulasi

    elif args.mode == "local":
        print("\n[2] Mode LOCAL — memuat file .nc dari data_himawari/ ...")
        all_files = sorted(DATA_DIR.glob("NC_H09_*.nc"))
        all_files = [f for f in all_files if f.stat().st_size > 10_000]

        # Filter hanya R21 (Full Disk — mencakup Indonesia)
        nc_files = [f for f in all_files if "_R21_" in f.name or "_r21_" in f.name]
        skipped  = len(all_files) - len(nc_files)
        print(f"  Total file   : {len(all_files)}")
        print(f"  File R21     : {len(nc_files)}")
        if skipped > 0:
            print(f"  Dilewati     : {skipped} file (bukan R21/Full Disk)")

        if not nc_files:
            print("  ❌ Tidak ada file R21. Pastikan download Full Disk dari JAXA.")
            return

        # Parse semua file
        print(f"\n[2b] Parsing {len(nc_files)} file R21 ...")
        raw_series_paired = []
        failed = 0
        for i, f in enumerate(nc_files, 1):
            result = parse_nc_to_ctt(f, verbose=(i <= 3))
            if result is not None:
                ctt, ts = result
                raw_series_paired.append((ts, ctt))
            else:
                failed += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(nc_files)}] berhasil={len(raw_series_paired)} gagal={failed}")

        print(f"\n  Total frame berhasil : {len(raw_series_paired)}")
        print(f"  Total frame gagal    : {failed}")

        if not raw_series_paired:
            print("\n  ❌ Semua file gagal diparse.")
            print("  Jalankan: python debug2.py")
            print("  untuk melihat detail error.")
            return

        raw_series_paired.sort(key=lambda x: x[0])
        t_start     = raw_series_paired[0][0]
        t_end       = raw_series_paired[-1][0]
        total_hours = (t_end - t_start).total_seconds() / 3600
        min_hours   = (FORECAST_HORIZON_MINUTES * 2) / 60

        print(f"  Rentang  : "
          f"{(t_start + timedelta(hours=7)).strftime('%H:%M WIB')} – "
          f"{(t_end   + timedelta(hours=7)).strftime('%H:%M WIB')}")
        print(f"  Durasi   : {total_hours:.1f} jam")

        if total_hours < min_hours:
            print(f"\n  ⚠️  Data hanya {total_hours:.1f} jam "
              f"(minimum {min_hours:.0f} jam untuk window+horizon 3+3 jam).")
            print(f"  Lanjut dengan data yang ada ...")

        nc_files = None  # raw_series_paired sudah siap

    elif args.mode == "auto":
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        nc_files  = download_one_day(yesterday, interval_minutes=args.interval)
        if not nc_files:
            print("  Download gagal, beralih ke simulasi")
            nc_files = None

    elif args.mode == "date":
        if not args.date:
            print("  Error: --date diperlukan untuk mode date")
            return
        target = datetime.strptime(args.date, "%Y-%m-%d")
        nc_files = download_one_day(target, interval_minutes=args.interval)

    # ── Step 2: Parse file → time series (skip jika simulasi) ─────────
    print("\n[2] Membangun time series ...")
    if nc_files is not None:
        raw_series_paired = []
        for f in nc_files:
            result = parse_nc_to_ctt(f)
            if result is not None:
                ctt, ts = result
                raw_series_paired.append((ts, ctt))
        raw_series_paired.sort(key=lambda x: x[0])

    print(f"  Total frame raw: {len(raw_series_paired)}")
    if len(raw_series_paired) < 20:
        print("  ⚠️  Frame terlalu sedikit untuk forecasting yang bermakna")
        print("     Minimal 20 frame diperlukan. Lanjut dengan data yang ada ...")

    # Tentukan folder output berdasarkan tanggal data yang digunakan
    if raw_series_paired:
        used_dates = sorted({ts.date() for ts, _ in raw_series_paired})
        if len(used_dates) == 1:
            date_folder = used_dates[0].strftime("%Y%m%d")
        else:
            date_folder = (
                f"{used_dates[0].strftime('%Y%m%d')}_"
                f"{used_dates[-1].strftime('%Y%m%d')}"
            )
    elif args.mode == "date" and args.date:
        date_folder = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y%m%d")
    else:
        date_folder = datetime.now().strftime("%Y%m%d")

    output_dir = OUTPUT_DIR / date_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 3: Load KNN model ─────────────────────────────────────────
    print("\n[3] Memuat model KNN ...")
    knn, scaler = build_knn_model()
    print("  Model siap")

    # ── Step 4: Jalankan eksperimen untuk 3 interval ───────────────────
    print("\n[4] Menjalankan eksperimen forecasting ...")
    all_metrics = {}

    for interval_label, interval_min in INTERVALS.items():
        # Selalu kirim raw_series_paired (sudah berupa list of tuple)
        ts_resampled = build_time_series(raw_series_paired, interval_min)

        min_frames = WINDOW_SIZE[interval_label] + (FORECAST_HORIZON_MINUTES // interval_min) + 1
        if len(ts_resampled) < min_frames:
            print(f"  [{interval_label}] Frame tidak cukup: "
                f"{len(ts_resampled)} < {min_frames} minimum, skip")
            continue

        metrics = run_forecast_experiment(ts_resampled, interval_min, knn, scaler)
        all_metrics[interval_label] = metrics

    # ── Step 5: Simpan PNG per frame + buat GIF ────────────────────────────
    print("\n[5] Membuat visualisasi ...")
    compute_global_color_range(all_metrics)

    for interval_label, metrics in all_metrics.items():
        interval_dir = output_dir / f"interval_{interval_label.replace(' ', '_')}"
        interval_dir.mkdir(exist_ok=True)

        results = metrics.get("results", [])
        if not results:
            print(f"  [{interval_label}] Tidak ada hasil, skip")
            continue

        max_frames = {"10 menit": 48, "30 menit": 24, "60 menit": 12}
        max_f      = max_frames.get(interval_label, 24)
        if len(results) <= max_f:
            selected = results
        else:
            step     = len(results) / max_f
            selected = [results[int(i * step)] for i in range(max_f)]

        print(f"  [{interval_label}] Membuat {len(selected)} frame ...",
            end=" ", flush=True)

        png_ctt   = []   # untuk GIF CTT
        png_class = []   # untuk GIF klasifikasi

        for res in selected:
            ts_str   = res["input_ts"].strftime("%Y%m%d_%H%M")
            out_path = interval_dir / f"forecast_{ts_str}.png"
            path_ctt, path_class = plot_single_frame(res, interval_label, out_path)
            png_ctt.append(path_ctt)
            png_class.append(path_class)

        print("OK")

        fps_map = {"10 menit": 3, "30 menit": 2, "60 menit": 1}
        fps     = fps_map.get(interval_label, 2)

        # GIF untuk CTT
        gif_ctt = output_dir / f"animasi_{interval_label.replace(' ', '_')}_ctt.gif"
        create_gif(png_ctt, gif_ctt, fps=fps)

        # GIF untuk klasifikasi
        gif_cls = output_dir / f"animasi_{interval_label.replace(' ', '_')}_class.gif"
        create_gif(png_class, gif_cls, fps=fps)

    # ── Step 6: Grafik perbandingan ────────────────────────────────────
    print("\n[6] Membuat grafik perbandingan interval ...")
    comparison_path = output_dir / "perbandingan_interval.png"
    plot_comparison_chart(all_metrics, comparison_path)

    # ── Step 7: Simpan ringkasan JSON ─────────────────────────────────
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "area": f"Bandung LAT {LAT_MIN}–{LAT_MAX} LON {LON_MIN}–{LON_MAX}",
        "intervals": {
            lbl: {
                "n_predictions":  m["n_predictions"],
                "acc_ctt_mean":   round(m["acc_ctt_mean"], 2),
                "acc_flood_mean": round(m["acc_flood_mean"], 2),
                "mae_mean":       round(m["mae_mean"], 2),
            }
            for lbl, m in all_metrics.items()
        }
    }

    summary_path = output_dir / "ringkasan_forecasting.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Step 8: Print ringkasan ────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RINGKASAN HASIL FORECASTING")
    print(f"{'='*65}")
    print(f"  {'Interval':<12} {'Akurasi Awan':>14} {'Akurasi Banjir':>16} {'MAE CTT':>10}")
    print(f"  {'-'*55}")
    for lbl, m in all_metrics.items():
        print(f"  {lbl:<12} {m['acc_ctt_mean']:>13.1f}% "
              f"{m['acc_flood_mean']:>15.1f}%  "
              f"{m['mae_mean']:>8.2f} K")

    if all_metrics:
        best = max(all_metrics.items(), key=lambda x: x[1]["acc_ctt_mean"])
        print(f"\n  ★ Interval terbaik: {best[0]} "
              f"(Akurasi {best[1]['acc_ctt_mean']:.1f}%)")

    print(f"\n  Output tersimpan di: {output_dir.resolve()}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()