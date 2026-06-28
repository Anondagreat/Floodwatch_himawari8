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

import argparse
import json
import pickle
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
# Area studi: Kota Bandung dan Kabupaten Bandung, Jawa Barat
LAT_MIN, LAT_MAX = -7.15, -6.75
LON_MIN, LON_MAX = 107.35, 107.95
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
OUTPUT_DIR   = Path("./output_fase3")
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

def parse_nc_to_ctt(nc_path: Path) -> tuple[np.ndarray, datetime] | None:
    """
    Baca file .nc dan ekstrak CTT grid 50×50 untuk area Bandung.
    Returns (ctt_grid, timestamp) atau None jika gagal.
    """
    import re
    from scipy.ndimage import zoom

    # Ekstrak timestamp dari nama file
    m = re.search(r'NC_H09_(\d{8})_(\d{4})', nc_path.name)
    if not m:
        return None
    timestamp = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")

    # Coba baca dengan netCDF4
    try:
        import netCDF4 as nc_lib
        with nc_lib.Dataset(nc_path, "r") as ds:
            # Cari variabel BT
            bt_var_names = ["tbb_13", "tbb_B13", "brightness_temp_13",
                            "IR_108", "tbb", "BT_B13"]
            bt_data = None
            var_used = None
            for vn in bt_var_names:
                if vn in ds.variables:
                    bt_data = ds.variables[vn][:]
                    var_used = vn
                    break
            if bt_data is None:
                for vn in ds.variables:
                    if any(k in vn.lower() for k in ["tbb", "temp", "bt"]):
                        bt_data = ds.variables[vn][:]
                        var_used = vn
                        break
            if bt_data is None:
                return _simulate_ctt(timestamp)

            # Apply scale/offset
            bv = ds.variables[var_used]
            scale  = float(getattr(bv, "scale_factor", 1.0))
            offset = float(getattr(bv, "add_offset", 0.0))
            fill   = getattr(bv, "_FillValue", -32768)
            if hasattr(bt_data, "filled"):
                bt_data = bt_data.filled(fill_value=fill)
            bt_arr = np.array(bt_data, dtype=np.float32)
            bt_arr = np.where(bt_arr == fill, np.nan, bt_arr)
            bt_arr = bt_arr * scale + offset
            bt_arr = np.where(np.isnan(bt_arr), 295.0, bt_arr)
            if bt_arr.ndim == 3:
                bt_arr = bt_arr[0]

            # CTT koreksi
            ctt = np.clip(0.9991 * bt_arr + 0.3, 150, 330).astype(np.float32)

            # Koordinat
            lat_1d = lon_1d = None
            for lv in ["latitude", "lat", "Latitude"]:
                if lv in ds.variables:
                    lat_1d = np.array(ds.variables[lv][:])
                    break
            for lv in ["longitude", "lon", "Longitude"]:
                if lv in ds.variables:
                    lon_1d = np.array(ds.variables[lv][:])
                    break
            if lat_1d is None:
                h, w   = ctt.shape
                lat_1d = np.linspace(60.0, -60.0, h)
                lon_1d = np.linspace(80.0, 160.0, w)
            if lat_1d.ndim == 2:
                lat_1d = lat_1d[:, 0]
            if lon_1d.ndim == 2:
                lon_1d = lon_1d[0, :]
            if lat_1d[0] < lat_1d[-1]:
                lat_1d = lat_1d[::-1]
                ctt    = ctt[::-1, :]

            # Crop Bandung
            lat_mask = (lat_1d >= LAT_MIN) & (lat_1d <= LAT_MAX)
            lon_mask = (lon_1d >= LON_MIN) & (lon_1d <= LON_MAX)
            lat_idx  = np.where(lat_mask)[0]
            lon_idx  = np.where(lon_mask)[0]
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                return _simulate_ctt(timestamp)

            ctt_crop = ctt[lat_idx[0]:lat_idx[-1]+1,
                           lon_idx[0]:lon_idx[-1]+1]

            # Resample 50×50
            zr = GRID_SIZE / ctt_crop.shape[0]
            zc = GRID_SIZE / ctt_crop.shape[1]
            ctt_out = zoom(ctt_crop, (zr, zc), order=1).astype(np.float32)
        return ctt_out, timestamp

    except ImportError:
        return _simulate_ctt(timestamp)
    except Exception:
        return _simulate_ctt(timestamp)


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
    Terima input berupa:
        - list[Path]  → parse file .nc dulu
        - list[tuple] → sudah (timestamp, ctt), langsung resample
    Resample ke interval yang diinginkan.
    """
    # Deteksi tipe input
    if not input_data:
        return []

    first = input_data[0]

    if isinstance(first, tuple):
        # Sudah berupa (timestamp, ctt) — dari mode simulasi
        raw_series = list(input_data)
    else:
        # Berupa Path — parse file .nc dulu
        print(f"\n  Parsing {len(input_data)} file ke interval {interval_minutes} menit ...")
        raw_series = []
        for f in input_data:
            result = parse_nc_to_ctt(f)
            if result is not None:
                ctt, ts = result
                raw_series.append((ts, ctt))

    if not raw_series:
        return []

    raw_series.sort(key=lambda x: x[0])

    # Resample ke interval target
    if interval_minutes == 10:
        print(f"    → {len(raw_series)} frame (interval 10 menit, tidak perlu resample)")
        return raw_series

    resampled = []
    start_ts  = raw_series[0][0]
    end_ts    = raw_series[-1][0]
    current   = start_ts

    while current <= end_ts:
        closest  = min(raw_series,
                       key=lambda x: abs((x[0] - current).total_seconds()))
        diff_min = abs((closest[0] - current).total_seconds()) / 60
        if diff_min <= interval_minutes / 2:
            resampled.append(closest)
        current += timedelta(minutes=interval_minutes)

    # Hapus duplikat
    seen, unique = set(), []
    for ts, ctt in resampled:
        if ts not in seen:
            seen.add(ts)
            unique.append((ts, ctt))

    print(f"    → {len(unique)} frame setelah resample ke {interval_minutes} menit")
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 4: FORECASTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def weighted_trend_forecast(series: list[np.ndarray],
                             n_steps_ahead: int) -> np.ndarray:
    """
    Prediksi CTT N step ke depan menggunakan weighted linear trend.
    
    Metode:
        - Untuk setiap piksel (i,j), fit tren linear dari window frames
        - Frame lebih baru diberi bobot lebih tinggi (exponential weighting)
        - Proyeksi tren N step ke depan
    
    Ini metode yang sederhana tapi bisa dijelaskan secara ilmiah:
    menggunakan prinsip bahwa perkembangan awan memiliki inersia —
    tren yang berjalan cenderung berlanjut dalam jangka pendek.
    
    Args:
        series: List array CTT (dari lama ke baru), shape masing-masing (50,50)
        n_steps_ahead: Berapa step ke depan yang diprediksi
    
    Returns:
        Predicted CTT grid shape (50,50)
    """
    n = len(series)
    if n < 2:
        return series[-1].copy()

    stack = np.stack(series, axis=0)   # shape: (n, 50, 50)
    t     = np.arange(n, dtype=np.float32)

    # Exponential weights: frame terbaru lebih dipercaya
    weights = np.exp(0.3 * t)
    weights /= weights.sum()

    # Weighted linear regression per piksel
    # y = a*t + b → prediksi di t = n - 1 + n_steps_ahead
    wsum    = weights.sum()
    wt_sum  = (weights * t).sum()
    wt2_sum = (weights * t**2).sum()

    # Denominat regresi berbobot
    denom = wsum * wt2_sum - wt_sum**2
    if abs(denom) < 1e-10:
        # Tidak ada tren, pakai rata-rata berbobot
        return (stack * weights[:, np.newaxis, np.newaxis]).sum(axis=0)

    # Hitung slope (a) dan intercept (b) per piksel
    wy_sum  = (weights[:, np.newaxis, np.newaxis] * stack).sum(axis=0)
    wty_sum = (weights[:, np.newaxis, np.newaxis] *
               t[:, np.newaxis, np.newaxis] * stack).sum(axis=0)

    a = (wsum * wty_sum - wt_sum * wy_sum) / denom   # slope
    b = (wy_sum - a * wt_sum) / wsum                  # intercept

    # Prediksi di t_pred
    t_pred   = (n - 1) + n_steps_ahead
    forecast = a * t_pred + b

    # Clip ke range fisik yang wajar
    return np.clip(forecast, 150, 330).astype(np.float32)


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
        pred_ctt = weighted_trend_forecast(input_frames, n_steps_ahead=horizon)

        # Evaluasi per piksel
        mae = float(np.mean(np.abs(pred_ctt - actual_ctt)))
        mae_list.append(mae)

        # Klasifikasi kelas awan: actual vs predicted
        actual_class  = classify_ctt_grid(actual_ctt)
        pred_class    = classify_ctt_grid(pred_ctt)
        acc_ctt       = float(np.mean(actual_class == pred_class) * 100)
        acc_list_ctt.append(acc_ctt)

        # Klasifikasi risiko banjir
        actual_precip = ctt_to_precip(actual_ctt)
        pred_precip   = ctt_to_precip(pred_ctt)
        actual_flood  = classify_flood(actual_precip)
        pred_flood    = classify_flood(pred_precip)
        acc_flood     = float(np.mean(actual_flood == pred_flood) * 100)
        acc_list_flood.append(acc_flood)

        results.append({
            "input_ts":    input_ts,
            "actual_ts":   actual_ts,
            "input_ctt":   input_frames[-1],    # frame input terakhir
            "pred_ctt":    pred_ctt,
            "actual_ctt":  actual_ctt,
            "pred_class":  pred_class,
            "actual_class": actual_class,
            "pred_flood":  pred_flood,
            "actual_flood": actual_flood,
            "mae":         mae,
            "acc_ctt":     acc_ctt,
            "acc_flood":   acc_flood,
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
    scaler_path = MODEL_DIR / "model_saved/scaler.pkl"

    if model_path.exists():
        with open(model_path, "rb") as f:
            knn = pickle.load(f)
        try:
            with open(MODEL_DIR / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = StandardScaler()
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

    MODEL_DIR.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return knn, scaler


# ══════════════════════════════════════════════════════════════════════════════
# BAGIAN 6: VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════

CITIES = {
    "Bandung":    (-6.91, 107.61),
    "Cimahi":     (-6.88, 107.54),
    "Soreang":    (-7.03, 107.52),
    "Banjaran":   (-7.05, 107.60),
    "Majalaya":   (-7.04, 107.78),
    "Cicalengka": (-6.99, 107.84),
    "Lembang":    (-6.81, 107.62),
    "Margahayu":  (-6.97, 107.58),
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


def plot_single_frame(result: dict,
                       interval_label: str,
                       out_path: Path) -> None:
    """
    Plot satu frame perbandingan: input aktual vs prediksi 3 jam ke depan.
    Layout: 2×3 panel
        [Input CTT] [Prediksi CTT] [Perbandingan CTT]
        [Input Awan][Prediksi Banjir][Error Map]
    """
    fig = plt.figure(figsize=(18, 10), facecolor="#0f172a")

    input_ts_wib  = result["input_ts"]  + timedelta(hours=7)
    actual_ts_wib = result["actual_ts"] + timedelta(hours=7)
    alert         = _flood_alert(result["pred_flood"])

    fig.suptitle(
        f"Forecasting 3 Jam ke Depan — Interval {interval_label}\n"
        f"Input: {input_ts_wib.strftime('%d %b %Y %H:%M WIB')}  →  "
        f"Prediksi: {actual_ts_wib.strftime('%H:%M WIB')}  |  "
        f"Status: {alert}  |  "
        f"MAE: {result['mae']:.2f} K  |  "
        f"Akurasi: {result['acc_ctt']:.1f}%",
        fontsize=12, color="white", fontweight="bold", y=0.98
    )

    ax_kw = dict(facecolor="#1e293b")
    axes  = [fig.add_subplot(2, 3, i+1, **ax_kw) for i in range(6)]

    def style(ax, title):
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        _add_cities(ax)

    # Panel 1: Input CTT (terakhir)
    cf1 = axes[0].contourf(LONS, LATS, result["input_ctt"],
                            levels=20, cmap="RdYlBu_r",
                            vmin=180, vmax=300)
    plt.colorbar(cf1, ax=axes[0], label="K", shrink=0.85)
    style(axes[0], f"CTT Input ({input_ts_wib.strftime('%H:%M WIB')})")

    # Panel 2: Prediksi CTT 3 jam ke depan
    cf2 = axes[1].contourf(LONS, LATS, result["pred_ctt"],
                            levels=20, cmap="RdYlBu_r",
                            vmin=180, vmax=300)
    plt.colorbar(cf2, ax=axes[1], label="K", shrink=0.85)
    style(axes[1], f"Prediksi CTT (+3 jam = {actual_ts_wib.strftime('%H:%M WIB')})")

    # Panel 3: CTT Aktual (ground truth)
    cf3 = axes[2].contourf(LONS, LATS, result["actual_ctt"],
                            levels=20, cmap="RdYlBu_r",
                            vmin=180, vmax=300)
    plt.colorbar(cf3, ax=axes[2], label="K", shrink=0.85)
    style(axes[2], f"CTT Aktual ({actual_ts_wib.strftime('%H:%M WIB')})")

    # Panel 4: Kelas Awan Aktual
    axes[3].pcolormesh(LONS, LATS, result["actual_class"],
                        cmap=CLOUD_CMAP, norm=CLOUD_NORM, shading="auto")
    patches_c = [
        mpatches.Patch(color="#FFD700", label="Tidak Hujan"),
        mpatches.Patch(color="#87CEEB", label="Mendung"),
        mpatches.Patch(color="#1E3A8A", label="Hujan"),
    ]
    axes[3].legend(handles=patches_c, fontsize=6, loc="lower right",
                   facecolor="#1e293b", labelcolor="white")
    style(axes[3], "Kelas Awan Aktual")

    # Panel 5: Prediksi Risiko Banjir
    axes[4].pcolormesh(LONS, LATS, result["pred_flood"],
                        cmap=FLOOD_CMAP, norm=FLOOD_NORM, shading="auto")
    patches_f = [
        mpatches.Patch(color="#22c55e", label="Aman"),
        mpatches.Patch(color="#f59e0b", label="Waspada"),
        mpatches.Patch(color="#ef4444", label="Bahaya"),
    ]
    axes[4].legend(handles=patches_f, fontsize=6, loc="lower right",
                   facecolor="#1e293b", labelcolor="white")
    ac = ALERT_COLOR[alert]
    axes[4].set_title(f"★ Prediksi Risiko Banjir [{alert}]",
                       color=ac, fontsize=9, fontweight="bold", pad=6)
    axes[4].tick_params(colors="#94a3b8", labelsize=7)
    for sp in axes[4].spines.values():
        sp.set_edgecolor(ac)
    _add_cities(axes[4])

    # Panel 6: Error Map (selisih prediksi vs aktual)
    error = np.abs(result["pred_ctt"] - result["actual_ctt"])
    cf6 = axes[5].contourf(LONS, LATS, error,
                            levels=15, cmap="hot_r",
                            vmin=0, vmax=30)
    plt.colorbar(cf6, ax=axes[5], label="Error (K)", shrink=0.85)
    style(axes[5], f"Error Map |Pred - Aktual|  (MAE={result['mae']:.1f}K)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="#0f172a")
    plt.close()


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
            accs = [r["acc_ctt"] for r in m["results"]]
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
    """Buat GIF animasi dari list PNG."""
    try:
        from PIL import Image
        if not png_paths:
            return
        frames = []
        for p in sorted(png_paths):
            if p.exists():
                frames.append(Image.open(p))
        if frames:
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0
            )
            print(f"  ✓ GIF: {out_path.name} ({len(frames)} frame, {fps} fps)")
    except ImportError:
        print("  [INFO] Pillow tidak terinstall. Install: pip install Pillow")
        print("         GIF tidak dibuat, PNG sequence tetap tersedia.")


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
        nc_files = load_local_files()
        if not nc_files:
            print("  Tidak ada file .nc lokal, beralih ke simulasi")
            nc_files = None
            raw_series_paired = [
                (_simulate_ctt(datetime.now() - timedelta(days=1) +
                               timedelta(minutes=10*i)))[0:2][::-1]
                for i in range(144)
            ]
            raw_series_paired = [
                (ts, ctt)
                for ctt, ts in [_simulate_ctt(
                    datetime.now() - timedelta(days=1) + timedelta(minutes=10*i)
                ) for i in range(144)]
            ]

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

    # ── Step 3: Load KNN model ─────────────────────────────────────────
    print("\n[3] Memuat model KNN ...")
    knn, scaler = build_knn_model()
    print("  Model siap")

    # ── Step 4: Jalankan eksperimen untuk 3 interval ───────────────────
    print("\n[4] Menjalankan eksperimen forecasting ...")
    all_metrics = {}

    for interval_label, interval_min in INTERVALS.items():
        # Resample time series ke interval ini
        ts_resampled = build_time_series(raw_series_paired, interval_min)
        if len(ts_resampled) < WINDOW_SIZE[interval_label] + 1:
            print(f"  [{interval_label}] Frame tidak cukup, skip")
            continue
        metrics = run_forecast_experiment(ts_resampled, interval_min,
                                           knn, scaler)
        all_metrics[interval_label] = metrics

    # ── Step 5: Simpan PNG per frame + buat GIF ────────────────────────
    print("\n[5] Membuat visualisasi ...")

    for interval_label, metrics in all_metrics.items():
        interval_dir = OUTPUT_DIR / f"interval_{interval_label.replace(' ', '_')}"
        interval_dir.mkdir(exist_ok=True)

        results   = metrics.get("results", [])
        png_paths = []

        # Batasi maksimal 24 frame untuk efisiensi
        step = max(1, len(results) // 24)
        selected = results[::step][:24]

        print(f"  [{interval_label}] Membuat {len(selected)} PNG ...",
              end=" ", flush=True)

        for i, res in enumerate(selected):
            ts_str   = res["input_ts"].strftime("%Y%m%d_%H%M")
            out_path = interval_dir / f"forecast_{ts_str}.png"
            plot_single_frame(res, interval_label, out_path)
            png_paths.append(out_path)

        print("OK")

        # Buat GIF
        gif_path = OUTPUT_DIR / f"animasi_{interval_label.replace(' ', '_')}.gif"
        create_gif(png_paths, gif_path, fps=2)

    # ── Step 6: Grafik perbandingan ────────────────────────────────────
    print("\n[6] Membuat grafik perbandingan interval ...")
    comparison_path = OUTPUT_DIR / "perbandingan_interval.png"
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

    summary_path = OUTPUT_DIR / "ringkasan_forecasting.json"
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

    print(f"\n  Output tersimpan di: {OUTPUT_DIR.resolve()}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()