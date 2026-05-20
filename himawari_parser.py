"""
himawari_parser.py
==================
Modul membaca dan memproses file NetCDF (.nc) dari Himawari-8.

Cara kerja:
    1. Baca file .nc menggunakan netCDF4
    2. Ekstrak variabel brightness temperature (kanal B13 = 10.4μm)
    3. Crop ke area studi (bounding box Jawa Barat)
    4. Konversi dari brightness temperature ke Cloud Top Temperature (CTT)
    5. Hasilkan array 2D siap digunakan oleh KNN

Kanal Himawari-8 yang relevan:
    B13 (10.4 μm) — Infrared Thermal, digunakan untuk Cloud Top Temperature
    B08 (6.2 μm)  — Water Vapor, opsional sebagai fitur tambahan
    B03 (0.64 μm) — Visible, untuk daytime cloud detection

Dependencies:
    pip install netCDF4 numpy scipy
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re

# Import netCDF4 dengan graceful fallback ke simulasi jika belum ada
try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False
    print("  [WARNING] netCDF4 belum terinstall. Jalankan: pip install netCDF4")
    print("            Saat ini menggunakan data simulasi sebagai fallback.\n")

# ─── KONFIGURASI AREA STUDI ──────────────────────────────────────────────────
LAT_MIN, LAT_MAX = -8.0, -5.0
LON_MIN, LON_MAX = 105.0, 109.0
OUTPUT_GRID_SIZE = 50   # grid output setelah crop & resample

# Threshold suhu awan (Kelvin) — sesuai paper
CTT_CLASS_THRESHOLDS = {
    0: (270, 300),   # Tinggi  → Tidak Hujan
    1: (230, 270),   # Sedang  → Mendung
    2: (180, 230),   # Rendah  → Hujan
}

# Konstanta konversi brightness temperature Himawari-8 kanal B13
# (Planck function constants untuk 10.4 μm)
PLANCK_C1 = 1.19104e-5   # mW / (m² sr cm⁻⁴)
PLANCK_C2 = 1.43877      # K cm
WAVENUMBER_B13 = 960.9   # cm⁻¹  (untuk B13 / 10.4 μm)
# ────────────────────────────────────────────────────────────────────────────


def brightness_to_ctt(bt_kelvin: np.ndarray) -> np.ndarray:
    """
    Konversi Brightness Temperature ke Cloud Top Temperature (CTT).
    Untuk kanal infrared B13 Himawari-8, perbedaannya kecil (<2K)
    namun tetap dikoreksi dengan koefisien kalibrasi AHI.
    
    Koreksi linear berdasarkan dokumen kalibrasi JAXA:
        CTT ≈ 0.9991 × BT + 0.3
    """
    ctt = 0.9991 * bt_kelvin + 0.3
    return np.clip(ctt, 150, 330)   # clip ke rentang fisik yang wajar


def radiance_to_brightness_temp(radiance: np.ndarray) -> np.ndarray:
    """
    Konversi radiance (mW m⁻² sr⁻¹ cm) ke brightness temperature (K).
    Menggunakan inverse Planck function untuk wavenumber B13.
    
    Formula: BT = C2 × ν / ln(C1 × ν³ / L + 1)
    """
    # Hindari log(0) atau nilai negatif
    radiance_safe = np.maximum(radiance, 0.001)
    bt = (PLANCK_C2 * WAVENUMBER_B13 /
          np.log(PLANCK_C1 * WAVENUMBER_B13**3 / radiance_safe + 1))
    return bt.astype(np.float32)


def crop_to_region(data_2d: np.ndarray,
                   lat_array: np.ndarray,
                   lon_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop array 2D ke bounding box area studi.
    
    Args:
        data_2d:   Array 2D (lat × lon)
        lat_array: Array 1D latitude (menurun atau menaik)
        lon_array: Array 1D longitude
    
    Returns:
        (data_cropped, lat_cropped, lon_cropped)
    """
    # Pastikan lat menurun (dari utara ke selatan) adalah standar HSD
    if lat_array[0] < lat_array[-1]:
        lat_array = lat_array[::-1]
        data_2d = data_2d[::-1, :]
    
    lat_mask = (lat_array >= LAT_MIN) & (lat_array <= LAT_MAX)
    lon_mask = (lon_array >= LON_MIN) & (lon_array <= LON_MAX)
    
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"Area studi tidak ada dalam data. "
            f"Lat range data: {lat_array.min():.1f}–{lat_array.max():.1f}, "
            f"Lon range data: {lon_array.min():.1f}–{lon_array.max():.1f}"
        )
    
    data_cropped = data_2d[lat_idx[0]:lat_idx[-1]+1,
                           lon_idx[0]:lon_idx[-1]+1]
    lat_cropped = lat_array[lat_mask]
    lon_cropped = lon_array[lon_mask]
    
    return data_cropped, lat_cropped, lon_cropped


def resample_grid(data: np.ndarray,
                  target_rows: int = OUTPUT_GRID_SIZE,
                  target_cols: int = OUTPUT_GRID_SIZE) -> np.ndarray:
    """
    Resample grid ke ukuran output standar menggunakan interpolasi bilinear.
    Himawari-8 HSD resolusi 0.02° (~2 km), setelah crop ke Jawa Barat
    hasilnya sekitar 150×200 piksel, kita resample ke 50×50.
    """
    from scipy.ndimage import zoom
    zoom_r = target_rows / data.shape[0]
    zoom_c = target_cols / data.shape[1]
    return zoom(data, (zoom_r, zoom_c), order=1).astype(np.float32)


def parse_nc_file(nc_path: Path,
                  channel: str = "B13",
                  verbose: bool = True) -> dict | None:
    """
    Baca satu file NetCDF Himawari-8 HSD dan ekstrak CTT.
    
    Args:
        nc_path: Path ke file .nc
        channel: Kanal yang dibaca, default B13 (10.4 μm / IR thermal)
        verbose: Print info saat proses
    
    Returns dict berisi:
        {
            "ctt":       np.ndarray (50×50) suhu awan dalam Kelvin,
            "lat":       np.ndarray (50,)   latitude,
            "lon":       np.ndarray (50,)   longitude,
            "timestamp": datetime           waktu observasi (UTC),
            "filename":  str                nama file sumber,
            "valid_pixels": int             jumlah piksel valid (non-masked),
        }
        atau None jika file tidak bisa dibaca.
    """
    if not NETCDF4_AVAILABLE:
        if verbose:
            print(f"  [FALLBACK] netCDF4 tidak ada, menggunakan simulasi untuk {nc_path.name}")
        return _simulate_parse(nc_path)
    
    if not nc_path.exists():
        print(f"  [ERROR] File tidak ditemukan: {nc_path}")
        return None
    
    if verbose:
        print(f"  [PARSE] Membaca: {nc_path.name} ...", end=" ", flush=True)
    
    try:
        with nc.Dataset(nc_path, "r") as ds:
            # ── 1. Ekstrak koordinat ──────────────────────────────────────
            # Format HSD Himawari-8: variabel 'latitude' dan 'longitude'
            # Ukuran full disk: 2401×2401 (untuk R10, resolusi 4 km)
            if "latitude" in ds.variables:
                lat_raw = ds.variables["latitude"][:]
                lon_raw = ds.variables["longitude"][:]
            elif "lat" in ds.variables:
                lat_raw = ds.variables["lat"][:]
                lon_raw = ds.variables["lon"][:]
            else:
                # Hitung dari atribut navigasi (beberapa varian file)
                nav = ds.variables.get("Navigation", None)
                if nav is None:
                    raise KeyError("Variabel koordinat tidak ditemukan dalam file .nc")
                lat_raw = nav["Latitude"][:]
                lon_raw = nav["Longitude"][:]
            
            # ── 2. Ekstrak data brightness temperature ────────────────────
            # Nama variabel bervariasi tergantung versi file HSD:
            # 'brightness_temperature', 'BT', 'tbb', 'albedo'
            bt_var_names = [
                f"brightness_temperature_{channel}",
                f"tbb_{channel.lower()}",
                "brightness_temperature",
                "tbb",
                "BT",
            ]
            bt_raw = None
            for vname in bt_var_names:
                if vname in ds.variables:
                    bt_raw = ds.variables[vname][:]
                    if verbose:
                        print(f"(variabel: '{vname}')", end=" ", flush=True)
                    break
            
            if bt_raw is None:
                # Fallback: cari variabel dengan 'temp' atau 'tbb' dalam nama
                for vname in ds.variables:
                    if any(k in vname.lower() for k in ["temp", "tbb", "bt"]):
                        bt_raw = ds.variables[vname][:]
                        if verbose:
                            print(f"(variabel fallback: '{vname}')", end=" ", flush=True)
                        break
            
            if bt_raw is None:
                raise KeyError(f"Tidak ada variabel brightness temperature dalam {nc_path.name}")
            
            # ── 3. Handle masked array ────────────────────────────────────
            if hasattr(bt_raw, "filled"):
                # Isi missing value dengan suhu tinggi (= tidak ada awan)
                bt_raw = bt_raw.filled(fill_value=295.0)
            bt_raw = np.array(bt_raw, dtype=np.float32)
            
            # ── 4. Scale factor & offset (jika ada) ──────────────────────
            bt_var = None
            for vname in bt_var_names:
                if vname in ds.variables:
                    bt_var = ds.variables[vname]
                    break
            if bt_var is not None:
                scale = getattr(bt_var, "scale_factor", 1.0)
                offset = getattr(bt_var, "add_offset", 0.0)
                fill_val = getattr(bt_var, "_FillValue", -32768)
                # Mask fill values sebelum apply scale
                bt_raw = np.where(bt_raw == fill_val, np.nan, bt_raw)
                bt_raw = bt_raw * scale + offset
                bt_raw = np.where(np.isnan(bt_raw), 295.0, bt_raw)
            
            # ── 5. Ekstrak timestamp dari nama file ───────────────────────
            match = re.search(r"H08_(\d{8})_(\d{4})", nc_path.name)
            if match:
                ts_str = match.group(1) + match.group(2)
                timestamp = datetime.strptime(ts_str, "%Y%m%d%H%M")
            else:
                timestamp = datetime.utcnow()
            
            # ── 6. Crop ke area studi ─────────────────────────────────────
            # lat_raw bisa berupa array 1D atau 2D
            if lat_raw.ndim == 2:
                lat_1d = lat_raw[:, 0]
                lon_1d = lon_raw[0, :]
            else:
                lat_1d = np.array(lat_raw)
                lon_1d = np.array(lon_raw)
            
            bt_cropped, lat_c, lon_c = crop_to_region(bt_raw, lat_1d, lon_1d)
            
            # ── 7. Konversi BT → CTT ──────────────────────────────────────
            ctt = brightness_to_ctt(bt_cropped)
            
            # ── 8. Resample ke OUTPUT_GRID_SIZE × OUTPUT_GRID_SIZE ────────
            ctt_resampled = resample_grid(ctt)
            lat_resampled = np.linspace(lat_c.min(), lat_c.max(), OUTPUT_GRID_SIZE)
            lon_resampled = np.linspace(lon_c.min(), lon_c.max(), OUTPUT_GRID_SIZE)
            
            valid_pct = np.sum(ctt_resampled > 150) / ctt_resampled.size * 100
            
            if verbose:
                print(f"OK | CTT: {ctt_resampled.min():.1f}–{ctt_resampled.max():.1f} K | Valid: {valid_pct:.1f}%")
            
            return {
                "ctt":          ctt_resampled,
                "lat":          lat_resampled,
                "lon":          lon_resampled,
                "timestamp":    timestamp,
                "filename":     nc_path.name,
                "valid_pixels": int(np.sum(ctt_resampled > 150)),
            }
    
    except Exception as e:
        if verbose:
            print(f"ERROR — {e}")
        return None


def parse_multiple_files(nc_paths: list[Path],
                          channel: str = "B13") -> list[dict]:
    """
    Parse beberapa file .nc sekaligus, cocok untuk batch processing.
    
    Returns:
        List dict hasil parse (file yang gagal dilewati/None dibuang)
    """
    results = []
    for i, p in enumerate(nc_paths, 1):
        print(f"  [{i}/{len(nc_paths)}]", end=" ")
        result = parse_nc_file(p, channel=channel)
        if result is not None:
            results.append(result)
    
    print(f"\n  ✓ Berhasil parse: {len(results)}/{len(nc_paths)} file")
    return results


def _simulate_parse(nc_path: Path) -> dict:
    """
    Fallback: generate data simulasi yang strukturnya sama
    dengan output parse_nc_file sungguhan. Dipakai saat netCDF4
    belum terinstall atau file .nc tidak tersedia.
    """
    # Ekstrak seed dari nama file agar konsisten
    match = re.search(r"H08_(\d{8})_(\d{4})", str(nc_path))
    if match:
        seed = int(match.group(1)[-4:] + match.group(2)) % (2**32)
        ts_str = match.group(1) + match.group(2)
        timestamp = datetime.strptime(ts_str, "%Y%m%d%H%M")
    else:
        seed = 42
        timestamp = datetime.utcnow()
    
    rng = np.random.default_rng(seed)
    G = OUTPUT_GRID_SIZE
    
    ctt = 260 + 20 * rng.standard_normal((G, G))
    # Tambah beberapa sel konvektif
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.integers(5, G-5, size=2)
        r = rng.integers(3, 8)
        for i in range(G):
            for j in range(G):
                d = np.sqrt((i-cx)**2 + (j-cy)**2)
                if d < r:
                    ctt[i, j] -= 40 * (1 - d/r)
    ctt = np.clip(ctt, 180, 300).astype(np.float32)
    
    return {
        "ctt":          ctt,
        "lat":          np.linspace(LAT_MIN, LAT_MAX, G),
        "lon":          np.linspace(LON_MIN, LON_MAX, G),
        "timestamp":    timestamp,
        "filename":     nc_path.name,
        "valid_pixels": int(G * G * 0.9),
    }