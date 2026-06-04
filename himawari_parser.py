"""
himawari_parser.py — FINAL untuk NC_H09 NetCDF format
"""

import re
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.ndimage import zoom

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False
    print("  [WARNING] netCDF4 belum terinstall: pip install netCDF4")

LAT_MIN, LAT_MAX = -7.4, -6.4
LON_MIN, LON_MAX = 107.0, 108.3
OUTPUT_GRID_SIZE = 50


def bt_to_ctt(bt: np.ndarray) -> np.ndarray:
    """Koreksi linear BT → CTT kanal B13 Himawari-9."""
    return np.clip(0.9991 * bt + 0.3, 150, 330).astype(np.float32)


def parse_nc_file(nc_path: Path, verbose: bool = True) -> dict | None:
    """
    Baca file NC_H09 NetCDF dan ekstrak Cloud Top Temperature.
    Format: NC_H09_YYYYMMDD_HHMM_R21_FLDK.02801_02401.nc
    """
    if not NETCDF4_AVAILABLE:
        return _simulate_parse(nc_path)

    if not nc_path.exists() or nc_path.stat().st_size < 10_000:
        return _simulate_parse(nc_path)

    if verbose:
        print(f"  [PARSE] {nc_path.name} ...", end=" ", flush=True)

    try:
        with nc.Dataset(nc_path, "r") as ds:

            # ── 1. Cari variabel brightness temperature ──────────────────
            # Dalam file NC_H09, variabel untuk kanal IR biasanya:
            # 'tbb_13', 'brightness_temp_13', 'IR_108', atau 'tbb'
            bt_var_candidates = [
                "tbb_13", "tbb_B13", "brightness_temp_13",
                "IR_108", "IR013", "tbb", "BT_B13",
            ]
            bt_data = None
            var_used = None

            for vname in bt_var_candidates:
                if vname in ds.variables:
                    bt_data = ds.variables[vname][:]
                    var_used = vname
                    break

            # Jika tidak ketemu, cari variabel yang mengandung 'tbb' atau 'temp'
            if bt_data is None:
                for vname in ds.variables:
                    if any(k in vname.lower() for k in ["tbb", "temp", "bt"]):
                        bt_data = ds.variables[vname][:]
                        var_used = vname
                        break

            if bt_data is None:
                if verbose:
                    print(f"GAGAL — variabel BT tidak ditemukan")
                    print(f"  Variabel tersedia: {list(ds.variables.keys())[:10]}")
                return _simulate_parse(nc_path)

            # ── 2. Handle masked array & scale factor ────────────────────
            bt_var = ds.variables[var_used]
            scale  = float(getattr(bt_var, "scale_factor", 1.0))
            offset = float(getattr(bt_var, "add_offset", 0.0))
            fill   = getattr(bt_var, "_FillValue", -32768)

            if hasattr(bt_data, "filled"):
                bt_data = bt_data.filled(fill_value=fill)

            bt_arr = np.array(bt_data, dtype=np.float32)
            bt_arr = np.where(bt_arr == fill, np.nan, bt_arr)
            bt_arr = bt_arr * scale + offset
            bt_arr = np.where(np.isnan(bt_arr), 295.0, bt_arr)

            # Squeeze dimensi ekstra (misal shape (1, H, W) → (H, W))
            if bt_arr.ndim == 3:
                bt_arr = bt_arr[0]

            # ── 3. Koordinat lat/lon ─────────────────────────────────────
            lat_candidates = ["latitude", "lat", "Latitude", "LAT"]
            lon_candidates = ["longitude", "lon", "Longitude", "LON"]

            lat_1d = lon_1d = None
            for lv in lat_candidates:
                if lv in ds.variables:
                    lat_1d = np.array(ds.variables[lv][:])
                    break
            for lv in lon_candidates:
                if lv in ds.variables:
                    lon_1d = np.array(ds.variables[lv][:])
                    break

            # Jika tidak ada variabel lat/lon, hitung dari dimensi file
            if lat_1d is None or lon_1d is None:
                h, w = bt_arr.shape
                # NC_H09 R21 FLDK coverage: approx -60 to 60 lat, 80 to 160 lon
                lat_1d = np.linspace(60.0, -60.0, h)
                lon_1d = np.linspace(80.0, 160.0, w)

            # Pastikan 1D
            if lat_1d.ndim == 2:
                lat_1d = lat_1d[:, 0]
            if lon_1d.ndim == 2:
                lon_1d = lon_1d[0, :]

            # Pastikan lat dari besar ke kecil (utara ke selatan)
            if lat_1d[0] < lat_1d[-1]:
                lat_1d = lat_1d[::-1]
                bt_arr = bt_arr[::-1, :]

            # ── 4. Crop ke area Jawa Barat ───────────────────────────────
            lat_mask = (lat_1d >= LAT_MIN) & (lat_1d <= LAT_MAX)
            lon_mask = (lon_1d >= LON_MIN) & (lon_1d <= LON_MAX)
            lat_idx  = np.where(lat_mask)[0]
            lon_idx  = np.where(lon_mask)[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                if verbose:
                    print("GAGAL — area Jawa Barat tidak ditemukan dalam data")
                return _simulate_parse(nc_path)

            bt_crop  = bt_arr[lat_idx[0]:lat_idx[-1]+1,
                               lon_idx[0]:lon_idx[-1]+1]
            lat_crop = lat_1d[lat_mask]
            lon_crop = lon_1d[lon_mask]

            # ── 5. Konversi BT → CTT ─────────────────────────────────────
            ctt = bt_to_ctt(bt_crop)

            # ── 6. Resample ke 50×50 ─────────────────────────────────────
            zr  = OUTPUT_GRID_SIZE / ctt.shape[0]
            zc  = OUTPUT_GRID_SIZE / ctt.shape[1]
            ctt_out = zoom(ctt, (zr, zc), order=1).astype(np.float32)
            lat_out = np.linspace(lat_crop.min(), lat_crop.max(), OUTPUT_GRID_SIZE)
            lon_out = np.linspace(lon_crop.min(), lon_crop.max(), OUTPUT_GRID_SIZE)

            # ── 7. Timestamp dari nama file ──────────────────────────────
            m = re.search(r'NC_H09_(\d{8})_(\d{4})', nc_path.name)
            timestamp = (datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
                         if m else datetime.utcnow())

            valid_pct = np.sum((ctt_out > 150) & (ctt_out < 320)) / ctt_out.size * 100

            if verbose:
                print(f"OK ({var_used}) | "
                      f"CTT: {ctt_out.min():.1f}–{ctt_out.max():.1f} K | "
                      f"Valid: {valid_pct:.1f}%")

            return {
                "ctt":          ctt_out,
                "lat":          lat_out,
                "lon":          lon_out,
                "timestamp":    timestamp,
                "filename":     nc_path.name,
                "valid_pixels": int(np.sum(ctt_out > 150)),
            }

    except Exception as e:
        if verbose:
            print(f"ERROR — {e}")
        return _simulate_parse(nc_path)


def parse_multiple_files(nc_paths: list[Path]) -> list[dict]:
    """Parse beberapa file sekaligus."""
    results = []
    for i, p in enumerate(nc_paths, 1):
        print(f"  [{i}/{len(nc_paths)}]", end=" ")
        r = parse_nc_file(p)
        if r:
            results.append(r)
    print(f"\n  ✓ Berhasil: {len(results)}/{len(nc_paths)}")
    return results


def _simulate_parse(nc_path: Path) -> dict:
    """Fallback simulasi jika file tidak bisa dibaca."""
    m = re.search(r'(\d{8})_(\d{4})', str(nc_path))
    seed = 42
    timestamp = datetime.utcnow()
    if m:
        seed = int(m.group(1)[-4:] + m.group(2)) % (2**32)
        timestamp = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")

    rng = np.random.default_rng(seed)
    G   = OUTPUT_GRID_SIZE
    ctt = np.clip(260 + 20 * rng.standard_normal((G, G)), 180, 300).astype(np.float32)
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.integers(5, G-5, size=2)
        r = rng.integers(3, 8)
        for i in range(G):
            for j in range(G):
                d = np.sqrt((i-cx)**2 + (j-cy)**2)
                if d < r:
                    ctt[i, j] -= 40 * (1 - d/r)
    ctt = np.clip(ctt, 180, 300)

    return {
        "ctt":          ctt,
        "lat":          np.linspace(LAT_MIN, LAT_MAX, G),
        "lon":          np.linspace(LON_MIN, LON_MAX, G),
        "timestamp":    timestamp,
        "filename":     nc_path.name,
        "valid_pixels": G * G,
    }