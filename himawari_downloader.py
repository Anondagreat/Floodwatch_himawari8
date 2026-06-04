"""
himawari_downloader.py — FINAL
Format: NC_H09_YYYYMMDD_HHMM_R21_FLDK.02801_02401.nc
Path  : /jma/netcdf/YYYYMM/DD/
"""

import ftplib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = os.environ.get("JAXA_USER", "asmodes123_gmail.com")
FTP_PASS = os.environ.get("JAXA_PASS", "SP+wari8")

LOCAL_DATA_DIR = Path("./data_himawari")
LAT_MIN, LAT_MAX = -7.4, -6.4
LON_MIN, LON_MAX = 107.0, 108.3


def get_ftp_path(dt: datetime) -> tuple[str, str]:
    """
    Hasilkan folder dan nama file berdasarkan datetime UTC.
    
    Contoh:
        dt = datetime(2026, 5, 21, 9, 0)
        folder = /jma/netcdf/202605/21/
        file   = NC_H09_20260521_0900_R21_FLDK.02801_02401.nc
    
    Himawari-9 merekam setiap 10 menit:
        00, 10, 20, 30, 40, 50 menit tiap jam
    """
    minute_rounded = (dt.minute // 10) * 10
    dt_r = dt.replace(minute=minute_rounded, second=0, microsecond=0)

    folder = f"/jma/netcdf/{dt_r.strftime('%Y%m')}/{dt_r.strftime('%d')}/"
    fname  = f"NC_H09_{dt_r.strftime('%Y%m%d_%H%M')}_R21_FLDK.02801_02401.nc"
    return folder, fname


def download_single(dt: datetime, force: bool = False) -> Path | None:
    """Download satu file .nc untuk satu timestamp UTC."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    folder, fname = get_ftp_path(dt)
    local_path = LOCAL_DATA_DIR / fname

    if local_path.exists() and not force and local_path.stat().st_size > 10_000:
        print(f"  [SKIP] {fname} sudah ada")
        return local_path

    print(f"  [FTP]  {fname} ...", end=" ", flush=True)

    try:
        with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
            ftp.login(FTP_USER, FTP_PASS)

            try:
                ftp.cwd(folder)
            except ftplib.error_perm:
                print(f"GAGAL — folder tidak ada: {folder}")
                return None

            # Validasi file ada di server
            available = ftp.nlst()
            if fname not in available:
                # Coba cari file dengan timestamp yang sama (beda resolusi/dimensi)
                ts_str = dt.replace(
                    minute=(dt.minute // 10) * 10,
                    second=0, microsecond=0
                ).strftime('%Y%m%d_%H%M')

                candidates = [f for f in available
                              if ts_str in f and "FLDK" in f and f.endswith(".nc")]

                if candidates:
                    fname = candidates[0]
                    local_path = LOCAL_DATA_DIR / fname
                    print(f"\n  [INFO] Pakai file: {fname} ...", end=" ", flush=True)
                else:
                    print(f"GAGAL — file tidak ada di server")
                    return None

            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR {fname}", f.write)

        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"OK ({size_mb:.1f} MB)")
        return local_path

    except Exception as e:
        print(f"GAGAL — {e}")
        if local_path.exists():
            local_path.unlink()
        return None


def download_latest(n_files: int = 3, max_retries: int = 6) -> list[Path]:
    """
    Download N file terbaru. Mundur 30 menit dari sekarang karena
    Himawari-9 punya delay ~20-30 menit dari waktu observasi ke tersedia di FTP.
    Coba beberapa slot waktu jika slot terbaru belum tersedia.
    """
    now_utc = datetime.now(timezone.utc) - timedelta(minutes=30)
    downloaded = []
    attempts = 0

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD DATA TERBARU HIMAWARI-9")
    print(f"  Referensi waktu: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*60}")

    current = now_utc
    while len(downloaded) < n_files and attempts < max_retries:
        result = download_single(current)
        if result:
            downloaded.append(result)
        current -= timedelta(minutes=10)
        attempts += 1

    print(f"\n  ✓ Berhasil: {len(downloaded)}/{n_files} file")
    return downloaded


def download_range(start_dt: datetime,
                   end_dt: datetime) -> list[Path]:
    """Download semua file dalam rentang waktu UTC."""
    downloaded = []
    current = start_dt
    total = int((end_dt - start_dt).total_seconds() / 600) + 1
    count = 0

    print(f"\n{'='*60}")
    print(f"  DOWNLOAD RANGE: {start_dt.strftime('%Y-%m-%d %H:%M')} — "
          f"{end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Target: {total} file")
    print(f"{'='*60}")

    while current <= end_dt:
        count += 1
        print(f"\n  [{count}/{total}]", end=" ")
        result = download_single(current)
        if result:
            downloaded.append(result)
        current += timedelta(minutes=10)

    print(f"\n  ✓ Selesai: {len(downloaded)}/{total} berhasil")
    return downloaded


def list_local_files() -> list[Path]:
    """List file .nc yang sudah ada di lokal."""
    if not LOCAL_DATA_DIR.exists():
        return []
    files = sorted(LOCAL_DATA_DIR.glob("NC_H09_*.nc"))
    print(f"\n  File lokal: {len(files)} file")
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        m = re.search(r'NC_H09_(\d{8})_(\d{4})', f.name)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
            wib = dt + timedelta(hours=7)
            print(f"    {f.name}  |  {size_mb:.1f} MB  |  "
                  f"{wib.strftime('%d %b %Y %H:%M')} WIB")
    return files


if __name__ == "__main__":
    list_local_files()
    print("\nMencoba download 1 file terbaru ...")
    result = download_latest(n_files=1)
    if result:
        list_local_files()