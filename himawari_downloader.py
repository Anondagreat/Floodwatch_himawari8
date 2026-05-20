"""
himawari_downloader.py
======================
Modul download data NetCDF dari FTP JAXA (Himawari-8 P-Tree System).

Cara mendapatkan akun:
    1. Daftar di: https://www.eorc.jaxa.jp/ptree/registration_top.html
    2. Tunggu email dari JAXA (1-2 hari kerja)
    3. Isi FTP_USER dan FTP_PASS di bawah atau lewat environment variable

Struktur folder FTP JAXA:
    /jma/hsd/YYYYMM/DD/HHmm/
    File: H08_YYYYMMDD_HHmm_R10_FLDK.02401_02401.nc

Referensi: https://www.eorc.jaxa.jp/ptree/
"""

import ftplib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ─── KONFIGURASI FTP — ISI DENGAN AKUN JAXA KAMU ───────────────────────────
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = os.environ.get("JAXA_USER", "your_email@example.com_jaxa")
FTP_PASS = os.environ.get("JAXA_PASS", "your_password")

# Folder lokal untuk menyimpan data yang didownload
LOCAL_DATA_DIR = Path("./data_himawari")

# Area studi: Jawa Barat (bounding box)
# Himawari-8 menyimpan full disk, kita crop saat parsing
LAT_MIN, LAT_MAX = -8.0, -5.0
LON_MIN, LON_MAX = 105.0, 109.0

# Kanal yang digunakan (B13 = IR 10.4μm → Cloud Top Temperature)
TARGET_CHANNEL = "B13"
# ────────────────────────────────────────────────────────────────────────────


def get_ftp_path(dt: datetime) -> tuple[str, str]:
    """
    Menghasilkan path FTP dan nama file berdasarkan datetime.
    
    Contoh:
        dt = datetime(2024, 4, 15, 3, 0)   ← UTC
        path = /jma/hsd/202404/15/0300/
        file = H08_20240415_0300_R10_FLDK.02401_02401.nc
    
    Himawari-8 merekam setiap 10 menit:
        00, 10, 20, 30, 40, 50 menit tiap jam
    """
    # Bulatkan ke 10 menit terdekat (ke bawah)
    minute_rounded = (dt.minute // 10) * 10
    dt_rounded = dt.replace(minute=minute_rounded, second=0, microsecond=0)
    
    folder_path = (
        f"/jma/hsd/"
        f"{dt_rounded.strftime('%Y%m')}/"
        f"{dt_rounded.strftime('%d')}/"
        f"{dt_rounded.strftime('%H%M')}/"
    )
    filename = (
        f"H08_{dt_rounded.strftime('%Y%m%d_%H%M')}"
        f"_R10_FLDK.02401_02401.nc"
    )
    return folder_path, filename


def download_single(dt: datetime, force_redownload: bool = False) -> Path | None:
    """
    Download satu file NetCDF untuk waktu tertentu (UTC).
    
    Returns:
        Path ke file yang didownload, atau None jika gagal.
    """
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    folder_path, filename = get_ftp_path(dt)
    local_path = LOCAL_DATA_DIR / filename
    
    # Cek apakah sudah ada (skip jika tidak force)
    if local_path.exists() and not force_redownload:
        print(f"  [SKIP] File sudah ada: {filename}")
        return local_path
    
    print(f"  [FTP]  Mendownload: {filename} ...", end=" ", flush=True)
    
    try:
        with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
            ftp.login(user=FTP_USER, passwd=FTP_PASS)
            ftp.cwd(folder_path)
            
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR {filename}", f.write)
        
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"OK ({size_mb:.1f} MB)")
        return local_path
    
    except ftplib.error_perm as e:
        print(f"GAGAL — File tidak ditemukan di server: {e}")
        if local_path.exists():
            local_path.unlink()
        return None
    
    except Exception as e:
        print(f"GAGAL — Error: {e}")
        if local_path.exists():
            local_path.unlink()
        return None


def download_range(
    start_dt: datetime,
    end_dt: datetime,
    interval_minutes: int = 10,
    max_retries: int = 3
) -> list[Path]:
    """
    Download semua file dalam rentang waktu tertentu.
    
    Args:
        start_dt: Waktu mulai (UTC)
        end_dt:   Waktu selesai (UTC)
        interval_minutes: Interval antar file (default 10 menit)
        max_retries: Jumlah percobaan ulang jika gagal
    
    Returns:
        List path file yang berhasil didownload
    
    Contoh penggunaan:
        from datetime import datetime, timezone
        start = datetime(2024, 4, 15, 7, 0, tzinfo=timezone.utc)   # 14:00 WIB
        end   = datetime(2024, 4, 15, 9, 0, tzinfo=timezone.utc)   # 16:00 WIB
        files = download_range(start, end)
    """
    downloaded = []
    current = start_dt
    total = int((end_dt - start_dt).total_seconds() / 60 / interval_minutes) + 1
    count = 0
    
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD HIMAWARI-8 DATA")
    print(f"  Dari : {start_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Sampai: {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Total : {total} file")
    print(f"{'='*60}")
    
    while current <= end_dt:
        count += 1
        print(f"\n  [{count}/{total}] {current.strftime('%Y-%m-%d %H:%M')} UTC")
        
        for attempt in range(1, max_retries + 1):
            result = download_single(current)
            if result is not None:
                downloaded.append(result)
                break
            if attempt < max_retries:
                print(f"  [RETRY] Percobaan {attempt+1}/{max_retries}...")
                time.sleep(5)
        
        current += timedelta(minutes=interval_minutes)
    
    print(f"\n{'='*60}")
    print(f"  ✓ Selesai: {len(downloaded)}/{total} file berhasil didownload")
    print(f"  Disimpan di: {LOCAL_DATA_DIR.resolve()}")
    print(f"{'='*60}\n")
    
    return downloaded


def download_latest(n_files: int = 6) -> list[Path]:
    """
    Download N file terbaru (waktu real sekarang, mundur 10 menit per step).
    Data Himawari-8 biasanya tersedia dengan delay ~30 menit dari waktu real.
    
    Args:
        n_files: Berapa file terakhir yang mau didownload (default 6 = 1 jam terakhir)
    
    Returns:
        List path file yang berhasil didownload
    """
    # Himawari-8 ada delay ~30 menit, jadi mundur 30 menit dari sekarang
    now_utc = datetime.now(timezone.utc) - timedelta(minutes=30)
    end_dt = now_utc.replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(minutes=10 * (n_files - 1))
    
    return download_range(start_dt, end_dt)


def list_local_files() -> list[Path]:
    """List semua file .nc yang sudah ada di folder lokal."""
    if not LOCAL_DATA_DIR.exists():
        return []
    files = sorted(LOCAL_DATA_DIR.glob("H08_*.nc"))
    print(f"\n  File lokal di {LOCAL_DATA_DIR}: {len(files)} file")
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        # Ekstrak timestamp dari nama file
        match = re.search(r"H08_(\d{8})_(\d{4})", f.name)
        if match:
            ts = match.group(1) + "_" + match.group(2)
            dt = datetime.strptime(ts, "%Y%m%d_%H%M")
            wib = dt + timedelta(hours=7)
            print(f"    {f.name}  |  {size_mb:.1f} MB  |  {wib.strftime('%d %b %Y %H:%M')} WIB")
    return files


if __name__ == "__main__":
    # Contoh: download 6 file terakhir (1 jam terakhir)
    print("Mode: Download data terbaru Himawari-8")
    print(f"FTP Host : {FTP_HOST}")
    print(f"FTP User : {FTP_USER}")
    print("\n⚠️  Pastikan FTP_USER dan FTP_PASS sudah diisi dengan akun JAXA!")
    print("    Daftar di: https://www.eorc.jaxa.jp/ptree/registration_top.html\n")
    
    # Cek file lokal yang ada
    list_local_files()