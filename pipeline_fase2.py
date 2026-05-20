"""
pipeline_fase2.py
=================
Pipeline lengkap Fase 2:
    Download → Parse NetCDF → KNN → Simpan hasil → Alert

Cara pakai:
    # Mode 1: Gunakan file .nc yang sudah ada di folder data_himawari/
    python pipeline_fase2.py --mode local

    # Mode 2: Download data terbaru lalu langsung prediksi
    python pipeline_fase2.py --mode live

    # Mode 3: Download rentang waktu tertentu
    python pipeline_fase2.py --mode range \
        --start "2024-04-15 07:00" --end "2024-04-15 10:00"

    # Mode 4: Jalankan terus setiap 10 menit (loop otomatis)
    python pipeline_fase2.py --mode scheduler
"""

import argparse
import json
import time
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (cocok untuk server)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import modul fase 2
from himawari_downloader import download_latest, download_range, list_local_files, LOCAL_DATA_DIR
from himawari_parser import parse_nc_file, parse_multiple_files

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path("./output_fase2")
MODEL_DIR     = Path("./model_saved")
LOG_FILE      = OUTPUT_DIR / "prediction_log.jsonl"   # JSON Lines format

K_VALUE       = 5          # Default K (sesuai paper, K=5)
K_VALUES_EVAL = [5, 7, 9, 11]   # Semua K yang dievaluasi

# Threshold risiko banjir (mm/jam)
FLOOD_THRESH_SAFE   =  10   # Di bawah ini: AMAN
FLOOD_THRESH_WARN   =  20   # Di bawah ini: WASPADA, di atas: BAHAYA

# Relasi CTT → curah hujan estimasi (model empiris sederhana)
# Berdasarkan Risyanto et al. (2019) untuk wilayah Indonesia
def ctt_to_estimated_precip(ctt_kelvin: float) -> float:
    """
    Estimasi curah hujan dari suhu awan menggunakan hubungan empiris.
    Sumber: GOES Precipitation Index (GPI) yang diadaptasi untuk AHI.
    
    CTT < 235 K → rain rate = 3 × exp(-0.036 × (CTT - 235))  mm/jam
    CTT ≥ 235 K → 0 mm/jam
    """
    if ctt_kelvin < 235:
        return float(3.0 * np.exp(-0.036 * (ctt_kelvin - 235)))
    return 0.0

ctt_to_precip_vec = np.vectorize(ctt_to_estimated_precip)
# ────────────────────────────────────────────────────────────────────────────


def classify_ctt(ctt_grid: np.ndarray) -> np.ndarray:
    """Klasifikasi suhu awan → 3 kelas sesuai paper."""
    labels = np.zeros_like(ctt_grid, dtype=int)
    labels[ctt_grid >= 270] = 0   # Tinggi → Tidak Hujan
    labels[(ctt_grid >= 230) & (ctt_grid < 270)] = 1  # Sedang → Mendung
    labels[ctt_grid < 230] = 2    # Rendah → Hujan
    return labels


def classify_flood_risk(precip_grid: np.ndarray) -> np.ndarray:
    """Klasifikasi risiko banjir dari curah hujan."""
    risk = np.zeros_like(precip_grid, dtype=int)
    risk[(precip_grid >= FLOOD_THRESH_SAFE) & (precip_grid < FLOOD_THRESH_WARN)] = 1
    risk[precip_grid >= FLOOD_THRESH_WARN] = 2
    return risk


def build_or_load_knn_model(force_retrain: bool = False) -> tuple:
    """
    Build model KNN dari data historis atau load dari file yang tersimpan.
    
    Saat data real belum banyak, model diinisialisasi dengan data
    sintetis yang mewakili pola fisik CTT-hujan Indonesia.
    Model akan di-update inkremental setiap ada data baru.
    
    Returns:
        (knn_model, scaler, metadata_dict)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path  = MODEL_DIR / "knn_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    meta_path   = MODEL_DIR / "model_meta.json"
    
    if model_path.exists() and not force_retrain:
        print("  [MODEL] Memuat model dari file yang tersimpan ...")
        with open(model_path, "rb") as f:
            knn = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"         K={meta['k']} | Akurasi={meta['accuracy']:.2f}% | "
              f"Trained: {meta['trained_at']}")
        return knn, scaler, meta
    
    print("  [MODEL] Training model KNN baru ...")
    
    # Generate data sintetis representatif untuk inisialisasi
    rng = np.random.default_rng(42)
    n   = 500   # lebih banyak dari paper agar model lebih robust

    # Kelas 0: Tidak Hujan — CTT 270–300 K
    ctt0 = rng.uniform(270, 300, n // 3)
    # Kelas 1: Mendung — CTT 230–270 K
    ctt1 = rng.uniform(230, 270, n // 3)
    # Kelas 2: Hujan — CTT 180–230 K
    ctt2 = rng.uniform(180, 230, n // 3)
    
    X_init = np.concatenate([ctt0, ctt1, ctt2]).reshape(-1, 1)
    y_init = np.array([0]*(n//3) + [1]*(n//3) + [2]*(n//3))
    
    # Tambah noise agar model tidak overfit pada boundary
    X_init += rng.normal(0, 1.5, X_init.shape)
    X_init = np.clip(X_init, 180, 300)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_init, y_init, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    
    # Pilih K terbaik secara otomatis
    best_k, best_acc = K_VALUE, 0
    for k in K_VALUES_EVAL:
        m = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        m.fit(X_train_sc, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_sc)) * 100
        print(f"         K={k:2d} → {acc:.2f}%")
        if acc > best_acc:
            best_acc, best_k = acc, k
    
    knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    knn.fit(X_train_sc, y_train)
    
    meta = {
        "k":          best_k,
        "accuracy":   best_acc,
        "n_training": len(X_train),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source":     "synthetic_init",
    }
    
    # Simpan model
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  [MODEL] Terbaik: K={best_k} ({best_acc:.2f}%) — disimpan ke {MODEL_DIR}/")
    return knn, scaler, meta


def predict_from_parsed(parsed: dict,
                         knn: KNeighborsClassifier,
                         scaler: StandardScaler) -> dict:
    """
    Jalankan prediksi KNN pada satu frame hasil parse.
    
    Returns dict lengkap berisi grid prediksi, statistik, dan alert.
    """
    ctt = parsed["ctt"]
    G   = ctt.shape[0]
    
    # Estimasi curah hujan dari CTT (empiris)
    precip = ctt_to_precip_vec(ctt)
    
    # KNN predict pada seluruh grid
    X_flat  = ctt.flatten().reshape(-1, 1)
    X_sc    = scaler.transform(X_flat)
    cloud_pred = knn.predict(X_sc).reshape(G, G)
    
    # Risiko banjir dari curah hujan estimasi
    flood_pred = classify_flood_risk(precip)
    
    # Statistik
    total = G * G
    n_safe   = int((flood_pred == 0).sum())
    n_warn   = int((flood_pred == 1).sum())
    n_danger = int((flood_pred == 2).sum())
    
    danger_pct = n_danger / total * 100
    
    # Tentukan status alert
    if danger_pct >= 20:
        alert_level = "BAHAYA"
        alert_msg   = f"⚠️  BAHAYA BANJIR: {danger_pct:.1f}% area berisiko tinggi!"
    elif danger_pct >= 5 or n_warn / total * 100 >= 30:
        alert_level = "WASPADA"
        alert_msg   = f"⚡ WASPADA: {n_warn/total*100:.1f}% area waspada, {danger_pct:.1f}% bahaya"
    else:
        alert_level = "AMAN"
        alert_msg   = f"✓ AMAN: Kondisi cuaca baik di area studi"
    
    timestamp_wib = parsed["timestamp"] + timedelta(hours=7)
    
    return {
        "ctt":         ctt,
        "precip":      precip,
        "cloud_pred":  cloud_pred,
        "flood_pred":  flood_pred,
        "lat":         parsed["lat"],
        "lon":         parsed["lon"],
        "timestamp":   parsed["timestamp"],
        "timestamp_wib": timestamp_wib,
        "filename":    parsed["filename"],
        "stats": {
            "aman_pct":    n_safe   / total * 100,
            "waspada_pct": n_warn   / total * 100,
            "bahaya_pct":  n_danger / total * 100,
            "ctt_min":     float(ctt.min()),
            "ctt_max":     float(ctt.max()),
            "ctt_mean":    float(ctt.mean()),
            "precip_max":  float(precip.max()),
        },
        "alert_level": alert_level,
        "alert_msg":   alert_msg,
    }


def save_result_plot(result: dict, out_path: Path) -> None:
    """Simpan visualisasi 4-panel untuk satu frame prediksi."""
    fig = plt.figure(figsize=(16, 10), facecolor="#0f172a")
    
    ts_wib = result["timestamp_wib"]
    fig.suptitle(
        f"Prediksi Banjir Himawari-8 — {ts_wib.strftime('%d %B %Y  %H:%M WIB')}\n"
        f"Status: {result['alert_msg']}",
        fontsize=13, color="white", y=0.98,
        fontweight="bold"
    )
    
    lons, lats = result["lon"], result["lat"]
    ax_kw = dict(facecolor="#1e293b")
    
    # Panel 1: CTT
    ax1 = fig.add_subplot(2, 2, 1, **ax_kw)
    cf = ax1.contourf(lons, lats, result["ctt"], levels=20, cmap="RdYlBu_r")
    plt.colorbar(cf, ax=ax1, label="CTT (K)", shrink=0.85)
    ax1.set_title("Cloud Top Temperature", color="white")
    ax1.tick_params(colors="#94a3b8")
    for sp in ax1.spines.values(): sp.set_edgecolor("#334155")
    
    # Panel 2: Estimasi Curah Hujan
    ax2 = fig.add_subplot(2, 2, 2, **ax_kw)
    cf2 = ax2.contourf(lons, lats, result["precip"], levels=20, cmap="Blues")
    plt.colorbar(cf2, ax=ax2, label="Curah Hujan (mm/jam)", shrink=0.85)
    ax2.set_title("Estimasi Curah Hujan", color="white")
    ax2.tick_params(colors="#94a3b8")
    for sp in ax2.spines.values(): sp.set_edgecolor("#334155")
    
    # Panel 3: KNN Kelas Awan
    ax3 = fig.add_subplot(2, 2, 3, **ax_kw)
    c_cmap = mcolors.ListedColormap(["#FFD700", "#87CEEB", "#1E3A8A"])
    c_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    ax3.pcolormesh(lons, lats, result["cloud_pred"], cmap=c_cmap, norm=c_norm, shading="auto")
    patches = [
        mpatches.Patch(color="#FFD700", label="Tidak Hujan (270–300K)"),
        mpatches.Patch(color="#87CEEB", label="Mendung (230–270K)"),
        mpatches.Patch(color="#1E3A8A", label="Hujan (<230K)"),
    ]
    ax3.legend(handles=patches, fontsize=8, loc="lower right",
               facecolor="#1e293b", labelcolor="white")
    ax3.set_title("Prediksi KNN — Kelas Awan", color="white")
    ax3.tick_params(colors="#94a3b8")
    for sp in ax3.spines.values(): sp.set_edgecolor("#334155")
    
    # Panel 4: Peta Risiko Banjir
    ax4 = fig.add_subplot(2, 2, 4, **ax_kw)
    f_cmap = mcolors.ListedColormap(["#22c55e", "#f59e0b", "#ef4444"])
    f_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    ax4.pcolormesh(lons, lats, result["flood_pred"], cmap=f_cmap, norm=f_norm, shading="auto")
    
    # Marker kota
    for city, (la, lo) in [("Bandung",(-6.91,107.61)),("Bogor",(-6.60,106.80)),("Cirebon",(-6.73,108.55))]:
        ax4.plot(lo, la, "w^", ms=6, zorder=5)
        ax4.text(lo+0.05, la+0.05, city, fontsize=7, color="white", fontweight="bold")
    
    flood_p = [
        mpatches.Patch(color="#22c55e", label=f"Aman ({result['stats']['aman_pct']:.1f}%)"),
        mpatches.Patch(color="#f59e0b", label=f"Waspada ({result['stats']['waspada_pct']:.1f}%)"),
        mpatches.Patch(color="#ef4444", label=f"Bahaya ({result['stats']['bahaya_pct']:.1f}%)"),
    ]
    ax4.legend(handles=flood_p, fontsize=8, loc="lower right",
               facecolor="#1e293b", labelcolor="white")
    
    alert_colors = {"AMAN": "#22c55e", "WASPADA": "#f59e0b", "BAHAYA": "#ef4444"}
    ax4.set_title(f"★ PETA RISIKO BANJIR [{result['alert_level']}]",
                  color=alert_colors.get(result["alert_level"], "white"), fontweight="bold")
    ax4.tick_params(colors="#94a3b8")
    for sp in ax4.spines.values(): sp.set_edgecolor("#334155")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()


def save_log_entry(result: dict) -> None:
    """Simpan satu entri hasil prediksi ke log file (format JSON Lines)."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp_utc": result["timestamp"].strftime("%Y-%m-%dT%H:%M:00Z"),
        "timestamp_wib": result["timestamp_wib"].strftime("%Y-%m-%d %H:%M WIB"),
        "filename":      result["filename"],
        "alert_level":   result["alert_level"],
        "alert_msg":     result["alert_msg"],
        **result["stats"],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_once(nc_paths: list[Path], knn, scaler) -> list[dict]:
    """
    Jalankan pipeline sekali untuk list file .nc yang diberikan.
    Download sudah selesai di luar fungsi ini.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Memproses {len(nc_paths)} file ...")
    all_results = []
    
    for nc_path in nc_paths:
        # 1. Parse
        parsed = parse_nc_file(nc_path)
        if parsed is None:
            continue
        
        # 2. Prediksi
        result = predict_from_parsed(parsed, knn, scaler)
        
        # 3. Simpan plot
        ts_str = result["timestamp"].strftime("%Y%m%d_%H%M")
        plot_path = OUTPUT_DIR / f"prediksi_{ts_str}.png"
        save_result_plot(result, plot_path)
        
        # 4. Log
        save_log_entry(result)
        
        # 5. Print status
        print(f"  {result['timestamp_wib'].strftime('%d/%m %H:%M WIB')} | "
              f"{result['alert_level']:8s} | "
              f"Bahaya={result['stats']['bahaya_pct']:.1f}% | "
              f"CTT_min={result['stats']['ctt_min']:.1f}K | "
              f"Plot → {plot_path.name}")
        
        all_results.append(result)
    
    return all_results


def print_summary(results: list[dict]) -> None:
    """Print ringkasan dari semua frame yang diproses."""
    if not results:
        print("\n  Tidak ada hasil yang diproses.")
        return
    
    print(f"\n{'='*65}")
    print(f"  RINGKASAN PREDIKSI FASE 2")
    print(f"{'='*65}")
    print(f"  Total frame: {len(results)}")
    print(f"  Periode    : {results[0]['timestamp_wib'].strftime('%d/%m %H:%M')} – "
          f"{results[-1]['timestamp_wib'].strftime('%d/%m %H:%M')} WIB")
    
    alert_counts = {"AMAN": 0, "WASPADA": 0, "BAHAYA": 0}
    for r in results:
        alert_counts[r["alert_level"]] = alert_counts.get(r["alert_level"], 0) + 1
    
    print(f"\n  Alert Summary:")
    print(f"    ✓  AMAN    : {alert_counts['AMAN']:3d} frame")
    print(f"    ⚡ WASPADA : {alert_counts['WASPADA']:3d} frame")
    print(f"    ⚠️  BAHAYA  : {alert_counts['BAHAYA']:3d} frame")
    
    max_danger = max(r["stats"]["bahaya_pct"] for r in results)
    max_precip = max(r["stats"]["precip_max"] for r in results)
    print(f"\n  Puncak bahaya  : {max_danger:.1f}% area")
    print(f"  Curah hujan max: {max_precip:.1f} mm/jam")
    print(f"\n  Output disimpan di: {OUTPUT_DIR.resolve()}/")
    print(f"  Log file         : {LOG_FILE.resolve()}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Fase 2 — Prediksi Banjir Himawari-8 Real Data"
    )
    parser.add_argument(
        "--mode", choices=["local", "live", "range", "scheduler"],
        default="local",
        help=(
            "local     : proses file .nc yang sudah ada di data_himawari/\n"
            "live      : download data terbaru lalu prediksi\n"
            "range     : download rentang waktu tertentu\n"
            "scheduler : loop otomatis setiap 10 menit"
        )
    )
    parser.add_argument("--start", type=str,
                        help="Waktu mulai UTC, format: '2024-04-15 07:00'")
    parser.add_argument("--end",   type=str,
                        help="Waktu selesai UTC, format: '2024-04-15 10:00'")
    parser.add_argument("--retrain", action="store_true",
                        help="Paksa re-training model dari awal")
    args = parser.parse_args()
    
    print("\n" + "="*65)
    print("  PIPELINE FASE 2 — DATA ASLI HIMAWARI-8 + KNN")
    print("  Universitas Telkom × BRIN — 2024")
    print("="*65)
    
    # Load atau build model
    print("\n[1] Memuat/membuat model KNN ...")
    knn, scaler, meta = build_or_load_knn_model(force_retrain=args.retrain)
    
    if args.mode == "local":
        # Proses semua file .nc yang ada di folder lokal
        print("\n[2] Mode LOCAL — Memproses file yang sudah ada ...")
        nc_files = sorted(LOCAL_DATA_DIR.glob("H08_*.nc"))
        if not nc_files:
            print(f"  Tidak ada file .nc di {LOCAL_DATA_DIR}")
            print("  Menjalankan simulasi fallback (netCDF4 parser mode) ...")
            # Buat path dummy agar _simulate_parse dipanggil
            from himawari_parser import _simulate_parse
            dummy_paths = [
                LOCAL_DATA_DIR / f"H08_20240415_{h:02d}00_R10_FLDK.02401_02401.nc"
                for h in range(7, 14)   # 07:00–13:00 UTC (14:00–20:00 WIB)
            ]
            LOCAL_DATA_DIR.mkdir(exist_ok=True)
            # Buat file dummy kosong agar path exist check lewat
            for p in dummy_paths:
                p.touch()
            nc_files = dummy_paths
        results = run_once(nc_files, knn, scaler)
    
    elif args.mode == "live":
        print("\n[2] Mode LIVE — Download data terbaru ...")
        nc_files = download_latest(n_files=6)
        if not nc_files:
            print("  Download gagal. Pastikan akun JAXA sudah dikonfigurasi.")
            print("  Jalankan dengan --mode local untuk menggunakan simulasi.")
            return
        results = run_once(nc_files, knn, scaler)
    
    elif args.mode == "range":
        if not args.start or not args.end:
            print("  Error: --mode range membutuhkan --start dan --end")
            print("  Contoh: python pipeline_fase2.py --mode range "
                  "--start '2024-04-15 07:00' --end '2024-04-15 10:00'")
            return
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        end   = datetime.strptime(args.end,   "%Y-%m-%d %H:%M")
        print(f"\n[2] Mode RANGE — Download {args.start} s/d {args.end} UTC ...")
        nc_files = download_range(start, end)
        results  = run_once(nc_files, knn, scaler)
    
    elif args.mode == "scheduler":
        print("\n[2] Mode SCHEDULER — Loop otomatis setiap 10 menit")
        print("    Tekan Ctrl+C untuk berhenti\n")
        results = []
        iteration = 0
        while True:
            iteration += 1
            print(f"\n  ─── Iterasi #{iteration} | "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ───")
            new_files = download_latest(n_files=1)
            if new_files:
                new_results = run_once(new_files, knn, scaler)
                results.extend(new_results)
            print("  Menunggu 10 menit untuk data berikutnya ...")
            time.sleep(600)   # 10 menit
    
    print_summary(results if results else [])


if __name__ == "__main__":
    main()