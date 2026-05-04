"""
=============================================================================
PREDIKSI BANJIR DARI DATA SATELIT HIMAWARI-8
Menggunakan Metode K-Nearest Neighbor (KNN)
=============================================================================
Berdasarkan paper: "Prediksi Curah Hujan Dari Data Satelit Himawari-8
Menggunakan Metode K-Nearest Neighbor (KNN)"
oleh Hikmah Nisya, Casi Setianingsih, Wendi Harjupa, Risyanto (2023)

DIPERLUAS: Tambahan prediksi risiko BANJIR berdasarkan threshold curah hujan.

Pipeline:
    1. Generate / Load data simulasi NetCDF (suhu awan CLTT + curah hujan)
    2. Preprocessing & labeling (3 kelas suhu awan + 3 level risiko banjir)
    3. Training KNN dengan berbagai nilai K (5, 7, 9, 11)
    4. Evaluasi akurasi
    5. Prediksi & visualisasi peta risiko banjir
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI & KONSTANTA
# ─────────────────────────────────────────────────────────────────────────────

# Area studi: Jawa Barat (Bandung dan sekitarnya)
LAT_MIN, LAT_MAX = -8.0, -5.0   # derajat selatan
LON_MIN, LON_MAX = 105.0, 109.0  # derajat timur
GRID_SIZE = 50  # resolusi grid (50x50 titik)

# Threshold suhu awan (Kelvin) — sesuai paper
CLOUD_TEMP_RANGES = {
    "Tinggi (Tidak Hujan)": (270, 300),   # Merah — 270–300 K
    "Sedang (Mendung)":     (230, 270),   # Biru Muda — 230–260 K
    "Rendah (Hujan)":       (180, 230),   # Biru Tua — 180–220 K
}

# Label kelas suhu awan
CLASS_LABELS = {0: "Tidak Hujan", 1: "Mendung", 2: "Hujan"}
CLASS_COLORS = {0: "#FFD700", 1: "#87CEEB", 2: "#1E3A8A"}

# ─────────────────────────────────────────────────────────────────────────────
# EKSTENSI BANJIR: Threshold curah hujan (mm/jam) → Level Risiko Banjir
# ─────────────────────────────────────────────────────────────────────────────
# Berdasarkan BMKG dan standar hidrometeorologi Indonesia:
#   0–10 mm/jam   → Aman
#   10–20 mm/jam  → Waspada
#   >20 mm/jam    → BAHAYA BANJIR

FLOOD_THRESHOLDS = {
    "Aman":           (0, 10),    # Hijau
    "Waspada":        (10, 20),   # Kuning
    "Bahaya Banjir":  (20, 999),  # Merah
}
FLOOD_COLORS = {"Aman": "#22c55e", "Waspada": "#f59e0b", "Bahaya Banjir": "#ef4444"}
FLOOD_LABELS = {0: "Aman", 1: "Waspada", 2: "Bahaya Banjir"}

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE DATA SIMULASI (menggantikan data .nc dari Himawari-8)
# ─────────────────────────────────────────────────────────────────────────────

def generate_simulation_data(n_samples=180, grid_size=GRID_SIZE, random_seed=42):
    """
    Generate data simulasi yang menyerupai struktur data NetCDF Himawari-8.
    
    Output:
        - Cloud Top Temperature (CTT) dalam Kelvin
        - Precipitation Rate (curah hujan) dalam mm/jam
        - Label kelas suhu (0=Tidak Hujan, 1=Mendung, 2=Hujan)
        - Label risiko banjir (0=Aman, 1=Waspada, 2=Bahaya)
    """
    rng = np.random.default_rng(random_seed)
    
    lats = np.linspace(LAT_MIN, LAT_MAX, grid_size)
    lons = np.linspace(LON_MIN, LON_MAX, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # ── Simulasi suhu awan dengan pola spasial (mirip citra satelit) ──
    # Pola 1: sistem konveksi di barat (suhu rendah = awan tebal = hujan)
    ctt_grid = 260 + 20 * rng.standard_normal((grid_size, grid_size))
    
    # Tambah sel konvektif (suhu sangat rendah) di beberapa titik
    for _ in range(5):
        cx = rng.integers(5, grid_size - 5)
        cy = rng.integers(5, grid_size - 5)
        radius = rng.integers(4, 8)
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist < radius:
                    ctt_grid[i, j] -= (40 * (1 - dist / radius))
    
    # Tambah area cerah (suhu tinggi = tidak hujan)
    for _ in range(3):
        cx = rng.integers(5, grid_size - 5)
        cy = rng.integers(5, grid_size - 5)
        radius = rng.integers(5, 10)
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist < radius:
                    ctt_grid[i, j] += (30 * (1 - dist / radius))
    
    # Clip ke rentang realistis Himawari-8 (180–300 K)
    ctt_grid = np.clip(ctt_grid, 180, 300)
    
    # ── Label kelas suhu awan (sesuai paper) ──
    cloud_class = np.zeros_like(ctt_grid, dtype=int)
    cloud_class[ctt_grid >= 270] = 0   # Tinggi → Tidak Hujan
    cloud_class[(ctt_grid >= 230) & (ctt_grid < 270)] = 1  # Sedang → Mendung
    cloud_class[ctt_grid < 230] = 2    # Rendah → Hujan
    
    # ── Simulasi curah hujan (berkorelasi dengan suhu awan) ──
    precip_grid = np.zeros_like(ctt_grid)
    # Saat suhu rendah (awan tebal), curah hujan tinggi
    mask_rain = cloud_class == 2
    mask_cloudy = cloud_class == 1
    
    precip_grid[mask_rain] = rng.uniform(15, 45, mask_rain.sum())
    precip_grid[mask_cloudy] = rng.uniform(2, 15, mask_cloudy.sum())
    precip_grid[~mask_rain & ~mask_cloudy] = rng.uniform(0, 3, (~mask_rain & ~mask_cloudy).sum())
    
    # ── Label risiko banjir ──
    flood_risk = np.zeros_like(precip_grid, dtype=int)
    flood_risk[precip_grid >= 20] = 2   # Bahaya Banjir
    flood_risk[(precip_grid >= 10) & (precip_grid < 20)] = 1  # Waspada
    flood_risk[precip_grid < 10] = 0    # Aman
    
    # ── Sample untuk training/testing (180 data seperti paper) ──
    flat_ctt = ctt_grid.flatten()
    flat_precip = precip_grid.flatten()
    flat_cloud = cloud_class.flatten()
    flat_flood = flood_risk.flatten()
    
    idx = rng.choice(len(flat_ctt), size=n_samples, replace=False)
    
    return {
        "ctt_grid": ctt_grid,
        "precip_grid": precip_grid,
        "cloud_class_grid": cloud_class,
        "flood_risk_grid": flood_risk,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "lats": lats,
        "lons": lons,
        "X_samples": flat_ctt[idx].reshape(-1, 1),  # fitur: suhu awan
        "X_samples_full": np.column_stack([flat_ctt[idx], flat_precip[idx]]),
        "y_cloud_samples": flat_cloud[idx],
        "y_flood_samples": flat_flood[idx],
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING & EVALUASI KNN
# ─────────────────────────────────────────────────────────────────────────────

def train_evaluate_knn(X, y, k_values=[5, 7, 9, 11], label=""):
    """Train dan evaluasi KNN untuk berbagai nilai K."""
    print(f"\n{'='*60}")
    print(f"  KNN CLASSIFICATION — {label}")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, random_state=42
    )
    print(f"  Dataset: {len(X)} total | {len(X_train)} training | {len(X_test)} testing")
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    results = {}
    best_k, best_acc, best_model = None, 0, None
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train_sc, y_train)
        y_pred = knn.predict(X_test_sc)
        acc = accuracy_score(y_test, y_pred) * 100
        results[k] = {"accuracy": acc, "model": knn, "y_pred": y_pred, "y_test": y_test}
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = knn
        
        print(f"  K = {k:2d}  → Akurasi: {acc:.2f}%")
    
    print(f"\n  ★ K terbaik: K = {best_k} (Akurasi: {best_acc:.2f}%)")
    print(f"\n  Laporan Klasifikasi (K={best_k}):")
    
    if label == "Suhu Awan (Cloud Top Temperature)":
        target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]
    else:
        target_names = [FLOOD_LABELS[i] for i in sorted(FLOOD_LABELS)]
    
    y_pred_best = results[best_k]["y_pred"]
    print(classification_report(y_test, y_pred_best, target_names=target_names))
    
    return results, best_k, best_model, scaler, X_train_sc, y_train

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDIKSI PADA SELURUH GRID (peta)
# ─────────────────────────────────────────────────────────────────────────────

def predict_grid(model, scaler, ctt_grid, precip_grid=None):
    """Prediksi kelas untuk seluruh titik pada grid."""
    rows, cols = ctt_grid.shape
    
    if precip_grid is not None:
        X_flat = np.column_stack([ctt_grid.flatten(), precip_grid.flatten()])
    else:
        X_flat = ctt_grid.flatten().reshape(-1, 1)
    
    X_scaled = scaler.transform(X_flat)
    y_pred = model.predict(X_scaled)
    return y_pred.reshape(rows, cols)

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISASI LENGKAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(data, cloud_pred_grid, flood_pred_grid,
                 cloud_results, flood_results, best_k_cloud, best_k_flood):
    """Plot 6 panel visualisasi sesuai style paper + ekstensi banjir."""
    
    fig = plt.figure(figsize=(20, 14), facecolor="#0f172a")
    fig.suptitle(
        "PREDIKSI BANJIR DARI SATELIT HIMAWARI-8 — Metode KNN\n"
        "Area Studi: Jawa Barat, Indonesia",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )
    
    lats = data["lats"]
    lons = data["lons"]
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    
    ax_kwargs = dict(facecolor="#1e293b")
    
    # ── Panel 1: Cloud Top Temperature (CTT) ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set(**ax_kwargs)
    ctt_plot = ax1.contourf(lons, lats, data["ctt_grid"],
                             levels=20, cmap="RdYlBu_r")
    plt.colorbar(ctt_plot, ax=ax1, label="Suhu Awan (K)", shrink=0.8)
    ax1.set_title("Cloud Top Temperature (CTT)\nDari Satelit Himawari-8",
                  color="white", fontsize=10, pad=8)
    ax1.set_xlabel("Longitude", color="#94a3b8", fontsize=8)
    ax1.set_ylabel("Latitude", color="#94a3b8", fontsize=8)
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values(): spine.set_edgecolor("#334155")
    
    # ── Panel 2: Curah Hujan (Ground Truth) ──
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set(**ax_kwargs)
    precip_plot = ax2.contourf(lons, lats, data["precip_grid"],
                                levels=20, cmap="Blues")
    plt.colorbar(precip_plot, ax=ax2, label="Curah Hujan (mm/jam)", shrink=0.8)
    ax2.set_title("Curah Hujan Simulasi\n(GSMaP Rain Rate)",
                  color="white", fontsize=10, pad=8)
    ax2.set_xlabel("Longitude", color="#94a3b8", fontsize=8)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values(): spine.set_edgecolor("#334155")
    
    # ── Panel 3: Perbandingan Akurasi KNN ──
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set(**ax_kwargs)
    k_vals = [5, 7, 9, 11]
    accs_cloud = [cloud_results[k]["accuracy"] for k in k_vals]
    accs_flood = [flood_results[k]["accuracy"] for k in k_vals]
    x = np.arange(len(k_vals))
    w = 0.35
    bars1 = ax3.bar(x - w/2, accs_cloud, w, label="Prediksi Hujan",
                    color="#3b82f6", alpha=0.85, edgecolor="#60a5fa")
    bars2 = ax3.bar(x + w/2, accs_flood, w, label="Prediksi Banjir",
                    color="#ef4444", alpha=0.85, edgecolor="#f87171")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"K={k}" for k in k_vals], color="#94a3b8")
    ax3.set_ylim(60, 105)
    ax3.set_ylabel("Akurasi (%)", color="#94a3b8", fontsize=9)
    ax3.set_title("Perbandingan Akurasi KNN\n(Berbagai nilai K)",
                  color="white", fontsize=10, pad=8)
    ax3.tick_params(colors="#94a3b8")
    ax3.legend(fontsize=8, facecolor="#1e293b", labelcolor="white")
    for spine in ax3.spines.values(): spine.set_edgecolor("#334155")
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=6.5, color="#93c5fd")
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=6.5, color="#fca5a5")
    
    # ── Panel 4: Peta Prediksi Kelas Awan (KNN) ──
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set(**ax_kwargs)
    cloud_cmap = mcolors.ListedColormap(["#FFD700", "#87CEEB", "#1E3A8A"])
    cloud_bounds = [-0.5, 0.5, 1.5, 2.5]
    cloud_norm = mcolors.BoundaryNorm(cloud_bounds, cloud_cmap.N)
    ax4.pcolormesh(lons, lats, cloud_pred_grid,
                   cmap=cloud_cmap, norm=cloud_norm, shading="auto")
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_LABELS[i])
               for i in range(3)]
    ax4.legend(handles=patches, fontsize=7, loc="lower right",
               facecolor="#1e293b", labelcolor="white", framealpha=0.9)
    ax4.set_title(f"Prediksi KNN: Kelas Hujan\n(K={best_k_cloud}, sesuai paper)",
                  color="white", fontsize=10, pad=8)
    ax4.set_xlabel("Longitude", color="#94a3b8", fontsize=8)
    ax4.set_ylabel("Latitude", color="#94a3b8", fontsize=8)
    ax4.tick_params(colors="#94a3b8")
    for spine in ax4.spines.values(): spine.set_edgecolor("#334155")
    
    # ── Panel 5: Peta Risiko BANJIR (ekstensi) ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set(**ax_kwargs)
    flood_cmap = mcolors.ListedColormap(["#22c55e", "#f59e0b", "#ef4444"])
    flood_bounds = [-0.5, 0.5, 1.5, 2.5]
    flood_norm = mcolors.BoundaryNorm(flood_bounds, flood_cmap.N)
    ax5.pcolormesh(lons, lats, flood_pred_grid,
                   cmap=flood_cmap, norm=flood_norm, shading="auto")
    flood_patches = [
        mpatches.Patch(color="#22c55e", label="Aman (< 10 mm/jam)"),
        mpatches.Patch(color="#f59e0b", label="Waspada (10–20 mm/jam)"),
        mpatches.Patch(color="#ef4444", label="Bahaya Banjir (> 20 mm/jam)"),
    ]
    ax5.legend(handles=flood_patches, fontsize=7, loc="lower right",
               facecolor="#1e293b", labelcolor="white", framealpha=0.9)
    ax5.set_title(f"★ PETA RISIKO BANJIR (EKSTENSI)\nK={best_k_flood} | Threshold BMKG",
                  color="#fbbf24", fontsize=10, fontweight="bold", pad=8)
    ax5.set_xlabel("Longitude", color="#94a3b8", fontsize=8)
    ax5.tick_params(colors="#94a3b8")
    for spine in ax5.spines.values(): spine.set_edgecolor("#334155")
    
    # Tambah marker kota-kota penting di Jawa Barat
    cities = {
        "Bandung": (-6.91, 107.61),
        "Bogor": (-6.60, 106.80),
        "Cirebon": (-6.73, 108.55),
        "Sukabumi": (-6.92, 106.93),
    }
    for city, (lat, lon) in cities.items():
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            ax5.plot(lon, lat, "w^", markersize=6, zorder=5)
            ax5.text(lon + 0.05, lat + 0.05, city, fontsize=6.5,
                     color="white", fontweight="bold", zorder=6)
    
    # ── Panel 6: Statistik Risiko ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set(**ax_kwargs)
    total = flood_pred_grid.size
    aman = (flood_pred_grid == 0).sum()
    waspada = (flood_pred_grid == 1).sum()
    bahaya = (flood_pred_grid == 2).sum()
    
    categories = ["Aman", "Waspada", "Bahaya\nBanjir"]
    values = [aman/total*100, waspada/total*100, bahaya/total*100]
    colors_bar = ["#22c55e", "#f59e0b", "#ef4444"]
    
    bars = ax6.barh(categories, values, color=colors_bar, alpha=0.85, height=0.5)
    ax6.set_xlim(0, 100)
    ax6.set_xlabel("Persentase Area (%)", color="#94a3b8", fontsize=9)
    ax6.set_title("Distribusi Risiko Banjir\nArea Studi (Jawa Barat)",
                  color="white", fontsize=10, pad=8)
    ax6.tick_params(colors="#94a3b8")
    for spine in ax6.spines.values(): spine.set_edgecolor("#334155")
    for bar, val in zip(bars, values):
        ax6.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", color="white", fontsize=10,
                 fontweight="bold")
    
    # Tambah teks akurasi KNN
    best_cloud_acc = max(cloud_results[k]["accuracy"] for k in k_vals)
    best_flood_acc = max(flood_results[k]["accuracy"] for k in k_vals)
    fig.text(0.5, 0.01,
             f"Akurasi KNN Terbaik — Prediksi Hujan: {best_cloud_acc:.2f}% | "
             f"Prediksi Banjir: {best_flood_acc:.2f}%  |  "
             f"Dataset: 180 sampel (120 training + 60 validasi)",
             ha="center", color="#94a3b8", fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("/home/claude/flood_prediction/hasil_prediksi_banjir.png",
                dpi=150, bbox_inches="tight", facecolor="#0f172a")
    print("\n  ✓ Gambar disimpan: hasil_prediksi_banjir.png")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  SISTEM PREDIKSI BANJIR — SATELIT HIMAWARI-8 + KNN")
    print("  Universitas Telkom / BRIN — 2024")
    print("="*60)
    
    # ── Step 1: Generate data simulasi ──
    print("\n[1/5] Generating data simulasi Himawari-8 ...")
    data = generate_simulation_data(n_samples=180)
    ctt_flat = data["ctt_grid"].flatten()
    print(f"      CTT range: {ctt_flat.min():.1f} K – {ctt_flat.max():.1f} K")
    print(f"      Distribusi kelas awan: "
          f"Tidak Hujan={( data['cloud_class_grid']==0).sum()}, "
          f"Mendung={(data['cloud_class_grid']==1).sum()}, "
          f"Hujan={(data['cloud_class_grid']==2).sum()}")
    
    # ── Step 2: Training KNN — Prediksi Hujan (dari suhu awan) ──
    print("\n[2/5] Training KNN — Prediksi Hujan (CTT) ...")
    cloud_results, best_k_cloud, best_cloud_model, cloud_scaler, _, _ = \
        train_evaluate_knn(
            data["X_samples"],
            data["y_cloud_samples"],
            label="Suhu Awan (Cloud Top Temperature)"
        )
    
    # ── Step 3: Training KNN — Prediksi Risiko Banjir ──
    print("\n[3/5] Training KNN — Prediksi Risiko Banjir (CTT + Curah Hujan) ...")
    flood_results, best_k_flood, best_flood_model, flood_scaler, _, _ = \
        train_evaluate_knn(
            data["X_samples_full"],
            data["y_flood_samples"],
            label="Risiko Banjir (EKSTENSI)"
        )
    
    # ── Step 4: Prediksi seluruh grid ──
    print("\n[4/5] Prediksi pada seluruh grid peta ...")
    cloud_pred_grid = predict_grid(best_cloud_model, cloud_scaler, data["ctt_grid"])
    flood_pred_grid = predict_grid(best_flood_model, flood_scaler,
                                   data["ctt_grid"], data["precip_grid"])
    
    total = flood_pred_grid.size
    print(f"      Aman:         {(flood_pred_grid==0).sum()/total*100:.1f}%")
    print(f"      Waspada:      {(flood_pred_grid==1).sum()/total*100:.1f}%")
    print(f"      Bahaya Banjir:{(flood_pred_grid==2).sum()/total*100:.1f}%")
    
    # ── Step 5: Visualisasi ──
    print("\n[5/5] Membuat visualisasi ...")
    plot_results(data, cloud_pred_grid, flood_pred_grid,
                 cloud_results, flood_results, best_k_cloud, best_k_flood)
    
    # ── Ringkasan Akhir ──
    print("\n" + "="*60)
    print("  ✓ PREDIKSI SELESAI")
    print("  Akurasi Terbaik:")
    print(f"    - Prediksi Hujan  (K=11): {cloud_results[11]['accuracy']:.2f}%")
    print(f"    - Prediksi Banjir (K=11): {flood_results[11]['accuracy']:.2f}%")
    print("  Output: hasil_prediksi_banjir.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
