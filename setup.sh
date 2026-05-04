#!/bin/bash
echo "====================================================="
echo "  SETUP: Prediksi Banjir Himawari-8 + KNN"
echo "  Universitas Telkom / BRIN"
echo "====================================================="
echo ""

# Cek Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python tidak ditemukan!"
        echo "Mac    : brew install python3"
        echo "Ubuntu : sudo apt install python3 python3-pip"
        exit 1
    fi
    PYTHON=python
else
    PYTHON=python3
fi
echo "[OK] Python ditemukan: $($PYTHON --version)"

# Install library
echo ""
echo "[1/3] Menginstall library Python..."
$PYTHON -m pip install numpy scikit-learn matplotlib scipy --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Gagal install. Coba: pip3 install numpy scikit-learn matplotlib scipy"
    exit 1
fi
echo "[OK] Library berhasil diinstall"

# Cek file
echo ""
echo "[2/3] Mengecek file..."
if [ -f "flood_prediction_knn.py" ]; then
    echo "[OK] flood_prediction_knn.py ditemukan"
else
    echo "[ERROR] flood_prediction_knn.py tidak ditemukan!"
    exit 1
fi

# Test run
echo ""
echo "[3/3] Menjalankan test prediksi..."
$PYTHON flood_prediction_knn.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Gagal menjalankan prediksi!"
    exit 1
fi

echo ""
echo "====================================================="
echo "  SETUP SELESAI!"
echo "====================================================="
echo ""
echo "Cara menjalankan:"
echo "  Web App : Buka index.html di browser (double-click)"
echo "  Python  : python3 flood_prediction_knn.py"
echo ""
