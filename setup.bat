@echo off
echo =====================================================
echo   SETUP: Prediksi Banjir Himawari-8 + KNN
echo   Universitas Telkom / BRIN
echo =====================================================
echo.

:: Cek Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan!
    echo Silakan install Python dari: https://www.python.org/downloads/
    echo Pastikan centang "Add Python to PATH" saat install
    pause
    exit /b 1
)
echo [OK] Python ditemukan

:: Install dependensi
echo.
echo [1/3] Menginstall library Python...
pip install numpy scikit-learn matplotlib scipy --quiet
if errorlevel 1 (
    echo [ERROR] Gagal install library. Coba jalankan sebagai Administrator.
    pause
    exit /b 1
)
echo [OK] Library Python berhasil diinstall

echo.
echo [2/3] Mengecek file sistem...
if exist "flood_prediction_knn.py" (
    echo [OK] flood_prediction_knn.py ditemukan
) else (
    echo [ERROR] flood_prediction_knn.py tidak ditemukan!
    echo Pastikan semua file ada di folder yang sama.
    pause
    exit /b 1
)

if exist "index.html" (
    echo [OK] index.html ditemukan
) else (
    echo [WARNING] index.html tidak ditemukan
)

echo.
echo [3/3] Menjalankan test prediksi...
python flood_prediction_knn.py
if errorlevel 1 (
    echo [ERROR] Gagal menjalankan prediksi!
    pause
    exit /b 1
)

echo.
echo =====================================================
echo   SETUP SELESAI!
echo =====================================================
echo.
echo Cara menjalankan sistem:
echo   1. Web App  : Buka file index.html di browser
echo   2. Python   : python flood_prediction_knn.py
echo.
echo Output yang dihasilkan:
echo   - hasil_prediksi_banjir.png (peta prediksi)
echo.
pause
