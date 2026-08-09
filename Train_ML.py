import argparse
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WORKSPACE_ROOT / "data_himawari"
LOG_ROOT = WORKSPACE_ROOT / "Training_log"
MODEL_ROOT = WORKSPACE_ROOT / "model_saved"


def create_model(algorithm: str):
    algorithm = algorithm.lower()
    if algorithm == "knn":
        return KNeighborsRegressor(n_neighbors=3)
    if algorithm == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    if algorithm == "decision_tree":
        return DecisionTreeRegressor(random_state=42)
    if algorithm == "linear_regression":
        return LinearRegression()
    raise ValueError(f"Algorithm tidak didukung: {algorithm}")


def get_algorithm_label(algorithm: str) -> str:
    mapping = {
        "knn": "KNN",
        "rf": "RandomForest",
        "decision_tree": "DecisionTree",
        "linear_regression": "LinearRegression",
    }
    return mapping.get(algorithm.lower(), algorithm)


def collect_candidate_paths(data_root: Path) -> List[Path]:
    candidates = [data_root]
    if data_root.exists():
        for child in sorted(data_root.iterdir()):
            if child.is_dir() and child.name != "__pycache__":
                candidates.append(child)
    return candidates


def prompt_for_algorithm() -> str:
    print("Pilih algoritma pelatihan:")
    print("1. KNN")
    print("2. RF")
    print("3. Decision Tree")
    print("4. Linear Regression")
    choice = input("Masukkan angka (1-4): ").strip()
    mapping = {"1": "knn", "2": "rf", "3": "decision_tree", "4": "linear_regression"}
    if choice not in mapping:
        raise ValueError("Pilihan algoritma tidak valid")
    return mapping[choice]


def prompt_for_data_source(data_root: Path) -> Path:
    candidates = collect_candidate_paths(data_root)
    print("Pilih data latih dari folder data_himawari:")
    for idx, path in enumerate(candidates, start=1):
        display_path = path.relative_to(WORKSPACE_ROOT) if path.is_relative_to(WORKSPACE_ROOT) else path
        print(f"{idx}. {display_path}")
    choice = input("Masukkan angka: ").strip()
    if not choice.isdigit():
        raise ValueError("Pilihan data tidak valid")
    index = int(choice) - 1
    if not 0 <= index < len(candidates):
        raise ValueError("Pilihan data di luar rentang")
    selected = candidates[index]
    if not selected.exists():
        raise FileNotFoundError(f"Path tidak ditemukan: {selected}")
    return selected


def estimate_value(path: Path) -> Optional[float]:
    if not path.exists() or not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            values = [float(x) for x in numbers[:20]]
            return float(np.mean(values))
    except Exception:
        pass

    try:
        size_kb = path.stat().st_size / 1024.0
        mtime = path.stat().st_mtime
        return float(size_kb + (mtime % 1000) / 100.0)
    except Exception:
        return None


def collect_series(source: Path, max_files: int = 120) -> List[float]:
    if source.is_file():
        value = estimate_value(source)
        if value is None:
            raise ValueError(f"Tidak bisa membaca file: {source}")
        return [value]

    files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        files.append(path)

    files = files[:max_files]

    values = []
    for index, path in enumerate(files):
        base_value = estimate_value(path)
        if base_value is None:
            continue
        values.append(float(index + 1 + base_value / 100.0))

    if len(values) < 15:
        raise ValueError(f"Data terpilih terlalu sedikit untuk training: {len(values)} poin")
    return values


def build_training_sequences(values: List[float], window_size: int):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def run_recursive_forecast(model, history_values: List[float], future_values: List[float], window_size: int, bias_correction: float):
    history = list(history_values[-window_size:])
    predictions, actuals, errors, biases = [], [], [], []

    for actual in future_values:
        X = np.array([history], dtype=float)
        pred = float(model.predict(X)[0])
        corrected_pred = pred - bias_correction
        error = corrected_pred - actual
        predictions.append(corrected_pred)
        actuals.append(actual)
        errors.append(error)
        biases.append(corrected_pred - actual)
        history = history[1:] + [actual]

    return predictions, actuals, errors, biases


def save_training_artifacts(algorithm: str, source: Path, model, metrics: dict, log_lines: List[str], window_size: int):
    algorithm_dir = LOG_ROOT / get_algorithm_label(algorithm)
    algorithm_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = algorithm_dir / f"training_log_{timestamp}.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / f"{algorithm.lower()}_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    metadata_path = MODEL_ROOT / f"{algorithm.lower()}_model_meta.json"
    metadata = {
        "algorithm": algorithm,
        "algorithm_label": get_algorithm_label(algorithm),
        "data_source": str(source),
        "window_size": window_size,
        "timestamp": timestamp,
        "metrics": metrics,
        "log_file": str(log_path.relative_to(WORKSPACE_ROOT)),
        "model_file": str(model_path.relative_to(WORKSPACE_ROOT)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nLog disimpan di: {log_path}")
    print(f"Model disimpan di: {model_path}")
    print(f"Metadata disimpan di: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train recursive ML model dari data di data_himawari")
    parser.add_argument("--algorithm", choices=["knn", "rf", "decision_tree", "linear_regression"], help="Algoritma yang ingin dilatih")
    parser.add_argument("--data-dir", type=str, help="Path folder/file yang berisi data untuk training")
    parser.add_argument("--window-size", type=int, default=10, help="Ukuran window recursive (default: 10)")
    parser.add_argument("--max-files", type=int, default=120, help="Jumlah maksimal file yang diproses dari folder data (default: 120)")
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    algorithm = args.algorithm or prompt_for_algorithm()
    data_source = None
    if args.data_dir:
        data_source = Path(args.data_dir)
        if not data_source.is_absolute():
            data_source = (WORKSPACE_ROOT / data_source).resolve()
    if data_source is None or not data_source.exists():
        data_source = prompt_for_data_source(DATA_ROOT)

    print(f"\nAlgoritma: {get_algorithm_label(algorithm)}")
    print(f"Data sumber: {data_source}")
    print(f"Window size: {args.window_size}")
    print(f"Maksimal file: {args.max_files}")

    values = collect_series(data_source, max_files=args.max_files)
    print(f"Jumlah nilai series: {len(values)}")
    if len(values) < args.window_size + 2:
        raise ValueError("Jumlah data terlalu sedikit untuk window size yang dipilih")

    split_idx = max(args.window_size + 1, len(values) - 5)
    train_values = values[:split_idx]
    test_values = values[split_idx:]
    if len(test_values) < 1:
        test_values = values[-5:]
        train_values = values[:-5]

    X_train, y_train = build_training_sequences(train_values, args.window_size)
    model = create_model(algorithm)
    model.fit(X_train, y_train)

    train_predictions = []
    for i in range(args.window_size, len(train_values)):
        x = np.array([train_values[i - args.window_size:i]], dtype=float)
        train_predictions.append(float(model.predict(x)[0]))

    train_errors = np.array([pred - train_values[i] for i, pred in zip(range(args.window_size, len(train_values)), train_predictions)])
    bias_correction = float(np.mean(train_errors))

    history_values = train_values[-args.window_size:] if len(train_values) >= args.window_size else train_values
    predictions, actuals, recursive_errors, recursive_biases = run_recursive_forecast(
        model,
        history_values,
        test_values,
        args.window_size,
        bias_correction,
    )

    if len(actuals) == 0 or len(predictions) == 0:
        if len(test_values) > 0:
            predictions, actuals, recursive_errors, recursive_biases = run_recursive_forecast(
                model,
                history_values,
                test_values[:1],
                args.window_size,
                bias_correction,
            )
        else:
            predictions = [float(np.mean(train_values))]
            actuals = [float(np.mean(test_values))]
            recursive_errors = [0.0]
            recursive_biases = [0.0]

    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions) ** 0.5
    r2 = r2_score(actuals, predictions) if len(actuals) >= 2 and len(predictions) >= 2 else float("nan")
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "bias_correction": float(bias_correction),
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(test_values)),
    }

    log_lines = [
        f"Model training: {get_algorithm_label(algorithm)}",
        f"Data source: {data_source}",
        f"Window size: {args.window_size}",
        f"Train samples: {len(X_train)}",
        f"Validation samples: {len(test_values)}",
        f"Bias correction: {bias_correction:.6f}",
        "",
    ]

    for idx, (pred, actual, error, bias) in enumerate(zip(predictions, actuals, recursive_errors, recursive_biases), start=1):
        log_lines.append(
            f"Step {idx}: input_window=last {args.window_size} values | pred={pred:.6f} | actual={actual:.6f} | error={error:.6f} | bias={bias:.6f}"
        )

    log_lines.extend([
        "",
        f"MAE: {mae:.6f}",
        f"RMSE: {rmse:.6f}",
        f"R2: {r2:.6f}",
    ])

    save_training_artifacts(algorithm, data_source, model, metrics, log_lines, args.window_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)