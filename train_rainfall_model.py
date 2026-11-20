"""
Train Machine Learning Models for Karnataka Rainfall Prediction

Models:
- XGBoost Regressor / Random Forest Regressor (Predicts next-day rainfall depth in mm)
- XGBoost Classifier (Predicts rainfall occurrence and intensity classification)

Evaluation Metrics:
- Regression: RMSE, MAE, R², Pearson Correlation
- Classification: Accuracy, F1-Score, Critical Success Index (CSI), Confusion Matrix

Plots:
- Feature Importance Chart
- Actual vs Predicted Rainfall Scatter & Time-Series Comparison
- Confusion Matrix

Usage:
    python train_rainfall_model.py --input-file ML_Ready_Karnataka_Rainfall.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def calculate_csi(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """
    Calculate Critical Success Index (Threat Score) for precipitation threshold:
    CSI = Hits / (Hits + False Alarms + Misses)
    """
    hits = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    false_alarms = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    misses = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    denominator = hits + false_alarms + misses
    return float(hits / denominator) if denominator > 0 else 0.0


def train_eval_regression(df: pd.DataFrame, feature_cols: list[str], output_dir: Path):
    """
    Train & evaluate regression model for next-day rainfall depth prediction.
    """
    print("\n" + "=" * 60)
    print("TRAINING REGRESSION MODEL (Next-Day Rainfall Depth in mm)")
    print("=" * 60)

    # Time-based Train-Test Split (Last 20% of time for testing)
    time_col = next((c for c in df.columns if "time" in c.lower()), "valid_time")
    df = df.sort_values(by=time_col).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["target_next_day_precip_mm"]
    X_test, y_test = test_df[feature_cols], test_df["target_next_day_precip_mm"]

    print(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

    if HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "XGBoost Regressor"
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model_name = "Random Forest Regressor"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0.0)  # Clip negative predictions

    # Calculate regression metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    pearson_corr = float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0.0

    # Binary rainfall detection metric (Precipitation > 0.1 mm)
    csi = calculate_csi((y_test >= 0.1).astype(int), (y_pred >= 0.1).astype(int))

    print(f"\n[{model_name} Results]")
    print(f"  |-- RMSE:  {rmse:.4f} mm")
    print(f"  |-- MAE:   {mae:.4f} mm")
    print(f"  |-- R2:    {r2:.4f}")
    print(f"  |-- Pearson r: {pearson_corr:.4f}")
    print(f"  +-- Critical Success Index (CSI > 0.1mm): {csi:.4f}")

    # Plot Actual vs Predicted
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="#1f77b4")
    plt.plot([0, max(y_test.max(), y_pred.max())], [0, max(y_test.max(), y_pred.max())], 'r--', label='Ideal 1:1')
    plt.xlabel("Actual Next-Day Rainfall (mm)")
    plt.ylabel("Predicted Next-Day Rainfall (mm)")
    plt.title(f"{model_name} - Actual vs Predicted Rainfall")
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "regression_actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved scatter plot: {plot_path}")

    # Feature Importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1][:12]
        plt.barh(range(len(indices)), importances[indices], align="center", color="#2ca02c")
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel("Relative Feature Importance")
        plt.title(f"{model_name} - Top 12 Features")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        fi_plot_path = output_dir / "regression_feature_importance.png"
        plt.savefig(fi_plot_path, dpi=300)
        plt.close()
        print(f"  [OK] Saved feature importance plot: {fi_plot_path}")

    metrics = {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "pearson_r": round(pearson_corr, 4),
        "csi_rain_threshold": round(csi, 4),
    }

    return metrics


def train_eval_classification(df: pd.DataFrame, feature_cols: list[str], output_dir: Path):
    """
    Train & evaluate classification model for rainfall intensity categories.
    """
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFICATION MODEL (Rainfall Intensity Category)")
    print("=" * 60)

    time_col = next((c for c in df.columns if "time" in c.lower()), "valid_time")
    df = df.sort_values(by=time_col).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(train_df["target_rain_class"].astype(int))
    y_test_encoded = le.transform(test_df["target_rain_class"].astype(int))

    if HAS_XGBOOST:
        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        clf_name = "XGBoost Classifier"
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        clf_name = "Random Forest Classifier"

    clf.fit(X_train, y_train_encoded)
    y_pred_encoded = clf.predict(X_test)

    # Inverse transform to original class integers for plotting
    y_test = le.inverse_transform(y_test_encoded)
    y_pred = le.inverse_transform(y_pred_encoded)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    print(f"\n[{clf_name} Results]")
    print(f"  |-- Accuracy:  {acc * 100:.2f}%")
    print(f"  +-- Macro F1:  {f1_macro:.4f}")
    labels = sorted(list(set(y_test) | set(y_pred)))
    target_names_map = {0: "No Rain", 1: "Light", 2: "Moderate", 3: "Heavy"}
    target_names = [target_names_map.get(i, f"Class {i}") for i in labels]

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{clf_name} - Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    cm_plot_path = output_dir / "classification_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved confusion matrix plot: {cm_plot_path}")

    return {
        "model": clf_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Karnataka rainfall prediction")
    parser.add_argument("--input-file", type=str, default="ML_Ready_Karnataka_Rainfall.csv", help="Path to ML dataset CSV")
    parser.add_argument("--output-dir", type=str, default="Model_Outputs", help="Directory for saved plots and metrics")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        if Path("ML_Ready_Karnataka_Rainfall.parquet").exists():
            input_path = Path("ML_Ready_Karnataka_Rainfall.parquet")
        elif Path("ML_Ready_Karnataka_Rainfall.csv").exists():
            input_path = Path("ML_Ready_Karnataka_Rainfall.csv")

    if not input_path.exists():
        print(f"ML dataset {input_path.resolve()} does not exist. Please run create_ml_dataset.py first.")
        return

    print(f"Loading ML dataset from: {input_path.name}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # Exclude non-feature columns (timestamps and targets)
    non_features = [
        "time", "valid_time", "target_next_day_precip_mm", "target_rain_class",
        "year"
    ]
    feature_cols = [c for c in df.columns if c not in non_features and c.lower() != "valid_time"]

    print(f"Loaded ML dataset with {len(df):,} rows.")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    reg_metrics = train_eval_regression(df, feature_cols, output_dir)
    clf_metrics = train_eval_classification(df, feature_cols, output_dir)

    summary = {
        "regression": reg_metrics,
        "classification": clf_metrics
    }

    metrics_file = output_dir / "model_evaluation_metrics.json"
    metrics_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Training Complete] Saved all evaluation metrics to: {metrics_file.resolve()}")


if __name__ == "__main__":
    main()
