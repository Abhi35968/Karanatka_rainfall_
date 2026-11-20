import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
data_dir = base_dir / "Stage3_4_Final_Release_Audit/data"
stage4_dir = base_dir / "Stage4_1_Model_Development"
models_dir = stage4_dir / "models"
audit_dir = base_dir / "Stage4_2_Independent_Model_Audit"
audit_dir.mkdir(parents=True, exist_ok=True)

print("Running Task 4: Paired Temporal Block Bootstrap Significance Audit (B=5,000 replicates)...")

df_test = pd.read_parquet(data_dir / "final_test_dataset.parquet")
df_test["date"] = pd.to_datetime(df_test["date"])

y_true = df_test["target_rainfall_mm"].values.astype(np.float64)

# Load test predictions for target models
preds = {
    "Hurdle_EV": np.load(models_dir / "pred_hurdle_ev_test.npy").astype(np.float64),
    "CatBoost_Tweedie": np.load(models_dir / "pred_cat_tweedie_test.npy").astype(np.float64),
    "LightGBM_Tweedie": np.load(models_dir / "pred_lgb_tweedie_test.npy").astype(np.float64),
    "Persistence": np.load(models_dir / "pred_baseline_persist_test.npy").astype(np.float64),
    "Raw_ERA5": np.load(models_dir / "pred_baseline_era5_test.npy").astype(np.float64),
    "EQM": np.load(models_dir / "pred_baseline_eqm_test.npy").astype(np.float64)
}

# Unique dates
unique_dates = np.sort(df_test["date"].unique())
n_dates = len(unique_dates)
block_len = 7
n_blocks = int(np.ceil(n_dates / block_len))

print(f"Total Test Days: {n_dates} days (2024–2025)")
print(f"Spatial Grid Cells: 252 IMD cells per day")
print(f"Total Test Instances: {len(y_true):,}")
print(f"Bootstrap Design: Paired Stationary Temporal Block Bootstrap (L={block_len} days, B=5,000 resamples)")

# Group indices by date for instantaneous spatial field lookup
# Create a 2D matrix of shape (n_dates, 252) for ultra-fast vectorized bootstrap
# Ensure each date has exactly 252 rows sorted identically
df_test["date_idx"] = pd.factorize(df_test["date"])[0]
n_stations = 252

# Pre-compute squared errors matrix of shape (n_dates, 252) for each model
se_matrices = {}
for m_name, y_p in preds.items():
    diff = np.clip(y_p, 0.0, None) - y_true
    se_matrices[m_name] = (diff ** 2).reshape(n_dates, n_stations)

# Comparisons requested in Task 4
comparisons = [
    ("Hurdle_EV", "Persistence", "Hurdle vs Persistence"),
    ("Hurdle_EV", "Raw_ERA5", "Hurdle vs Raw ERA5"),
    ("Hurdle_EV", "EQM", "Hurdle vs EQM"),
    ("CatBoost_Tweedie", "Persistence", "Best Regression (CatBoost Tweedie) vs Persistence"),
    ("CatBoost_Tweedie", "EQM", "Best Regression (CatBoost Tweedie) vs EQM"),
    ("LightGBM_Tweedie", "Persistence", "Best Regression (LightGBM Tweedie) vs Persistence"),
    ("LightGBM_Tweedie", "EQM", "Best Regression (LightGBM Tweedie) vs EQM")
]

# Set seed for strict reproducibility
np.random.seed(42)
B = 5000

# Sample block start indices of shape (B, n_blocks)
start_indices = np.random.randint(0, n_dates - block_len + 1, size=(B, n_blocks))

# Construct array of sampled date indices for each replicate
# Each block contributes [start, start+1, ..., start+block_len-1]
date_offsets = np.arange(block_len)
# sampled_days_matrix of shape (B, n_blocks * block_len)
sampled_days = (start_indices[:, :, None] + date_offsets[None, None, :]).reshape(B, n_blocks * block_len)
# Trim to exact n_dates (731 days) to match sample size
sampled_days = sampled_days[:, :n_dates]

# Precompute mean squared error per day for each model: shape (n_dates,)
mse_per_day = {m: np.mean(se_matrices[m], axis=1) for m in preds}

# Compute replicate RMSEs: shape (B,) for each model
rmse_replicates = {}
for m_name in preds:
    # daily_mse_sampled of shape (B, n_dates)
    daily_mse_sampled = mse_per_day[m_name][sampled_days]
    # mean over all sampled days and take sqrt
    rmse_replicates[m_name] = np.sqrt(np.mean(daily_mse_sampled, axis=1))

# Now compute paired differences across all 5,000 replicates
records = []
for model_a, model_b, comp_label in comparisons:
    diff_reps = rmse_replicates[model_a] - rmse_replicates[model_b] # Delta RMSE = Model - Baseline (negative is good)
    
    mean_diff = float(np.mean(diff_reps))
    median_diff = float(np.median(diff_reps))
    std_diff = float(np.std(diff_reps))
    
    ci_low = float(np.percentile(diff_reps, 2.5))
    ci_high = float(np.percentile(diff_reps, 97.5))
    
    # Empirical two-sided bootstrap p-value
    # Null hypothesis: Delta RMSE >= 0 (Model does not improve over baseline)
    count_non_negative = np.sum(diff_reps >= 0.0)
    p_val_one_sided = float((count_non_negative + 1.0) / (B + 1.0))
    p_val_two_sided = min(1.0, 2.0 * p_val_one_sided)
    
    # Model A and Model B individual CIs
    a_mean = float(np.mean(rmse_replicates[model_a]))
    a_ci_low = float(np.percentile(rmse_replicates[model_a], 2.5))
    a_ci_high = float(np.percentile(rmse_replicates[model_a], 97.5))
    
    b_mean = float(np.mean(rmse_replicates[model_b]))
    b_ci_low = float(np.percentile(rmse_replicates[model_b], 2.5))
    b_ci_high = float(np.percentile(rmse_replicates[model_b], 97.5))
    
    # Relative reduction in RMSE (%)
    pct_reduction = (abs(mean_diff) / b_mean) * 100.0
    
    records.append({
        "comparison": comp_label,
        "model_A": model_a,
        "model_B": model_b,
        "bootstrap_replicates": B,
        "block_length_days": block_len,
        "sample_size_instances": len(y_true),
        "model_A_RMSE_mean": round(a_mean, 4),
        "model_A_RMSE_95CI": f"[{a_ci_low:.4f}, {a_ci_high:.4f}]",
        "model_B_RMSE_mean": round(b_mean, 4),
        "model_B_RMSE_95CI": f"[{b_ci_low:.4f}, {b_ci_high:.4f}]",
        "paired_diff_RMSE_mean": round(mean_diff, 4),
        "paired_diff_RMSE_std": round(std_diff, 4),
        "paired_diff_95CI_low": round(ci_low, 4),
        "paired_diff_95CI_high": round(ci_high, 4),
        "paired_diff_RMSE_95CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "pct_RMSE_reduction": round(pct_reduction, 2),
        "replicates_diff_ge_zero": int(count_non_negative),
        "empirical_p_value": f"< {p_val_one_sided:.5f}" if count_non_negative == 0 else f"{p_val_two_sided:.5f}",
        "statistical_significance_verdict": "STATISTICALLY_SIGNIFICANT" if ci_high < 0.0 else "NOT_SIGNIFICANT"
    })

df_bootstrap = pd.DataFrame(records)
df_bootstrap.to_csv(audit_dir / "paired_bootstrap_rmse_differences.csv", index=False)
print("\nSaved paired_bootstrap_rmse_differences.csv")
print(df_bootstrap[["comparison", "paired_diff_RMSE_mean", "paired_diff_RMSE_95CI", "pct_RMSE_reduction", "empirical_p_value", "statistical_significance_verdict"]].to_string())

