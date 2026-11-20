import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
stage3_dir = base_dir / "Stage3_4_Final_Release_Audit"
stage4_1_dir = base_dir / "Stage4_1_Model_Development"
stage7_dir = base_dir / "Stage7_Publication_Ready_Package"
repro_dir = stage7_dir / "reproducibility"
repro_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. EXPERIMENTAL CONFIGURATION JSON
# ==============================================================================
print("--- 1. Generating experimental_configuration.json ---")
config = {
    "project_metadata": {
        "title": "Machine-Learning-Guided Daily Rainfall Estimation and Hindcasting for Karnataka",
        "study_area": "Karnataka State, India (252 in-state 0.25° IMD grid cells)",
        "bounding_box": {
            "latitude_min": 11.75,
            "latitude_max": 18.50,
            "longitude_min": 74.25,
            "longitude_max": 78.50
        },
        "target_definition": "Daily rainfall accumulation (mm) ending at 03:00 UTC (08:30 IST)",
        "prediction_issue_time": "03:00 UTC on Day d-1 (24-hour lead hindcasting cycle)",
        "framework": "Retrospective reanalysis hindcasting and diagnostic downscaling (not operational real-time NWP)",
        "random_seed": 42
    },
    "temporal_partitioning": {
        "full_dataset": {
            "start_date": "2019-01-15",
            "end_date": "2025-12-31",
            "total_days": 2543,
            "total_instances": 640836,
            "warmup_quarantine": "2019-01-01 to 2019-01-14 (3,528 records quarantined for 14-day rolling lags)"
        },
        "training_partition": {
            "start_date": "2019-01-15",
            "end_date": "2022-12-31",
            "total_days": 1447,
            "total_instances": 364644,
            "usage": "Model parameter optimization, gradient tree building, fitting all preprocessors (scalers, encoders, EQM knots)"
        },
        "validation_partition": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "total_days": 365,
            "total_instances": 91980,
            "climatological_context": "Severe regional drought year (27.39% wet days vs 36.95% train)",
            "usage": "Early stopping (patience=50), decision threshold calibration (p*), permutation importance, SHAP attribution"
        },
        "test_partition": {
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "total_days": 731,
            "total_instances": 184212,
            "days_2024_leap_year": 366,
            "days_2025": 365,
            "usage": "Frozen forward benchmark evaluation and descriptive reporting only; zero tuning or model re-selection"
        }
    },
    "spatial_discretization": {
        "grid_cell_count": 252,
        "grid_resolution_deg": 0.25,
        "subdivisions": {
            "Coastal Karnataka": {"grid_cells": 13, "test_instances": 9503},
            "Malnad": {"grid_cells": 35, "test_instances": 25585},
            "North Interior Karnataka": {"grid_cells": 118, "test_instances": 86258},
            "South Interior Karnataka": {"grid_cells": 86, "test_instances": 62866}
        },
        "coastal_fallback_cells": {
            "cell_count": 3,
            "station_ids": [52, 68, 125],
            "locations": {
                "52": {"lat": 13.25, "lon": 74.75, "zone": "Udupi / Malpe", "coast_dist_km": 5.94},
                "68": {"lat": 13.50, "lon": 74.75, "zone": "Kundapura", "coast_dist_km": 15.11},
                "125": {"lat": 14.75, "lon": 74.25, "zone": "Karwar / Ankola", "coast_dist_km": 6.98}
            },
            "physical_reason": "IMD centroid located over Arabian Sea water in ERA5-Land land-sea mask; extracted via nearest-inland point",
            "test_instances": 2193
        }
    },
    "target_thresholds_mm": {
        "rain_occurrence": 0.1,
        "custom_diagnostic": 35.5,
        "official_heavy_rain": 64.5,
        "official_very_heavy_rain": 115.6,
        "official_extremely_heavy_rain": 204.5
    },
    "model_hyperparameters": {
        "LightGBM_Tweedie": {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 1000,
            "early_stopping_round": 50,
            "metric": "rmse",
            "random_state": 42
        },
        "CatBoost_Tweedie": {
            "loss_function": "Tweedie:variance_power=1.5",
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "early_stopping_rounds": 50,
            "random_seed": 42
        },
        "LightGBM_Focal_Loss": {
            "objective": "custom_focal_loss",
            "gamma": 2.0,
            "alpha": 0.75,
            "surrogate_hessian": "positive_definite_h_i = w_i * p * (1 - p)",
            "calibrated_decision_threshold": 0.53,
            "threshold_selection_criterion": "Max CSI on 2023 Validation"
        },
        "LightGBM_Rain_Classifier": {
            "objective": "binary",
            "calibrated_decision_threshold": 0.44,
            "threshold_selection_criterion": "Max HSS on 2023 Validation"
        }
    },
    "statistical_bootstrapping": {
        "primary_replicates": 2000,
        "audit_replicates": 5000,
        "block_lengths_days": [7, 14],
        "resampling_protocol": "Stationary block bootstrap preserving all 252 IMD grid cells simultaneously per resampled date",
        "empirical_p_value_resolution": "p < 1 / (B + 1)"
    }
}

with open(repro_dir / "experimental_configuration.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)
print("Saved experimental_configuration.json")

# ==============================================================================
# 2. DATA DICTIONARY CSV (ALL 78 COLUMNS)
# ==============================================================================
print("\n--- 2. Generating data_dictionary.csv ---")
df_man = pd.read_csv(stage3_dir / "final_model_input_manifest.csv")

# Add physical unit mapping
unit_map = {
    "elevation_dem_m": "meters",
    "slope_dem_deg": "degrees",
    "latitude": "decimal degrees",
    "longitude": "decimal degrees",
    "dist_to_coast_km": "kilometers",
    "dist_to_ghats_crest_km": "kilometers",
    "subdivision": "categorical text",
    "zone_id": "categorical integer",
    "month": "calendar month (1-12)",
    "day_of_year": "day of year (1-366)",
    "sin_day_of_year": "dimensionless [-1, 1]",
    "cos_day_of_year": "dimensionless [-1, 1]",
    "sin_month": "dimensionless [-1, 1]",
    "cos_month": "dimensionless [-1, 1]",
    "is_monsoon": "binary indicator (0/1)",
    "target_rainfall_mm": "millimeters",
    "target_is_rain": "binary indicator (0/1)",
    "target_is_heavy_rain": "binary indicator (0/1)",
    "target_rain_ge_35_5mm": "binary indicator (0/1)",
    "is_coastal_fallback": "binary indicator (0/1)"
}

dict_rows = []
for _, r in df_man.iterrows():
    cname = r["column_name"]
    # Infer unit
    if cname in unit_map:
        unit = unit_map[cname]
    elif "rain" in cname or "tp" in cname or cname.startswith("ro") or cname.startswith("sro"):
        unit = "millimeters"
    elif "t2m" in cname or "d2m" in cname or "temp" in cname:
        unit = "degrees Celsius"
    elif "swvl" in cname:
        unit = "m3/m3 (volumetric water content)"
    elif "flux" in cname or "ssrd" in cname or "strd" in cname or "slhf" in cname or "sshf" in cname:
        unit = "Joules per square meter (J/m2)"
    elif "wind" in cname:
        unit = "meters per second (m/s)"
    elif "sp" in cname:
        unit = "hectopascals (hPa)"
    elif cname.startswith("e_") or cname.startswith("pev") or cname.startswith("eva") or cname.startswith("src"):
        unit = "meters (water equivalent)"
    elif "fal" in cname:
        unit = "dimensionless [0, 1] (albedo)"
    else:
        unit = "dimensionless / metadata"
        
    dict_rows.append({
        "column_name": cname,
        "role_in_dataset": r["role_in_dataset"],
        "eligible_for_X_primary": r["eligible_for_X_primary"],
        "feature_group": r["feature_group"],
        "data_type": r["data_type"],
        "unit": unit,
        "availability_timestamp": r["availability_timestamp"],
        "pipeline_handling": r["pipeline_handling"]
    })

df_dict = pd.DataFrame(dict_rows)
df_dict.to_csv(repro_dir / "data_dictionary.csv", index=False)
print(f"Saved data_dictionary.csv ({len(df_dict)} rows)")

# ==============================================================================
# 3. REPRODUCIBILITY MANIFEST CSV (SHA-256 HASHES)
# ==============================================================================
print("\n--- 3. Generating reproducibility_manifest.csv ---")

def calc_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

manifest_files = [
    # Stage 3 Datasets
    (stage3_dir / "data/final_train_dataset.parquet", "Stage 3.4", "Training Dataset Parquet (N=364,644)"),
    (stage3_dir / "data/final_validation_dataset.parquet", "Stage 3.4", "Validation Dataset Parquet (N=91,980)"),
    (stage3_dir / "data/final_test_dataset.parquet", "Stage 3.4", "Test Dataset Parquet (N=184,212)"),
    (stage3_dir / "final_model_input_manifest.csv", "Stage 3.4", "Feature and Column Metadata Manifest"),
    
    # Stage 4 Models & Predictions
    (stage4_1_dir / "models/lgb_tweedie_regressor.joblib", "Stage 4.1", "Trained LightGBM Tweedie Regressor"),
    (stage4_1_dir / "models/cat_tweedie_regressor.joblib", "Stage 4.1", "Trained CatBoost Tweedie Regressor"),
    (stage4_1_dir / "models/lgb_heavy_focal.joblib", "Stage 4.1", "Trained LightGBM Focal Loss Classifier"),
    (stage4_1_dir / "models/pred_lgb_tweedie_test.npy", "Stage 4.1", "Frozen Test Predictions: LightGBM Tweedie"),
    (stage4_1_dir / "models/pred_baseline_persist_test.npy", "Stage 4.1", "Frozen Test Predictions: Persistence"),
    (stage4_1_dir / "models/pred_baseline_era5_test.npy", "Stage 4.1", "Frozen Test Predictions: Raw ERA5-Land"),
    (stage4_1_dir / "models/pred_baseline_eqm_test.npy", "Stage 4.1", "Frozen Test Predictions: EQM"),
    
    # Stage 7 Tables
    (stage7_dir / "master_results_table.csv", "Stage 7.0", "Master Audited Results Table (44 models/strata)"),
    (stage7_dir / "publication_tables/table1_dataset_characteristics.csv", "Stage 7.0", "Table 1: Dataset Characteristics"),
    (stage7_dir / "publication_tables/table2_feature_groups.csv", "Stage 7.0", "Table 2: Feature Groups"),
    (stage7_dir / "publication_tables/table3_model_benchmarking.csv", "Stage 7.0", "Table 3: Model Benchmarking"),
    (stage7_dir / "publication_tables/table4_heavy_rain_classification.csv", "Stage 7.0", "Table 4: Heavy Rain Contingency"),
    (stage7_dir / "publication_tables/table5_regional_seasonal_performance.csv", "Stage 7.0", "Table 5: Regional and Seasonal Performance"),
    (stage7_dir / "publication_tables/table6_statistical_robustness.csv", "Stage 7.0", "Table 6: Statistical Bootstrap Uncertainty"),
    (stage7_dir / "publication_tables/table7_feature_ablation.csv", "Stage 7.0", "Table 7: Feature Ablation"),
    (stage7_dir / "publication_tables/table8_rainfall_intensity_limitations.csv", "Stage 7.0", "Table 8: Intensity Limitations"),
    
    # Stage 7 Figures
    (stage7_dir / "publication_figures/figure1_study_area.png", "Stage 7.0", "Figure 1: Study Area and Grid Cells Map"),
    (stage7_dir / "publication_figures/figure2_methodology.png", "Stage 7.0", "Figure 2: Methodological Workflow"),
    (stage7_dir / "publication_figures/figure3_observed_predicted.png", "Stage 7.0", "Figure 3: Observed vs. Predicted Parity"),
    (stage7_dir / "publication_figures/figure4_model_benchmarking.png", "Stage 7.0", "Figure 4: Model Benchmarking"),
    (stage7_dir / "publication_figures/figure5_heavy_rain_detection.png", "Stage 7.0", "Figure 5: Heavy Rain Detection Contingency"),
    (stage7_dir / "publication_figures/figure6_feature_interpretation.png", "Stage 7.0", "Figure 6: Feature Importance & Domain Sensitivity"),
    (stage7_dir / "publication_figures/figure7_regional_seasonal.png", "Stage 7.0", "Figure 7: Regional and Seasonal Breakdown"),
    (stage7_dir / "publication_figures/figure8_extreme_rainfall_errors.png", "Stage 7.0", "Figure 8: Extreme Rainfall Underestimation"),
    (stage7_dir / "publication_figures/figure9_temporal_robustness.png", "Stage 7.0", "Figure 9: Temporal Robustness & Bootstrap CIs")
]

manifest_rows = []
for pth, stage, desc in manifest_files:
    if pth.exists():
        rel_path = pth.relative_to(base_dir).as_posix()
        sha = calc_sha256(pth)
        size_bytes = pth.stat().st_size
        manifest_rows.append({
            "artifact_name": pth.name,
            "source_stage": stage,
            "relative_path": rel_path,
            "file_type": pth.suffix.replace(".", "").upper(),
            "size_bytes": size_bytes,
            "sha256_checksum": sha,
            "description": desc
        })
    else:
        print(f"Warning: file {pth} not found.")

df_manifest = pd.DataFrame(manifest_rows)
df_manifest.to_csv(repro_dir / "reproducibility_manifest.csv", index=False)
print(f"Saved reproducibility_manifest.csv ({len(df_manifest)} tracked artifacts)")

# ==============================================================================
# 4. METHODOLOGY REPRODUCTION MARKDOWN
# ==============================================================================
print("\n--- 4. Generating methodology_reproduction.md ---")
methodology_content = """# Methodology and Protocol Reproduction Guide

**Project:** Karnataka Hydrometeorological Daily Rainfall Hindcasting  
**Repository:** `d:\\PROJECT\\karnataka_rainfall_cds\\`  
**Purpose:** End-to-end procedural guide detailing data pipelines, de-accumulation, spatial collocation, model training, evaluation, and statistical uncertainty estimation.

---

## 1. Raw Data Requirements and Sourcing

1. **ERA5-Land Reanalysis (ECMWF / C3S):**
   - Downloaded via `cdsapi` from the Climate Data Store.
   - Domain: Latitude 11.0°N to 19.0°N, Longitude 73.5°E to 79.0°E (covering Karnataka and maritime buffers).
   - Variables: 30 parameters including total precipitation (`tp`), 2m temperature (`t2m`), 2m dewpoint (`d2m`), surface pressure (`sp`), 10m wind speed (`u10`, `v10`), soil moisture (`swvl1` to `swvl4`), surface runoff (`sro`), total runoff (`ro`), skin reservoir content (`src`), and energy fluxes (`ssrd`, `strd`, `slhf`, `sshf`).
   - Temporal range: 84 monthly files from 2019-01 to 2025-12.

2. **IMD Gridded Rainfall Data (IMD Pune):**
   - Source: National Climate Centre, India Meteorological Department, Pune.
   - Format: 0.25° × 0.25° binary gridded daily rainfall (`.grd` files).
   - Extracted using `imdlib` for the Karnataka bounding box.
   - Exactly 252 in-state land grid cells identified via official Karnataka administrative boundary shapefile masking.

3. **Copernicus Digital Elevation Model (GLO-90):**
   - Source: European Space Agency (ESA) Copernicus Programme.
   - Native 90m point elevation extracted at the 252 IMD centroids.
   - Macro-slope (degrees) computed across a 27.75 km baseline (matching the 0.25° IMD grid resolution).

---

## 2. Temporal Alignment and Precipitation De-accumulation

A critical scientific pitfall in reanalysis studies is misaligning the reporting cycle.
1. **The Alignment Problem:** IMD daily rainfall accumulations represent rain falling between **08:30 IST on Day d-1 and 08:30 IST on Day d**, corresponding exactly to **03:00 UTC Day d-1 to 03:00 UTC Day d**.
2. **De-accumulation Protocol:** Raw ERA5-Land precipitation accumulates from 00:00 UTC. To prevent leakage and mismatch, hourly fluxes are de-accumulated:
   $$\\Delta P(t) = P(t) - P(t-1)$$
   Hourly values are then aggregated over the 24 hours spanning 03:00 UTC (d-1) to 03:00 UTC (d).
3. **Forensic Verification:** All temporal splits and cumulative counters were audited in Stage 2.1 to guarantee that no cumulative counter resets bleed across calendar days.

---

## 3. Spatial Collocation and the Coastal Fallback Protocol

1. **Native Land Cells:** 249 of the 252 IMD grid cells fell directly over native land cells in the ERA5-Land land-sea mask.
2. **Coastal Fallback Protocol:** Exactly 3 grid cells had centroids located over open Arabian Sea water:
   - Station ID 52 (Lat 13.25°N, Lon 74.75°E, Udupi / Malpe)
   - Station ID 68 (Lat 13.50°N, Lon 74.75°E, Kundapura)
   - Station ID 125 (Lat 14.75°N, Lon 74.25°E, Karwar / Ankola)
3. For these 3 cells, nearest-neighbor inland extraction was applied, and the indicator `is_coastal_fallback = 1` was permanently attached. The remaining 249 cells carry `is_coastal_fallback = 0`.

---

## 4. Leakage-Free Chronological Splitting

1. **Quarantine Buffer:** The first 14 days of 2019 (2019-01-01 to 2019-01-14, 3,528 instances) were quarantined to ensure all 14-day antecedent lags (`api_14d`, `target_rain_lag_14d`, `target_rain_roll_14d_sum`) had 100% complete historical records without imputation.
2. **Training Partition (2019-01-15 to 2022-12-31):** $N = 364,644$ records across 1,447 days.
3. **Validation Partition (2023-01-01 to 2023-12-31):** $N = 91,980$ records across 365 days.
4. **Test Partition [FROZEN] (2024-01-01 to 2025-12-31):** $N = 184,212$ records across 731 days (366 days in 2024 leap year + 365 days in 2025).
5. **Leakage Gate:** Target encodings for `subdivision` and `zone_id` were fitted strictly on Training data.

---

## 5. Model Architecture and Training

1. **Champion Regressor (LightGBM Tweedie):**
   - Objective: Compound Poisson-Gamma (`tweedie`, variance power $p = 1.5$).
   - Log-link: $\\ln(\\mu) = \\sum f_k(X)$.
   - Learning rate: $0.05$, max leaves: $31$, stopping patience: $50$ iterations on Validation RMSE.
2. **Two-Stage Hurdle Architecture:**
   - Stage 1: LightGBM binary classifier ($p^* = 0.44$, calibrated on Validation HSS).
   - Stage 2: LightGBM conditional regressor trained strictly on wet instances ($y > 0.1\\text{ mm}$).
   - Formulation: $\\hat{y}_{EV} = P(\\text{Rain} \\ge 0.1) \\times \\hat{y}_{cond}$.
3. **Extreme Event Classifier (LightGBM Focal Loss):**
   - Custom focal loss objective: $\\mathcal{L}_{FL} = -\\alpha_t (1 - p_t)^\\gamma \\ln(p_t)$ with $\\gamma = 2.0$, $\\alpha = 0.75$.
   - Stabilized using positive-definite surrogate Hessian: $h_i = w_i p (1 - p)$.
   - Decision threshold: $p^* = 0.53$, calibrated strictly on 2023 Validation CSI.

---

## 6. Statistical Uncertainty: Paired Block Bootstrap Protocol

To account for spatial cross-correlation and multi-day temporal autocorrelation:
1. Resample blocks of consecutive calendar days with replacement.
2. **Crucial Rule:** When date $d$ is selected, all 252 statewide grid cells for date $d$ are resampled together, perfectly preserving the statewide spatial correlation matrix.
3. Block lengths tested: $L = 7$ days (synoptic memory) and $L = 14$ days (bi-weekly / MJO intraseasonal memory).
4. Replicates: $B = 2,000$ (and $B = 5,000$ in master audit).
5. Paired difference: $\\Delta \\text{RMSE}^{(b)} = \\text{RMSE}_{ML}^{(b)} - \\text{RMSE}_{Baseline}^{(b)}$.
6. 95% Confidence Intervals: computed from empirical 2.5th and 97.5th percentiles.
7. Resolution limit: Reported as $p < \\frac{1}{B+1}$.
"""

with open(repro_dir / "methodology_reproduction.md", "w", encoding="utf-8") as f:
    f.write(methodology_content)
print("Saved methodology_reproduction.md")

# ==============================================================================
# 5. README MARKDOWN
# ==============================================================================
print("\n--- 5. Generating README.md ---")
readme_content = """# Reproducibility Package: Karnataka Rainfall Retrospective Hindcasting

**Project:** Machine-Learning-Guided Daily Rainfall Estimation and Hindcasting for Karnataka Using ERA5-Land Reanalysis and IMD Gridded Rainfall (2019–2025)  
**Lead Repository:** `d:\\PROJECT\\karnataka_rainfall_cds\\`  
**Package Path:** `d:\\PROJECT\\karnataka_rainfall_cds\\Stage7_Publication_Ready_Package\\`

---

## 1. Project Purpose and Scientific Framing

This research establishes a rigorous, leakage-free machine learning framework for retrospective daily rainfall hindcasting and diagnostic downscaling across the 252 IMD 0.25° grid cells covering Karnataka State, India.

### Key Scientific Framing:
- **Task:** Retrospective hindcasting and reconstruction using ERA5-Land reanalysis predictors.
- **Not NWP:** This framework is a retrospective diagnostic tool, not an operational real-time NWP forecasting harness.
- **Spatial Units:** 252 in-state 0.25° IMD grid cells (not point weather stations).
- **Temporal Generalisation:** Rigorously proven on 2-year out-of-sample forward evaluation (2024–2025, N=184,212).
- **Spatial Generalisation:** Not claimed; spatial transfer to unseen basins is unproven.

---

## 2. Directory Structure

```text
Stage7_Publication_Ready_Package/
│
├── stage7_final_report.md                # Comprehensive master scientific report
├── master_results_table.csv              # Audited results across 44 model/partition rows
│
├── publication_tables/                   # Formatted tables ready for publication
│   ├── table1_dataset_characteristics.csv
│   ├── table2_feature_groups.csv
│   ├── table3_model_benchmarking.csv
│   ├── table4_heavy_rain_classification.csv
│   ├── table5_regional_seasonal_performance.csv
│   ├── table6_statistical_robustness.csv
│   ├── table7_feature_ablation.csv
│   └── table8_rainfall_intensity_limitations.csv
│
├── publication_figures/                  # 300 DPI PNG and vector PDF figures
│   ├── figure1_study_area.png (+ .pdf)
│   ├── figure2_methodology.png (+ .pdf)
│   ├── figure3_observed_predicted.png (+ .pdf)
│   ├── figure4_model_benchmarking.png (+ .pdf)
│   ├── figure5_heavy_rain_detection.png (+ .pdf)
│   ├── figure6_feature_interpretation.png (+ .pdf)
│   ├── figure7_regional_seasonal.png (+ .pdf)
│   ├── figure8_extreme_rainfall_errors.png (+ .pdf)
│   └── figure9_temporal_robustness.png (+ .pdf)
│
└── reproducibility/                      # Exact execution and verification materials
    ├── README.md                         # This file
    ├── environment_requirements.txt      # Exact pip freeze package versions
    ├── experimental_configuration.json   # Full experiment configuration
    ├── data_dictionary.csv               # 78-column metadata dictionary
    ├── reproducibility_manifest.csv      # SHA-256 checksums of all key artifacts
    └── methodology_reproduction.md       # Detailed procedural reproduction guide
```

---

## 3. Principal Verified Findings

1. **Continuous Regression Superiority:**
   - **LightGBM Tweedie ($p=1.5$):** Test RMSE = **$7.4427\\text{ mm}$**, MAE = **$2.8993\\text{ mm}$**, Pearson $r = **0.7363**$, Modified KGE-2012 = **$0.6245$**, Mean Bias = **$-0.0179\\text{ mm}$**.
   - Decisively outperforms Persistence (RMSE $10.3582\\text{ mm}$), Raw ERA5-Land (RMSE $10.4550\\text{ mm}$), and Empirical Quantile Mapping (RMSE $11.1970\\text{ mm}$).
2. **Statistical Significance via Paired Block Bootstrapping:**
   - 5,000 paired block replicates ($L=7\\text{d}$) and 2,000 paired block replicates ($L=14\\text{d}$) produced zero non-negative differences ($p < 0.0005$), demonstrating consistent paired RMSE reductions of **$2.93\\text{ mm}$** over Persistence and **$3.76\\text{ mm}$** over EQM.
3. **Heavy Rain Classification ($\ge 64.5\\text{ mm}$):**
   - **LightGBM Focal Loss ($p^*=0.53$):** Precision = **$58.22\\%$**, Recall = **$46.96\\%$**, CSI = **$0.3512$**, with only **$427$ false alarms** (compared to $722$ false alarms for class weighting).
4. **Dominance of Land-Surface Fluxes:**
   - Surface runoff (`sro_lag_1d`) and total runoff (`ro_lag_1d`) account for **$54.52\\%$ of total gain** in heavy rain classification and lead permutation sensitivity.

---

## 4. Documented Weaknesses & Boundaries

1. **Extreme Tail Quantile Shrinkage:** Continuous models underestimate $100\\%$ of observations exceeding $115.6\\text{ mm}$, plateauing around $70–90\\text{ mm}$.
2. **False Drizzle on Dry Days:** Continuous models output an average of $+0.77\\text{ mm}$ on true zero-rain days.
3. **Interior Plains Predictability Floor:** Convective rainfall in North and South Interior Karnataka exhibits a correlation ceiling ($r \\approx 0.53$) due to sub-grid convective clouds unresolved by hydrostatic reanalysis.
4. **Spatial Generalisation Limitation:** The experimental design does not evaluate spatial transfer to unseen geographical cells outside Karnataka.
"""

with open(repro_dir / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Saved README.md")

print("\n--- Reproducibility Package Fully Built ---")

