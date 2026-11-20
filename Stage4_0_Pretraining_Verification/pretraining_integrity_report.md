# STAGE 4.0 PRE-TRAINING INTEGRITY CHECK AND EXPERIMENT SETUP REPORT

**Project:** Machine-Learning-Guided Daily Rainfall Estimation and Forecasting for Karnataka Using High-Resolution ERA5-Land Reanalysis and IMD Observations  
**Document Code:** PRETRAIN-INTEGRITY-STAGE4.0-REP  
**Date:** November 2026  
**Status:** **AUDITED, SANITIZED & APPROVED FOR MODEL TRAINING**  
**Lead Investigators:** Senior Hydrometeorological Research Scientist, Machine Learning Engineer, and Data Systems Architect  

---

## 1. Executive Summary & Stage Gate Decision

Prior to initiating model training in Stage 4, an exhaustive pre-training integrity audit was executed under **Stage 4.0** to verify the physical datasets, validate the input matrix reconstruction, prove zero temporal leakage, and freeze the experimental evaluation protocol.

In strict adherence to project governance rules:
* **No machine learning models have been trained.**
* **All prior datasets and scripts (Stage 2 through Stage 3.4) remain preserved untouched.**
* **The test partition (184,212 instances) remains completely frozen and quarantined.**

### Pre-Training Gate Verdict: **UNCONDITIONAL PASS (CERTIFIED FOR STAGE 4)**

All verification assertions across dataset row counts, unique grid-cell identifiers, missing values, non-negativity of rainfall targets, terrain value fidelity against Copernicus GLO-90, schema dimensionality, and 50-sample temporal causality checks have passed with a **100.0% success rate**.

---

## 2. Task 1: Final Dataset Dimensionality and Physical Integrity

The finalized datasets in `Stage3_4_Final_Release_Audit/data/` were loaded and inspected directly from disk via PyArrow:

### 2.1. Partition Row and Column Dimensions
* **Full Dataset (`final_stage3_dataset.parquet`):** Exactly **640,836 rows** and **78 columns**.
* **Training Partition (`final_train_dataset.parquet`):** Exactly **364,644 rows** (2019-01-15 to 2022-12-31, 1,447 days; initial 14-day warm-up of 3,528 records quarantined).
* **Validation Partition (`final_validation_dataset.parquet`):** Exactly **91,980 rows** (2023-01-01 to 2023-12-31, 365 days).
* **Test Partition (`final_test_dataset.parquet`):** Exactly **184,212 rows** (2024-01-01 to 2025-12-31, 731 days).
* **Row Sum Verification:** $364,644 + 91,980 + 184,212 = \mathbf{640,836}$ (**PASSED**).

### 2.2. Spatial Uniqueness and Physical Validity
* **Unique IMD Grid Cells:** Exactly **252 unique station IDs** (1 to 252) matching the official in-state Karnataka mask.
* **Duplicate Grid-Cell / Date Combinations:** Exactly **0 duplicates**. Every grid-cell has exactly one observation per calendar day.
* **Missing and Infinite Values in Predictors:** Exactly **0 NaNs** and **0 Infs** across all 63 primary model predictors and auxiliary features across all 640,836 rows.
* **Target Non-Negativity:** Exactly **0 negative rainfall values** ($\min(\text{target\_rainfall\_mm}) = 0.000\text{ mm}$, $\max(\text{target\_rainfall\_mm}) = 398.24\text{ mm}$).

### 2.3. Terrain Feature Verification Against Authoritative Copernicus GLO-90 Source
The terrain features in `final_stage3_dataset.parquet` were compared directly against the authoritative Copernicus DEM GLO-90 extraction records from `dem_source_harmonization_comparison.csv`:
* **`elevation_dem_m` Fidelity:** **252 out of 252 grid cells match exactly** (Max difference: $0.00\text{ m}$, correlation $r = 1.00000$).
* **`slope_dem_deg` Fidelity:** **252 out of 252 grid cells match exactly** (Max difference: $0.0000^\circ$, correlation $r = 1.00000$).
* **Verification Log:** Archived in [`terrain_value_verification_252_cells.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/terrain_value_verification_252_cells.csv).

---

## 3. Task 2: Feature Matrix Schema Reconstruction & Soil Moisture Audit

The feature matrix was reconstructed directly from the Parquet schema and compared against `final_model_input_manifest.csv`. The exact accounting of all 78 columns is certified below:

```mermaid
flowchart TD
    ALL["Final Dataset Parquet Schema<br/>(78 Columns)"]
    
    ALL --> X_PRIM["63 Primary Model Predictors<br/>Strictly Causal & Available <= 03:00 UTC Day d-1"]
    ALL --> X_AUX["1 Auxiliary Feature<br/>is_coastal_fallback (binary 0/1)"]
    ALL --> PROXY["2 Atmospheric Pressure Proxies (Ablation Only)<br/>elevation_pressure_proxy_m, slope_pressure_proxy_deg"]
    ALL --> TGT["4 Target Labels (Strictly Quarantined)<br/>target_rainfall_mm, target_is_rain,<br/>target_is_heavy_rain, target_rain_ge_35_5mm"]
    ALL --> ID["4 Identifiers (Non-features)<br/>date, station_id, district, zone_name"]
    ALL --> PROT["2 Protocol Metadata (Text)<br/>prediction_issue_time, target_period"]
    ALL --> STR["2 Excluded Redundant Strings<br/>interpolation_method, season"]
```

### 3.1. Primary Predictor Space Breakdown (63 Features):
1. **Verified DEM Topography (2):** `elevation_dem_m`, `slope_dem_deg` (Copernicus GLO-90 point elevation at centroid and boundary-corrected regional macro-slope).
2. **Categorical Climate Zones (2):** `subdivision` (4 classes), `zone_id` (1–10).
3. **Spatial Geometry (4):** `latitude`, `longitude`, `dist_to_coast_km`, `dist_to_ghats_crest_km`.
4. **Temporal Solar Ephemeris (7):** `month`, `day_of_year`, `sin/cos_day_of_year`, `sin/cos_month`, `is_monsoon`.
5. **Autoregressive IMD Rainfall Memory (13):** `target_rain_lag_1d`, `2d`, `3d`, `7d`, `14d`; rolling sums, means, and max (3d, 7d, 14d); `api_14d`.
6. **ERA5 Precip Memory (5):** `era5_tp_lag_1d`, `2d`, `3d`; `era5_tp_roll_3d_sum`, `era5_tp_roll_7d_sum`.
7. **Atmospheric Thermodynamic & Land Surface Fluxes (30):**
   * *Thermal & Moisture:* `t2m_mean_lag_1d`, `min`, `max`, `temp_range_lag_1d`, `d2m_mean_lag_1d`, `dewpoint_depression_lag_1d`, `t2m_roll_3d_mean`, `7d_mean`, `sp_lag_1d`.
   * *Wind Circulation:* `wind_speed_lag_1d`, `wind_roll_3d_mean`, `7d_mean`.
   * *Radiation & Energy:* `ssrd_lag_1d`, `strd_lag_1d`, `slhf_lag_1d`, `sshf_lag_1d`.
   * *Water Balance & Evaporation:* `ro_lag_1d`, `sro_lag_1d`, `e_lag_1d`, `pev_lag_1d`, `evabs_lag_1d`, `evatc_lag_1d`, `evavt_lag_1d`, `src_lag_1d`, `fal_lag_1d`.
   * *Soil Moisture Profile (5):* Detailed below.

### 3.2. Detailed Audit of Soil Moisture Profile Features
In response to Task 2, the soil moisture features were audited against the raw ERA5-Land NetCDF variables:
* `swvl1_lag_1d`: Volumetric soil water layer 1 ($0\text{–}7\text{ cm}$ depth), lag 1 day ($m^3/m^3$, active surface layer).
* `swvl1_lag_3d`: Volumetric soil water layer 1, lag 3 days ($m^3/m^3$).
* `swvl1_lag_7d`: Volumetric soil water layer 1, lag 7 days ($m^3/m^3$, surface moisture memory).
* `swvl2_lag_1d`: Volumetric soil water layer 2 ($7\text{–}28\text{ cm}$ depth), lag 1 day ($m^3/m^3$, primary agricultural root zone).
* `swvl3_lag_1d`: Volumetric soil water layer 3 ($28\text{–}100\text{ cm}$ depth), lag 1 day ($m^3/m^3$, subsoil recharge layer).
* *Exclusion of Layer 4 (`swvl4`, $100\text{–}289\text{ cm}$):* Layer 4 represents deep gravitational drainage that exhibits minimal daily variance and negligible correlation with daily surface convective triggering. Excluding `swvl4` prevents multicollinearity while preserving the active hydrological profile.

The complete machine-readable inventory is archived in [`final_feature_matrix_manifest.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/final_feature_matrix_manifest.csv).

---

## 4. Task 3: Temporal Leakage Preflight Findings

A comprehensive temporal preflight audit evaluated 50 stratified random samples covering all 7 calendar years (2019 to 2025), all 4 seasons, and all 4 subdivisions directly against the raw Stage 2 NetCDF extract tables:

* **Prediction Issue Time:** Fixed strictly at **03:00 UTC on Day $d-1$** (08:30 IST).
* **Target Observation Interval:** 24-hour accumulation ending at **03:00 UTC on Day $d$**.
* **Pre-Rolling Shift Rule:** All rolling aggregates (`roll_3d_sum`, `roll_7d_mean`, etc.) and the 14-day Antecedent Precipitation Index (`api_14d`) shift by at least 1 day before aggregation:
  $$\text{roll\_3d\_sum}(d) = \text{Rain}(d-1) + \text{Rain}(d-2) + \text{Rain}(d-3)$$
  $$\text{api\_14d}(d) = \sum_{j=1}^{14} 0.85^j \cdot \text{Rain}(d-j)$$
  Zero shift 0 ($\text{Rain}(d)$) is included.
* **Latest Source Timestamp:** Every single predictor observation ended at or before **03:00 UTC Day $d-1$** (100% compliant).
* **Target Label Fidelity:** `target_is_rain` ($y \ge 0.1\text{ mm}$), `target_is_heavy_rain` ($y \ge 64.5\text{ mm}$), and `target_rain_ge_35_5mm` ($y \ge 35.5\text{ mm}$) match ground-truth accumulation across all samples.
* **Preflight Pass Rate:** **50 out of 50 samples PASSED (100.0%)**.

The full 50-row evidence log is archived in [`temporal_leakage_preflight.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/temporal_leakage_preflight.csv).

### 4.1. Formal Documentation of Operational Latency Assumption
In this research framework, we operate under a **retrospective reanalysis hindcasting convention**: ground truth IMD observations and ERA5-Land reanalysis features ending at $(d-1)$ 03:00 UTC are assumed to be finalized and accessible at issue time. In an operational real-time forecasting environment, manual raingauge telemetry and gridded interpolation incur an operational transmission latency of several hours to 1–2 days, and ERA5-Land is released with a 5-day latency (ERA5T). Operational deployment would require numerical weather prediction (NWP) forecasts and automated weather station (AWS) real-time feeds.

---

## 5. Task 4: Chronological Partition Quarantine

The partition boundaries are officially certified and frozen:
1. **Training Partition (364,644 records; 2019-01-15 to 2022-12-31):** Used exclusively for model fitting and preprocessor calibration (`.fit()`).
2. **Validation Partition (91,980 records; 2023-01-01 to 2023-12-31):** Used exclusively for hyperparameter tuning, early stopping (patience = 50), and hurdle classification threshold calibration ($p^*$).
3. **Test Partition (184,212 records; 2024-01-01 to 2025-12-31):** **Completely frozen.** Pure out-of-sample forward evaluation. Evaluated in a single forward pass per final model. Zero parameter updates, zero threshold tuning, and zero feature selection based on test outcomes.
4. **Identical Test Mask Principle:** Every baseline and ML model will be evaluated across the **exact same 184,212 test instances**.

---

## 6. Task 5: Summary of Stage 4 Experiment Configuration

The full experiment configuration has been codified in machine-readable format in [`stage4_experiment_configuration.yaml`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/stage4_experiment_configuration.yaml) and documented in [`stage4_evaluation_protocol.md`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/stage4_evaluation_protocol.md):

### 6.1. Benchmarking Model Hierarchy:
1. **Naive Baselines:** Rainfall Persistence (`target_rain_lag_1d`) & Station Day-of-Year Climatology (Gaussian 7-day smoothed historical DOY mean calibrated on Train).
2. **Reanalysis Baseline:** Antecedent ERA5-Land precipitation accumulation ending 03:00 UTC Day $d-1$ (`era5_tp_lag_1d`).
3. **Statistical Bias Correction:** Empirical Quantile Mapping (EQM) calibrated per subdivision on the Training partition.
4. **Regularized Linear Models:** Ridge & ElasticNet Regression (StandardScaler fitted on Train).
5. **Gradient Boosted Decision Trees:** LightGBM, XGBoost, and CatBoost optimizing Tweedie compound Poisson-Gamma deviance ($p \in [1.1, 1.7]$) and RMSE.
6. **Two-Stage Hurdle Architecture (Primary ML Model):**
   * *Stage 1:* LightGBM Binary Classifier on `target_is_rain` ($\ge 0.1\text{ mm}$) with balanced class weighting. Optimal decision threshold $p^*$ tuned on 2023 Validation set to maximize HSS and CSI.
   * *Stage 2:* LightGBM Regressor trained exclusively on wet observations ($y \ge 0.1\text{ mm}$) with target $\ln(y + 1)$.
   * *Synthesis:* $\hat{y} = 0.0$ if $\hat{p} < p^*$ else $\exp(\hat{y}_{\text{reg}}) - 1$.

### 6.2. Severe Class Imbalance Strategy for Official Heavy Rain ($\ge 64.5\text{ mm}$):
* Heavy rainfall occurs in only **0.666%** of observations state-wide (1 in 150 events; 2,672 in train, 332 in validation, 1,267 in test).
* Models will utilize:
  1. Cost-sensitive weighting (`scale_pos_weight = 135.5`, matching the training negative-to-positive ratio).
  2. Focal loss ($\gamma = 2.0, \alpha = 0.25$).
  3. Probability threshold tuning on Validation data to maximize the Critical Success Index ($\text{CSI}_{64.5}$) rather than unweighted accuracy.

### 6.3. Comprehensive Metric Suite:
* **Continuous:** RMSE, MAE, Pearson $r$, NSE, Kling-Gupta Efficiency (KGE-2012), Percent Bias (PBias).
* **Categorical Contingency:** POD (Recall), FAR, CSI (Threat Score), ETS (Equitable Threat Score), Heidke Skill Score (HSS), F1-Score at $0.1\text{ mm}$, $15.6\text{ mm}$, $35.5\text{ mm}$, and $64.5\text{ mm}$.
* **Calibration:** Brier Score, Expected Calibration Error (ECE), Reliability Diagrams.
* **Subgroup Breakdowns:** Evaluated statewide and reported separately for all 4 meteorological subdivisions and 4 seasons.

---

## 7. Deliverables Checklist (Saved in `Stage4_0_Pretraining_Verification/`)

1. [`pretraining_integrity_report.md`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/pretraining_integrity_report.md) — Master pre-training audit and integrity report.
2. [`final_feature_matrix_manifest.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/final_feature_matrix_manifest.csv) — Complete 78-column role, eligibility, and pipeline handling manifest.
3. [`terrain_value_verification_252_cells.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/terrain_value_verification_252_cells.csv) — 252-cell terrain elevation and slope verification against Copernicus GLO-90.
4. [`temporal_leakage_preflight.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/temporal_leakage_preflight.csv) — 50-sample empirical temporal preflight audit log.
5. [`stage4_experiment_configuration.yaml`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/stage4_experiment_configuration.yaml) — Frozen experiment configuration for all 9 model tiers.
6. [`stage4_evaluation_protocol.md`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_0_Pretraining_Verification/stage4_evaluation_protocol.md) — Frozen evaluation protocol, partition rules, and metric suite.

---

## 8. Final Decision & Sign-Off

* **Dataset Dimensions & Integrity:** 100% verified (640,836 rows; 78 columns; 252 grid cells; 0 NaNs; 0 negative rain).
* **Terrain Values:** 100% verified against Copernicus GLO-90 (252/252 cells match).
* **Feature Schema:** 100% verified (63 primary predictors, 1 auxiliary flag, 14 excluded columns).
* **Temporal Causality:** 100% verified (0 leakage; 50/50 samples passed).
* **Protocol & Partitions:** 100% frozen (Train: 364,644; Val: 91,980; Test: 184,212).
* **Unresolved Issues:** **NONE.**

**Pre-Training Gate Status:** **FULLY APPROVED & CERTIFIED FOR STAGE 4 MODEL TRAINING.**

Per non-negotiable guidelines, **no models have been trained**. I await your explicit approval to initiate **Stage 4: Baseline Benchmarking, Machine Learning Model Training (LightGBM, XGBoost, CatBoost, Hurdle Architecture), and Out-of-Sample Evaluation**.

