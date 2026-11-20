# STAGE 4 EXPERIMENTAL PROTOCOL FINAL SCIENTIFIC REVIEW & CERTIFICATION

**Project:** Machine-Learning-Guided Daily Rainfall Estimation and Forecasting for Karnataka Using High-Resolution ERA5-Land Reanalysis and IMD Observations  
**Document Code:** PROTOCOL-STAGE4-REVIEW-FINAL  
**Date:** November 2026  
**Auditor:** Senior Hydrometeorological Research Scientist, ML Engineer, and Scientific Reviewer  
**Status:** FULLY COMPLIANT & CERTIFIED FOR EXECUTION  

---

## 1. Executive Summary

This document provides the final scientific review and certification of the **Stage 4 Experimental Protocol**. Following the findings of Stage 3.1 and Stage 3.2, this review verifies that all experimental procedures adhere to the highest standards of hydrometeorological research integrity, eliminating data leakage, test-set snooping, threshold ambiguity, and unsupported climatological generalizations.

---

## 2. Strict Partition Quarantine and Boundary Hygiene

The dataset partitions prepared in Stage 3 (`stage3_dataset.parquet`, 640,836 valid station-day observations across 252 IMD grid cells over 2,543 calendar days) are strictly quarantined as follows:

| Partition | Temporal Span | Calendar Days | In-State Grid Cells | Total Records | Permitted Operations | Prohibited Operations |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Training Set** | 2019-01-15 to 2022-12-31 | 1,447 | 252 | **364,644** | Parameter optimization; loss minimization; fitting all preprocessors (scalers, power transformers, categorical encoders). Initial 14 days (3,528 records) quarantined as warm-up buffer. | Inference-time evaluation; hyperparameter tuning; decision threshold calibration. |
| **Validation Set** | 2023-01-01 to 2023-12-31 | 365 | 252 | **91,980** | Hyperparameter search (Optuna/Bayesian); early stopping (patience = 50 iterations); probability threshold calibration ($p^*$ for Hurdle classification); feature ablation validation. | Updating model weights or preprocessors; fitting scalers or encoders. |
| **Test Set** | 2024-01-01 to 2025-12-31 | 731 | 252 | **184,212** | **Completely frozen.** Pure out-of-sample forward evaluation. Evaluated exactly once per benchmarked model. | **Zero parameter updates, zero threshold tuning, zero feature selection based on test outcomes.** |

### 2.1. Non-Negotiable Quarantine Rules:
1. **Zero Pre-Scaling / Pre-Encoding Leakage:** Any preprocessor (StandardScaler, RobustScaler, TargetEncoder) must be fitted exclusively on the Training partition (`.fit(X_train)`), and applied downstream via `.transform()` on Validation and Test sets. Applying `.fit_transform()` across concatenated partitions constitutes an immediate protocol violation.
2. **Strict Test Partition Quarantine:** The Test set will not be examined, scored, or used in any iterative feedback loop during model development. Once models are finalized using the Validation set, test predictions are generated in a single pass.
3. **Identical Test Mask Principle:** Every baseline (Persistence, Climatology, Raw ERA5, Empirical Quantile Mapping) and ML model (Ridge, Random Forest, LightGBM, XGBoost, CatBoost, Hurdle Architecture) must be evaluated across the **exact same 184,212 test instances**. Filtering out dry days, dropping coastal stations, or computing metrics on custom subsets without reporting whole-mask metrics is strictly disallowed.

---

## 3. Topographic and Feature Integration Requirements

Following the Stage 3.2 DEM audit, the following adjustments are mandatory for Stage 4 model feature ingestion:

1. **Primary Topographic Features:** Models will train on the verified DEM features extracted from SRTM 3-arc-second data:
   * `elevation_dem_m`: NASA/USGS SRTM 90m point elevation at the 0.25° centroid.
   * `slope_dem_deg`: Regional topographic macro-gradient computed from DEM elevation differences.
2. **Handling of the Barometric Proxy:**
   * The existing columns derived from surface pressure (`sp_mean_hpa`) must be renamed to `elevation_pressure_proxy_m` and `slope_pressure_proxy_deg` if retained.
   * To prevent multicollinearity and physical misattribution, primary models should utilize `elevation_dem_m` and `slope_dem_deg`. If the barometric proxy is evaluated in ablation studies, it must be documented as an atmospheric surface-pressure feature, not terrain height.
3. **Quality Flag Handling:**
   * `is_coastal_fallback` (binary indicator, 0/1) may be included as an auxiliary feature to allow tree-based models to learn specific land-sea coastal boundary corrections.
   * String columns (`interpolation_method`, `season`, `district`, `zone_name`, `date`, `prediction_issue_time`, `target_period`) are strictly excluded from $X$.

---

## 4. Authoritative Regional Context for the 2023 Validation Deficit

In the preliminary Stage 3.1 report, the 2023 validation set was informally described with broad statements regarding "El Niño drought". In this revised protocol, this statement is rigorously supported with authoritative regional meteorological documentation and empirical ground-truth context:

### 4.1. Official Regional Documentation
* **IMD Southwest Monsoon End-of-Season Report 2023 (NWFC, New Delhi):** Documents that August 2023 was the driest August in India since 1901 (36% nationwide deficit). South Peninsular India experienced significant monsoonal failure, with South Interior Karnataka recording a $-24\%$ to $-28\%$ seasonal deficit and Coastal Karnataka recording below-normal rainfall.
* **Karnataka State Natural Disaster Monitoring Centre (KSNDMC) Annual Report 2023:** Confirms widespread agricultural and hydrological drought across the state.
* **Government of Karnataka Revenue Department (Disaster Management) Orders (Sept–Oct 2023):** Officially notified **223 out of 236 taluks** in Karnataka as drought-affected (196 severely affected, 27 moderately affected).

### 4.2. Empirical Ground-Truth Manifestation in the Dataset
In our verified 252-station ground-truth observations (`target_rainfall_mm`):
* Rain occurrence ($\ge 0.1\text{ mm}$) dropped from **$36.95\%$** in the 2019–2022 Training partition to **$27.39\%$** in 2023 (a relative drop of $25.9\%$).
* Standard heavy rainfall ($\ge 64.5\text{ mm}$) dropped by more than half, from **$0.73\%$** in Train to **$0.36\%$** in 2023 (332 total station-days across the entire state).
* **Protocol Value:** Rather than an invalidating defect, the 2023 validation partition serves as a **rigorous out-of-distribution stress test** for evaluating whether models can accurately suppress false alarms and maintain calibration during an extreme dry anomaly.

---

## 5. Retrospective Hindcasting vs. Operational Latency

The protocol explicitly formalizes the distinction between retrospective reanalysis hindcasting and operational real-time forecasting:

1. **Retrospective Hindcasting Convention (Current Research Scope):**
   * Target Date: $D$ (precipitation accumulated from $D-1$ 03:00 UTC to $D$ 03:00 UTC).
   * Prediction Issue Time: $D-1$ 03:00 UTC (08:30 IST).
   * Predictors: All IMD observations and ERA5-Land aggregations ending at or before $D-1$ 03:00 UTC.
   * Assumption: Ground truth observation at $D-1$ 03:00 UTC is available retrospectively in archived datasets.
2. **Operational Real-Time Constraint:**
   * In operational real-time forecasting, IMD manual raingauge telemetry and gridded interpolation incur an operational transmission latency of several hours to 1–2 days.
   * Similarly, ERA5-Land reanalysis is released with a 5-day latency (ERA5T). In an operational environment, reanalysis would be substituted by operational ECMWF IFS numerical weather forecasts.
   * **Mandatory Reporting Requirement:** Any academic paper or technical report emerging from this project must explicitly declare that this framework evaluates **reanalysis-guided hindcast skill under retrospective conditions**, and establish that real-time operational deployment would require either telemetry fast-paths or NWP forecast predictors.

---

## 6. Model Evaluation Metric Suite

Performance will be quantified simultaneously across three operational dimensions:

### 6.1. Continuous Hydrological Accuracy
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **Pearson Correlation ($r$)**
* **Nash-Sutcliffe Efficiency (NSE)**
* **Kling-Gupta Efficiency (KGE-2012)**:
  $$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\beta - 1)^2 + (\gamma - 1)^2}, \quad \beta = \frac{\mu_{\hat{y}}}{\mu_y}, \; \gamma = \frac{\text{CV}_{\hat{y}}}{\text{CV}_y}$$
* **Percent Bias (PBias)**

### 6.2. Categorical Contingency Metrics
Evaluated at official IMD thresholds:
* $T_0 = 0.1\text{ mm}$ (Rain Occurrence / Wet Day)
* $T_1 = 15.6\text{ mm}$ (Moderate Rain)
* $T_2 = 64.5\text{ mm}$ (Standard IMD Heavy Rain)
* Custom $T_{\text{custom}} = 35.5\text{ mm}$ (`rain_ge_35_5mm`)

Metrics computed:
* **Probability of Detection (POD / Recall):** $\frac{H}{H+M}$
* **False Alarm Ratio (FAR):** $\frac{F}{H+F}$
* **Critical Success Index (CSI / Threat Score):** $\frac{H}{H+M+F}$
* **Equitable Threat Score (ETS)**
* **Heidke Skill Score (HSS)**

### 6.3. Spatial Breakdown
All metrics reported statewide and across all 4 meteorological subdivisions: Coastal, Malnad, South Interior Plain, and North Interior.

---

## 7. Stage Gate Certification

**Protocol Status:** APPROVED & FROZEN  
**Sign-off Criteria:**
1. All preprocessing and modeling will adhere strictly to the frozen partition boundaries.
2. Test partition remains untouched until the final single-pass evaluation.
3. Evaluated across all 184,212 test instances under identical masks.
4. Models will ingest verified DEM terrain metrics.

