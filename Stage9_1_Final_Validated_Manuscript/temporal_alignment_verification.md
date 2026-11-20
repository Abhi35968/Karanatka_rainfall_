# Spatiotemporal Alignment and Causality Verification Report

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Investigation Scope:** Stage 2 Preprocessing (`corrected_era5_processing.py`), Stage 3 Feature Engineering (`Stage3_Feature_Engineering/build_stage3_dataset.py`), and Final Released Datasets (`Stage3_4_Final_Release_Audit/data/*.parquet`).  
**Auditor:** Senior Hydrometeorological Research Scientist, Atmospheric Data Engineer, and Statistical Auditor  
**Date:** November 2026  
**Final Audit Conclusion:** **PASSED — ZERO DATA LEAKAGE CONFIRMED**

---

## 1. Executive Summary and Causality Verdict

A forensic, line-by-line inspection of the underlying executable Python scripts (`corrected_era5_processing.py` and `build_stage3_dataset.py`) and parquet data files was conducted to resolve apparent contradictions in the manuscript's description of temporal intervals.

### The Apparent Manuscript Contradiction
Earlier manuscript drafts stated:
1. Daily IMD ground-truth rainfall accumulation ends at 03:00 UTC on day $t$.
2. ERA5-Land hourly increments are integrated from 04:00 UTC on day $t-1$ through 03:00 UTC on day $t$.
3. All predictors are restricted to the antecedent 24-hour cycle ending at 03:00 UTC on day $t-1$.

To an external reviewer, statements 2 and 3 appeared irreconcilable because if the reanalysis accumulation spans $t-1$ 04:00 UTC to $t$ 03:00 UTC, ingesting it to predict day $t$ would violate causal ordering and leak contemporaneous target-window information.

### Empirical Code Investigation Finding
The underlying code strictly enforces causal lag separation. The apparent contradiction arose entirely from a semantic conflation between two distinct data products:
- **Baseline Reanalysis Product ($X_t^{\text{IMD}}$):** In `corrected_era5_processing.py`, an IMD-aligned daily ERA5-Land series was constructed for the interval **04:00 UTC day $t-1$ to 03:00 UTC day $t$** (`tp_daily_imd_mm` on date $t$). This series was created exclusively as a **benchmarking baseline (raw ERA5-Land / EQM downscaling)** to compare reanalysis rainfall against observed IMD rainfall over the exact same 24-hour accumulation window.
- **Machine Learning Feature Matrix ($X_{t-1}^{\text{IMD}}$):** In `build_stage3_dataset.py` (lines 228–296), all dynamic ERA5-Land fluxes and IMD gauge totals were explicitly lagged using `.shift(1)`. For target date $t$, the ingested feature `era5_tp_lag_1d` corresponds to the antecedent date $t-1$, which was integrated over the interval **04:00 UTC day $t-2$ to 03:00 UTC day $t-1$**.
- **Contemporaneous Dropping:** All unlagged, contemporaneous columns for date $t$ (`imd_observed_mm`, `tp_daily_imd_mm`, `corrected_era5_imd_aligned_mm`, and all unlagged hourly flux totals) were explicitly purged (`df_final = df_merged.drop(columns=drop_unlagged)`, line 305).

**Conclusion:** The maximum information timestamp of all predictor variables for sample date $t$ is strictly **03:00 UTC on day $t-1$** (08:30 IST on day $t-1$). The target observation accumulation window begins exactly at **03:00 UTC on day $t-1$** and concludes at **03:00 UTC on day $t$**. There is zero temporal overlap, zero target-window intrusion, and zero future data leakage.

---

## 2. Itemized Investigation of the Eight Specific Temporal Questions

### 1. How accumulated ERA5-Land variables are de-accumulated
In ECMWF ERA5-Land, surface fluxes and total precipitation are initialized as continuous integrals from 00:00 UTC of each daily forecast cycle. Consecutive backward differencing is executed along the hourly axis:
$$\Delta x_1 = x_1 \quad (\text{at 01:00 UTC})$$
$$\Delta x_h = x_h - x_{h-1}, \quad h \in \{2, 3, \dots, 24\}$$
Each resulting increment $\Delta x_h$ represents the discrete physical flux or precipitation depth across the single hour $(h-1):00$ to $h:00$ UTC.

### 2. How forecast-cycle resets are handled
The numerical forecast resets to zero at 00:00 UTC. At step 24 (00:00 UTC of the subsequent day), the accumulated value $x_{24}$ represents the complete 24-hour integral for that calendar day. For month boundaries, `corrected_era5_processing.py` (lines 296–338, 575–599) bridges the final 24th hour by loading hour 0 from the succeeding month's NetCDF archive (`next_head_ds`), ensuring seamless differencing across calendar months.

### 3. Which hourly increments are included in each daily total
For an IMD-aligned daily aggregation labeled with date $d$, the integrated total encompasses 24 consecutive hourly increments:
$$X_d^{\text{IMD}} = \sum_{h=4}^{24} \Delta x_{d-1, h} + \sum_{h=1}^{3} \Delta x_{d, h}$$
In terms of raw accumulated forecast steps:
$$X_d^{\text{IMD}} = \left[ A_{d-1}(24) - A_{d-1}(3) \right] + A_d(3)$$
This spans hour 04:00 UTC of day $d-1$ through hour 03:00 UTC of day $d$.

### 4. The exact UTC interval used for each predictor date
For any model prediction evaluating target date $t$:
- Single-day lagged dynamic predictors (`_lag_1d`) ingest records stamped with date $d = t-1$.
- Because date $t-1$ spans from $t-2$ 03:00 UTC to $t-1$ 03:00 UTC, the aggregation interval for dynamic reanalysis predictors is strictly:
  $$\mathbf{[03:00\text{ UTC day } t-2 \;\longrightarrow\; 03:00\text{ UTC day } t-1]}$$
- Static geographical features (coordinates, DEM elevation, slope, distances to coast/crest) and deterministic orbital ephemeris (`sin_day_of_year`, `month`) are constant or known a priori at issue time.

### 5. The exact UTC interval used for the IMD target
The IMD ground-truth target for date $t$ (`target_rainfall_mm`) represents rainfall recorded at 08:30 IST on day $t$. Converted to Coordinated Universal Time, the target accumulation interval is strictly:
$$\mathbf{[03:00\text{ UTC day } t-1 \;\longrightarrow\; 03:00\text{ UTC day } t]}$$

### 6. Availability of features prior to target accumulation
Every dynamic predictor feature used to predict day $t$ is fully accumulated, de-accumulated, and available at or before **03:00 UTC on day $t-1$**. The target accumulation window opens at **03:00 UTC on day $t-1$**. Thus, the predictor boundary and target window meet at a singular temporal inflection point ($T_{\text{issue}} = 03:00\text{ UTC day } t-1$) without overlapping.

### 7. Correct shifting of rolling and antecedent features
All rolling window aggregations were inspected in `build_stage3_dataset.py` (lines 235–272). Every rolling transformation explicitly applies `.shift(1)` prior to computing `.rolling(N)`:
- `target_rain_roll_3d_mean`: `x.shift(1).rolling(3).mean()` $\to$ spans days $t-3, t-2, t-1$.
- `target_rain_roll_7d_sum`: `x.shift(1).rolling(7).sum()` $\to$ spans days $t-7$ through $t-1$.
- `target_rain_roll_14d_sum`: `x.shift(1).rolling(14).sum()` $\to$ spans days $t-14$ through $t-1$.
- `api_14d` (Antecedent Precipitation Index): computed via $\sum_{i=1}^{14} 0.85^i \cdot y_{t-i}$, strictly utilizing lags $1 \le i \le 14$.
No rolling feature contains contemporaneous day $t$ information.

### 8. Target-day information entering the predictor matrix
Zero target-day information enters the feature matrix. Unlagged columns (`imd_observed_mm`, `tp_daily_imd_mm`, `t2m_mean_c`, etc.) are dropped in line 305 of `build_stage3_dataset.py`. The final feature matrix $X$ consists strictly of the 63 primary lagged and static predictors.

---

## 3. Comprehensive Variable Group Temporal Mapping Table

The table below documents the source timestamps, aggregation intervals, operational roles, and leakage assessments for every variable family in the retrospective modeling pipeline:

| Variable Group | Source Timestamps | Aggregation Interval | Target / Predictor Role | Leakage Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Ground-Truth Target Rainfall** (`target_rainfall_mm`, `target_is_rain`, `target_is_heavy_rain`) | IMD Daily Gridded Archive (0.25°) | Day $t-1$ 03:00 UTC to Day $t$ 03:00 UTC (08:30 IST to 08:30 IST) | **Target Variable ($y_t$)**; strictly quarantined from feature matrix $X$ | **Verified Ground Truth.** Evaluated out-of-sample on frozen partitions; zero backward propagation into $X$. |
| **Antecedent IMD Gauge Memory** (`target_rain_lag_1d` to `_14d`, `target_rain_roll_3d/7d/14d`, `api_14d`) | IMD Daily Gridded Archive (0.25°) | Day $t-14$ 03:00 UTC through Day $t-1$ 03:00 UTC (all shifted by $\ge 1$ day) | **Primary Predictors ($X_t$)** (13 features) | **Zero Leakage.** Maximum timestamp is Day $t-1$ 03:00 UTC. Concludes exactly as target window begins. |
| **ERA5-Land Precipitation Memory** (`era5_tp_lag_1d` to `_3d`, `era5_tp_roll_3d/7d`) | Hourly ERA5-Land Reanalysis (`tp`) | Day $t-7$ 04:00 UTC through Day $t-1$ 03:00 UTC (all shifted by $\ge 1$ day) | **Primary Predictors ($X_t$)** (5 features) | **Zero Leakage.** De-accumulated and shifted by 1 day; maximum timestamp is Day $t-1$ 03:00 UTC. |
| **ERA5-Land Raw Benchmark** (`Antecedent_ERA5_Land`, `EQM`) | Hourly ERA5-Land Reanalysis (`tp`) | Day $t-1$ 04:00 UTC through Day $t$ 03:00 UTC (unlagged relative to target window) | **Benchmarking Baseline Only**; NOT included in ML feature matrix $X_t$ | **Zero Leakage.** Evaluated as independent physical baseline product ($X_t^{\text{IMD}}$) against $y_t$; never supplied as an input feature to ML models. |
| **ERA5-Land Surface Runoff & Water Balance** (`sro_lag_1d`, `ro_lag_1d`, `e_lag_1d`, `pev_lag_1d`, `evabs/evatc/evavt_lag_1d`, `src_lag_1d`) | Hourly ERA5-Land De-accumulated Fluxes | Day $t-2$ 04:00 UTC through Day $t-1$ 03:00 UTC | **Primary Predictors ($X_t$)** (9 features) | **Zero Leakage.** Diagnostically derived from HTESSEL land model; maximum timestamp is Day $t-1$ 03:00 UTC. |
| **Atmospheric Thermodynamics & Wind** (`t2m_mean/min/max_lag_1d`, `d2m_mean_lag_1d`, `dewpoint_depression_lag_1d`, `sp_lag_1d`, `wind_speed_lag_1d`, rolling means) | Hourly ERA5-Land Instantaneous Fields | Day $t-2$ 04:00 UTC through Day $t-1$ 03:00 UTC | **Primary Predictors ($X_t$)** (12 features) | **Zero Leakage.** Averaged/min/max over the antecedent 24 hours ending Day $t-1$ 03:00 UTC. |
| **Soil Moisture Dynamics** (`swvl1_lag_1d/3d/7d`, `swvl2_lag_1d`, `swvl3_lag_1d`) | Hourly ERA5-Land Volumetric Water Layers | Day $t-7$ to Day $t-1$ 03:00 UTC | **Primary Predictors ($X_t$)** (5 features) | **Zero Leakage.** Instantaneous state fields shifted by $\ge 1$ day; maximum timestamp is Day $t-1$ 03:00 UTC. |
| **Surface Radiative & Turbulent Energy Fluxes** (`ssrd_lag_1d`, `strd_lag_1d`, `slhf_lag_1d`, `sshf_lag_1d`, `fal_lag_1d`) | Hourly ERA5-Land De-accumulated Energy Fluxes | Day $t-2$ 04:00 UTC through Day $t-1$ 03:00 UTC | **Primary Predictors ($X_t$)** (5 features) | **Zero Leakage.** Energy integrals converted to mean $\text{W/m}^2$; maximum timestamp is Day $t-1$ 03:00 UTC. |
| **Static Orography & Geography** (`latitude`, `longitude`, `elevation_dem_m`, `slope_dem_deg`, `dist_to_coast_km`, `dist_to_ghats_crest_km`) | Copernicus GLO-90 DEM & Geodesic Calculation | Invariant across all dates | **Primary Predictors ($X_t$)** (6 features) | **Zero Leakage.** Fixed geographical constants established prior to model training. |
| **Temporal Ephemeris** (`month`, `day_of_year`, `sin/cos_day_of_year`, `sin/cos_month`, `is_monsoon`) | Deterministic Calendar Calculation | Day $t$ Calendar Ephemeris | **Primary Predictors ($X_t$)** (7 features) | **Zero Leakage.** Perfectly known astronomical date attributes at issue time $T_{\text{issue}}$. |
| **Regional Categorical Encodings** (`zone_id`, `subdivision`) | Administrative & ICAR / UAS Agroclimatic Maps | Fixed spatial zones (target encoding fitted strictly on 2019–2022 Train set) | **Primary Predictors ($X_t$)** (2 raw / 2 encoded) | **Zero Leakage.** Out-of-fold target encoding strictly computed on training data; validation and test sets transformed using frozen training means. |
| **Auxiliary Quality Flag** (`is_coastal_fallback`) | Land-Sea Mask Forensic Analysis | Invariant indicator (0/1) for Grid-Cell IDs 52, 68, 125 | **Auxiliary Flag** (separate from 63 primary predictors) | **Zero Leakage.** Static flag indicating cells where bilinear interpolation touched ocean cells. |

---

## 4. Required Manuscript Adjustments

To ensure the published manuscript is completely unambiguous and prevent future readers from confusing baseline evaluation intervals with predictor intervals, the text must incorporate the following precise clarifications:

1. **Clarify Dual Role in Section 3.1:**
   - Define $X_t^{\text{IMD}}$ (accumulated from 04:00 UTC $t-1$ to 03:00 UTC $t$) explicitly as the **contemporaneous reanalysis benchmark** used to evaluate raw ERA5-Land and EQM against ground-truth IMD observations $y_t$.
   - Explicitly define the machine-learning predictor feature as $X_{t-1}^{\text{IMD}}$ (`era5_tp_lag_1d`), integrated over **04:00 UTC $t-2$ to 03:00 UTC $t-1$**, terminating at 03:00 UTC on day $t-1$.
2. **Explicit Formula for the Issue Time Boundary:**
   State clearly that for any target day $t$:
   $$T_{\text{issue}} = \text{03:00 UTC on day } t-1$$
   $$\text{Predictor Information Interval: } [-\infty, \, T_{\text{issue}}]$$
   $$\text{Target Evaluation Interval: } [T_{\text{issue}}, \, T_{\text{issue}} + 24\text{ hours}]$$
3. **Operational Publication Latency Disclaimer:**
   Reiterate that official ERA5-Land data has an operational release latency of 2 to 3 months. Therefore, this framework constitutes an **offline retrospective hydrometeorological reconstruction and gap-filling engine**, rather than an operational real-time forecasting system.

---
*End of Spatiotemporal Alignment and Causality Verification Report.*

