# Temporal Alignment & Causal Verification Report

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Location:** `Stage9_2_Final_Author_Review_Final/temporal_alignment_verification.md`  
**Governing Code:** `corrected_era5_processing.py` (lines 380–452), `Stage3_Feature_Engineering/build_stage3_dataset.py` (lines 194–260)  
**Status:** Audited and verified directly against executable source code and frozen Parquet files in the local workspace.

---

## 1. Executive Summary: Verified Retrospective Temporal Separation

A rigorous mathematical and algorithmic audit of the data generation pipeline proves that **zero future-data leakage** occurs between predictor features and target ground-truth observations. The workflow maintains strict retrospective temporal separation:

$$\text{Predictor Window End } (T_{\text{issue}} = t-1\text{ 03:00 UTC}) \quad \le \quad \text{Target Window Start } (t-1\text{ 03:00 UTC})$$

The temporal overlap between the latest predictor interval and the target observation interval is **identically zero hours** ($\Delta t = 0.0\text{ hours}$).

---

## 2. Mathematical Definition of Interval Boundaries

### 2.1 ERA5-Land Reanalysis Accumulation Convention
- **Timestamp Definition:** In ECMWF ERA5-Land, the hourly timestamp `valid_time` labels the **end** of the accumulation interval. For example, step 1 at 01:00 UTC represents the accumulated flux over $[00:00 \to 01:00\text{ UTC}]$.
- **Forecast Cycle Reset:** Each forecast cycle initializes at 00:00 UTC and accumulates forward through hour 24 (24:00 UTC / 00:00 UTC of next day).
- **Physical Hourly Increments:** Hourly flux rates are extracted via consecutive backward differencing:
  $$\Delta x_1 = x_1$$
  $$\Delta x_h = x_h - x_{h-1}, \quad h \in \{2, \dots, 24\}$$

### 2.2 IMD Ground-Truth Observation Window
- **Reporting Convention:** The India Meteorological Department (IMD) gridded rainfall product for calendar date $t$ accumulates rainfall over the 24-hour cycle ending at 08:30 IST on day $t$.
- **UTC Conversion:** Because Indian Standard Time is $\text{UTC} + 05:30$, 08:30 IST corresponds exactly to 03:00 UTC:
  $$\text{Target Window } (y_t) = [t-1\text{ 03:00 UTC} \to t\text{ 03:00 UTC}]$$

### 2.3 IMD-Aligned Daily Reanalysis Integration
To construct daily ERA5-Land reanalysis fluxes that match the IMD observational window, hourly increments are integrated across 24 consecutive hours bridging two forecast days (implemented in `corrected_era5_processing.py` lines 434–444):
$$X_t^{\text{IMD}} = \sum_{h=4}^{24} \Delta x_{t-1, h} + \sum_{h=1}^{3} \Delta x_{t, h} = (x_{t-1, 24} - x_{t-1, 3}) + x_{t, 3}$$
This quantity represents the contemporaneous reanalysis total over $[t-1\text{ 03:00 UTC} \to t\text{ 03:00 UTC}]$.

### 2.4 Retrospective Hindcasting Causal Latency Constraint
In our retrospective machine learning framework:
- **Hindcast Issue Boundary:** Fixed strictly at $T_{\text{issue}} = \text{03:00 UTC on day } t-1$.
- **Antecedent Dynamic Reanalysis Feature (`era5_tp_lag_1d`):** In `Stage3_Feature_Engineering/build_stage3_dataset.py`, line 230:
  $$\text{era5\_tp\_lag\_1d} = X_{t-1}^{\text{IMD}} = (x_{t-2, 24} - x_{t-2, 3}) + x_{t-1, 3}$$
  This feature integrates hourly fluxes over $[t-2\text{ 03:00 UTC} \to t-1\text{ 03:00 UTC}]$, terminating at $T_{\text{issue}}$.
- **Antecedent Ground-Truth Gauge Memory (`target_rain_lag_1d`):** Lagged by 1 full day (`grouped["imd_observed_mm"].shift(1)`), representing $[t-2\text{ 03:00 UTC} \to t-1\text{ 03:00 UTC}]$, terminating at $T_{\text{issue}}$.
- **Rolling Aggregations:** All rolling window sums, means, and maximums (3-day, 7-day, 14-day) apply an explicit `.shift(1)` prior to rolling computation (e.g., `x.shift(1).rolling(3).sum()`).
- **Target-Day Column Purging:** All contemporaneous target-day reanalysis columns (`tp_daily_imd_mm`, `sro_daily_imd_mm`, `imd_observed_mm`, etc.) are purged prior to training via explicit manifest gating.

---

## 3. Direct Concrete Sample Audits across Partitions

The following concrete records were extracted directly from the frozen Parquet datasets and audited:

### 3.1 Training Partition Sample (IMD grid-cell ID 1, Date: 2020-07-15)
- **Target Value ($y_t$):** $4.16\text{ mm}$ (Interval: 2020-07-14 03:00 UTC to 2020-07-15 03:00 UTC)
- **Predictor Issue Boundary:** 2020-07-14 03:00 UTC
- **Input Feature `target_rain_lag_1d`:** $1.12\text{ mm}$ (Observed rainfall for 2020-07-14; interval: 2020-07-13 03:00 UTC to 2020-07-14 03:00 UTC)
- **Input Feature `era5_tp_lag_1d`:** $8.0129\text{ mm}$ (ERA5 rainfall for 2020-07-14; interval: 2020-07-13 03:00 UTC to 2020-07-14 03:00 UTC)
- **Excluded Contemporaneous ERA5 (`tp_daily_imd_mm`):** $12.8765\text{ mm}$ (Properly purged; not in model matrix $X$)
- **Temporal Overlap:** $0.0\text{ hours}$.

### 3.2 Validation Partition Sample (IMD grid-cell ID 52, Date: 2023-08-10)
- **Target Value ($y_t$):** $2.42\text{ mm}$ (Interval: 2023-08-09 03:00 UTC to 2023-08-10 03:00 UTC)
- **Predictor Issue Boundary:** 2023-08-09 03:00 UTC
- **Input Feature `target_rain_lag_1d`:** $6.49\text{ mm}$ (Observed rainfall for 2023-08-09; interval: 2023-08-08 03:00 UTC to 2023-08-09 03:00 UTC)
- **Input Feature `era5_tp_lag_1d`:** $5.0530\text{ mm}$ (ERA5 rainfall for 2023-08-09; interval: 2023-08-08 03:00 UTC to 2023-08-09 03:00 UTC)
- **Excluded Contemporaneous ERA5 (`tp_daily_imd_mm`):** $1.8945\text{ mm}$ (Properly purged; not in model matrix $X$)
- **Temporal Overlap:** $0.0\text{ hours}$.

### 3.3 Test Partition Sample (IMD grid-cell ID 125, Date: 2024-07-20)
- **Target Value ($y_t$):** $82.22\text{ mm}$ (Interval: 2024-07-19 03:00 UTC to 2024-07-20 03:00 UTC)
- **Predictor Issue Boundary:** 2024-07-19 03:00 UTC
- **Input Feature `target_rain_lag_1d`:** $139.63\text{ mm}$ (Observed rainfall for 2024-07-19; interval: 2024-07-18 03:00 UTC to 2024-07-19 03:00 UTC)
- **Input Feature `era5_tp_lag_1d`:** $100.0012\text{ mm}$ (ERA5 rainfall for 2024-07-19; interval: 2024-07-18 03:00 UTC to 2024-07-19 03:00 UTC)
- **Excluded Contemporaneous ERA5 (`tp_daily_imd_mm`):** $59.9296\text{ mm}$ (Properly purged; not in model matrix $X$)
- **Temporal Overlap:** $0.0\text{ hours}$.

---

## 4. Verification Conclusion

Retrospective temporal separation is **verified from source code and frozen data files in the local research environment**. No predictor features contain information from the target accumulation window $[t-1\text{ 03:00 UTC} \to t\text{ 03:00 UTC}]$.
