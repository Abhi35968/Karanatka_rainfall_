# Temporal Alignment Source Evidence & Causal Verification Audit

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Location:** `Stage9_2_Final_Author_Review/temporal_alignment_source_evidence.md`  
**Audit Date:** November 2026  
**Auditor:** Atmospheric Science & Computational Audit Team  
**Governing Standard:** Strict Hindcast Causality & Zero Data Leakage Protocol  

---

## 1. Executive Answers to the 7 Governing Temporal Alignment Questions

### Question 1: Does ERA5-Land timestamp indicate beginning or end of accumulation?
**Answer:** In ECMWF ERA5-Land hourly GRIB/NetCDF products, the `valid_time` timestamp designates the **end** of the hourly accumulation interval.
- Specifically, the forecast integration step initializes each day at `00:00 UTC` ($A(0) = 0$).
- The first recorded accumulation step at `01:00 UTC` represents:
  $$A(1) = \int_{\text{00:00 UTC}}^{\text{01:00 UTC}} \phi(t) \, dt$$
- Step $h \in \{1, \dots, 24\}$ labels the accumulation interval $[(h-1)\text{:00 UTC} \to h\text{:00 UTC}]$.
- At step 24 (`24:00 UTC` or `00:00 UTC` of the next calendar day), the accumulated value $A(24)$ represents the full calendar day integral from `00:00 UTC` to `24:00 UTC`.
- Therefore, when extracting the interval between 03:00 UTC on day $t-1$ and 03:00 UTC on day $t$, the accumulated slice spans from hour 04 of day $t-1$ (accumulation from 03:00 to 04:00 UTC) through hour 03 of day $t$ (accumulation from 02:00 to 03:00 UTC).

---

### Question 2: When IMD daily rain is labeled date $t$, what exact UTC and IST interval does it cover?
**Answer:** The India Meteorological Department (IMD) gridded daily rainfall archive defines daily rainfall on date $t$ as the precipitation accumulated over the 24-hour observation window ending at **08:30 IST on date $t$**.
- Because Indian Standard Time (IST) is $\text{UTC} + \text{05:30}$:
  $$\text{08:30 IST on date } t = \text{03:00 UTC on date } t$$
- The 24-hour observational accumulation window for IMD rainfall labeled date $t$ ($y_t$) is:
  $$\text{Target Window } (y_t) = [\text{08:30 IST day } t-1 \to \text{08:30 IST day } t] \equiv [\text{03:00 UTC day } t-1 \to \text{03:00 UTC day } t]$$

---

### Question 3: What is the hindcast prediction issue time $T_{\text{issue}}$?
**Answer:** The hindcast prediction issue time is defined strictly as:
$$T_{\text{issue}} = \text{03:00 UTC on date } t-1 \quad (\text{08:30 IST on date } t-1)$$
- At this exact boundary, the ground-truth observation period for target rainfall $y_t$ commences.
- Any observation or reanalysis field occurring after $T_{\text{issue}}$ belongs to the target accumulation window and is strictly inaccessible to the model feature matrix.

---

### Question 4: For target date $t$, what is the exact time window of the most recent dynamic predictor?
**Answer:** All dynamic hydrometeorological predictors ingested into the model feature matrix for target date $t$ are derived from lag-1 (or greater) antecedent histories:
- **Antecedent IMD Rainfall (`target_rain_lag_1d`):**
  $$\text{Window: } [\text{03:00 UTC day } t-2 \to \text{03:00 UTC day } t-1] \quad (\text{08:30 IST } t-2 \to \text{08:30 IST } t-1)$$
  Reported by IMD at 08:30 IST on day $t-1$ ($T_{\text{issue}}$).
- **Antecedent ERA5-Land Precipitation & Surface Fluxes (`era5_tp_lag_1d`, `ro_lag_1d`, `sro_lag_1d`, `e_lag_1d`, `ssrd_lag_1d`, `strd_lag_1d`, `slhf_lag_1d`, `sshf_lag_1d`):**
  $$\text{Window: } [\text{04:00 UTC day } t-2 \to \text{03:00 UTC day } t-1]$$
  Ending at 03:00 UTC on day $t-1$ ($T_{\text{issue}}$).
- **Antecedent Instantaneous Fields (`t2m_mean_lag_1d`, `sp_lag_1d`, `wind_speed_lag_1d`, `swvl1_lag_1d`):**
  $$\text{Window: } 24 \text{ hourly steps from 04:00 UTC day } t-2 \text{ through 03:00 UTC day } t-1$$
  Mean/Min/Max terminating at 03:00 UTC on day $t-1$ ($T_{\text{issue}}$).

---

### Question 5: Does any predictor window overlap with the target accumulation window?
**Answer:** **No. Zero overlap ($\Delta t = 0\text{ hours}$).**
$$\text{Predictor Window: } [\text{03:00/04:00 UTC day } t-2 \to \mathbf{03:00\text{ UTC day } t-1}]$$
$$\text{Target Window: } [\mathbf{03:00\text{ UTC day } t-1} \to \text{03:00 UTC day } t]$$
The predictor window terminates at the exact instant the target accumulation begins. There is zero temporal overlap, zero shared hourly increments, and zero future data leakage.

---

## 2. Source Code Evidence & Line-by-Line Excerpts

### A. ERA5-Land IMD-Aligned De-Accumulation Logic (`corrected_era5_processing.py`)

In `corrected_era5_processing.py`, lines 376–444 demonstrate how 24-hour IMD-aligned aggregates are constructed across daily 00:00 UTC accumulation resets:

```python
# Lines 376-381:
# PRODUCT B: IMD-Aligned Daily Aggregation (03:00 to 03:00 UTC)
# P_IMD(d) = [A_prev(24h) - A_prev(03h)] + A_curr(03h)
curr_step03 = tp_raw[24 * d + 3]
if d == 0:
    # Boundary logic for first day of month using previous month tail...
...
else:
    # Lines 416-419:
    prev_step24 = tp_raw[24 * d]
    prev_step03 = tp_raw[24 * (d - 1) + 3]

# Lines 434-443:
p_imd_val = (prev_step24 - prev_step03) + curr_step03
imd_tp.append(np.maximum(p_imd_val, 0.0))
imd_ro.append(np.maximum((prev_ro24 - prev_ro03) + ro_raw[24 * d + 3], 0.0))
imd_sro.append(np.maximum((prev_sro24 - prev_sro03) + sro_raw[24 * d + 3], 0.0))
imd_e.append(np.abs((prev_e24 - prev_e03) + e_raw[24 * d + 3]))

imd_ssrd.append(((prev_ssrd24 - prev_ssrd03) + ssrd_raw[24 * d + 3]) / 86400.0)
imd_strd.append(((prev_strd24 - prev_strd03) + strd_raw[24 * d + 3]) / 86400.0)
imd_slhf.append(((prev_slhf24 - prev_slhf03) + slhf_raw[24 * d + 3]) / 86400.0)
imd_sshf.append(((prev_sshf24 - prev_sshf03) + sshf_raw[24 * d + 3]) / 86400.0)
```

### B. Strict Causal Lag Shifting & Dropping Contemporaneous Columns (`build_stage3_dataset.py`)

In `Stage3_Feature_Engineering/build_stage3_dataset.py`, lines 194–306 prove that target labels and predictor matrices are strictly separated:

```python
# Lines 194-202:
# Target is IMD observed rainfall for date d (window 03:00 UTC d-1 to 03:00 UTC d)
df_merged["target_rainfall_mm"] = df_merged["imd_observed_mm"].astype(np.float32)
df_merged["target_is_rain"] = (df_merged["target_rainfall_mm"] >= 0.1).astype(int)
df_merged["target_is_heavy_rain"] = (df_merged["target_rainfall_mm"] >= 35.5).astype(int)
df_merged["target_period"] = "03:00 UTC (d-1) to 03:00 UTC (d)"
df_merged["prediction_issue_time"] = "03:00 UTC (d-1)"

# Lines 226-256:
grouped = df_merged.groupby("station_id")

# 1. Target IMD Rainfall History (strictly shifted by >= 1 day)
df_merged["target_rain_lag_1d"] = grouped["imd_observed_mm"].shift(1).astype(np.float32)
df_merged["target_rain_lag_2d"] = grouped["imd_observed_mm"].shift(2).astype(np.float32)
...
# Rolling sums and means apply shift(1) prior to rolling
df_merged["target_rain_roll_3d_sum"] = grouped["imd_observed_mm"].transform(lambda x: x.shift(1).rolling(3).sum()).astype(np.float32)
df_merged["target_rain_roll_7d_sum"] = grouped["imd_observed_mm"].transform(lambda x: x.shift(1).rolling(7).sum()).astype(np.float32)

# 2. Antecedent ERA5 Precipitation History (strictly shifted by >= 1 day)
df_merged["era5_tp_lag_1d"] = grouped["tp_daily_imd_mm"].shift(1).astype(np.float32)
df_merged["era5_tp_lag_2d"] = grouped["tp_daily_imd_mm"].shift(2).astype(np.float32)
df_merged["era5_tp_roll_3d_sum"] = grouped["tp_daily_imd_mm"].transform(lambda x: x.shift(1).rolling(3).sum()).astype(np.float32)

# Lines 296-305:
# Drop unlagged source flux columns to guarantee zero leakage
drop_unlagged = [
    "imd_observed_mm", "corrected_era5_imd_aligned_mm", "tp_daily_imd_mm",
    "t2m_mean_c", "t2m_min_c", "t2m_max_c", "d2m_mean_c", "temp_range_c",
    "dewpoint_depression_c", "sp_hpa", "wind_speed_ms", "swvl1", "swvl2", "swvl3",
    "ssrd_wm2", "strd_wm2", "slhf_wm2", "sshf_wm2", "ro_daily_mm", "sro_daily_mm",
    "e_daily_mm", "pev_daily_mm", "evabs_daily_mm", "evatc_daily_mm", "evavt_daily_mm",
    "src_mm", "fal"
]
df_final = df_merged.drop(columns=drop_unlagged).copy()
```

---

## 3. Concrete Sample Audits across Partitions

The executable verification script (`verify_three_samples.py`) audited 3 sample grid-cells across the training, validation, and test partitions directly against the frozen Parquet files:

```text
================================================================================
SAMPLE AUDIT 1: Train Partition | Grid-Cell ID: 1 | Target Hindcast Date: 2020-07-15
================================================================================
1. Target Observation (y_t):
   - Target Variable Name: `target_rainfall_mm`
   - Target Value: 4.16 mm
   - IMD Observation Timestamp: 2020-07-15 08:30 IST (03:00 UTC)
   - Target Accumulation Period: 2020-07-14 03:00 UTC ---> 2020-07-15 03:00 UTC

2. Prediction Issue Boundary:
   - Issue Time (T_issue): 2020-07-14 03:00 UTC (08:30 IST on day t-1)
   - Max Information Timestamp: 2020-07-14 03:00 UTC

3. Ingested Predictor Features (All Shifted by >= 1 Day):
   - `target_rain_lag_1d` = 1.12 mm
     Source: IMD observation on 2020-07-14 = 1.12 mm (Difference: 5.59e-08)
     Accumulation Window: 2020-07-13 03:00 UTC ---> 2020-07-14 03:00 UTC
   - `era5_tp_lag_1d` = 8.0129 mm
     Source: ERA5-Land IMD-aligned on 2020-07-14 = 8.0129 mm (Difference: 3.52e-07)
     Accumulation Window: 2020-07-13 04:00 UTC ---> 2020-07-14 03:00 UTC
   - `t2m_mean_lag_1d` = 21.31 °C
     Source: ERA5-Land t2m mean on 2020-07-14 = 21.31 °C (Difference: 5.34e-07)
   - `sro_lag_1d` = 1.0241 mm
     Source: ERA5-Land surface runoff on 2020-07-14 = 1.0241 mm (Difference: 5.40e-08)

4. Contemporaneous Target-Day Verification:
   - Same-day unlagged ERA5 total on 2020-07-15: 12.8765 mm
   - Is contemporaneous ERA5 rain present in model input X? False
   - Is contemporaneous IMD target rain present in model input X? False
   - Causal Status: STRICT PASS

================================================================================
SAMPLE AUDIT 2: Validation Partition | Grid-Cell ID: 52 | Target Hindcast Date: 2023-08-10
================================================================================
1. Target Observation (y_t):
   - Target Variable Name: `target_rainfall_mm`
   - Target Value: 2.42 mm
   - IMD Observation Timestamp: 2023-08-10 08:30 IST (03:00 UTC)
   - Target Accumulation Period: 2023-08-09 03:00 UTC ---> 2023-08-10 03:00 UTC

2. Prediction Issue Boundary:
   - Issue Time (T_issue): 2023-08-09 03:00 UTC (08:30 IST on day t-1)
   - Max Information Timestamp: 2023-08-09 03:00 UTC

3. Ingested Predictor Features (All Shifted by >= 1 Day):
   - `target_rain_lag_1d` = 6.49 mm
     Source: IMD observation on 2023-08-09 = 6.49 mm (Difference: 2.33e-07)
     Accumulation Window: 2023-08-08 03:00 UTC ---> 2023-08-09 03:00 UTC
   - `era5_tp_lag_1d` = 5.0530 mm
     Source: ERA5-Land IMD-aligned on 2023-08-09 = 5.0530 mm (Difference: 2.67e-08)
     Accumulation Window: 2023-08-08 04:00 UTC ---> 2023-08-09 03:00 UTC
   - `t2m_mean_lag_1d` = 26.22 °C
     Source: ERA5-Land t2m mean on 2023-08-09 = 26.22 °C (Difference: 6.87e-07)
   - `sro_lag_1d` = 0.0443 mm
     Source: ERA5-Land surface runoff on 2023-08-09 = 0.0443 mm (Difference: 1.11e-09)

4. Contemporaneous Target-Day Verification:
   - Same-day unlagged ERA5 total on 2023-08-10: 1.8945 mm
   - Is contemporaneous ERA5 rain present in model input X? False
   - Is contemporaneous IMD target rain present in model input X? False
   - Causal Status: STRICT PASS

================================================================================
SAMPLE AUDIT 3: Test Partition | Grid-Cell ID: 125 | Target Hindcast Date: 2024-07-20
================================================================================
1. Target Observation (y_t):
   - Target Variable Name: `target_rainfall_mm`
   - Target Value: 82.22 mm
   - IMD Observation Timestamp: 2024-07-20 08:30 IST (03:00 UTC)
   - Target Accumulation Period: 2024-07-19 03:00 UTC ---> 2024-07-20 03:00 UTC

2. Prediction Issue Boundary:
   - Issue Time (T_issue): 2024-07-19 03:00 UTC (08:30 IST on day t-1)
   - Max Information Timestamp: 2024-07-19 03:00 UTC

3. Ingested Predictor Features (All Shifted by >= 1 Day):
   - `target_rain_lag_1d` = 139.63 mm
     Source: IMD observation on 2024-07-19 = 139.63 mm (Difference: 2.69e-06)
     Accumulation Window: 2024-07-18 03:00 UTC ---> 2024-07-19 03:00 UTC
   - `era5_tp_lag_1d` = 100.0012 mm
     Source: ERA5-Land IMD-aligned on 2024-07-19 = 100.0012 mm (Difference: 2.19e-06)
     Accumulation Window: 2024-07-18 04:00 UTC ---> 2024-07-19 03:00 UTC
   - `t2m_mean_lag_1d` = 23.78 °C
     Source: ERA5-Land t2m mean on 2024-07-19 = 23.78 °C (Difference: 6.87e-07)
   - `sro_lag_1d` = 9.0704 mm
     Source: ERA5-Land surface runoff on 2024-07-19 = 9.0704 mm (Difference: 2.38e-07)

4. Contemporaneous Target-Day Verification:
   - Same-day unlagged ERA5 total on 2024-07-20: 59.9296 mm
   - Is contemporaneous ERA5 rain present in model input X? False
   - Is contemporaneous IMD target rain present in model input X? False
   - Causal Status: STRICT PASS
================================================================================
```

---

## 4. Formal Audit Conclusion

The temporal alignment between ERA5-Land hourly integrals, IMD 24-hour observation windows, and machine-learning feature representations conforms strictly to physical hindcast causality:
1. Every dynamic predictor $X_t$ was frozen and available by $T_{\text{issue}} = \text{03:00 UTC on day } t-1$.
2. Target observation $y_t$ accumulates strictly after $T_{\text{issue}}$ until $\text{03:00 UTC on day } t$.
3. All contemporaneous reanalysis fields were dropped prior to model ingestion.
4. No data leakage occurred in any training, validation, or testing partition.

