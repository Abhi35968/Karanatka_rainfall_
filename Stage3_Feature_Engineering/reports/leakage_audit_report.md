# Stage 3 Spatiotemporal Data Leakage and Causality Audit Report

**Auditor:** Senior Hydrometeorological Research Scientist & Data Engineer  
**Evaluation Dates:** 1 January 2019 to 31 December 2025  
**Clean Sample Pool:** 640,836 station-days across 252 in-state IMD stations  
**Audit Decision:** **PASSED (Zero Data Leakage Detected)**

---

## 1. Problem Formulation & Information Availability Boundary
In an operational forecasting or retrospective hindcast framework, the prediction issue time $T_{\text{issue}}$ marks the boundary of permitted information:

- **Target Observation Window ($y_d$):** Date $d-1$ 03:00 UTC to Date $d$ 03:00 UTC (labeled as IMD rainfall of Date $d$).
- **Prediction Issue Time ($T_{\text{issue}}$):** Date $d-1$ 03:00 UTC ($08:30\text{ IST}$ on Day $d-1$).
- **Causality Rule:** Every predictor variable $X_{d, s}$ for station $s$ on prediction date $d$ must have its maximum information timestamp $T_{X} \le T_{\text{issue}}$.
- **Permitted Predictors:** Meteorological observations, soil moisture states, fluxes, and IMD antecedent rain completed on or before Date $d-1$ (lag $\ge 1\text{ day}$).
- **Prohibited Information:** Any observation from Day $d$ (including same-day ERA5 fluxes or same-day IMD target observations).

---

## 2. Automated Feature Timestamp Audit
- **Total Features Audited:** 64
- **Causal Compliance Rate:** **100.0% (64 / 64)**
- **Causality Violations Detected:** **0**
- Complete feature audit mapping saved to [`feature_timestamp_audit.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage3_Feature_Engineering/feature_timestamp_audit.csv).

---

## 3. Manual Spot-Check Verification on 20 Random Grid-Cell / Date Samples

The table below documents 20 randomly drawn samples verifying that predictor values strictly correspond to completed antecedent observations:

| Sample | Station | Target Date | Prediction Issue Time | Target Window | Target (mm) | Lag-1 Rain (mm) | API 14d (mm) | Lag-1 T2m (°C) | Audit Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 434817 | 171 | 2025-11-26 | 2025-11-25 03:00 UTC | 2025-11-25 03:00 UTC -> 2025-11-26 03:00 UTC | 0.00 | 0.00 | 0.00 | 23.24 | **PASSED (Zero Future Leakage)** |
| 116485 | 46 | 2024-08-26 | 2024-08-25 03:00 UTC | 2024-08-25 03:00 UTC -> 2024-08-26 03:00 UTC | 0.00 | 0.00 | 20.52 | 22.91 | **PASSED (Zero Future Leakage)** |
| 526248 | 207 | 2025-08-01 | 2025-07-31 03:00 UTC | 2025-07-31 03:00 UTC -> 2025-08-01 03:00 UTC | 0.00 | 0.00 | 12.48 | 24.68 | **PASSED (Zero Future Leakage)** |
| 38192 | 16 | 2019-03-03 | 2019-03-02 03:00 UTC | 2019-03-02 03:00 UTC -> 2019-03-03 03:00 UTC | 0.00 | 0.00 | 0.00 | 27.69 | **PASSED (Zero Future Leakage)** |
| 534006 | 210 | 2025-12-08 | 2025-12-07 03:00 UTC | 2025-12-07 03:00 UTC -> 2025-12-08 03:00 UTC | 0.00 | 0.00 | 0.00 | 23.36 | **PASSED (Zero Future Leakage)** |
| 113111 | 45 | 2022-05-18 | 2022-05-17 03:00 UTC | 2022-05-17 03:00 UTC -> 2022-05-18 03:00 UTC | 76.95 | 20.64 | 33.98 | 23.11 | **PASSED (Zero Future Leakage)** |
| 597325 | 235 | 2025-03-27 | 2025-03-26 03:00 UTC | 2025-03-26 03:00 UTC -> 2025-03-27 03:00 UTC | 0.00 | 0.00 | 12.32 | 31.26 | **PASSED (Zero Future Leakage)** |
| 157999 | 63 | 2019-12-14 | 2019-12-13 03:00 UTC | 2019-12-13 03:00 UTC -> 2019-12-14 03:00 UTC | 0.00 | 0.00 | 0.84 | 21.53 | **PASSED (Zero Future Leakage)** |
| 257421 | 102 | 2020-08-15 | 2020-08-14 03:00 UTC | 2020-08-14 03:00 UTC -> 2020-08-15 03:00 UTC | 0.00 | 0.00 | 2.15 | 23.77 | **PASSED (Zero Future Leakage)** |
| 401012 | 158 | 2023-11-11 | 2023-11-10 03:00 UTC | 2023-11-10 03:00 UTC -> 2023-11-11 03:00 UTC | 0.00 | 1.12 | 17.54 | 24.35 | **PASSED (Zero Future Leakage)** |
| 552304 | 218 | 2020-05-02 | 2020-05-01 03:00 UTC | 2020-05-01 03:00 UTC -> 2020-05-02 03:00 UTC | 1.23 | 0.21 | 3.73 | 31.79 | **PASSED (Zero Future Leakage)** |
| 395770 | 156 | 2023-06-08 | 2023-06-07 03:00 UTC | 2023-06-07 03:00 UTC -> 2023-06-08 03:00 UTC | 13.08 | 0.00 | 9.38 | 31.73 | **PASSED (Zero Future Leakage)** |
| 282980 | 112 | 2020-12-22 | 2020-12-21 03:00 UTC | 2020-12-21 03:00 UTC -> 2020-12-22 03:00 UTC | 0.00 | 0.00 | 0.36 | 20.24 | **PASSED (Zero Future Leakage)** |
| 461602 | 182 | 2022-08-26 | 2022-08-25 03:00 UTC | 2022-08-25 03:00 UTC -> 2022-08-26 03:00 UTC | 0.00 | 0.00 | 1.75 | 24.11 | **PASSED (Zero Future Leakage)** |
| 231635 | 92 | 2019-08-25 | 2019-08-24 03:00 UTC | 2019-08-24 03:00 UTC -> 2019-08-25 03:00 UTC | 20.96 | 0.00 | 12.86 | 23.49 | **PASSED (Zero Future Leakage)** |
| 86214 | 34 | 2025-04-28 | 2025-04-27 03:00 UTC | 2025-04-27 03:00 UTC -> 2025-04-28 03:00 UTC | 12.85 | 0.00 | 7.25 | 28.29 | **PASSED (Zero Future Leakage)** |
| 425066 | 168 | 2020-02-04 | 2020-02-03 03:00 UTC | 2020-02-03 03:00 UTC -> 2020-02-04 03:00 UTC | 0.00 | 0.00 | 0.00 | 23.94 | **PASSED (Zero Future Leakage)** |
| 146948 | 58 | 2024-07-04 | 2024-07-03 03:00 UTC | 2024-07-03 03:00 UTC -> 2024-07-04 03:00 UTC | 0.00 | 0.12 | 5.82 | 23.10 | **PASSED (Zero Future Leakage)** |
| 4175 | 2 | 2023-07-05 | 2023-07-04 03:00 UTC | 2023-07-04 03:00 UTC -> 2023-07-05 03:00 UTC | 4.18 | 0.58 | 9.25 | 21.22 | **PASSED (Zero Future Leakage)** |
| 255930 | 101 | 2023-07-03 | 2023-07-02 03:00 UTC | 2023-07-02 03:00 UTC -> 2023-07-03 03:00 UTC | 0.75 | 6.61 | 17.47 | 23.52 | **PASSED (Zero Future Leakage)** |

---

## 4. Leakage Safeguard Checklist
- [x] Target rainfall $y_d$ is strictly taken from IMD observations; ERA5 reanalysis precipitation is never used as ground truth.
- [x] All rolling operations (`rolling(3)`, `rolling(7)`, `rolling(14)`) applied explicit `.shift(1)` before computing sums or means.
- [x] Zero contemporaneous (same-day) ERA5 predictors are present in the final ML feature matrix.
- [x] Initial 14-day warm-up period (2019-01-01 to 2019-01-14) was completely removed to prevent `NaN` leakage or uninitialized rolling states.
- [x] Temporal partitioning strictly splits Train (2019–2022), Validation (2023), and Test (2024–2025) chronologically; zero random shuffling across time.
