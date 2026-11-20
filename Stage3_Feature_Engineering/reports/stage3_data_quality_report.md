# Stage 3 Data Quality and Quality Control Report

**Temporal Scope:** 15 January 2019 to 31 December 2025 (2,543 consecutive calendar days)  
**Spatial Network:** 252 Collocated In-State IMD Stations across Karnataka  
**Retained ML Samples:** 640,836 Station-Days  
**Excluded Warm-Up Samples:** 3,528 Station-Days (14 days x 252 stations, 2019-01-01 to 2019-01-14)  

---

## 1. Quality Control & Integrity Audit

| QC Check | Criterion | Observed Value | QC Status |
| :--- | :--- | :--- | :---: |
| **Missing Predictors** | Exactly 0 NaNs across all features | 0 NaNs | **PASSED** |
| **Missing Targets** | Exactly 0 NaNs in target rainfall | 0 NaNs | **PASSED** |
| **Infinite Values** | Exactly 0 Inf / -Inf across all numeric fields | 0 Inf | **PASSED** |
| **Duplicate Records** | 0 duplicated (date, station_id) pairs | 0 Duplicates | **PASSED** |
| **Negative Rainfall** | Rainfall values must be >= 0.0 mm | Min rain: 0.00 mm | **PASSED** |
| **Unit Verification** | Temperatures in °C, Pressure in hPa, Fluxes in W/m² | Verified | **PASSED** |
| **Physical Value Bounds** | Features lie within standard planetary boundary layer bounds | Verified | **PASSED** |

---

## 2. Partition Summary and Record Allocation

| Split Name | Calendar Date Range | Total Days | Unique Grid Cells | Record Count | Missing Data % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Warm-Up (Excluded)** | 2019-01-01 to 2019-01-14 | 14 | 252 | 3,528 | N/A (Warm-Up) |
| **Training Split** | 2019-01-15 to 2022-12-31 | 1,447 | 252 | 364,644 | **0.0%** |
| **Validation Split** | 2023-01-01 to 2023-12-31 | 365 | 252 | 91,980 | **0.0%** |
| **Test Split** | 2024-01-01 to 2025-12-31 | 731 | 252 | 184,212 | **0.0%** |
| **Total Clean Dataset** | **2019-01-15 to 2025-12-31** | **2,543** | **252** | **640,836** | **0.0%** |

---

## 3. Class Imbalance and Extreme Rainfall Distribution in ML Targets
Evaluating ground-truth IMD observed rainfall classes across the 640,836 clean records:

- **Dry Days ($< 0.1\text{ mm}$):** 460,782 records (71.9%)
- **Light Rain ($0.1 \le y < 2.5\text{ mm}$):** 68,914 records (10.8%)
- **Moderate Rain ($2.5 \le y < 15.6\text{ mm}$):** 68,342 records (10.7%)
- **Heavy Rain ($15.6 \le y < 64.5\text{ mm}$):** 36,928 records (5.8%)
- **Very Heavy / Extreme Rain ($y \ge 64.5\text{ mm}$):** 5,870 records (0.92%)
