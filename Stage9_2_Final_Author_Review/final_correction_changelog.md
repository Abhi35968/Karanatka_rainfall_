# Final Correction Changelog: Author-Review Correction Pass (Stage 9.2)

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Revision Stage:** Final Minor Author-Review Correction Pass (Stage 9.2)  
**Date:** November 2026  
**Governing Standard:** Absolute Scientific Rigor, Incontrovertible Source Evidence, and Verified Retrospective Temporal Separation  
**Output Directory:** `Stage9_2_Final_Author_Review/`  

---

## 1. Executive Summary

This final author-review correction pass resolves all remaining scientific ambiguities, editorial inconsistencies, and narrative qualifications identified during independent review of the Stage 9.2 deliverables. All corrections were executed directly using executable source code (`corrected_era5_processing.py`, `build_stage3_dataset.py`, `train_regression_models.py`, `train_heavy_rain_models.py`), authoritative experimental results, and frozen Parquet tables.

In accordance with strict research restrictions:
1. **Zero models were retrained.**
2. **Zero hyperparameters, loss functions, or decision thresholds were modified.**
3. **Zero historical source datasets were altered.**
4. **Zero results were invented or selectively omitted.**
5. **All 56 numerical metrics were audited and verified to match authoritative experimental CSVs within $\le 10^{-4}$ tolerance.**

---

## 2. Itemized Changelog by Task

### Task 1: Resolution of Temporal Alignment & Causal Proof
* **Identified Ambiguity:** Prior draft descriptions did not unambiguously distinguish between the timestamp convention of ERA5-Land hourly accumulations and the 24-hour IMD ground-truth accumulation window, raising potential concerns regarding future data leakage.
* **Source Evidence Examined:**
  - `corrected_era5_processing.py` (lines 196–225, 376–444)
  - `Stage3_Feature_Engineering/build_stage3_dataset.py` (lines 194–202, 226–305)
  - `verify_three_samples.py` (executed against `final_train_dataset.parquet`, `final_validation_dataset.parquet`, and `final_test_dataset.parquet`)
* **Resolution & Physics Established:**
  1. *Timestamp Convention:* In ERA5-Land, the hourly `valid_time` timestamp designates the **end** of the accumulation interval (e.g., step 1 at 01:00 UTC is accumulation from 00:00 to 01:00 UTC).
  2. *Target Window:* Ground truth IMD rainfall for date $t$ accumulates over $[t-1\text{ 03:00 UTC} \to t\text{ 03:00 UTC}]$ ($08:30\text{ IST } t-1 \to 08:30\text{ IST } t$).
  3. *Prediction Issue Boundary:* The hindcast issue time is strictly $T_{\text{issue}} = \text{03:00 UTC on day } t-1$.
  4. *Predictor Window:* The most recent dynamic predictors (`era5_tp_lag_1d`, `target_rain_lag_1d`, `ro_lag_1d`, `sro_lag_1d`) accumulate over $[t-2\text{ 03/04:00 UTC} \to t-1\text{ 03:00 UTC}]$, terminating at $T_{\text{issue}}$.
  5. *Zero Temporal Overlap:* The predictor interval terminates at the exact instant the target interval begins ($\Delta t = 0\text{ hours}$ overlap). Contemporaneous target-day columns were purged before model ingestion (`drop_unlagged`).
  6. *Concrete Sample Audits:* Formally verified three sample grid-cells:
     - *Train:* Station 1, Date 2020-07-15 ($y_t = 4.16\text{ mm}$, lag-1 IMD = 1.12 mm, lag-1 ERA5 = 8.0129 mm; contemporaneous ERA5 = 12.8765 mm excluded).
     - *Validation:* Station 52, Date 2023-08-10 ($y_t = 2.42\text{ mm}$, lag-1 IMD = 6.49 mm, lag-1 ERA5 = 5.0530 mm; contemporaneous ERA5 = 1.8945 mm excluded).
     - *Test:* Station 125, Date 2024-07-20 ($y_t = 82.22\text{ mm}$, lag-1 IMD = 139.63 mm, lag-1 ERA5 = 100.0012 mm; contemporaneous ERA5 = 59.9296 mm excluded).
* **Deliverable Created:** `temporal_alignment_source_evidence.md`.

---

### Task 2: Accumulation Quality Control & Negative Residuals Reconciliation
* **Identified Inconsistency:** 
  1. The manuscript text previously claimed that all negative increments become zero after clipping, whereas the QC CSV reported hundreds of millions of negative values remaining for turbulent heat fluxes (`slhf`, `sshf`).
  2. A mathematical contradiction existed between the reported detection threshold ($\Delta x_h < -1.0 \times 10^{-5}\text{ mm}$) and the previously reported median negative residual ($-3.70 \times 10^{-6}\text{ mm}$, which is algebraically greater than $-1.0 \times 10^{-5}\text{ mm}$).
* **Source Evidence Examined:**
  - `corrected_era5_processing.py` (lines 453–463, 701–703)
  - `Stage2_Corrected_Processing/reports/accumulation_validation_report.md`
  - Re-evaluation of backward differences across 221,798,424 hourly increments.
* **Resolution & Clarification:**
  1. *Underlying QC Statistics:* Backward differencing across 221,798,424 increments revealed that 1,136,648 differences ($0.512469\%$) fell below the numerical check threshold of $\Delta x_h < -1.0 \times 10^{-5}\text{ mm}$. The true median of detected negative residuals is $-1.49 \times 10^{-5}\text{ mm}$ (minimum $-4.21 \times 10^{-4}\text{ mm}$), fully consistent with IEEE 754 single-precision float representation artifacts ($2^{-23} \approx 1.19 \times 10^{-7}$).
  2. *Distinct Variable Family Clipping Rules:*
     - **Non-Negative Hydrological & Radiative Depths:** For precipitation (`tp`), surface runoff (`sro`), total runoff (`ro`), and downward solar radiation (`ssrd`), variables are physically non-negative. Single-precision truncation residuals were clipped via `np.maximum(x, 0.0)`. Post-clipping negatives are strictly zero ($0.0\%$). Downward thermal radiation (`strd`) is strictly positive (>280 W/m²) and required zero clipping.
     - **Evaporation Mass Fluxes (Sign Inversion):** ECMWF stores evaporation (`e`) and potential evaporation (`pev`) as upward mass fluxes with negative sign convention. Absolute values were taken via `np.abs(x)` to represent positive water-equivalent depths. Post-transformation negatives are strictly zero ($0.0\%$).
     - **Signed Turbulent Energy Fluxes:** Latent heat flux (`slhf`) and sensible heat flux (`sshf`) are signed physical energy fluxes ($\text{W/m}^2$, dividing accumulated Joules by 86,400 s). Physical bounds were enforced ([-300.0, 50.0] W/m² for `slhf`, [-250.0, 100.0] W/m² for `sshf`), but **their physical signs were validly retained**. `slhf` retains 220,950,112 negative values ($99.6175\%$) representing upward evaporative cooling; `sshf` retains 168,452,109 negative values ($75.9483\%$) representing surface sensible heating during convective daytime conditions.
* **Deliverables Updated:** `accumulation_quality_control_reconciliation.csv`, Section 3.1 in DOCX/PDF/Markdown manuscripts.

---

### Task 3: Feature Accounting & Model Input Architecture Reconciliation
* **Identified Inconsistency:** 
  1. Section 3.2 text previously listed five feature groups totaling 57 features, contradicting Table 2 whose seven predictor groups totaled 63 features.
  2. Clarification was required on whether categorical columns (`subdivision`, `zone_id`) are expanded or replaced by target encodings in the model feature matrix $X$.
* **Source Evidence Examined:**
  - `Stage4_1_Model_Development/train_regression_models.py` (lines 23–45, target encoding dictionary mapping)
  - `Stage4_1_Model_Development/train_heavy_rain_models.py` (lines 25–48)
  - `Stage3_4_Final_Release_Audit/final_model_input_manifest.csv`
  - Direct execution of `verify_model_columns.py` confirming column shapes.
* **Resolution & Reconstructed Schema:**
  1. *Model Feature Matrix Architecture:* In the training pipeline, the model input matrix $X$ takes the 61 numeric primary features and applies target-mean encoding to the 2 raw string categorical columns (`subdivision` and `zone_id`), producing `subdivision_enc` and `zone_id_enc`. These 2 numerical encodings **replace** the raw string columns in $X$, yielding an exact 63-column numerical matrix ($X \in \mathbb{R}^{N \times 63}$) fed to LightGBM, CatBoost, XGBoost, and linear models.
  2. *Reconciled Seven Predictor Groups:*
     - Group 3: Antecedent IMD Gauge Memory (13 features)
     - Group 4: ERA5-Land Precipitation & Hydrological Balance (14 features)
     - Group 5: Atmospheric Thermodynamic State & Wind (12 features)
     - Group 6: Soil Moisture Profiles & Radiative/Turbulent Energy Fluxes (9 features)
     - Group 7: Static Topography, Elevation & Spatial Proximity (6 features)
     - Group 8: Cyclic Temporal Ephemeris & Astronomical Cycles (7 features)
     - Group 9: Primary Regional Categorical Encodings (2 features: `subdivision_enc`, `zone_id_enc`)
     - **Total Primary Predictors ($X$):** $13 + 14 + 12 + 9 + 6 + 7 + 2 = 63\text{ features}$.
  3. *Full Schema Reconciliation (78 Columns):*
     - 63 Primary Gated Features ($X$)
     - 4 Ground-Truth Target Labels
     - 4 Spatial & Administrative Identifiers (`station_id`, `district`, `zone_name`, `target_period`)
     - 2 Protocol Metadata Strings (`date`, `prediction_issue_time`)
     - 1 Auxiliary Spatial Quality Flag (`is_coastal_fallback`)
     - 4 Excluded Features (2 elevation/slope pressure proxies, `interpolation_method`, `season`)
     - Total = 78 columns.
* **Deliverables Updated:** `complete_63_feature_manifest.csv`, `table2_feature_groups.csv`, Section 3.2 in DOCX/PDF/Markdown manuscripts.

---

### Task 4: Scientific Interpretations, 2023 Neutralization & Terminology Sanitization
* **2023 Drought Characterization Neutralized:**
  - Removed "regional drought year" from Table 1 (`table1_dataset_characteristics.csv`) and all manuscript text.
  - Cites both the daily rainfall mean ($2.3571\text{ mm/day}$ in validation vs. $3.7388\text{ mm/day}$ in training) alongside reduced wet-day frequency ($27.39\%$ vs. $36.95\%$) so wet-day frequency alone is not presented as sole proof.
* **Terminology Sanitization:**
  - Replaced residual "station latitude", "station longitude", and station descriptions with IMD grid-cell terminology: "IMD 0.25° grid-cell centroid latitude/longitude coordinate".
  - Replaced "DOY Station Climatology" with "Grid-Cell DOY Climatology".
  - Replaced "station-wise non-parametric quantile mapping" with "grid-cell-wise non-parametric quantile mapping".
  - Replaced "Station-Wise Knots" with "Grid-Cell Knots" in Table 4 and text.
  - Replaced unqualified "leakage-free" with "retrospective temporal separation verified".
* **Decision Threshold Terminology Corrected:**
  - Replaced "calibrated threshold" with "validation-selected threshold" or "validation-selected decision threshold" ($p^* = 0.53$).
* **Operational Readiness Formally Disclaimed:**
  - Framed the framework as a retrospective hydrometeorological reconstruction and gap-filling engine operating on 24-hour daily aggregated boundaries under a 2- to 3-month reanalysis latency, and explicitly disclaimed real-time operational nowcasting.
* **Extreme Quantile Shrinkage Formally Documented:**
  - Highlighted that continuous models underestimate $100\%$ of test observations $\ge 115.6\text{ mm}$ ($200$ events, comprising 192 Very Heavy and 8 Extremely Heavy events), plateauing at $70–90\text{ mm}$.

---

### Task 5: Document Build and Cross-Format Parity
* **Natively Exported PDF:**
  - Built `Karnataka_Rainfall_Research_Paper_Final.docx` (3.41 MB) with all updated tables, figures, and narrative.
  - Exported `Karnataka_Rainfall_Research_Paper_Final.pdf` (19 pages, 9 embedded figures, verified via PyMuPDF) using native Microsoft Word COM automation.
  - Built `Karnataka_Rainfall_Research_Paper_Final.md` with complete mathematical expressions, synchronized text, and identical table layouts.
* **Zero Local Paths:** Verified zero local machine file paths in all published manuscript text, tables, and captions.

---
*End of Final Correction Changelog.*
