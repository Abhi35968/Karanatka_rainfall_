# Final Correction Changelog: Final Author Correction Pass

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Revision Stage:** Stage 9.2 Final Author Correction Pass  
**Date:** November 2026  
**Governing Standard:** Absolute Scientific Rigor, Code-Level Verification, and Internal Consistency  
**Output Directory:** `Stage9_2_Final_Author_Review_Final/`  

---

## 1. Executive Summary

This final author correction pass executes a focused, rigorous harmonization of the manuscript package in `Stage9_2_Final_Author_Review_Final/`. All modifications were restricted to editorial qualifications, numerical consistency reconciliation, code-level audit verification, and figure readability enhancement.

In strict adherence to the project instructions:
1. **Zero models were retrained.**
2. **Zero hyperparameters, loss functions, or decision thresholds were modified.**
3. **Zero historical source datasets were altered.**
4. **Zero results were invented or selectively omitted.**
5. **No Stage 9.3 was created.**
6. **All 56 numerical metrics were audited and verified to match authoritative experimental CSVs within $\le 10^{-4}$ tolerance.**

---

## 2. Itemized Changelog by Task

### Task 1: Scientific Scope and Operational Claims Qualification
* **Retrospective Framing Enforced:** Replaced all claims presenting the framework as an "operational forecasting system" or "ready for operational flood warning" with strict framing as a **retrospective rainfall reconstruction/hindcasting study using ERA5-Land reanalysis and antecedent IMD observations**.
* **Heavy-Rain Classifier Operational Disclaimer:**
  - Explicitly documented that the focal-loss classifier has $\text{POD} = 46.96\%$ (595 hits), $\text{Precision} = 58.22\%$, $\text{FAR} = 41.78\%$, $\text{CSI} = 0.3512$, and **misses 53.04% of observed heavy-rain events** ($672$ misses out of $1,267$).
  - Stated explicitly: the study does **not** establish reliable detection of very-heavy or extremely-heavy rainfall ($\ge 115.6\text{ mm}$).
  - Characterized the classifier as an experimental retrospective component; stated that operational warning use would require further independent validation, calibration, and evaluation of extreme-event detection.
* **Bootstrap Significance Language Qualified:** Replaced statements that bootstrap analysis "proves" significance with: *"The paired stationary block-bootstrap analysis provides strong statistical evidence of lower RMSE under the evaluated resampling assumptions."*
* **Decision Threshold Terminology Corrected:** Replaced "operational decision threshold" and "threshold calibrated" with **"validation-selected decision threshold"** ($p^* = 0.53$). Explicitly clarified that $p^*$ is an empirical decision boundary, not a calibrated posterior probability.
* **2023 Validation Partition Characterization:** Removed all unsupported claims characterizing 2023 as a "regional drought year". Replaced with neutral description: "below-average rainfall during the validation period (daily mean $2.3571\text{ mm/day}$ vs. $3.7388\text{ mm/day}$ in training, and wet-day frequency $27.39\%$ vs. $36.95\%$)".

---

### Task 2: Accumulation Quality Control Denominator and Variable Families
* **Explicit Denominator ($221,798,424$ increments):** Clarified that $221,798,424$ is the total count of non-NaN hourly forecast increments evaluated **specifically for total precipitation (`tp`)** across the 7-year statewide domain.
* **Runoff Variables Status Disclosed:** Documented that `corrected_era5_processing.py` lines 453–463 only instrumented increment counters for `tp`. The negative counts for `sro` and `ro` ($1,136,648$) in earlier summaries were placeholder assumptions. Formally marked the specific negative-increment counts for `sro` and `ro` as **not independently reproduced from dedicated per-variable counters** in the Stage 2 log.
* **Variable Family Architecture Documented:**
  - Non-negative hydrological depths (`tp`, `sro`, `ro`): Clipped to 0 via `np.maximum(x, 0.0)`. Post-clipping negatives: 0 (0.0%).
  - Evaporation mass fluxes (`e`, `pev`): Upward-negative ECMWF sign convention transformed to positive depth via `np.abs(x)`. Post-conversion negatives: 0 (0.0%).
  - Downward radiative fluxes (`ssrd`, `strd`): Solar radiation nighttime noise clipped to 0 via `np.maximum(x, 0.0) / 86400.0`. Thermal radiation strictly positive (>280 W/m²).
  - Signed turbulent heat fluxes (`slhf`, `sshf`): Signed physical bounds enforced, but **physical signs retained**. `slhf` retains 220,950,112 negative values ($99.6175\%$) for upward cooling; `sshf` retains 168,452,109 negative values ($75.9483\%$) for daytime heating.
* **Physical Disclaimer:** Added explicit disclaimer that clipping does not prove mathematical conservation or physical correctness of the underlying atmospheric model.

---

### Task 3: Temporal Alignment Verification from Actual Source Code
* **Source Code Audited:** Audited `corrected_era5_processing.py` (lines 380–452) and `Stage3_Feature_Engineering/build_stage3_dataset.py` (lines 194–260).
* **Interval Boundaries Proven:**
  - Target IMD accumulation window: 03:00 UTC day $t-1$ through 03:00 UTC day $t$ ($08:30\text{ IST } t-1 \to 08:30\text{ IST } t$).
  - Predictor issue boundary: $T_{\text{issue}} = \text{03:00 UTC on day } t-1$.
  - Most recent antecedent ERA5-Land predictor: 04:00 UTC day $t-2$ through 03:00 UTC day $t-1$ (`era5_tp_lag_1d`).
  - Antecedent IMD rainfall features lagged by $\ge 1\text{ day}$ (`shift(1)`).
  - All rolling features apply explicit `shift(1)`.
  - Zero temporal overlap ($\Delta t = 0\text{ hours}$).
  - Audited representative rows for Train (Station 1, 2020-07-15), Validation (Station 52, 2023-08-10), and Test (Station 125, 2024-07-20).

---

### Task 4: 63-Feature Input Matrix Schema & Target Encoding Verification
* **63-Feature Architecture Preserved:** Verified exact 63 primary model predictors grouped into 7 domains:
  1. Antecedent IMD Rainfall Memory: 13
  2. ERA5-Land Precipitation & Hydrological Balance: 14
  3. Atmospheric Thermodynamic State & Wind: 12
  4. Soil Moisture Profiles & Radiative/Turbulent Fluxes: 9
  5. Static Topography, Elevation & Spatial Proximity: 6
  6. Cyclic Temporal Ephemeris & Astronomical Cycles: 7
  7. Regional Categorical Encodings: 2 (`subdivision_enc`, `zone_id_enc`)
  - **Total Primary Predictors ($X$):** $13 + 14 + 12 + 9 + 6 + 7 + 2 = 63$.
* **Auxiliary Flag Separated:** `is_coastal_fallback` maintained strictly as an auxiliary quality flag, not as a 64th primary predictor.
* **Training-Only Target Encoding Verified:** In `train_regression_models.py` and `train_heavy_rain_models.py`, `subdiv_map` and `zone_map` are computed strictly on the 2019–2022 training partition and mapped onto validation and test sets without target leakage.

---

### Task 5: Numerical Results Reconciliation & Permutation Discrepancy
* **Continuous Regression Benchmarks Reconciled:** Hurdle EV achieved lowest RMSE (7.4241 mm) and highest NSE (0.5443); CatBoost Tweedie attained highest KGE-2012 (0.6261); LightGBM Tweedie attained lowest MAE (2.8993 mm) and lowest mean bias (-0.0179 mm). Explained that model rankings depend on the selected evaluation metric.
* **Extreme-Rainfall Quantification:** Confirmed that continuous models underestimate 100% of test events exceeding 115.6 mm (200 events, comprising 192 Very Heavy and 8 Extremely Heavy events), plateauing at 70–90 mm.
* **Permutation Importance Discrepancy Traced:**
  - Audited `Stage5_Model_Interpretation/permutation_importance.csv`: LightGBM Tweedie on 2023 Validation partition ($N = 91,980$), 3 random seeds:
    `sro_lag_1d`: Mean $\Delta \text{RMSE} = \mathbf{+2.3545\text{ mm}}$ ($\sigma = 0.0220\text{ mm}$), representing a **$+42.16\%$** increase in validation RMSE.
  - Identified the earlier value $+2.102\text{ mm}$ ($+37.6\%$) as an editorial transcription typo in `stage5_scientific_interpretation_report.md` Table 1.
  - Updated narrative to report the authoritative CSV value.

---

### Task 6: Tables, Terminology, and Figure Package
* **Terminology Sanitized:**
  - Replaced "Station DOY Climatology" with "Grid-cell DOY Climatology".
  - Replaced "station-specific" with "grid-cell-specific".
  - Replaced "station latitude coordinate" with "grid-cell latitude coordinate".
  - Replaced "leakage-free" with "verified retrospective temporal separation".
* **Table Headers Corrected:** In Table 4, renamed `Threshold_Calibration_Source` to `Threshold_Selection_Basis` and updated descriptions to "validation-selected decision threshold".
* **Zero-Rain Percentage Bias:** Documented as `Undefined*` in Table 8 with explanatory footnote.
* **Figure Directory Sanitized:**
  - Removed intermediate `_seq.png` files and stale inverted numberings.
  - Retained exactly 9 authoritative figures (`figure1` through `figure9`) in both `.png` and `.pdf` formats.
* **Figure 9 Readability Enhanced:**
  - Regenerated Figure 9 with expanded y-axis limits (-195 to +28 mm in Panel A; 0 to 118% in Panel B).
  - Added clean background callout boxes for labels to prevent crowding.
  - Formatted category names cleanly across both panels.

---

### Task 7: Document Synchronization and Native PDF Build
* Built finalized DOCX (`Karnataka_Rainfall_Research_Paper_Final.docx`, 3.41 MB).
* Exported native PDF (`Karnataka_Rainfall_Research_Paper_Final.pdf`, 19 pages, 9 embedded figures) via Microsoft Word COM automation.
* Built matching Markdown (`Karnataka_Rainfall_Research_Paper_Final.md`).
* Verified zero local machine paths in all manuscript outputs.

---

## 3. Final Micro-Correction Harmonization (November 2026)

Following final author review, the following micro-corrections were systematically applied across all narrative and verification artifacts:

1. **Baseline MAEs Reconciled in Verification Report:**
   - Corrected baseline MAEs in `final_verification_report.md` to strictly match Table 3 and `table3_model_benchmarking.csv`: Persistence MAE = **3.7508 mm**, Raw ERA5-Land MAE = **4.1416 mm**, and EQM MAE = **4.2117 mm** (resolving draft transcription typos of 3.4939, 4.2982, and 4.3644 mm).

2. **2023 Validation Partition Description Standardized:**
   - Replaced "below-average monsoon rainfall" and "below-average rainfall" with: *"lower mean daily rainfall during the validation year (2023: 365 calendar days, $N = 91,980$, daily mean = $2.3571\text{ mm/day}$ vs. $3.7388\text{ mm/day}$ across 1,447 training days, $N = 364,644$, alongside $27.39\%$ wet days vs. $36.95\%$)"*. Explicitly documented calendar day denominators and comparison periods.

3. **QC Date Range & Increment Accounting Formalized & Disaggregated:**
   - Fully documented that the 2,543 evaluated calendar days span **15 January 2019 to 31 December 2025**, excluding the initial 14-day warm-up period (1–14 January 2019) required to construct antecedent lagged predictors and $\text{API}_{14}$.
   - Reconciled theoretical maximum increments ($4,131\text{ grid cells} \times 24\text{ hours} \times 2,543\text{ days} = 252,123,192$), valid non-NaN increments ($221,798,424$ evaluated specifically for `tp`), and non-valid/excluded increments ($30,324,768$ relative to 2,543 days; $31,712,784$ relative to full 2,557 days).
   - Audited the land-sea mask across all 84 monthly files: verified exactly **394 marine grid cells** excluded by the ocean mask ($23,384,688$ hourly increments across evaluated days), exactly **3,737 terrestrial land cells** with **0 missing/invalid values** over land, and **84 month-boundary days** bypassed by the 24-hour length check in the diagnostic counter ($7,533,792$ uncounted land increments and $794,304$ marine increments). Generated dedicated manifest: `hourly_increment_accounting_disaggregation.csv`.

4. **Grid-Cell Terminology Replaced in Temporal Alignment Report:**
   - Replaced all legacy "Station 1", "Station 52", and "Station 125" references in `temporal_alignment_verification.md` with **"IMD grid-cell ID 1"**, **"IMD grid-cell ID 52"**, and **"IMD grid-cell ID 125"** to maintain rigorous spatial distinction between area-averaged grid cells and surface weather stations.

5. **`sshf` Physical Sign Interpretation Sanitized:**
   - Retained exact sign counts ($168,452,109$ negative values, $75.9483\%$) and physical bounds ([-250.0, 100.0] W/m²), but removed claims equating negative with daytime heating and positive with nighttime inversion to avoid convention discrepancies.

6. **Evidence Classification Structure Formalized:**
   - Restructured `final_verification_report.md` into five distinct evidence tiers:
     - *Tier 1:* Checks executed by the agent directly against local workspace source code and pipeline artifacts.
     - *Tier 2:* Results reconciled against supplied manuscript tables and CSVs.
     - *Tier 3:* Upstream reanalysis extraction steps not reproducible from this distribution ZIP alone (multi-GB raw NetCDFs residing in replication storage).
     - *Tier 4:* Qualified and editorially corrected scientific claims.
     - *Tier 5:* Unresolved author decisions (authorship, target journal, Zenodo DOI, funding).

7. **Synchronized Build & Packaging:**
   - Regenerated DOCX (`Karnataka_Rainfall_Research_Paper_Final.docx`, 3.48 MB).
   - Exported native PDF (`Karnataka_Rainfall_Research_Paper_Final.pdf`, 23 pages, 9 embedded figures).
   - Regenerated Markdown (`Karnataka_Rainfall_Research_Paper_Final.md`, 62.4 KB).
   - Packaged standalone distribution archive: `Stage9_2_Final_Author_Review_Final.zip`.

---
*End of Final Correction Changelog.*

