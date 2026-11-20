# Final Scientific Verification & Evidence Audit Report

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Location:** `Stage9_2_Final_Author_Review_Final/final_verification_report.md`  
**Governing Standard:** Absolute Scientific Rigor, Code-Level Verification, and Transparent Evidence Classification  

---

## 1. Executive Summary and Critical Evidence Classification

This report provides the formal evidence audit for all claims, equations, numerical values, and narrative interpretations across the finalized research manuscript package.

### Transparent Evidence Hierarchy
Because this final manuscript deliverable package (`Stage9_2_Final_Author_Review_Final.zip`) contains the formatted manuscripts, publication tables, and figures—and does **not** package the multi-gigabyte raw NetCDF/GRIB reanalysis files, intermediate processing scripts, training pipeline scripts, or full multi-year tabular Parquet datasets ($N = 640,836 \times 78$ columns)—we rigorously distinguish:

1. **Checks Executed by the AI Agent Against Local Workspace Source Artifacts:** Analyses performed directly against source code (`corrected_era5_processing.py`, `build_stage3_dataset.py`, `train_regression_models.py`, `train_heavy_rain_models.py`, `stage5_task1_feature_shap.py`) and frozen Parquet files in the local research environment.
2. **Results Reconciled Against Supplied Manuscript Tables:** Quantitative metrics and counts that an independent reviewer can directly verify from the CSV tables in `manuscript_tables/` and `final_numerical_reconciliation.csv` included in this package.
3. **Checks Not Independently Reproducible from this Final ZIP Alone:** Computational checks that an external reviewer cannot reproduce without access to the primary multi-gigabyte raw data and execution repository.
4. **Editorially Corrected Claims:** Terminology, scientific framing, operational disclaimers, and metric contextualizations refined to ensure scientific defensibility.
5. **Unresolved Author Decisions:** Non-technical declarations (author metadata, target journal, Zenodo DOI minting, grant acknowledgments) reserved strictly for the human authors.

---

## 2. Categorized Verification Matrix

### Category 1: Checks Executed by the AI Agent Against Local Workspace Source Artifacts
The following verifications were executed by the AI agent against executable source scripts and frozen Parquet files in the local project directory:

1. **Temporal Alignment & Causal Latency ($T_{\text{issue}} = 03:00\text{ UTC day } t-1$):**
   - Verified from `corrected_era5_processing.py` (lines 380–452) and `Stage3_Feature_Engineering/build_stage3_dataset.py` (lines 194–260).
   - Proven: IMD daily rainfall accumulates over $[t-1\text{ 03:00 UTC} \to t\text{ 03:00 UTC}]$ ($08:30\text{ IST } t-1 \to 08:30\text{ IST } t$).
   - Antecedent ERA5-Land predictors accumulate over $[t-2\text{ 03:00 UTC} \to t-1\text{ 03:00 UTC}]$ (`era5_tp_lag_1d`), terminating at $T_{\text{issue}}$.
   - Ground truth antecedent IMD memory shifted by $\ge 1\text{ day}$ (`shift(1)`).
   - All rolling features apply an explicit `shift(1)`.
   - Overlap between predictor interval and target interval is identically $0.0\text{ hours}$.
   - Audited concrete rows for IMD grid-cell ID 1 (2020-07-15), IMD grid-cell ID 52 (2023-08-10), and IMD grid-cell ID 125 (2024-07-20).

2. **Feature Matrix Architecture ($X \in \mathbb{R}^{N \times 63}$):**
   - Verified from `Stage4_1_Model_Development/train_regression_models.py` (lines 23–45) and `train_heavy_rain_models.py` (lines 20–48).
   - 61 numeric primary features are retained.
   - 2 raw string categorical columns (`subdivision`, `zone_id`) are transformed via target mean encoding strictly fitted on the 2019–2022 training partition (`subdiv_map`, `zone_map`), producing `subdivision_enc` and `zone_id_enc`.
   - The 2 encodings **replace** the raw string columns in $X$, yielding an exact 63-column numerical matrix fed to the models.
   - Auxiliary coastal fallback flag (`is_coastal_fallback`) is strictly excluded from $X$.

3. **Accumulation QC Date Range and Increment Accounting:**
   - **Curated Dataset Period:** The statewide dataset spans **2,543 calendar days** (15 January 2019 to 31 December 2025). The initial 14 calendar days (1–14 January 2019) were excluded as a warm-up period to compute antecedent 14-day rainfall lag features and $\text{API}_{14}$.
   - **Theoretical Maximum Increments:** Over 2,543 days $\times 24\text{ hours/day} \times 4,131\text{ grid cells}$ ($81 \times 51$ bounding box), theoretical maximum spatial increments = **$252,123,192$** (or $253,511,208$ across all 2,557 calendar days).
   - **Valid Non-NaN Increments Evaluated:** Exactly **$221,798,424$ increments** were evaluated specifically for total precipitation (`tp`) across 2,473 days in `corrected_era5_processing.py` (lines 453–463).
   - **Non-Valid or Excluded Increments Disaggregation:** The excluded increments ($30,324,768$ relative to 2,543 days; $31,712,784$ relative to 2,557 days) do not solely represent marine cells. They are rigorously disaggregated into:
     * **Marine Water-Masked Increments:** Invariant ERA5-Land land-sea mask excludes **394 marine grid cells** over the Arabian Sea, accounting for $23,384,688$ hourly increments across evaluated days ($24,179,008$ across all 2,557 days).
     * **Valid Terrestrial Land Grid Cells:** Exactly **3,737 grid cells** ($4,131 - 394 = 3,737$), with **0 additional missing/invalid values** over land.
     * **Month-Boundary Bypassed Increments:** The 24-hour check (`if len(day_steps) == 24:`) bypassed the 84 monthly forecast boundary days (where step 24 was stored in the subsequent month's NetCDF file), accounting for $7,533,792$ uncounted land increments and $794,304$ marine increments (totaling $8,327,136$ increments across 84 days).
     * *(Full reference table: `hourly_increment_accounting_disaggregation.csv`)*.
   - In `tp`, $1,136,648$ increments ($0.512469\%$) fell below $\Delta x_h < -1.0 \times 10^{-5}\text{ mm}$ (median $-1.49 \times 10^{-5}\text{ mm}$, minimum $-4.21 \times 10^{-4}\text{ mm}$), verified as IEEE 754 single-precision float32 representation artifacts.

4. **Retained Signed Negative Turbulent Heat Fluxes:**
   - Verified from `corrected_era5_processing.py` (lines 440–444) and `accumulation_quality_control_reconciliation.csv`.
   - `slhf` retains 220,950,112 negative values ($99.6175\%$) representing upward evaporative cooling.
   - `sshf` retains 168,452,109 negative values ($75.9483\%$) within physical bounds ([-250.0, 100.0] W/m²). Claims attributing diurnal heating/inversion cycles to `sshf` signs were removed to ensure strict physical defensibility across sign conventions.

5. **Permutation Importance Forensic Audit:**
   - Verified from `Stage5_Model_Interpretation/permutation_importance.csv` generated by `stage5_task1_feature_shap.py`.
   - Model: LightGBM Tweedie on 2023 Validation partition ($N = 91,980$, 3 random seeds: 42, 101, 2024).
   - `sro_lag_1d`: Mean $\Delta \text{RMSE} = \mathbf{+2.3545\text{ mm}}$ ($\sigma = 0.0220\text{ mm}$), $+42.16\%$ RMSE increase.

---

### Category 2: Results Reconciled Against Supplied Manuscript Tables
The following 56 numerical metrics match the authoritative CSV tables in `manuscript_tables/` and can be verified directly by any reader of this package:

1. **Continuous Regression Benchmarks (Table 3 / `table3_model_benchmarking.csv`):**
   - Hurdle Expected Value: $\text{RMSE} = 7.4241\text{ mm}$, $\text{MAE} = 2.9300\text{ mm}$, $\text{Bias} = +0.0342\text{ mm}$, $r = 0.7380$, $\text{NSE} = 0.5443$, $\text{KGE-2012} = 0.6114$.
   - CatBoost Tweedie: $\text{RMSE} = 7.4335\text{ mm}$, $\text{MAE} = 2.9186\text{ mm}$, $\text{Bias} = +0.0388\text{ mm}$, $r = 0.7370$, $\text{NSE} = 0.5431$, $\text{KGE-2012} = \mathbf{0.6261}$.
   - LightGBM Tweedie: $\text{RMSE} = 7.4427\text{ mm}$, $\text{MAE} = \mathbf{2.8993\text{ mm}}$, $\text{Bias} = \mathbf{-0.0179\text{ mm}}$ ($-0.51\%$), $r = 0.7363$, $\text{NSE} = 0.5420$, $\text{KGE-2012} = 0.6245$.
   - Naive Persistence: $\text{RMSE} = 10.3582\text{ mm}$, $\mathbf{\text{MAE} = 3.7508\text{ mm}}$, $\text{NSE} = 0.1129$.
   - Raw ERA5-Land: $\text{RMSE} = 10.4550\text{ mm}$, $\mathbf{\text{MAE} = 4.1416\text{ mm}}$, $\text{NSE} = 0.0963$.
   - Empirical Quantile Mapping (EQM): $\text{RMSE} = 11.1970\text{ mm}$, $\mathbf{\text{MAE} = 4.2117\text{ mm}}$, $\text{NSE} = -0.0366$.

2. **Heavy-Rain Classification Metrics (Table 4 / `table4_heavy_rain_classification.csv`):**
   - LightGBM Focal Loss ($\gamma = 2.0$, $\alpha = 0.75$, validation-selected decision threshold $p^* = 0.53$):
     * Hits: $595$
     * Misses: $672$ (**Miss rate: $53.04\%$**)
     * False Alarms: $427$
     * Correct Negatives: $182,518$
     * Precision: $58.22\%$
     * POD / Recall: $46.96\%$
     * False Alarm Ratio (FAR): $41.78\%$
     * Critical Success Index (CSI): $0.3512$
     * Heidke Skill Score (HSS): $0.5169$
     * Equitable Threat Score (ETS): $0.3485$
     * PR-AUC: $0.5084$

3. **Extreme-Rainfall Event Counts & Underestimation (Table 8 / `table8_rainfall_intensity_limitations.csv`):**
   - No Rain ($<0.1\text{ mm}$): $N = 119,669$ ($64.96\%$), Bias $+0.77\text{ mm}$, Percentage Bias `Undefined*` (division by 0.00 mm mean).
   - Heavy ($64.5–115.5\text{ mm}$): $N = 1,067$, Observed Mean $81.68\text{ mm}$, Predicted Mean $46.98\text{ mm}$, Underestimated: $88.6\%$.
   - Very Heavy ($115.6–204.4\text{ mm}$): $N = 192$, Observed Mean $141.52\text{ mm}$, Predicted Mean $71.53\text{ mm}$, Underestimated: **$100.0\%$**.
   - Extremely Heavy ($\ge 204.5\text{ mm}$): $N = 8$, Observed Mean $225.02\text{ mm}$, Predicted Mean $73.66\text{ mm}$, Underestimated: **$100.0\%$**.
   - Total events $\ge 115.6\text{ mm}$: $192 + 8 = \mathbf{200\text{ events}}$, with **$100\%$ conditional mean underestimation**.

4. **Stationary Block Bootstrap Uncertainty (Table 7B / `table6_statistical_robustness.csv`):**
   - $B = 2,000$ paired stationary bootstrap iterations on test partition (2024–2025).
   - $L = 7\text{ days}$ (synoptic): LightGBM vs Persistence $\Delta \text{RMSE} = -2.9155\text{ mm}$ ($95\%\text{ CI } [-3.3089, -2.5539]$, empirical resolution $p < 0.0005$).
   - $L = 14\text{ days}$ (intraseasonal): LightGBM vs Persistence $\Delta \text{RMSE} = -2.9155\text{ mm}$ ($95\%\text{ CI } [-3.4542, -2.4344]$, empirical resolution $p < 0.0005$).

---

### Category 3: Checks Not Independently Reproducible from this Final ZIP Alone
An independent reviewer working strictly from this final deliverable ZIP cannot reproduce the following checks because their underlying multi-gigabyte source artifacts are stored in the primary research replication repository:

1. **Re-execution of ERA5-Land NetCDF Extraction:** Requires downloading the original multi-terabyte Copernicus ERA5-Land GRIB/NetCDF hourly archive (2019–2025) and re-running `corrected_era5_processing.py`.
2. **Re-training and Inference of Tree Ensembles:** Requires the full tabular Parquet datasets (`final_train_dataset.parquet`, `final_validation_dataset.parquet`, `final_test_dataset.parquet`; $N = 640,836 \times 78$ columns) and executing `train_regression_models.py` and `train_heavy_rain_models.py`.
3. **Pre-Clip Runoff Negative Increment Counts:** As documented, `corrected_era5_processing.py` did not instrument separate counters for `sro` or `ro`. Their pre-clip negative increment counts are therefore **not independently reproduced**.

---

### Category 4: Editorially Corrected Claims
1. **Retrospective Reconstruction Framing:** Framed strictly as a **retrospective rainfall reconstruction/hindcasting study using ERA5-Land reanalysis and antecedent IMD observations**. All claims of operational real-time forecasting or flood warning readiness were removed.
2. **Heavy-Rain Classifier Operational Disclaimer:** Documented that the classifier misses **$53.04\%$ of observed heavy events** ($672$ misses) and does **not** establish reliable detection of very-heavy or extremely-heavy precipitation ($\ge 115.6\text{ mm}$).
3. **Bootstrap Significance Qualification:** Replaced claims that bootstrap "proves" significance with: *"The paired stationary block-bootstrap analysis provides strong statistical evidence of lower RMSE under the evaluated resampling assumptions."*
4. **Decision Threshold Terminology:** Replaced "operational decision threshold" and "threshold calibrated" with **"validation-selected decision threshold"** ($p^* = 0.53$), an empirical decision boundary rather than a calibrated posterior probability.
5. **2023 Validation Partition Characterization:** Replaced "below-average monsoon rainfall" with **"lower mean daily rainfall during the validation year (2023: 365 calendar days, $N = 91,980$, daily mean = $2.3571\text{ mm/day}$, wet-day frequency = $27.39\%$) compared to the multi-year training period (2019–2022: 1,447 calendar days, $N = 364,644$, daily mean = $3.7388\text{ mm/day}$, wet-day frequency = $36.95\%$)"**.
6. **IMD Grid-Cell Terminology:** Sanitized all references from "station" to "IMD 0.25° grid cell" (e.g., "IMD grid-cell ID 1", "Grid-cell DOY Climatology", "grid-cell-specific quantile mapping").
7. **SSHF Diurnal Claims Removed:** Removed claims attributing daytime heating or nocturnal inversion to positive/negative signs of `sshf` while retaining verified physical units and signed numerical bounds.

---

### Category 5: Unresolved Author Decisions
The following items remain open for human author determination prior to journal submission:
1. **Author Names and Institutional Affiliations:** Blinded placeholders remain in the manuscript. Authors must insert full names, affiliations, and ORCID iDs if targeting a single-blind journal.
2. **Target Journal Selection:** Primary recommendation: *Journal of Hydrometeorology* (AMS); alternative: *Atmospheric Research* (Elsevier).
3. **Data Archival & Zenodo DOI:** Replication scripts and Parquet data must be uploaded to Zenodo/Figshare to mint a permanent DOI before final acceptance.
4. **Grant & Funding Numbers:** Insert specific grant identification numbers into Section 8.
5. **Permutation Importance Narrative Citation:** Confirm narrative citation of the authoritative CSV value ($\Delta \text{RMSE} = +2.35\text{ mm}$, $+42.2\%$) resolving the earlier report typo ($+2.102\text{ mm}$).

---
*End of Final Scientific Verification & Evidence Audit Report.*
