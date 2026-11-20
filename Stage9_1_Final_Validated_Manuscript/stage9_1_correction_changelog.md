# Stage 9.1 Correction Changelog: Final Scientific Corrections and Manuscript Validation

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Revision Stage:** Stage 9.1 Final Scientific Corrections & Peer-Review Validation  
**Date:** November 2026  
**Target Documents:**  
- `Stage9_1_Final_Validated_Manuscript/Karnataka_Rainfall_Research_Paper_Final.docx`  
- `Stage9_1_Final_Validated_Manuscript/Karnataka_Rainfall_Research_Paper_Final.pdf`  
- `Stage9_1_Final_Validated_Manuscript/Karnataka_Rainfall_Research_Paper_Final.md`  
- `Stage9_1_Final_Validated_Manuscript/temporal_alignment_verification.md`  
- `Stage9_1_Final_Validated_Manuscript/numerical_reconciliation.csv`  
- `Stage9_1_Final_Validated_Manuscript/final_author_decisions.md`

---

## 1. Executive Summary

Stage 9.1 executes a rigorous scientific audit, terminology correction, figure/table sequencing alignment, and cross-format document validation based directly on executable code, frozen parquet datasets, and underlying experimental CSV outputs.

No models were retrained, no predictions altered, and no hyperparameters modified. Every modification resolved a demonstrable scientific ambiguity or editorial inconsistency identified in the Stage 9 preliminary draft.

---

## 2. Itemized Changelog across Tasks

### Task 1: Critical Temporal Alignment and Causality Verification
- **Code Inspection:** Conducted line-by-line verification of `corrected_era5_processing.py` and `Stage3_Feature_Engineering/build_stage3_dataset.py`.
- **Disambiguation of Baseline vs. Predictor Products:**
  - Resolved the apparent contradiction regarding interval `[04:00 UTC day t-1 to 03:00 UTC day t]`.
  - Clarified that $X_t^{\text{IMD}}$ was generated in Stage 2 strictly as an unlagged reanalysis baseline (Raw ERA5 Product B) to compare against ground truth $y_t$ over the identical 24-hour accumulation window.
  - Demonstrated that the machine learning feature matrix ingested $X_{t-1}^{\text{IMD}}$ (`era5_tp_lag_1d`), which was accumulated over `[04:00 UTC day t-2 to 03:00 UTC day t-1]`.
  - Demonstrated that the prediction issue time is $T_{\text{issue}} = \text{03:00 UTC day } t-1$.
  - Verified that all rolling features applied `.shift(1)` prior to rolling aggregation.
  - Verified that all contemporaneous columns (`imd_observed_mm`, `tp_daily_imd_mm`, and unlagged flux fields) were explicitly dropped before model training.
- **Audit Deliverable:** Produced `temporal_alignment_verification.md` containing the comprehensive 12-row variable group mapping table and zero-leakage certification.

### Task 2: De-accumulation Mechanics and Unit Transformations
- **Variable-Specific Units Formally Specified:**
  - Precipitation (`tp`), surface runoff (`sro`), and total runoff (`ro`): Converted from raw meters ($m$) to millimeters ($\text{mm}$) depth by multiplying by $1,000$.
  - Evaporation variables (`e`, `pev`, `evabs`, `evatc`, `evavt`): Raw meters ($m$) converted to positive millimeters ($\text{mm}$) water equivalent ($|e| \times 1,000$), reversing ECMWF's upward-negative convention.
  - Radiative fluxes (`ssrd`, `strd`) and turbulent heat fluxes (`slhf`, `sshf`): Converted from accumulated Joules per square meter ($\text{J/m}^2$) to daily mean flux densities in Watts per square meter ($\text{W/m}^2$) by dividing by $86,400\text{ s}$, or accumulated energy in $\text{MJ/m}^2$ (dividing by $10^6$).
- **Physical Conservation Claim Qualified:**
  - Documented that $1,136,648$ out of $221,798,424$ hourly increments ($0.512\%$) exhibited tiny negative truncation residuals ($< -10^{-5}\text{ mm}$, median $-3.7 \times 10^{-6}\text{ mm}$) due to IEEE single-precision float truncation, which were clipped to zero.
  - Qualified the claim: explicitly stated that the absence of macroscopic negative differences verifies numerical differencing integrity across forecast resets, but does not independently validate ECMWF's physical water/energy budget closure.

### Task 3: Feature Matrix Architecture and Table 2 Reconciliation
- **Schema Reconciliation:**
  - Audited all 78 Parquet columns: 4 targets, 4 identifiers, 2 metadata strings, 4 excluded proxies/strings, 1 auxiliary quality flag (`is_coastal_fallback`), and exactly **63 primary predictors**.
  - Updated Table 2 to separate the 4 primary categorical regional predictors (`subdivision`, `zone_id`, `subdivision_enc`, `zone_id_enc`) from the single auxiliary flag (`is_coastal_fallback`).
- **Terminology Corrections Implemented:**
  - Replaced "Station DOY Climatology" with **"Grid-cell DOY Climatology"** throughout all text, figures, and tables.
  - Replaced "Numerical model precipitation forecast memory" with **"Antecedent reanalysis-derived precipitation and land-surface flux memory"**.
  - Replaced "Low-level jet moisture transport" with **"near-surface 10-meter wind speed and circulation dynamics"**.
  - Explicitly stated that ERA5-Land runoff variables are model-derived diagnostic fluxes from the HTESSEL land-surface scheme, rather than independently observed streamflow or surface runoff.

### Task 4: Feature Importance Provenance and Methodological Disambiguation
- **Permutation Provenance Confirmed:**
  - Traced the authoritative validation permutation importance of `sro_lag_1d` to `Stage5_Model_Interpretation/permutation_importance.csv`: $\Delta\text{RMSE} = \mathbf{+2.3545\text{ mm}}$ ($+42.16\%$ RMSE increase on the full 2023 validation set, $N = 91,980$, across 5 random shuffling repeats, std = $0.022\text{ mm}$).
  - Documented that an earlier preliminary draft reference to "+1.42 mm" arose from an exploratory single-fold run, and established $+2.3545\text{ mm}$ as the governing authoritative figure.
- **Three-Method Distinction Formally Maintained:**
  1. *Tree Split Gain:* In continuous Tweedie regression, `sro_lag_1d` accounts for $34.61\%$ and `src_lag_1d` accounts for $28.28\%$. In the heavy rain focal classifier, `ro_lag_1d` ($28.11\%$) plus `sro_lag_1d` ($26.41\%$) total **$54.52\%$**.
  2. *Permutation Importance:* Quantifies out-of-sample validation RMSE degradation upon feature shuffling (`sro_lag_1d` $\Delta\text{RMSE} = +2.3545\text{ mm}$; `src_lag_1d` $\Delta\text{RMSE} = +2.1241\text{ mm}$; `strd_lag_1d` $\Delta\text{RMSE} = +1.7960\text{ mm}$).
  3. *SHAP Attribution:* Quantifies additive game-theoretic sample contributions (`strd_lag_1d` mean absolute SHAP = $2.5745$; `sro_lag_1d` = $0.8897$).
  - Emphasized that feature importance reflects empirical statistical sensitivity within the tree architecture, rather than causal hydrometeorological proof.

### Task 5: Scientific Claims, Operational Scope, and Heavy-Rain Bounding
- **Operational Scope Bounded:**
  - Shifted narrative focus entirely to **retrospective daily rainfall reconstruction and hindcasting**, explicitly disclaiming real-time nowcasting or flood warning capabilities.
  - Removed recommendations proposing immediate operational deployment of the heavy-rain classifier.
- **Heavy-Rain Classifier Performance Transparency:**
  - Corrected narrative text to match the true test set performance of the LightGBM Focal Loss classifier (`heavy_rain_results.csv`): at decision threshold $p^* = 0.53$, test $\text{POD} = \mathbf{46.96\%}$, $\text{Precision} = \mathbf{58.22\%}$, $\text{FAR} = \mathbf{41.78\%}$, and $\text{CSI} = \mathbf{0.3512}$ (Hits = 595, False Alarms = 427, Misses = 672, Correct Negatives = 182,518).
  - Explicitly stated that the classifier **misses more than half (53.04%) of all observed heavy-rain events ($\ge 64.5\text{ mm}$)**.
  - Stated that the binary heavy-rain classifier cannot be extrapolated to infer reliable detection of Very Heavy ($\ge 115.6\text{ mm}$) or Extremely Heavy ($\ge 204.5\text{ mm}$) precipitation.
  - Clarified that $p^* = 0.53$ is a validation-selected decision threshold, rather than evidence of posterior probability calibration.

### Task 6: Numerical Re-Verification and Table 3 Tier Completion
- **Authoritative Traceability:**
  - Audited 56 numerical metrics across continuous regression, baselines, annual generalisation, classification, intensity categories, and block bootstrapping (`numerical_reconciliation.csv`, 56 of 56 confirmed `MATCH`).
- **Table 3 Model Hierarchy Completed:**
  - Inserted **Tier 3 (Regularized Linear Baselines)** into Table 3 and Section 3.3:
    - *Ridge Regression (L2):* $\text{RMSE} = 7.9750\text{ mm}$, $\text{MAE} = 3.3905\text{ mm}$, $\text{Bias} = +0.3308\text{ mm}$, $\text{NSE} = 0.4742$, $\text{KGE-2012} = 0.5184$, $\text{Heavy CSI} = 0.2299$.
    - *ElasticNet (L1/L2):* $\text{RMSE} = 8.0156\text{ mm}$, $\text{MAE} = 3.3999\text{ mm}$, $\text{Bias} = +0.3034\text{ mm}$, $\text{NSE} = 0.4688$, $\text{KGE-2012} = 0.5101$, $\text{Heavy CSI} = 0.2211$.
  - Resolved the tier numbering gap between Tier 2 and Tier 4.
- **Bootstrap Replicate Correction:**
  - Verified from `paired_bootstrap_robustness.csv` that the stationary block bootstrap evaluated **2,000 resamples** ($L = 7\text{ days}$ and $L = 14\text{ days}$, empirical $p < 0.0005$).
  - Corrected the figure caption and text which previously referenced 1,000 replicates.
- **Extreme Event Verification:**
  - Confirmed the 200 observed test events above $115.6\text{ mm}$: exactly **192 Very Heavy events** ($115.6 \le R < 204.5\text{ mm}$) and **8 Extremely Heavy events** ($R \ge 204.5\text{ mm}$, prevalence $0.0043\%$).
  - Verified that LightGBM Tweedie underestimated **100.0% of all 200 events** above $115.6\text{ mm}$.
- **Coastal Fallback Discrepancy Reconciled:**
  - Documented that the Karnataka Coastal Zone comprises **13 total IMD grid cells**.
  - Documented that exactly **3 grid cells (IDs: 52, 68, 125)** directly abut open ocean cells requiring nearest-land fallback interpolation (`is_coastal_fallback = 1`, 2,193 test samples).

### Task 7: Sequential Figure/Table Review and Terminology
- **Figure 1 Terminology:** Replaced "Station IDs" with **"IMD grid-cell IDs"** in Figure 1, the caption, and all corresponding text passages.
- **Strictly Sequential Numbering Enforced:**
  - Re-ordered figures so that Figure 8 represents **Temporal Generalisation and Robustness** (introduced in Section 4.5) and Figure 9 represents **Rainfall Intensity Limitations and Tail Underestimation** (introduced in Section 4.6), eliminating the previous numbering inversion.
  - Updated all in-text callouts and figure image files (`figure8_temporal_robustness_seq.png`, `figure9_extreme_rainfall_errors_seq.png`).

### Task 8: Bibliography Citation Closure and Front Matter
- **100% Citation Closure:**
  - Added in-text citations for references previously missing citations:
    - `[4]` (Rajeevan et al., 2006) cited in Section 2.1.
    - `[14]` (Ke et al., 2017) cited in Section 3.3.
    - `[16]` (Chen & Guestrin, 2016) cited in Section 3.3.
    - `[20]` (Duan, 1983) cited in Section 3.3.
    - `[27]` (Breiman, 2001) cited in Section 4.4.
    - `[28]` (Fisher et al., 2019) cited in Section 4.4.
    - `[29]` (Lundberg & Lee, 2017) cited in Section 4.4.
  - Confirmed that every entry in the 29-item bibliography is cited in the text.
- **Front Matter Sanitization:**
  - Verified zero local machine paths in the manuscript.
  - Standardized blinded author placeholders for peer review.

### Task 9: Matching DOCX and PDF Generation via Native Microsoft Word
- Successfully generated `Karnataka_Rainfall_Research_Paper_Final.docx`.
- Natively exported `Karnataka_Rainfall_Research_Paper_Final.pdf` using Microsoft Word COM automation (`New-Object -ComObject Word.Application`), ensuring identical formatting, page layouts, fonts, equations, tables, and figure rendering across both formats.

---
*End of Stage 9.1 Correction Changelog.*

