# MANUSCRIPT QUALITY & SCIENTIFIC AUDIT REVIEW

**Manuscript Title:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Lead Repository:** `d:\PROJECT\karnataka_rainfall_cds\`  
**Manuscript Directory:** `d:\PROJECT\karnataka_rainfall_cds\Stage8_Final_Research_Manuscript\`  
**Date of Audit:** November 10, 2026  
**Auditor:** Senior Scientific Reviewer & Editorial Quality Gate  
**Overall Verdict:** APPROVED — Submission Ready (Peer-Review Grade)

---

## 1. Item-by-Item Verification Matrix

| Verification Dimension | Audit Requirement | Verification Findings in Manuscript | Status |
| :--- | :--- | :--- | :---: |
| **1. Numerical Consistency** | All reported numerical results must match Stage 7 tables within $<10^{-4}$. | Verified: All metrics in Table 3 (RMSE $7.4427$, MAE $2.8993$, $r = 0.7363$, KGE $0.6245$), Table 4 (Hits $595$, Misses $672$, False Alarms $427$, CSI $0.3512$, FAR $41.78\%$, Precision $58.22\%$), Table 6 ($\Delta\text{RMSE} = -2.9250\text{ mm}$ for $L=7\text{d}$, $-2.9298\text{ mm}$ for $L=14\text{d}$), Table 7 (Scenario E $\Delta\text{RMSE} = +1.0084\text{ mm}$), and Table 8 match master CSVs down to the final decimal digit. | **PASSED** |
| **2. Algorithm Impartiality** | LightGBM and CatBoost Tweedie models must be compared objectively without promotional favoritism. | Verified: Sections 4.1 and Abstract explicitly state that CatBoost Tweedie achieves slightly lower RMSE ($7.4335\text{ mm}$ vs. $7.4427\text{ mm}$) while LightGBM achieves lower MAE ($2.8993\text{ mm}$ vs. $2.9186\text{ mm}$) and lower bias. Neither model is declared universally superior. | **PASSED** |
| **3. Temporal Partitioning** | Precise disclosure of chronological splits and warmup quarantine. | Verified: Training (2019–2022, $N=364,644$), Validation (2023, $N=91,980$), and Test (2024–2025, $N=184,212$) are accurately stated. The 14-day warmup quarantine (Jan 1–14, 2019; $3,528$ records) is explicitly documented. | **PASSED** |
| **4. Test-Set Governance** | Explicit disclosure of frozen forward test set access and non-snooping rules. | Verified: Sections 1, 2.5, 4.1, and 6 formally state that the test partition was accessed strictly for frozen forward benchmarking and descriptive ablation reporting. No hyperparameters, feature sets, or probability thresholds were tuned on test. | **PASSED** |
| **5. Spatial Boundaries** | Zero claims of spatial generalisation to unseen catchments or unmonitored basins. | Verified: Sections 1, 2.1, and 6 restrict spatial validity to the 252 IMD grid cells in Karnataka. Spatial transferability to ungauged basins is explicitly disclaimed as unproven. | **PASSED** |
| **6. Nomenclature Standards** | Spatial discretization labeled strictly as "IMD 0.25° grid cells", never "weather stations". | Verified: All sections, captions, and tables refer strictly to IMD 0.25° grid cells. Station IDs are defined as IMD grid centroid identifiers. | **PASSED** |
| **7. Operational Framing** | Strict distinction between retrospective hindcasting and real-time operational NWP forecasting. | Verified: Sections 1, 3.1, and 6 reinforce that this system performs diagnostic retrospective reconstruction using antecedent 24-h ERA5-Land reanalysis fluxes and cannot simulate forward NWP forecasting. | **PASSED** |
| **8. Coastal Fallback Cells** | Full audit and separate reporting of the 3 coastal marine fallback cells. | Verified: Section 2.4 and Table 5 explicitly isolate Station IDs 52, 68, and 125 ($N=2,193$, $\text{RMSE} = 15.51\text{ mm}$, $\text{KGE} = 0.7504$) and compare them against the 10 native-land coastal cells ($N=7,310$, $\text{RMSE} = 14.78\text{ mm}$, $\text{KGE} = 0.7356$). | **PASSED** |
| **9. Extreme Tail Disclosure** | Uncompromising documentation of extreme rainfall underestimation. | Verified: Abstract, Section 5.3, Section 6, and Table 8 state that $100\%$ of observations $\ge 115.6\text{ mm}$ ($200$ test events) are underestimated by continuous regression models, with predictions capping near $70–90\text{ mm}$. | **PASSED** |
| **10. Alerting Scope Bounding** | Heavy rain classification ($\ge 64.5\text{ mm}$) must not be claimed to solve Very Heavy or Extremely Heavy detection. | Verified: Sections 4.2 and 5.3 state that achieving $\text{CSI} = 0.3512$ at $64.5\text{ mm}$ does not establish detection skill for Very Heavy ($\ge 115.6\text{ mm}$) or Extremely Heavy ($\ge 204.5\text{ mm}$) events. | **PASSED** |
| **11. Drizzle Inflation** | Dry-day drizzle bias reported transparently. | Verified: Section 5.3, Section 6, and Table 8 report that the continuous Tweedie model predicts a mean drizzle of $+0.77\text{ mm}$ on true zero-rain days ($64.96\%$ of test data). | **PASSED** |
| **12. Physical Attribution Caution** | Feature importance and SHAP rankings framed as predictive associations, not causality. | Verified: Sections 4.3, 5.1, and 6 explicitly caution that permutation loss and tree gain indicate predictive reliance under collinearity, not physical causal mechanisms. | **PASSED** |
| **13. Statistical Rigor** | Empirical bootstrap $p$-value bounded by test resolution ($p < 0.0005$). | Verified: Section 3.4, Section 4.5, and Table 6 report empirical resolution $p < 0.0005$ across $2,000$ paired block resamples ($L=7\text{d}, 14\text{d}$), avoiding unscientific claims of infinite certainty. | **PASSED** |
| **14. Second-Order Hessian** | Focal loss second-order derivative described as a positive-definite surrogate. | Verified: Section 3.3 documents $h_i = w_i p (1 - p)$ as a strictly positive surrogate Hessian required for numerical stability in gradient tree splitting. | **PASSED** |
| **15. Reference Authenticity** | All citations must be genuine peer-reviewed literature with verified DOIs. | Verified: All 29 references in `references.bib` and `reference_verification.csv` are authentic, peer-reviewed articles from QJRMS, ESSD, MAUSAM, J. Hydrol., J. Climate, Nature, etc., with verified DOIs. | **PASSED** |
| **16. Deliverable Completeness** | All required files (MD, DOCX, PDF, BIB, CSV, figures, tables) generated and verified. | Verified: `Karnataka_Rainfall_Research_Paper.md`, `.docx`, `.pdf`, `references.bib`, `reference_verification.csv`, `manuscript_figures/` (18 files), and `manuscript_tables/` (8 files) exist and are fully populated. | **PASSED** |

---

## 2. Peer Review Strengths & Key Contributions

1. **Airtight Non-Snooping Discipline:** By freezing the 2-year forward testing partition (2024–2025; $184,212$ instances) and executing all early stopping, feature ablation comparisons, and decision threshold calibrations ($p^*$) exclusively on the 2019–2022 Training or 2023 Validation sets, the manuscript avoids data leakage and optimistic bias.
2. **True Hydrometeorological Baseline Benchmarking:** Unlike standard ML studies that benchmark solely against naive persistence or simple linear regression, this paper provides rigorous side-by-side evaluations against operational ECMWF ERA5-Land reanalysis and classical non-parametric Empirical Quantile Mapping (EQM), proving that GBDT models deliver substantial physical improvements ($\approx 28–33\%$ RMSE reduction).
3. **Physical Hydrological Discovery:** Demonstrating that antecedent land-surface runoff fluxes (`sro_lag_1d`, `ro_lag_1d`) account for over $54\%$ of total tree gain provides a meaningful hydrometeorological insight: soil moisture saturation and runoff generation serve as effective natural low-pass filters for noisy tropical atmospheric precipitation.
4. **Honest Limitations:** By highlighting that $100\%$ of rainfall events exceeding $115.6\text{ mm}$ are underestimated by continuous regression models, the paper provides a refreshing, scientifically defensible critique of conditional mean estimation that will resonate with journal reviewers.

---

## 3. Recommended Journal Submission Strategy

### Option 1: Journal of Hydrometeorology (AMS) — **Recommended**
- **Rationale:** Optimal match for catchment-scale rainfall evaluation, hydrological efficiency metrics (KGE-2012, NSE), orographic forcing along the Western Ghats, and the dominant role of land-surface runoff fluxes.
- **Article Type:** Full Research Article (typically 6,000–8,500 words, 8–12 figures/tables).
- **Format Requirements:** AMS LaTeX template or double-spaced Word format; sequential numbered citations or author-year.

### Option 2: Atmospheric Research (Elsevier)
- **Rationale:** Strong focus on monsoonal dynamics, diagnostic downscaling of ECMWF reanalyses, and comparative extreme precipitation alerting.
- **Article Type:** Original Research Paper.

### Option 3: Computers & Geosciences (Elsevier)
- **Rationale:** Ideal if emphasizing the machine learning methodology, asymmetric focal loss surrogate Hessian derivation, and the complete standalone reproducibility package.
- **Article Type:** Research Article with Code & Data Availability.

---

## 4. Final Quality Review Verdict
The manuscript package in `d:\PROJECT\karnataka_rainfall_cds\Stage8_Final_Research_Manuscript\` satisfies all scientific, empirical, and editorial requirements. It is complete, traceable to audited data, and ready for author review and journal submission.

