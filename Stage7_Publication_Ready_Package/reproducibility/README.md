# Reproducibility Package: Karnataka Rainfall Retrospective Hindcasting

**Project:** Machine-Learning-Guided Daily Rainfall Estimation and Hindcasting for Karnataka Using ERA5-Land Reanalysis and IMD Gridded Rainfall (2019–2025)  
**Lead Repository:** `d:\PROJECT\karnataka_rainfall_cds\`  
**Package Path:** `d:\PROJECT\karnataka_rainfall_cds\Stage7_Publication_Ready_Package\`

---

## 1. Project Purpose and Scientific Framing

This research establishes a rigorous, leakage-free machine learning framework for retrospective daily rainfall hindcasting and diagnostic downscaling across the 252 IMD 0.25° grid cells covering Karnataka State, India.

### Key Scientific Framing:
- **Task:** Retrospective hindcasting and reconstruction using ERA5-Land reanalysis predictors.
- **Not NWP:** This framework is a retrospective diagnostic tool, not an operational real-time NWP forecasting harness.
- **Spatial Units:** 252 in-state 0.25° IMD grid cells (not point weather stations).
- **Temporal Generalisation:** Rigorously proven on 2-year out-of-sample forward evaluation (2024–2025, N=184,212).
- **Spatial Generalisation:** Not claimed; spatial transfer to unseen basins is unproven.

---

## 2. Directory Structure

```text
Stage7_Publication_Ready_Package/
│
├── stage7_final_report.md                # Comprehensive master scientific report
├── master_results_table.csv              # Audited results across 44 model/partition rows
│
├── publication_tables/                   # Formatted tables ready for publication
│   ├── table1_dataset_characteristics.csv
│   ├── table2_feature_groups.csv
│   ├── table3_model_benchmarking.csv
│   ├── table4_heavy_rain_classification.csv
│   ├── table5_regional_seasonal_performance.csv
│   ├── table6_statistical_robustness.csv
│   ├── table7_feature_ablation.csv
│   └── table8_rainfall_intensity_limitations.csv
│
├── publication_figures/                  # 300 DPI PNG and vector PDF figures
│   ├── figure1_study_area.png (+ .pdf)
│   ├── figure2_methodology.png (+ .pdf)
│   ├── figure3_observed_predicted.png (+ .pdf)
│   ├── figure4_model_benchmarking.png (+ .pdf)
│   ├── figure5_heavy_rain_detection.png (+ .pdf)
│   ├── figure6_feature_interpretation.png (+ .pdf)
│   ├── figure7_regional_seasonal.png (+ .pdf)
│   ├── figure8_extreme_rainfall_errors.png (+ .pdf)
│   └── figure9_temporal_robustness.png (+ .pdf)
│
└── reproducibility/                      # Exact execution and verification materials
    ├── README.md                         # This file
    ├── environment_requirements.txt      # Exact pip freeze package versions
    ├── experimental_configuration.json   # Full experiment configuration
    ├── data_dictionary.csv               # 78-column metadata dictionary
    ├── reproducibility_manifest.csv      # SHA-256 checksums of all key artifacts
    └── methodology_reproduction.md       # Detailed procedural reproduction guide
```

---

## 3. Principal Verified Findings

1. **Continuous Regression Superiority:**
   - **LightGBM Tweedie ($p=1.5$):** Test RMSE = **$7.4427\text{ mm}$**, MAE = **$2.8993\text{ mm}$**, Pearson $r = **0.7363**$, Modified KGE-2012 = **$0.6245$**, Mean Bias = **$-0.0179\text{ mm}$**.
   - Decisively outperforms Persistence (RMSE $10.3582\text{ mm}$), Raw ERA5-Land (RMSE $10.4550\text{ mm}$), and Empirical Quantile Mapping (RMSE $11.1970\text{ mm}$).
2. **Statistical Significance via Paired Block Bootstrapping:**
   - 5,000 paired block replicates ($L=7\text{d}$) and 2,000 paired block replicates ($L=14\text{d}$) produced zero non-negative differences ($p < 0.0005$), demonstrating consistent paired RMSE reductions of **$2.93\text{ mm}$** over Persistence and **$3.76\text{ mm}$** over EQM.
3. **Heavy Rain Classification ($\ge 64.5\text{ mm}$):**
   - **LightGBM Focal Loss ($p^*=0.53$):** Precision = **$58.22\%$**, Recall = **$46.96\%$**, CSI = **$0.3512$**, with only **$427$ false alarms** (compared to $722$ false alarms for class weighting).
4. **Dominance of Land-Surface Fluxes:**
   - Surface runoff (`sro_lag_1d`) and total runoff (`ro_lag_1d`) account for **$54.52\%$ of total gain** in heavy rain classification and lead permutation sensitivity.

---

## 4. Documented Weaknesses & Boundaries

1. **Extreme Tail Quantile Shrinkage:** Continuous models underestimate $100\%$ of observations exceeding $115.6\text{ mm}$, plateauing around $70–90\text{ mm}$.
2. **False Drizzle on Dry Days:** Continuous models output an average of $+0.77\text{ mm}$ on true zero-rain days.
3. **Interior Plains Predictability Floor:** Convective rainfall in North and South Interior Karnataka exhibits a correlation ceiling ($r \approx 0.53$) due to sub-grid convective clouds unresolved by hydrostatic reanalysis.
4. **Spatial Generalisation Limitation:** The experimental design does not evaluate spatial transfer to unseen geographical cells outside Karnataka.
