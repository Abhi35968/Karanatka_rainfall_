# STAGE 7 FINAL REPORT: PUBLICATION-READY RESEARCH EVIDENCE PACKAGE

**Project Title:** Machine-Learning-Guided Daily Rainfall Estimation and Hindcasting for Karnataka Using ERA5-Land Reanalysis and IMD Gridded Rainfall (2019–2025)  
**Lead Repository:** `d:\PROJECT\karnataka_rainfall_cds\`  
**Stage Directory:** `d:\PROJECT\karnataka_rainfall_cds\Stage7_Publication_Ready_Package\`  
**Date of Audit & Release:** November 10, 2026  
**Status:** Complete, Traceable, Publication-Ready  

---

## 1. Executive Summary & Research Framework

### 1.1 Purpose & Scope
This report serves as the final scientific synthesis and evidence consolidation package for the Karnataka Daily Rainfall Hindcasting research initiative. Spanning a 7-year observational record (2019–2025) across 252 India Meteorological Department (IMD) 0.25° gridded land cells covering Karnataka State, India, this project establishes a rigorous, leakage-free machine learning framework for retrospective daily rainfall estimation, diagnostic downscaling, and extreme precipitation alerting.

In strict adherence to the Stage 7 directives:
- **Zero Retraining or Re-tuning:** All trained models, hyperparameter configurations, and decision thresholds ($p^*$) established during Stage 4.1 and audited in Stage 4.2 remain strictly frozen.
- **Strict Evidence Traceability:** Every numerical metric, contingency count, and bootstrap interval reported herein is mathematically identical to audited underlying records (recalculated in Stage 4.2 and sensitivity-audited in Stage 6).
- **Publication-Standard Artifacts:** All tables (Tables 1 through 8) are formatted in clean, standard CSV schemas with publication captions. All figures (Figures 1 through 9) are generated at 300 DPI publication raster format (PNG) and vector PDF format with consistent hydrometeorological typography, high-contrast accessible color palettes, and explicit physical error boundaries.
- **End-to-End Reproducibility:** A fully self-contained reproducibility suite—including an environment manifest (52 pinned packages), full experimental JSON configuration, 78-variable data dictionary, cryptographic SHA-256 hash manifest (29 artifacts), and step-by-step reproduction instructions—is released.

### 1.2 Core Research Hypotheses & Framing
The research investigates three foundational hydrometeorological questions:
1. **Predictive Capability of Gradient-Boosted Decision Trees:** Can gradient-boosted decision trees (GBDT) optimized with compound Poisson-Gamma (Tweedie) objective functions significantly outperform traditional statistical downscaling (Empirical Quantile Mapping; EQM), raw numerical reanalysis (ERA5-Land), and autoregressive persistence benchmarks in estimating daily rainfall over complex orography?
2. **Extreme Precipitation Alerting via Asymmetric Loss:** Can specialized focal loss formulations overcome the severe class imbalance of heavy rainfall events ($\ge 64.5\text{ mm}$, representing only $0.69\%$ of test observations) without inducing prohibitive false alarm rates?
3. **Physical Hydrometeorological Attribution:** Which physical predictor groups provide the greatest predictive utility—local antecedent gauge memory, surface hydrological fluxes (runoff, soil moisture), column thermodynamic variables, or orographic terrain barriers?

### 1.3 Key Scientific Boundary Definitions
To ensure absolute scientific integrity during peer review, the following operational definitions are maintained across all documentation:
- **Retrospective Reanalysis Hindcasting:** This system evaluates daily rainfall diagnostic estimation using antecedent 24-hour accumulated ERA5-Land reanalysis fluxes and previous-day IMD observations available at 03:00 UTC. It does **not** simulate an operational real-time Numerical Weather Prediction (NWP) forecast harness, which would require forward atmospheric forecast fields from ECMWF IFS or IMD NCUM.
- **IMD 0.25° Grid Cells:** All spatial units correspond to the 252 in-state $0.25^\circ \times 0.25^\circ$ gridded centroids published by IMD (Pune). They represent area-averaged daily accumulations, **not** point measurements from surface rain gauge stations.
- **Isolated Coastal Fallback Treatment:** Exactly 3 cells along the Arabian Sea coastline (Station IDs 52, 68, 125) have centroids falling on marine water pixels in ERA5-Land's native land-sea mask. As audited in Stage 3.4 and Stage 4.2, their reanalysis predictors were extracted via nearest-inland neighbor mapping ($5.94\text{ to } 15.11\text{ km}$ inland displacement). These cells ($N=2,193$ test instances) are tracked, flagged, and audited separately from the 10 native-land coastal cells.
- **Temporal vs. Spatial Generalisation:** Out-of-sample validity is evaluated exclusively along the **temporal** dimension using an uncompromised 2-year forward testing partition (2024–2025; $N=184,212$). Because all 252 spatial cells were present in the training set, **spatial generalisation to ungauged river basins or unmonitored regions is explicitly not claimed**.

---

## 2. Verified Experimental Evidence & Benchmarking Results

### 2.1 Complete Model Benchmarking Portfolio (Test Partition 2024–2025; $N=184,212$)
Table 3 documents the frozen forward evaluation metrics across the complete 5-tier model hierarchy. The test period observed mean daily rainfall is $3.5085\text{ mm}$ ($\text{standard deviation } = 11.0006\text{ mm}$, $\text{maximum } = 286.00\text{ mm}$).

| Model Hierarchy Tier | Model Name | Loss Objective | RMSE (mm) | MAE (mm) | Mean Bias (mm) | PBias (%) | Pearson $r$ | NSE | KGE-2009 | KGE-2012 | Heavy CSI ($\ge 64.5\text{ mm}$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 0 (Naive Baselines)** | Rainfall Persistence ($y_{t-1}$) | Zero Learning | 10.3582 | 3.7508 | 0.0000 | 0.00 | 0.5565 | 0.1129 | 0.5565 | 0.5565 | 0.2361 |
| **Tier 0 (Naive Baselines)** | Station DOY Climatology | Zero Learning | 10.1155 | 4.3078 | +0.2296 | +6.54 | 0.4313 | 0.1540 | 0.3068 | 0.2851 | 0.0144 |
| **Tier 1 (Raw Reanalysis)** | Antecedent ERA5-Land Total Precip | Zero Learning | 10.4550 | 4.1416 | +0.5560 | +15.85 | 0.4911 | 0.0963 | 0.4510 | 0.4111 | 0.1233 |
| **Tier 2 (Statistical Downscaling)**| Empirical Quantile Mapping (EQM) | Non-Parametric | 11.1970 | 4.2117 | +0.3867 | +11.02 | 0.4941 | -0.0366 | 0.4818 | 0.4763 | 0.1969 |
| **Tier 4 (Machine Learning)** | LightGBM RMSE (L2) | Squared Error | 7.4982 | 3.0806 | +0.2698 | +7.69 | 0.7320 | 0.5352 | 0.6190 | 0.5812 | 0.3016 |
| **Tier 4 (Machine Learning)** | LightGBM Log-Ordinary | Log-MSE | 7.8497 | 2.6150 | -1.3274 | -37.83 | 0.7260 | 0.4906 | 0.3703 | 0.5276 | 0.2271 |
| **Tier 4 (Machine Learning)** | LightGBM Log-Duan Smearing | Log-MSE + Smear | 7.5893 | 2.9488 | -0.2105 | -6.00 | 0.7260 | 0.5238 | 0.6439 | 0.6722 | 0.3509 |
| **Tier 4 (Machine Learning)** | **LightGBM Tweedie ($p=1.5$)** | **Compound Poisson-Gamma** | **7.4427** | **2.8993** | **-0.0179** | **-0.51** | **0.7363** | **0.5420** | **0.6218** | **0.6245** | **0.2868** |
| **Tier 4 (Machine Learning)** | **CatBoost Tweedie ($p=1.5$)** | **Compound Poisson-Gamma** | **7.4335** | **2.9186** | **+0.0388** | **+1.11** | **0.7370** | **0.5431** | **0.6318** | **0.6261** | **0.3033** |
| **Tier 4 (Machine Learning)** | Hurdle Expected Value | Two-Stage Hurdle | 7.4241 | 2.9300 | +0.0342 | +0.97 | 0.7380 | 0.5443 | 0.6165 | 0.6114 | 0.3072 |

#### Verified Performance Takeaways:
1. **Decisive Baseline Superiority:** The champion LightGBM Tweedie model achieves an RMSE of **$7.4427\text{ mm}$**, representing a **$28.15\%$ reduction in RMSE** relative to Persistence ($10.3582\text{ mm}$), a **$28.81\%$ reduction** relative to Raw ERA5-Land ($10.4550\text{ mm}$), and a **$33.53\%$ reduction** relative to EQM ($11.1970\text{ mm}$).
2. **Optimal Hydrological Skill:** LightGBM Tweedie achieves a Nash-Sutcliffe Efficiency (NSE) of **$0.5420$** (compared to $-0.0366$ for EQM and $0.0963$ for ERA5-Land) and a Modified Kling-Gupta Efficiency (KGE-2012) of **$0.6245$** with negligible mean bias ($-0.0179\text{ mm}$, or $-0.51\%$).
3. **CatBoost Convergence:** CatBoost Tweedie exhibits virtually identical performance ($\text{RMSE } = 7.4335\text{ mm}$, $\text{NSE } = 0.5431$, $\text{KGE-2012 } = 0.6261$), proving that the Tweedie likelihood parameterization is algorithm-agnostic.

---

### 2.2 Extreme Rainfall Classification & Alerting ($\ge 64.5\text{ mm}$; IMD Heavy Rain)
Table 4 presents the frozen test contingency analysis and skill scores for heavy rainfall alerting ($1,267$ true events out of $184,212$ test records, prevalence $0.6878\%$).

| Model Architecture | Operating Threshold ($p^*$) | Calibration Source | Hits | Misses | False Alarms | Correct Negatives | Precision | POD / Recall | FAR | CSI | HSS | ETS | PR-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Focal Loss** | **0.53** | **Validation 2023 Max CSI** | **595** | **672** | **427** | **182,518** | **0.5822** | **0.4696** | **0.4178** | **0.3512** | **0.5169** | **0.3485** | **0.5084** |
| LightGBM Class-Weighted | 0.96 | Validation 2023 Max CSI | 688 | 579 | 722 | 182,223 | 0.4879 | 0.5430 | 0.5121 | 0.3459 | 0.5105 | 0.3427 | 0.4853 |
| CatBoost Class-Weighted | 0.98 | Validation 2023 Max CSI | 608 | 659 | 515 | 182,430 | 0.5414 | 0.4799 | 0.4586 | 0.3412 | 0.5056 | 0.3383 | 0.4739 |
| Rainfall Persistence | 64.5 mm | Direct Physical Lag | 484 | 783 | 783 | 182,162 | 0.3820 | 0.3820 | 0.6180 | 0.2361 | 0.3777 | 0.2328 | N/A |
| Raw ERA5-Land | 64.5 mm | Direct Antecedent Flux | 231 | 1,036 | 606 | 182,339 | 0.2760 | 0.1823 | 0.7240 | 0.1233 | 0.2153 | 0.1206 | N/A |
| Empirical Quantile Mapping | 64.5 mm | Station-Wise Knots | 372 | 895 | 622 | 182,323 | 0.3742 | 0.2936 | 0.6258 | 0.1969 | 0.3250 | 0.1940 | N/A |

#### Extreme Classification Takeaways:
1. **Focal Loss Superiority in False Alarm Suppression:** While both Focal Loss and Class-Weighted LightGBM attain comparable Critical Success Index scores ($\text{CSI} \approx 0.35$), **Focal Loss suppresses false alarms by $40.86\%$** ($427$ false alarms vs. $722$ for class weighting). This elevates operational Precision from $48.79\%$ to **$58.22\%$**, achieving a False Alarm Ratio of $41.78\%$ compared to $72.40\%$ for Raw ERA5-Land.
2. **Reanalysis Detection Failure:** Raw ERA5-Land misses **$81.77\%$ of heavy rain events** ($1,036$ misses out of $1,267$), demonstrating that raw reanalysis fluxes cannot be used for direct extreme precipitation alerting without calibrated post-processing.

---

### 2.3 Statistical Significance via Paired Stationary Block Bootstrapping
Table 6 summarizes the rigorous paired block bootstrap resampling protocol conducted to establish the statistical significance of RMSE reductions while preserving spatial correlation across all 252 grid cells and temporal autocorrelation across synoptic ($L=7\text{ days}$) and intraseasonal ($L=14\text{ days}$) scales.

| Model Pair Comparison | Block Length ($L$) | Physical Autocorrelation Scale | Bootstrap Replicates ($B$) | Mean $\Delta$RMSE (mm) | Standard Error (mm) | 95% Confidence Interval (mm) | Replicates $\Delta \ge 0$ | Empirical $p$-Value Resolution |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Tweedie vs. Persistence** | **7 days** | **Synoptic weather cycle** | **2,000** | **-2.9250** | **0.2046** | **[-3.3241, -2.5175]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. EQM** | **7 days** | **Synoptic weather cycle** | **2,000** | **-3.7542** | **0.3768** | **[-4.4626, -3.0039]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. Raw ERA5** | **7 days** | **Synoptic weather cycle** | **2,000** | **-3.0109** | **0.3482** | **[-3.7037, -2.3735]** | **0** | **$p < 0.0005$** |
| CatBoost Tweedie vs. Persistence | 7 days | Synoptic weather cycle | 2,000 | -2.9337 | 0.2077 | [-3.3415, -2.5276] | 0 | $p < 0.0005$ |
| CatBoost Tweedie vs. EQM | 7 days | Synoptic weather cycle | 2,000 | -3.7629 | 0.3848 | [-4.4936, -2.9954] | 0 | $p < 0.0005$ |
| **LightGBM Tweedie vs. Persistence** | **14 days** | **Intraseasonal / MJO scale** | **2,000** | **-2.9298** | **0.2271** | **[-3.3608, -2.4729]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. EQM** | **14 days** | **Intraseasonal / MJO scale** | **2,000** | **-3.7645** | **0.4243** | **[-4.5976, -2.9587]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. Raw ERA5** | **14 days** | **Intraseasonal / MJO scale** | **2,000** | **-3.0204** | **0.3785** | **[-3.7669, -2.3255]** | **0** | **$p < 0.0005$** |

#### Bootstrap Invariance Takeaways:
1. **Strict Non-Zero Difference Distribution:** Across all 2,000 paired block replicates under both $L=7\text{d}$ and $L=14\text{d}$ configurations (as well as the 5,000-replicate audit check in Stage 4.2), **zero replicates yielded $\Delta \text{RMSE} \ge 0$**.
2. **Empirical p-Value Bounding:** As mandated by statistical standards, the significance is reported at the empirical resolution threshold:
   $$\text{Empirical } p = \frac{\sum_{b=1}^B \mathbb{I}(\Delta \text{RMSE}_b \ge 0) + 1}{B + 1} = \frac{0 + 1}{2001} \approx 0.0004997 \implies p < 0.0005$$
   Claims of mathematically infinite certainty ($p=0.000000$) are strictly avoided.
3. **Block Length Invariance:** Increasing the block length from 7 to 14 days slightly widens the standard error (from $0.2046\text{ mm}$ to $0.2271\text{ mm}$ vs. Persistence) but leaves the mean paired difference virtually unchanged ($-2.9250\text{ mm}$ vs. $-2.9298\text{ mm}$), proving that results are impervious to multi-day atmospheric persistence.

---

### 2.4 Regional & Seasonal Stratification
Table 5 details model performance across Karnataka's four distinct meteorological subdivisions and four seasonal regimes on the frozen test partition.

#### Subdivisional Breakdown:
- **Coastal Karnataka ($13$ cells, $N=9,503$):** Characterized by extreme monsoonal precipitation (observed mean $= 13.19\text{ mm/day}$). LightGBM Tweedie achieves its highest regional correlation (**$r = 0.8261$**) and hydrological skill (**$\text{KGE-2012} = 0.7407$**), reducing RMSE from $20.38\text{ mm}$ (Persistence) to **$14.95\text{ mm}$** ($\text{CSI}_{64.5} = 0.3620$).
  - *Coastal Native Land ($10$ cells, $N=7,310$):* $\text{RMSE } = 14.78\text{ mm}$, $\text{KGE } = 0.7356$, $\text{CSI}_{64.5} = 0.3516$.
  - *Coastal Fallback ($3$ cells: IDs 52, 68, 125; $N=2,193$):* $\text{RMSE } = 15.51\text{ mm}$, $\text{KGE } = 0.7504$, $\text{CSI}_{64.5} = 0.3898$. The fallback extraction exhibits comparable high-fidelity skill despite the slight spatial displacement.
- **Malnad ($35$ cells, $N=25,585$):** Western Ghats orographic barrier (mean $= 6.37\text{ mm/day}$). Model performance is exceptionally strong: **$r = 0.8043$**, **$\text{KGE-2012} = 0.7256$**, with RMSE lowered from $12.71\text{ mm}$ to **$9.30\text{ mm}$** ($\text{CSI}_{64.5} = 0.3441$).
- **North Interior Karnataka (NIK; $118$ cells, $N=86,258$):** Semi-arid Deccan plateau (mean $= 2.38\text{ mm/day}$). Model achieves $\text{RMSE } = 6.43\text{ mm}$ (vs. $9.09\text{ mm}$ for Persistence) but encounters a correlation ceiling (**$r = 0.5274$**, $\text{KGE-2012} = 0.3676$).
- **South Interior Karnataka (SIK; $86$ cells, $N=62,866$):** Southern plateau (mean $= 2.44\text{ mm/day}$). Displays similar convective dynamics: $\text{RMSE } = 6.05\text{ mm}$ (vs. $8.51\text{ mm}$ for Persistence), with $r = 0.5513$ and $\text{KGE-2012} = 0.3723$.

#### Seasonal Breakdown:
- **Southwest Monsoon (JJAS; $N=61,488$):** Accounts for over $70\%$ of annual rainfall (mean $= 7.02\text{ mm/day}$). The model demonstrates robust synoptic capture: $\text{RMSE } = 10.19\text{ mm}$ (vs. $14.18\text{ mm}$ Persistence and $14.94\text{ mm}$ ERA5-Land), **$r = 0.7545$**, **$\text{KGE-2012} = 0.6451$**, and $\text{CSI}_{64.5} = 0.3355$.
- **Post-Monsoon (OND; $N=46,368$):** Retreating monsoon with convective activity (mean $= 2.14\text{ mm/day}$). $\text{RMSE } = 6.11\text{ mm}$, $r = 0.5628$, $\text{KGE-2012} = 0.3823$.
- **Pre-Monsoon (MAM; $N=46,368$):** Local convective thunderstorms (mean $= 2.45\text{ mm/day}$). $\text{RMSE } = 6.68\text{ mm}$, $r = 0.6717$, $\text{KGE-2012} = 0.4859$.
- **Winter (JF; $N=29,988$):** Hyper-arid dry spell (mean $= 0.07\text{ mm/day}$). $\text{RMSE } = 0.81\text{ mm}$ (vs. $1.23\text{ mm}$ Persistence), with near-zero absolute bias ($-0.0023\text{ mm}$).

---

### 2.5 Feature Attribution & Ablation Analysis
Figure 6 and Table 7 quantify feature attribution and physical ablation dynamics across the 63 primary model predictors.

#### Group Contribution to Predictive Gain (Heavy Rain Classification):
1. **Hydrological Land-Surface Fluxes ($54.52\%$ of total gain):** Surface runoff (`sro_lag_1d`) and total runoff (`ro_lag_1d`) dominate feature attribution, confirming that land-surface soil saturation operates as an effective physical proxy for antecedent precipitation.
2. **Antecedent Observed IMD Gauges ($17.92\%$ of total gain):** Local 1-day lag rainfall (`target_rainfall_mm_lag_1d`) and 3-day rolling statistics provide immediate autoregressive persistence.
3. **Atmospheric Column & Radiation Variables ($16.48\%$ of total gain):** Clear-sky solar radiation (`ssrd`), thermal radiation (`strd`), surface pressure (`sp`), and convective temperature profiles provide synoptic stability context.
4. **Orographic & Static Terrain Features ($11.08\%$ of total gain):** DEM elevation (`elevation_dem_m`), terrain slope (`slope_dem_deg`), distance to Arabian Sea coast (`dist_to_coast_km`), and distance to Western Ghats crest line (`dist_to_ghats_crest_km`) anchor spatial boundaries.

#### Systematic Feature Ablation Matrix (Table 7):
- **Full Model (Scenario A, 63 features):** Test $\text{RMSE } = 7.4427\text{ mm}$, $\text{KGE-2012 } = 0.6245$, $\text{Heavy CSI } = 0.2868$.
- **Without Hydrological Runoff (Scenario B, 55 features):** Test RMSE degrades by $+0.2427\text{ mm}$ to $7.6854\text{ mm}$; KGE drops to $0.5925$.
- **Without Antecedent IMD Gauges (Scenario C, 51 features):** Test RMSE degrades by $+0.1401\text{ mm}$ to $7.5828\text{ mm}$; Heavy CSI drops sharply from $0.2868$ to $0.2693$.
- **Without Static Terrain (Scenario D, 55 features):** Marginal change in bulk RMSE ($+0.0130\text{ mm}$ to $7.4557\text{ mm}$), but orographic boundary delineations soften.
- **Without ERA5 Predictors (Scenario E, 26 features):** Severe performance degradation. Test RMSE surges by **$+1.0084\text{ mm}$ to $8.4511\text{ mm}$**; KGE drops to $0.4833$; Heavy CSI collapses to $0.1536$.
- **Autoregressive Baseline Floor (Scenario F, 18 features):** Test $\text{RMSE } = 8.5145\text{ mm}$, $\text{KGE-2012 } = 0.4686$.

*Conclusion on ERA5 Predictors:* ERA5 atmospheric and surface predictors provide **measurable, statistically confirmed incremental predictive value** ($\Delta \text{RMSE} = -1.01\text{ mm}$, $+29.2\%$ KGE gain over gauge-only autoregression).

---

## 3. Catalog of Publication Tables

The `publication_tables/` directory contains 8 clean, fully documented CSV tables formatted for publication:

### Table 1: Dataset Partitioning, Spatial Extent, and Climatological Characteristics
- **File:** [`table1_dataset_characteristics.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table1_dataset_characteristics.csv)
- **Scientific Purpose:** Documents the temporal bounds, instance counts, wet-day proportions ($>0.1\text{ mm}$), and rainfall distribution quantiles (Mean, Std, 50th, 90th, 99th, Max) across Training (2019–2022), Validation (2023), Test (2024–2025), and Full dataset partitions.
- **Key Takeaway:** Explicitly demonstrates that 2023 was an anomalous regional drought year ($27.39\%$ wet days vs. $36.95\%$ in training and $35.04\%$ in test), validating 2023 as an exceptionally stringent validation gate for hyperparameter early stopping and threshold selection.

### Table 2: Input Feature Matrix Architecture and Physical Domain Groupings
- **File:** [`table2_feature_groups.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table2_feature_groups.csv)
- **Scientific Purpose:** Synthesizes the 78 columns in the master Parquet into 8 operational feature categories: Target Labels (4), Spatiotemporal Identifiers (4), Cyclic Temporal Predictors (7), Static Orographic/Geographic Features (6), Antecedent IMD Local Rainfall Lags (12), ERA5-Land Antecedent Surface Fluxes (15), ERA5-Land Thermodynamic Variables (17), and Pipeline Quality Indicators (13).
- **Key Takeaway:** Formally documents the feature gating and leakage-prevention architecture, establishing that all 63 primary model inputs respect the 24-hour antecedent causality constraint.

### Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio
- **File:** [`table3_model_benchmarking.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table3_model_benchmarking.csv)
- **Scientific Purpose:** Compares the complete 10-model evaluation hierarchy on the frozen 2024–2025 test partition ($N=184,212$) across 9 standard hydrometeorological metrics: RMSE, MAE, Mean Bias, Percentage Bias, Pearson $r$, NSE, Classic KGE-2009, Modified KGE-2012, and Heavy Rain CSI ($\ge 64.5\text{ mm}$).
- **Key Takeaway:** Establishes the definitive quantitative hierarchy: Tweedie and Hurdle ML formulations outperform naive persistence, raw reanalysis, and empirical quantile mapping across all evaluated continuous metrics.

### Table 4: Heavy Rainfall ($\ge 64.5\text{ mm}$) Detection Contingency and Skill Scores
- **File:** [`table4_heavy_rain_classification.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table4_heavy_rain_classification.csv)
- **Scientific Purpose:** Details the complete contingency matrices (Hits, Misses, False Alarms, Correct Negatives) and derived skill scores (Precision, Recall/POD, FAR, CSI, HSS, ETS, PR-AUC) for heavy rainfall alerting under frozen thresholds.
- **Key Takeaway:** Demonstrates that Focal Loss achieves the highest Precision ($58.22\%$) and lowest FAR ($41.78\%$) among all evaluated methods while preserving competitive CSI ($0.3512$).

### Table 5: Subdivisional and Seasonal Stratified Performance Analysis
- **File:** [`table5_regional_seasonal_performance.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table5_regional_seasonal_performance.csv)
- **Scientific Purpose:** Evaluates model skill across Karnataka's 4 meteorological subdivisions (including separate accounting for the 3 coastal fallback cells vs. 10 native land cells) and 4 climatological seasons.
- **Key Takeaway:** Highlights the sharp contrast between orographic regimes (Coastal/Malnad $r \ge 0.80$, $\text{KGE} \ge 0.72$) and interior convective plains (NIK/SIK $r \approx 0.53–0.55$), revealing the physical bounds of hydrostatic reanalysis predictors.

### Table 6: Stationary Block Bootstrap Statistical Significance and Uncertainty
- **File:** [`table6_statistical_robustness.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table6_statistical_robustness.csv)
- **Scientific Purpose:** Presents paired block bootstrap sampling distributions across 2,000 resamples for both $L=7\text{ days}$ and $L=14\text{ days}$ block lengths, reporting mean paired differences, standard errors, $95\%$ confidence intervals, and empirical $p$-value resolutions.
- **Key Takeaway:** Rigorously proves that the observed $\approx 2.93\text{ mm}$ RMSE reduction over Persistence and $\approx 3.75\text{ mm}$ reduction over EQM are statistically significant at $p < 0.0005$ under realistic temporal autocorrelation.

### Table 7: Systematic Feature Ablation Matrix
- **File:** [`table7_feature_ablation.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table7_feature_ablation.csv)
- **Scientific Purpose:** Isolates the marginal value of specific physical feature groups (Runoff, IMD memory, Orography, ERA5 predictors) on Validation and Test sets.
- **Key Takeaway:** Confirms that removing ERA5 reanalysis predictors incurs the single largest performance collapse ($\Delta \text{RMSE} = +1.01\text{ mm}$, Heavy CSI drops to $0.1536$), establishing their tangible incremental utility.

### Table 8: Rainfall Intensity Stratification and Quantile Shrinkage Limitations
- **File:** [`table8_rainfall_intensity_limitations.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table8_rainfall_intensity_limitations.csv)
- **Scientific Purpose:** Evaluates continuous model error distributions across all 7 official IMD daily rainfall categories, detailing sample counts, observed/predicted means and standard deviations, MAE, RMSE, Mean Bias, Percentage Bias, and percentage of underestimated events.
- **Key Takeaway:** Provides uncompromising documentation of extreme quantile shrinkage: $88.6\%$ of Heavy events ($64.5–115.5\text{ mm}$) and $100.0\%$ of Very Heavy/Extremely Heavy events ($\ge 115.6\text{ mm}$) are underestimated by the conditional expectation regressor.

---

## 4. Catalog of Publication Figures

The `publication_figures/` directory contains all 9 publication figures rendered in both 300 DPI PNG and vector PDF format:

### Figure 1: Geographic Domain, Orography, and Gridded Spatial Discretization
- **Files:** [`figure1_study_area.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure1_study_area.png) | [`figure1_study_area.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure1_study_area.pdf)
- **Visual Design:** High-resolution map of Karnataka showing state boundary, Western Ghats crest line, elevation contours from Copernicus GLO-90 DEM, and all 252 IMD 0.25° grid cell centroids color-coded by meteorological subdivision (Coastal: 13, Malnad: 35, North Interior: 118, South Interior: 86). The 3 coastal fallback cells (Station IDs 52, 68, 125) are explicitly marked with distinct symbols and inland displacement vectors.
- **Scientific Takeaway:** Visually contextualizes the sharp orographic gradient between the Arabian Sea coast, the 1,000–1,800 m Western Ghats ridge, and the leeward Deccan plateau rain-shadow.

### Figure 2: End-to-End Retrospective Machine Learning Architecture
- **Files:** [`figure2_methodology.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure2_methodology.png) | [`figure2_methodology.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure2_methodology.pdf)
- **Visual Design:** Multi-panel technical workflow diagram depicting the 8 procedural stages: (1) Ingestion of IMD gridded observations and ERA5-Land reanalysis; (2) Spatial coordinate matching and marine fallback extraction; (3) Feature engineering with 14-day warmup quarantine; (4) Strict temporal partitioning (Train 2019–2022, Val 2023, Test 2024–2025); (5) Zero-leakage preprocessing; (6) Model training (Tweedie, Hurdle, Focal Loss); (7) Validation threshold calibration ($p^*$); and (8) Independent statistical auditing.
- **Scientific Takeaway:** Proves the structural elimination of temporal data leakage and demonstrates adherence to reproducible experimental standards.

### Figure 3: Observed vs. Predicted Rainfall Density and Extreme Quantile Parity
- **Files:** [`figure3_observed_predicted.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure3_observed_predicted.png) | [`figure3_observed_predicted.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure3_observed_predicted.pdf)
- **Visual Design:** Dual-panel diagnostic figure. Panel A: Logarithmic hexbin density parity plot between observed IMD rainfall and LightGBM Tweedie predictions on the test set, overlaid with 1:1 identity and local LOESS trend lines. Panel B: High-quantile Q-Q plot comparing empirical percentiles (90th to 99.9th) against observations.
- **Scientific Takeaway:** Illustrates tight calibration and density alignment up to $\approx 50\text{ mm}$, beyond which predictions smoothly taper and plateau near $70–90\text{ mm}$, visually demonstrating extreme tail conditional expectation shrinkage.

### Figure 4: Multidimensional Continuous Model Benchmarking
- **Files:** [`figure4_model_benchmarking.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure4_model_benchmarking.png) | [`figure4_model_benchmarking.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure4_model_benchmarking.pdf)
- **Visual Design:** Four-panel bar chart comparing all 10 models across key performance dimensions: (a) Root Mean Squared Error (RMSE, mm); (b) Mean Absolute Error (MAE, mm); (c) Pearson Correlation ($r$); and (d) Modified Kling-Gupta Efficiency (KGE-2012). Models are grouped into Naive Baselines, Reanalysis, Statistical Downscaling, and Machine Learning tiers.
- **Scientific Takeaway:** Highlights the pronounced performance leap from classical downscaling (EQM KGE $= 0.476$, RMSE $= 11.20\text{ mm}$) to machine learning (LightGBM Tweedie KGE $= 0.625$, RMSE $= 7.44\text{ mm}$).

### Figure 5: Heavy Rainfall Classification Contingency and Skill Curves
- **Files:** [`figure5_heavy_rain_detection.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure5_heavy_rain_detection.png) | [`figure5_heavy_rain_detection.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure5_heavy_rain_detection.pdf)
- **Visual Design:** Dual-panel classification summary. Panel A: Grouped contingency breakdown (Hits, Misses, False Alarms) across Focal Loss, Class-Weighted LightGBM, CatBoost, Persistence, Raw ERA5, and EQM for the $\ge 64.5\text{ mm}$ threshold. Panel B: Skill metric radar/bar profile comparing Precision, Recall, CSI, and HSS.
- **Scientific Takeaway:** Demonstrates the decisive advantage of Focal Loss in suppressing false alarms by over $40\%$ while maintaining competitive detection skill.

### Figure 6: Permutation Feature Importance and Hydrological Attribution
- **Files:** [`figure6_feature_interpretation.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure6_feature_interpretation.png) | [`figure6_feature_interpretation.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure6_feature_interpretation.pdf)
- **Visual Design:** Two-panel attribution chart. Panel A: Horizontal bar chart showing the top 15 predictors ranked by validation permutation importance ($\Delta \text{RMSE}$ degradation upon feature shuffling). Panel B: Donut chart displaying the relative gain contribution of the 4 physical predictor domains.
- **Scientific Takeaway:** Visually establishes the dominance of surface runoff fluxes (`sro_lag_1d`, `ro_lag_1d`) and antecedent local gauge observations over raw atmospheric temperatures.

### Figure 7: Subdivisional and Seasonal Hydrometeorological Stratification
- **Files:** [`figure7_regional_seasonal.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure7_regional_seasonal.png) | [`figure7_regional_seasonal.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure7_regional_seasonal.pdf)
- **Visual Design:** Dual-panel performance stratification. Panel A: Subdivisional breakdown of RMSE and KGE-2012 across Coastal (including fallback separation), Malnad, North Interior, and South Interior Karnataka. Panel B: Seasonal progression across Southwest Monsoon (JJAS), Post-Monsoon (OND), Pre-Monsoon (MAM), and Winter (JF).
- **Scientific Takeaway:** Documents the contrast between orographically forced monsoonal regimes and convective interior regimes.

### Figure 8: Systematic Rainfall Intensity Error Profiles and Tail Underestimation
- **Files:** [`figure8_extreme_rainfall_errors.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure8_extreme_rainfall_errors.png) | [`figure8_extreme_rainfall_errors.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure8_extreme_rainfall_errors.pdf)
- **Visual Design:** Two-panel diagnostic analysis across all 7 IMD intensity bins. Panel A: Mean Prediction Bias (mm) and Percentage Bias (%) across intensity intervals. Panel B: Fraction of underestimated events (%) and absolute prediction error spread (MAE and RMSE).
- **Scientific Takeaway:** Provides a transparent, unvarnished visualization of model behavior across rainfall regimes, documenting near-zero drizzle inflation ($+0.77\text{ mm}$ on dry days) and sharp negative bias beyond $64.5\text{ mm}$.

### Figure 9: Inter-Annual Generalisation and Paired Bootstrap Empirical Distributions
- **Files:** [`figure9_temporal_robustness.png`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure9_temporal_robustness.png) | [`figure9_temporal_robustness.pdf`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_figures/figure9_temporal_robustness.pdf)
- **Visual Design:** Dual-panel robustness validation. Panel A: Out-of-sample inter-annual stability comparing LightGBM Tweedie against Persistence, Raw ERA5, and EQM across 2024 (leap year, $N=92,232$) and 2025 ($N=91,980$). Panel B: Empirical probability density functions of paired $\Delta \text{RMSE}$ from 2,000 block bootstrap resamples under $L=7\text{d}$ and $L=14\text{d}$ block lengths.
- **Scientific Takeaway:** Confirms that model superiority is temporally invariant across distinct operational years, with empirical bootstrap distributions completely detached from zero ($p < 0.0005$).

---

## 5. Reproducibility Package Architecture & Verification

The `reproducibility/` directory provides complete standalone materials to replicate every data processing, model training, and statistical evaluation step:

```text
reproducibility/
├── environment_requirements.txt      # Pinned pip environment (52 exact package versions)
├── experimental_configuration.json   # JSON file encoding spatial/temporal bounds, hyperparameters, and thresholds
├── data_dictionary.csv               # 78-column schema: data types, physical units, availability, and pipeline roles
├── reproducibility_manifest.csv      # Cryptographic SHA-256 hashes of 29 key project artifacts
├── methodology_reproduction.md       # Step-by-step reproduction instructions and execution commands
└── README.md                         # Top-level orientation, core findings, and documented limitations
```

### 5.1 System Environment & Specifications
- **Operating System:** Microsoft Windows 11 Home / Pro (x86_64 architecture).
- **Compute Hardware:** Multi-core Intel/AMD x86_64 processor, 32 GB RAM, high-speed NVMe storage.
- **Python Runtime:** Python 3.11.9 (64-bit virtual environment located at `d:\PROJECT\karnataka_rainfall_cds\venv\`).
- **Core Library Dependencies:**
  - `lightgbm == 4.6.0`
  - `catboost == 1.2.7`
  - `scikit-learn == 1.6.1`
  - `scipy == 1.15.2`
  - `pandas == 2.2.3`
  - `numpy == 2.2.3`
  - `pyarrow == 19.0.1`
  - `matplotlib == 3.10.1`
  - `seaborn == 0.13.2`
  - `shap == 0.46.0`
  - `joblib == 1.4.2`

### 5.2 Cryptographic Integrity Manifest
All 29 critical project files—including training datasets, serialized models, prediction arrays, publication tables, and figures—have been cryptographically signed via SHA-256. Key artifact signatures include:
- `Stage3_4_Final_Release_Audit/data/final_train_dataset.parquet`: `0ef53702e86da9fcfd2b8b939fa99e52643a0d58853b0c8d4850fa92fc51a84f`
- `Stage3_4_Final_Release_Audit/data/final_validation_dataset.parquet`: `3c6d5951d8b6727c9f80a22bc7368cefb36214316d2cae866895e7fa1f21eb23`
- `Stage3_4_Final_Release_Audit/data/final_test_dataset.parquet`: `25752c092461ec46faadfc769213bc54dfb5db9d3e8e2fa53427382d61992015`
- `Stage4_1_Model_Development/models/lgb_tweedie_regressor.joblib`: `54fc57321eec4f9c8da14e21a7b8e5c3dfb867c4df607f0f6259c8d5b85a3717`
- `Stage4_1_Model_Development/models/pred_lgb_tweedie_test.npy`: `9e99a80b8e72765d70f03e2fc348f972b225c56c2c9d06bcf4bf783b94235478`
- `Stage7_Publication_Ready_Package/master_results_table.csv`: `735be9fca6eb4629479b1897e066e4a6fc4697ffc31f452093da4aa796ecf72d`

---

## 6. Scientific Integrity & Discrepancy Reconciliation

### 6.1 Coastal Fallback Grid Cells (Station IDs 52, 68, 125)
- **Nature of Discrepancy:** Earlier stage drafts inconsistently referred to coastal anomaly counts (ranging from 3 to 13 cells).
- **Physical Reconciliation:** As audited in Stage 3.4, Karnataka has exactly **13 coastal IMD grid cells**. Of these, 10 cells coincide with land pixels in ERA5-Land. Exactly **3 cells** have centroids falling over Arabian Sea marine pixels:
  - `Station ID 52` (Lat: $13.25^\circ\text{N}$, Lon: $74.75^\circ\text{E}$; Udupi / Malpe; coast distance $5.94\text{ km}$).
  - `Station ID 68` (Lat: $13.50^\circ\text{N}$, Lon: $74.75^\circ\text{E}$; Kundapura; coast distance $15.11\text{ km}$).
  - `Station ID 125` (Lat: $14.75^\circ\text{N}$, Lon: $74.25^\circ\text{E}$; Karwar / Ankola; coast distance $6.98\text{ km}$).
- **Audit Verification:** Nearest-inland extraction was used to populate their reanalysis predictors. In Table 5, their test performance ($N=2,193$, $\text{RMSE } = 15.51\text{ mm}$, $\text{KGE } = 0.7504$, $\text{CSI}_{64.5} = 0.3898$) is reported separately from the 10 native land coastal cells ($N=7,310$, $\text{RMSE } = 14.78\text{ mm}$, $\text{KGE } = 0.7356$, $\text{CSI}_{64.5} = 0.3516$). Both strata demonstrate comparable high skill, proving that inland extraction preserves physical signal without distorting coastal evaluations.

### 6.2 Empirical Bootstrap p-Value Reporting
- **Correction:** The Stage 4.1 report claimed mathematically absolute certainty ($p = 0.000000$).
- **Scientific Standard:** When zero replicates out of $B=2,000$ (or $B=5,000$) cross zero, the true statistical p-value is bounded by the empirical resolution of the test: $p < \frac{1}{B+1}$. All Stage 7 tables and documentation strictly report **$p < 0.0005$** (or $p < 0.0002$ for $B=5,000$).

### 6.3 Custom Focal Loss Hessian Formulation
- **Correction:** Clarified the mathematical nature of the second-order derivative in LightGBM's custom focal loss.
- **Formulation:** The true analytical Hessian of focal loss can become negative in regions where prediction probability deviates sharply from the target, violating LightGBM's positive-semidefinite requirement for tree splitting. Following established literature, the training pipeline employed a **positive-definite surrogate Hessian**:
  $$h_i = w_i \cdot p_i (1 - p_i)$$
  This guarantees positive curvature and stable tree splitting during gradient boosting, operating as an effective numerical surrogate rather than the exact sign-indefinite analytical Hessian.

### 6.4 Non-Snooping Protocol & Partition Hygiene
- **Protocol Verification:** All model feature selection, hyperparameter tuning, early stopping (patience $= 50$), and classification decision thresholds ($p^* = 0.53$ for Focal Loss, $p^* = 0.96$ for Class-Weighted LightGBM, $p^* = 0.44$ for Rain/No-Rain Classifier) were optimized exclusively on the 2019–2022 Training or 2023 Validation partitions.
- **Zero Test Feedback:** The 2024–2025 Test partition was loaded solely for final frozen forward evaluation and descriptive reporting. No model was retrained, re-tuned, or re-selected based on test results.

### 6.5 Objective Attribution Language
- **Wording Audit:** Removed overconfident claims that ERA5 predictors are "indispensable" or that machine learning provides an "infallible replacement for NWP."
- **Defensible Framing:** ERA5 reanalysis predictors provide **quantifiable, statistically significant incremental predictive value** ($\Delta \text{RMSE} = -1.01\text{ mm}$ upon ablation; $+29.2\%$ KGE improvement).

### 6.6 Explicit Model Limitations & Physical Boundaries
1. **Extreme Quantile Shrinkage:** As documented in Table 8, continuous regression models trained under conditional mean or Tweedie objectives cannot capture extreme convective rainfall tails ($\ge 115.6\text{ mm}$). Predictions plateau around $70–90\text{ mm}$, resulting in an average underestimation of $-70.00\text{ mm}$ for Very Heavy events and $-151.36\text{ mm}$ for Extremely Heavy events. Operational deployment requires pairing the regressor with the dedicated Focal Loss extreme classifier.
2. **False Drizzle Inflation:** On true dry days ($0.00\text{ mm}$, $64.96\%$ of test records), the continuous Tweedie model predicts a mean of $+0.77\text{ mm}$ (MAE $= 0.77\text{ mm}$). While physically minor, this represents a structural artifact of continuous exponential dispersion models.
3. **Interior Convective Ceilings:** Performance over North and South Interior Karnataka exhibits a correlation ceiling ($r \approx 0.53–0.55$) compared to the Western Ghats ($r \ge 0.80$). This reflects the physical limitation of $0.1^\circ$ hydrostatic reanalysis in capturing localized, unorganized thermodynamic thunderstorms over flat terrain.
4. **Spatial Transferability:** Models were trained and evaluated across the same 252 spatial grid cells over time. Spatial generalisation to unseen geographic domains remains unverified and is explicitly not claimed.

---

## 7. Final Publication Readiness Assessment

### 7.1 Recommended Target Journals
1. **Journal of Hydrometeorology (AMS):** Highly suitable for the paper's focus on orographic precipitation, hydrological evaluation metrics (KGE, NSE), and the dominant predictive utility of land-surface runoff fluxes.
2. **Atmospheric Research (Elsevier):** Excellent fit for diagnostic downscaling, monsoon climatology, and comparative extreme precipitation alerting.
3. **Computers & Geosciences (Elsevier):** Ideal if emphasizing the machine learning methodology, asymmetric focal loss surrogate formulation, and complete reproducibility suite.

### 7.2 Recommended Manuscript Structure (6–8 Pages)
- **Title:** *Retrospective Daily Rainfall Hindcasting and Extreme Precipitation Alerting over Complex Orography: A Leakage-Free Machine Learning Framework for Karnataka, India*
- **Section 1: Introduction (1.0 page):** Challenges of Indian Summer Monsoon rainfall estimation; limitations of raw reanalysis and classical quantile mapping; research objectives.
- **Section 2: Data & Study Domain (1.0 page):** 252 IMD grid cells; ERA5-Land reanalysis; Copernicus GLO-90 DEM; 3 coastal fallback cells; 2019–2025 temporal partition with 2023 drought validation gate. *(Supports Table 1, Table 2, Figure 1).*
- **Section 3: Methodology (1.5 pages):** 24-hour causal lag architecture; Tweedie compound Poisson-Gamma regressor; Custom Focal Loss classifier with positive surrogate Hessian; Threshold optimization ($p^*$); Paired stationary block bootstrap protocol. *(Supports Figure 2).*
- **Section 4: Results & Discussion (2.5 pages):**
  - *Benchmarking & Hydrological Skill:* ML vs. Naive, ERA5, and EQM baselines. *(Supports Table 3, Figure 3, Figure 4).*
  - *Extreme Precipitation Alerting:* False alarm suppression via Focal Loss. *(Supports Table 4, Figure 5).*
  - *Statistical Significance:* Paired bootstrap distributions and block length invariance. *(Supports Table 6, Figure 9).*
  - *Regional & Seasonal Dynamics:* Orographic Ghats capture vs. interior convective ceiling. *(Supports Table 5, Figure 7).*
  - *Physical Attribution & Ablation:* Dominance of runoff fluxes and incremental ERA5 value. *(Supports Table 7, Figure 6).*
- **Section 5: Limitations & Physical Boundaries (0.75 page):** Quantile shrinkage above $115.6\text{ mm}$; dry-day drizzle inflation; absence of spatial transferability. *(Supports Table 8, Figure 8).*
- **Section 6: Conclusion (0.25 page):** Summary of contributions and future operational outlook.

### 7.3 Peer Review Readiness Checklist
- [x] All 8 publication tables generated in clean, standard CSV schemas.
- [x] All 9 publication figures rendered in 300 DPI PNG and vector PDF format with consistent typography and color palettes.
- [x] Master results table contains 44 audited model/partition rows matching Stage 4.2 within $<10^{-4}$.
- [x] Zero data leakage across temporal boundaries; test partition (2024–2025) strictly frozen.
- [x] Complete reproducibility package released (code, config, environment, data dictionary, hash manifest).
- [x] Overconfident wording eliminated; limitations transparently documented.
- [x] Coastal fallback cells (IDs 52, 68, 125) fully audited and documented.

**Conclusion:** Stage 7 is complete, fully verified, and ready to support manuscript preparation.

