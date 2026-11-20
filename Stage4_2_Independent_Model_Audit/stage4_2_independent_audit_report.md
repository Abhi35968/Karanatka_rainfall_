# Stage 4.2 — Independent Model Results, Metric and Statistical Audit Report

**Project:** Karnataka Hydrometeorological Daily Rainfall Forecasting Research  
**Root Repository:** [`d:\PROJECT\karnataka_rainfall_cds\`](file:///d:/PROJECT/karnataka_rainfall_cds/)  
**Audit Directory:** [`d:\PROJECT\karnataka_rainfall_cds\Stage4_2_Independent_Model_Audit\`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/)  
**Primary Auditor:** Senior Hydrometeorological Research Scientist, Scientific Reviewer & Independent Data Auditor  
**Audit Scope:** Independent recalculation of all metrics, mathematical audit of focal loss, 2x2 confusion matrix verification, 5,000-replicate paired block bootstrap significance analysis, feature ablation experiments, Duan smearing audit, coastal fallback cell reconciliation, and test set governance verification.

---

## 1. Executive Summary & Audit Verdict

This audit was conducted independently to verify the empirical validity, mathematical correctness, and scientific reproducibility of the models and benchmarks developed in **Stage 4.1**. In strict accordance with the Stage 4.2 protocol, all existing Stage 4.1 datasets, scripts, models, and results were preserved completely untouched.

### Definitive Audit Verdict: **STAGE 4.1 RESULTS ARE INDEPENDENTLY RECONCILED AND METHODOLOGICALLY RIGOROUS.**

### Summary of Audit Determinations:
1. **100% Metric Reconciliation (Task 1):** Independent recalculation from raw predictions and ground-truth observations across all **44 reported model/partition rows** in Stage 4.1 (`baseline_results.csv`, `regression_model_results.csv`, `hurdle_model_results.csv`, `rain_occurrence_results.csv`, `heavy_rain_results.csv`) confirmed exact mathematical agreement ($< 0.001$ discrepancy) across all continuous, contingency, and probabilistic metrics.
2. **Focal Loss Objective & Hessian Formulation (Task 2):** Analytical differentiation and numerical finite-difference testing confirmed that the implemented `focal_loss_lgb` uses a **positive-definite surrogate Hessian approximation** ($h_i = w_i p(1-p)$). This audit establishes that the true mathematical Hessian of binary focal loss has negative curvature in boundary regions ($p < 0.05$ for $y=1$ and $p > 0.95$ for $y=0$), which would cause standard Newton-Raphson tree splitting in LightGBM to encounter negative denominators. The implemented surrogate is a recognized heuristic that stabilizes tree-splitting by guaranteeing strictly positive curvature, but it should be reported as an empirical surrogate approximation rather than the exact mathematical second derivative or a theoretical proof of optimization convergence.
3. **Heavy Rain Classification Verification (Task 3):** Full 2x2 confusion matrices on the Test partition ($184,212$ instances, $1,267$ positive events $\ge 64.5\text{ mm}$) confirmed that **LightGBM with Focal Loss** achieves higher Precision ($58.22\%$) and lower False Alarm Ratio ($41.78\%$) than standard class-weighting, yielding a Critical Success Index of **$\text{CSI} = 0.3512$** at a frozen decision threshold of $p^* = 0.53$ calibrated strictly on Validation data.
4. **Statistical Significance via Paired Block Bootstrapping (Task 4):** A 5,000-replicate paired stationary temporal block bootstrap ($L = 7$ days, preserving statewide spatial fields) demonstrated consistent paired reductions in RMSE for machine learning models over Persistence ($\Delta \text{RMSE} = -2.935\text{ mm}$, $95\%\text{ CI } [-3.158, -2.728]\text{ mm}$) and over statistical Quantile Mapping EQM ($\Delta \text{RMSE} = -3.778\text{ mm}$, $95\%\text{ CI } [-3.994, -3.582]\text{ mm}$). Given that zero of the 5,000 resamples produced non-negative differences, the empirical test achieves its resolution limit of **$p < 0.0002$**, providing strong empirical support under the exchangeability and block-length assumptions of the stationary bootstrap.
5. **Measurable Incremental Value of ERA5 Predictors (Task 5):** Systematic ablation across 6 scenarios demonstrated that removing ERA5 meteorological variables causes test RMSE to increase from **$7.44\text{ mm}$ to $8.45\text{ mm}$** ($+13.5\%$) and decreases extreme detection skill ($\text{CSI}_{64.5}$ drops from $0.2868$ to $0.1536$). This confirms that physical atmospheric predictors provide measurable incremental predictive value beyond purely autoregressive IMD rainfall and calendar features.
6. **Duan Smearing Limitations Documented (Task 6):** The smearing factor $S = 1.35112$ was verified to be computed strictly from training residuals. However, residual heteroscedasticity analysis proved that conditional smearing $S(\hat{z})$ varies from $1.008$ on dry days to $1.813$ on wet days. Consequently, a single global scalar cannot universally correct conditional bias across all rainfall regimes. Raw-scale compound Poisson-Gamma (Tweedie) regression ($p=1.5$) is methodologically preferred, achieving lower RMSE ($7.44\text{ mm}$) and negligible mean bias ($-0.018\text{ mm}$).
7. **Reconciliation of Coastal Fallback Cells:** The apparent discrepancy between the 3 fallback cells identified in Stage 2 and the 13 cells mentioned in Stage 4.1 text was resolved: exactly 13 grid cells belong to the IMD Coastal Karnataka subdivision, and exactly 3 of those 13 cells (Station IDs 52, 68, 125) carry the `is_coastal_fallback == 1` flag.
8. **Test Set Governance & Transparency (Task 7):** All feature scalers, categorical target encodings, EQM knots, and decision thresholds were fitted strictly on Train/Validation. The test set was accessed strictly for frozen forward evaluation in Stage 4.1 and frozen feature ablation evaluation in Stage 4.2; no hyperparameters, feature subsets, or decision thresholds were tuned or altered in response to test set metrics.

---

## 2. Task 1: Independent Metric Recalculation & Reconciliation

### Metric Definitions and Formulations:
Let $y$ be the observed IMD rainfall (mm), $\hat{y}$ be the model prediction (mm), $N$ be the sample count, $\mu_y, \mu_{\hat{y}}$ be the sample means, and $\sigma_y, \sigma_{\hat{y}}$ be the sample standard deviations.

1. **Continuous Regression Metrics:**
   - $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$
   - $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$
   - $\text{Mean Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) = \mu_{\hat{y}} - \mu_y$
   - $\text{Percentage Bias (PBias)} = 100 \times \frac{\sum (\hat{y}_i - y_i)}{\sum y_i} = 100 \times \frac{\text{Bias}}{\mu_y}$
   - $\text{Pearson Correlation } r = \frac{\sum (y_i - \mu_y)(\hat{y}_i - \mu_{\hat{y}})}{\sqrt{\sum (y_i - \mu_y)^2 \sum (\hat{y}_i - \mu_{\hat{y}})^2}}$
   - $\text{Nash-Sutcliffe Efficiency (NSE)} = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \mu_y)^2}$
   - **Kling-Gupta Efficiency (Classic Gupta et al., 2009):**
     $$\text{KGE}_{2009} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$
     where $\beta = \frac{\mu_{\hat{y}}}{\mu_y}$ (bias ratio) and $\alpha = \frac{\sigma_{\hat{y}}}{\sigma_y}$ (variability ratio).
   - **Modified Kling-Gupta Efficiency (Kling et al., 2012):**
     $$\text{KGE}_{2012} = 1 - \sqrt{(r - 1)^2 + (\gamma - 1)^2 + (\beta - 1)^2}$$
     where $\gamma = \frac{CV_{\hat{y}}}{CV_y} = \frac{\sigma_{\hat{y}}/\mu_{\hat{y}}}{\sigma_y/\mu_y} = \frac{\alpha}{\beta}$.

### Handling of Edge Cases:
- **Zero Means:** If $\mu_y \le 10^{-8}$, $\beta$ is set to $1.0$ (or clipped) to prevent division by zero.
- **Zero Variance:** If $\sigma_y \le 10^{-8}$ or $\sigma_{\hat{y}} \le 10^{-8}$, $r$ is set to $0.0$, and KGE is assigned the standard invalid flag $-999.0$.
- **Missing Values:** Zero missing values exist in the audited datasets ($100\%$ complete records across all 252 IMD grid cells).

### Reconciled Metrics Audit Table:
From [`independent_metric_recalculation.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/independent_metric_recalculation.csv):

| Model Name | Partition | Reported RMSE | Recalculated RMSE | Reported Pearson $r$ | Recalculated Pearson $r$ | Reported KGE-2012 | Recalculated KGE-2012 | Classic KGE-2009 | Reconciled Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rainfall_Persistence** | Validation_2023 | 7.6674 | 7.6674 | 0.6032 | 0.6032 | 0.6032 | 0.6032 | 0.6032 | **MATCH** |
| **Rainfall_Persistence** | Test_2024_2025 | 10.3582 | 10.3582 | 0.5565 | 0.5565 | 0.5565 | 0.5565 | 0.5565 | **MATCH** |
| **Station_DOY_Climatology** | Validation_2023 | 8.6516 | 8.6516 | 0.3993 | 0.3993 | 0.0169 | 0.0169 | 0.1307 | **MATCH** |
| **Station_DOY_Climatology** | Test_2024_2025 | 10.1155 | 10.1155 | 0.4313 | 0.4313 | 0.2851 | 0.2851 | 0.3068 | **MATCH** |
| **Antecedent_ERA5_Land** | Validation_2023 | 7.4361 | 7.4361 | 0.5647 | 0.5647 | 0.4905 | 0.4905 | 0.5200 | **MATCH** |
| **Antecedent_ERA5_Land** | Test_2024_2025 | 10.4550 | 10.4550 | 0.4911 | 0.4911 | 0.4111 | 0.4111 | 0.4510 | **MATCH** |
| **Empirical_Quantile_Mapping_EQM** | Validation_2023 | 8.1535 | 8.1535 | 0.5380 | 0.5380 | 0.5373 | 0.5373 | 0.5364 | **MATCH** |
| **Empirical_Quantile_Mapping_EQM** | Test_2024_2025 | 11.1970 | 11.1970 | 0.4941 | 0.4941 | 0.4763 | 0.4763 | 0.4818 | **MATCH** |
| **LightGBM_RMSE** | Validation_2023 | 5.6442 | 5.6442 | 0.7554 | 0.7554 | 0.5907 | 0.5907 | 0.6341 | **MATCH** |
| **LightGBM_RMSE** | Test_2024_2025 | 7.4982 | 7.4982 | 0.7320 | 0.7320 | 0.5812 | 0.5812 | 0.6190 | **MATCH** |
| **LightGBM_Log_OrdinaryInverse** | Validation_2023 | 6.0075 | 6.0075 | 0.7505 | 0.7505 | 0.5025 | 0.5025 | 0.3356 | **MATCH** |
| **LightGBM_Log_OrdinaryInverse** | Test_2024_2025 | 7.8497 | 7.8497 | 0.7260 | 0.7260 | 0.5276 | 0.5276 | 0.3703 | **MATCH** |
| **LightGBM_Log_DuanSmearing** | Validation_2023 | 5.6913 | 5.6913 | 0.7505 | 0.7505 | 0.6831 | 0.6831 | 0.6414 | **MATCH** |
| **LightGBM_Log_DuanSmearing** | Test_2024_2025 | 7.5893 | 7.5893 | 0.7260 | 0.7260 | 0.6722 | 0.6722 | 0.6439 | **MATCH** |
| **LightGBM_Tweedie_p1.5** | Validation_2023 | 5.5850 | 5.5850 | 0.7618 | 0.7618 | 0.6494 | 0.6494 | 0.6349 | **MATCH** |
| **LightGBM_Tweedie_p1.5** | Test_2024_2025 | 7.4427 | 7.4427 | 0.7363 | 0.7363 | 0.6245 | 0.6245 | 0.6218 | **MATCH** |
| **CatBoost_Tweedie_p1.5** | Validation_2023 | 5.5868 | 5.5868 | 0.7613 | 0.7613 | 0.6401 | 0.6401 | 0.6401 | **MATCH** |
| **CatBoost_Tweedie_p1.5** | Test_2024_2025 | 7.4335 | 7.4335 | 0.7370 | 0.7370 | 0.6261 | 0.6261 | 0.6318 | **MATCH** |
| **Hurdle_ExpectedValue** | Validation_2023 | 5.5740 | 5.5740 | 0.7634 | 0.7634 | 0.6329 | 0.6329 | 0.6305 | **MATCH** |
| **Hurdle_ExpectedValue** | Test_2024_2025 | 7.4241 | 7.4241 | 0.7380 | 0.7380 | 0.6114 | 0.6114 | 0.6165 | **MATCH** |

*Audit Observation on KGE:* The Stage 4.1 engine reported `KGE_2012` using the variability ratio $\gamma = CV_{\hat{y}}/CV_y$. In classic KGE-2009, $\alpha = \sigma_{\hat{y}}/\sigma_y$ is used. For models suffering from retransformation compression (e.g. `LightGBM_Log_OrdinaryInverse`), classic KGE drops to **0.3703** (due to $\alpha \approx 0.62$). For the champion Tweedie models, both `KGE_2012` ($0.6261$) and `KGE_2009` ($0.6318$) exceed $0.62$, confirming robust hydrological fidelity under both definitions.

---

## 3. Reconciliation of Coastal Fallback Cells (Stage 2 vs. Stage 4.1)

A specific discrepancy was identified between the 3 coastal fallback cells documented in Stage 2 and the 13 cells referenced in certain narrative sections of Stage 4.1:

### Forensic Reconciliation:
1. **Subdivision Definition:** Exactly **13 IMD grid cells** comprise the entire **Coastal Karnataka** subdivision ($13 \text{ cells} \times 731 \text{ days} = 9,503\text{ test records}$).
2. **Fallback Flag Definition:** Exactly **3 of those 13 cells** carry the indicator flag `is_coastal_fallback == 1` ($3 \text{ cells} \times 731 \text{ days} = 2,193\text{ test records}$).
3. **Exact Grid-Cell Inventory:**
   - **Station ID 52:** Latitude $13.25^\circ\text{N}$, Longitude $74.75^\circ\text{E}$ (Udupi / Malpe coastal zone, distance to coast = $5.94\text{ km}$).
   - **Station ID 68:** Latitude $13.50^\circ\text{N}$, Longitude $74.75^\circ\text{E}$ (Kundapura coastal zone, distance to coast = $15.11\text{ km}$).
   - **Station ID 125:** Latitude $14.75^\circ\text{N}$, Longitude $74.25^\circ\text{E}$ (Karwar / Ankola coastal zone, distance to coast = $6.98\text{ km}$).
4. **Physical Definition:** The indicator `is_coastal_fallback == 1` marks cells where the IMD 0.25° grid-cell center fell over Arabian Sea open water in the native ERA5-Land land-sea mask, requiring nearest-neighbor inland extraction (as established in Stage 2). The remaining 10 coastal cells had native land representation.
5. **Resolution of Narrative Discrepancy:** The Stage 4.1 prose inadvertently conflated the 13 cells of the Coastal Karnataka subdivision with the 3 fallback cells, even though the computed test sample count ($2,193 = 3 \times 731$) accurately reflected the 3 fallback cells.

---

## 4. Task 2: Focal Loss Implementation & Surrogate Hessian Audit

The Stage 4.1 script [`train_heavy_rain_models.py`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_1_Model_Development/train_heavy_rain_models.py) defined a custom objective function for LightGBM to handle class imbalance ($0.73\%$ heavy rain events).

### Mathematical Derivation of True Binary Focal Loss:
Binary focal loss (Lin et al., 2017) is defined as:
$$\mathcal{L}_{FL}(y, p) = - \alpha_t (1 - p_t)^\gamma \ln(p_t)$$
where $p = \sigma(z) = \frac{1}{1 + e^{-z}}$, and for $y \in \{0, 1\}$:
$$p_t = y p + (1 - y)(1 - p), \quad \alpha_t = y \alpha + (1 - y)(1 - \alpha)$$

Differentiating with respect to logit margin $z$:
1. For $y = 1$:
   $$\frac{\partial \mathcal{L}_1}{\partial z} = \alpha (1 - p)^\gamma \left[ \gamma p \ln(p) + p - 1 \right]$$
2. For $y = 0$:
   $$\frac{\partial \mathcal{L}_0}{\partial z} = (1 - \alpha) p^\gamma \left[ p - \gamma (1 - p) \ln(1 - p) \right]$$

### The Non-Convex Curvature Problem in the True Hessian:
Evaluating the true second derivative $\frac{\partial^2 \mathcal{L}}{\partial z^2}$ across logit margins $z \in [-8, 8]$ reveals that the true Hessian becomes **negative** in boundary regions:
- For $y = 1$ when $z \le -3.0$ ($p \le 0.047$), $\frac{\partial^2 \mathcal{L}_1}{\partial z^2} < 0$.
- For $y = 0$ when $z \ge 3.0$ ($p \ge 0.953$), $\frac{\partial^2 \mathcal{L}_0}{\partial z^2} < 0$.

### Numerical Audit from `focal_loss_gradient_hessian_tests.csv`:
| Label $y$ | Logit $z$ | Probability $p$ | Implemented Gradient | True Analytical Gradient | Finite-Diff Gradient | Implemented Hessian | True Analytical Hessian | True Hessian Negative? | Implemented Positive? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **-6.0** | 0.002473 | -0.744450 | -0.766603 | -0.766603 | **+0.001841** | **-0.012785** | **YES (Negative)** | **YES (Positive)** |
| **1** | **-4.0** | 0.017986 | -0.710255 | -0.814797 | -0.814797 | **+0.012775** | **-0.035027** | **YES (Negative)** | **YES (Positive)** |
| **1** | **0.0** | 0.500000 | -0.093750 | -0.223715 | -0.223715 | +0.046875 | +0.299358 | NO | YES |
| **1** | **2.0** | 0.880797 | -0.001270 | -0.003653 | -0.003653 | +0.001119 | +0.009508 | NO | YES |
| **0** | **-2.0** | 0.119203 | +0.000423 | +0.001218 | +0.001218 | +0.000373 | +0.003169 | NO | YES |
| **0** | **0.0** | 0.500000 | +0.031250 | +0.074572 | +0.074572 | +0.015625 | +0.099786 | NO | YES |
| **0** | **4.0** | 0.982014 | +0.236752 | +0.271599 | +0.271599 | **+0.004258** | **-0.011676** | **YES (Negative)** | **YES (Positive)** |
| **0** | **6.0** | 0.997527 | +0.248150 | +0.255534 | +0.255534 | **+0.000614** | **-0.004262** | **YES (Negative)** | **YES (Positive)** |

### Clarification on the Hessian Approximation:
1. **Surrogate Nature:** In gradient tree boosting, leaf values are computed via Newton-Raphson descent:
   $$w_{leaf} = - \frac{\sum_{i \in leaf} g_i}{\sum_{i \in leaf} h_i + \lambda}$$
   Because negative true Hessians would destabilize this denominator, `focal_loss_lgb` uses a **positive-definite surrogate approximation**:
   $$g_i = w_i (p_i - y_i), \quad h_i = w_i p_i (1 - p_i) > 0$$
   where $w_i = \alpha_t (1 - p_{t, i})^\gamma$ acts as a detached sample weight.
2. **Methodological Framing:** This surrogate ensures strictly positive curvature and empirical stability during tree growth. However, it is an empirical heuristic rather than the exact mathematical second derivative of the focal loss, and it should not be described as a proof of optimization convergence.
3. **Probability Conversion:** Because custom objectives in LightGBM output raw margins $z$, the code correctly applies the logistic sigmoid function $p = \frac{1}{1 + e^{-z}}$ prior to probability calibration and thresholding.

---

## 5. Task 3: Heavy-Rain Classifier Confusion Matrix Verification

Official IMD threshold $\ge 64.5\text{ mm}$ evaluated on the Test partition ($184,212$ instances, $1,267$ positive events, $0.6878\%$ base rate):

From [`heavy_rain_confusion_matrices.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/heavy_rain_confusion_matrices.csv):

| Model Name | Calibrated Threshold $p^*$ | Hits ($H$) | False Alarms ($F$) | Misses ($M$) | Correct Negatives ($C$) | Precision | POD (Recall) | FAR | CSI | HSS | ETS | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Focal Loss** | **0.53** | **595** | **427** | 672 | 182,518 | **0.5822** | 0.4696 | **0.4178** | **0.3512** | **0.5169** | **0.3485** | **0.5084** |
| **LightGBM Class-Weighted** | 0.96 | 688 | 722 | 579 | 182,223 | 0.4879 | **0.5430** | 0.5121 | 0.3459 | 0.5105 | 0.3427 | 0.4853 |
| **CatBoost Class-Weighted** | 0.98 | 608 | 515 | 659 | 182,430 | 0.5414 | 0.4799 | 0.4586 | 0.3412 | 0.5056 | 0.3383 | 0.4739 |
| **Rainfall Persistence** | Fixed 64.5 mm | 484 | 783 | 783 | 182,162 | 0.3820 | 0.3820 | 0.6180 | 0.2361 | 0.3777 | 0.2328 | N/A |
| **Raw ERA5-Land** | Fixed 64.5 mm | 231 | 606 | 1,036 | 182,339 | 0.2760 | 0.1823 | 0.7240 | 0.1233 | 0.2153 | 0.1206 | N/A |
| **Empirical Quantile Mapping** | Fixed 64.5 mm | 372 | 622 | 895 | 182,323 | 0.3742 | 0.2936 | 0.6258 | 0.1969 | 0.3250 | 0.1940 | N/A |

- **Threshold Governance Confirmed:** All decision thresholds ($p^* = 0.53, 0.96, 0.98$) were derived exclusively from the 2023 Validation partition to maximize CSI.
- **Empirical Advantage:** LightGBM Focal Loss reduced false alarms from $722$ (Class-Weighted) down to $427$ while achieving higher Precision ($58.22\%$) and the highest Test CSI ($0.3512$).

---

## 6. Task 4: Statistical Significance Audit (Paired Block Bootstrapping)

To rigorously test whether the observed performance differences are robust against temporal autocorrelation and spatial dependence, a paired stationary temporal block bootstrap was conducted with $B = 5,000$ replicates ($L = 7$ days).

### Methodological Design:
- **Spatial Field Preservation:** Daily rainfall fields across all 252 IMD grid cells were resampled together for each date, preserving statewide spatial dependence.
- **Temporal Memory Preservation:** Blocks of 7 consecutive days were resampled with replacement to preserve synoptic persistence.
- **Paired Statistic:** For each replicate $b$, the paired difference in test RMSE was computed:
  $$\Delta \text{RMSE}^{(b)} = \text{RMSE}_{\text{Model A}}^{(b)} - \text{RMSE}_{\text{Baseline B}}^{(b)}$$

### Paired Bootstrap Results ($B = 5,000$ Replicates):
From [`paired_bootstrap_rmse_differences.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/paired_bootstrap_rmse_differences.csv):

| Comparison Pair | Model A Mean RMSE | Baseline B Mean RMSE | Paired $\Delta \text{RMSE}$ Mean | Paired $\Delta \text{RMSE}$ 95% Confidence Interval | Percent RMSE Reduction | Replicates with $\Delta \text{RMSE} \ge 0$ | Empirical Bootstrap $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hurdle (EV) vs. Persistence** | 7.4241 | 10.3582 | **-2.9353 mm** | **[-3.1580, -2.7278] mm** | **28.35%** | **0 / 5,000** | **$p < 0.0002$** |
| **Hurdle (EV) vs. Raw ERA5** | 7.4241 | 10.4550 | **-3.0346 mm** | **[-3.2327, -2.8435] mm** | **29.03%** | **0 / 5,000** | **$p < 0.0002$** |
| **Hurdle (EV) vs. EQM** | 7.4241 | 11.1970 | **-3.7776 mm** | **[-3.9941, -3.5820] mm** | **33.74%** | **0 / 5,000** | **$p < 0.0002$** |
| **CatBoost Tweedie vs. Persistence** | 7.4335 | 10.3582 | **-2.9262 mm** | **[-3.1515, -2.7163] mm** | **28.26%** | **0 / 5,000** | **$p < 0.0002$** |
| **CatBoost Tweedie vs. EQM** | 7.4335 | 11.1970 | **-3.7685 mm** | **[-3.9907, -3.5684] mm** | **33.66%** | **0 / 5,000** | **$p < 0.0002$** |
| **LightGBM Tweedie vs. Persistence** | 7.4427 | 10.3582 | **-2.9166 mm** | **[-3.1388, -2.7100] mm** | **28.17%** | **0 / 5,000** | **$p < 0.0002$** |
| **LightGBM Tweedie vs. EQM** | 7.4427 | 11.1970 | **-3.7589 mm** | **[-3.9682, -3.5672] mm** | **33.57%** | **0 / 5,000** | **$p < 0.0002$** |

### Statistical Framing and Limitations:
- **Empirical Resolution Limit:** Because zero of the 5,000 bootstrap replicates yielded $\Delta \text{RMSE} \ge 0$, the empirical one-sided $p$-value is bounded by the finite resolution of the test: $p < \frac{1}{B+1} = \frac{1}{5001} \approx 0.0002$.
- **Appropriate Scientific Interpretation:** These results provide strong empirical evidence of consistent paired improvement across resampled test conditions. However, this should not be characterized as "mathematically irrefutable" proof, as it remains conditional on the exchangeability and block length ($L = 7$ days) assumptions of the stationary block bootstrap.

---

## 7. Task 5: Feature Ablation Experiments

To evaluate the incremental contribution of specific predictor subsets, feature ablation was performed across 6 scenarios using LightGBM Tweedie ($p=1.5$):

### Ablation Results:
From [`feature_ablation_validation_results.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/feature_ablation_validation_results.csv) and [`feature_ablation_test_results.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/feature_ablation_test_results.csv):

| Ablation Scenario | Feature Count | Validation RMSE (mm) | Validation KGE-2012 | Validation $\text{CSI}_{64.5}$ | Test RMSE (mm) | Test KGE-2012 | Test $\text{CSI}_{64.5}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. All 63 Primary Predictors (Full Model)** | **63** | **5.5850** | **0.6494** | **0.4098** | **7.4427** | **0.6245** | **0.2868** |
| **B. Without Runoff & Hydrological Fluxes** | 55 | 5.7149 | 0.6463 | 0.4129 | 7.6854 | 0.5925 | 0.2721 |
| **C. Without Antecedent IMD Rainfall** | 51 | 5.7147 | 0.6235 | 0.3316 | 7.5828 | 0.6070 | 0.2693 |
| **D. Without Terrain & Spatial Predictors** | 55 | 5.5809 | 0.6479 | 0.4097 | 7.4557 | 0.6252 | 0.2843 |
| **E. Without ERA5 Predictors** | 26 | 6.3909 | 0.4181 | 0.2652 | 8.4511 | 0.4833 | 0.1536 |
| **F. Only Antecedent IMD & Temporal** | 18 | 6.4289 | 0.4193 | 0.3043 | 8.5145 | 0.4686 | 0.1456 |

### Defensible Physical Findings:
1. **Incremental Predictive Value of ERA5:** Removing all ERA5 predictors (Scenario E) increases test RMSE by **$1.01\text{ mm}$ ($+13.5\%$)**, from $7.44\text{ mm}$ to $8.45\text{ mm}$, and reduces $\text{CSI}_{64.5}$ from $0.2868$ to $0.1536$. While not describing them as "indispensable", the empirical data confirm that ERA5 atmospheric predictors provide **measurable incremental predictive value** beyond autoregressive IMD precipitation and calendar features.
2. **Antecedent Rainfall Memory:** Omitting antecedent IMD ground observations (Scenario C) degrades extreme rainfall prediction, dropping Validation CSI from $0.4098$ to $0.3316$.
3. **Runoff Fluxes:** Removing surface and total runoff features (`sro`, `ro`, `src`) worsens Test KGE from $0.6245$ to $0.5925$, confirming the utility of land-surface moisture proxies.

---

## 8. Task 6: Duan Smearing & Heteroscedasticity Audit

- **Derivation Integrity:** The smearing factor $S = 1.35112$ was confirmed to be computed strictly from training residuals ($N = 364,644$).
- **Heteroscedasticity Analysis:** Evaluating the conditional smearing factor $S(\hat{z}) = \mathbb{E}[\exp(e) | \hat{z}]$ across prediction bins revealed that $S(\hat{z})$ ranges from **$1.008$** for dry predictions ($\hat{z} \le 0.1$) up to **$1.813$** for moderate-to-heavy predictions ($\hat{z} \in [1.0, 2.0]$).
- **Scientific Limitation:** Because $38.1\%$ of the training records are near-zero days where $S \approx 1.0$, the global average is pulled down to $1.3511$. A single global scalar smearing factor cannot universally correct conditional bias across distinct hydrometeorological regimes.
- **Tweedie Comparison:** Raw-scale Tweedie regression ($p=1.5$) models $\mathbb{E}[Y|X]$ directly on the physical millimeter scale, avoiding retransformation bias altogether, achieving lower test RMSE ($7.44\text{ mm}$ vs. $7.59\text{ mm}$) and negligible bias ($-0.018\text{ mm}$ vs. $-0.211\text{ mm}$).

---

## 9. Test Set Governance & Transparency Statement

- **Leakage Prevention:** All scalers, encodings, and EQM transfer functions were fitted strictly on the Training partition (2019–2022).
- **Threshold Freezing:** All decision thresholds were calibrated strictly on the 2023 Validation partition.
- **Evaluation Disclosure:** The test set was accessed for frozen benchmark evaluation in Stage 4.1 and for frozen feature ablation evaluation in Stage 4.2. **No model architecture, hyperparameter, feature subset, or decision threshold was altered or tuned based on test set results.**
- **Governance Closure:** Having completed both benchmark and ablation reporting on the Test set, model development on this temporal partition is now strictly closed to prevent post-hoc snooping.

---

## 10. Audit Sign-Off

The Stage 4.2 audit is **COMPLETE AND CLOSED**. All 8 deliverables are persisted in [`Stage4_2_Independent_Model_Audit/`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_2_Independent_Model_Audit/). The findings provide solid, scientifically defensible empirical support for manuscript publication.
