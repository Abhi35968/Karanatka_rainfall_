# Test Set Usage, Data Governance, and Leakage Audit (Stage 4.2)

**Document Identifier:** STAGE-4.2-TEST-GOVERNANCE-AUDIT  
**Project:** Karnataka Hydrometeorological Daily Rainfall Forecasting Research  
**Author:** Independent Scientific Audit Team  
**Evaluation Scope:** Temporal partitioning protocol, data leakage verification, hyperparameter governance, and transparent disclosure of test set evaluations.

---

## 1. Executive Summary

A critical standard for publication in top-tier hydrometeorological and machine learning journals is ensuring that out-of-sample test results represent genuine generalization rather than artifacts of data leakage, iterative tuning, or test snooping.

This audit conducted a line-by-line inspection of data pipelines, preprocessors, loss functions, stopping criteria, and decision threshold calibration routines across Stages 3.4, 4.0, 4.1, and 4.2.

### Definitive Governance Verdict:
- **Zero Feature Leakage:** All scalers (`StandardScaler`), target encodings (`subdivision`, `zone_id`), Empirical Quantile Mapping (EQM) transfer functions, and Duan's smearing correction factor ($S = 1.35112$) were fitted strictly on the **Training partition (2019–2022, $N = 364,644$)**.
- **Zero Threshold Snooping:** All operational decision thresholds ($p^*$) for rain occurrence ($\ge 0.1\text{ mm}$) and official IMD heavy rain ($\ge 64.5\text{ mm}$) were calibrated strictly on the **2023 Validation partition ($N = 91,980$)** prior to test evaluation.
- **Zero Hyperparameter Tuning on Test:** Early stopping (50 rounds) and model architecture selection were governed exclusively by Validation metrics.
- **Full Disclosure of Test Partition Access:** The test partition (2024–2025, $N = 184,212$) was accessed twice:
  1. In Stage 4.1, to execute the frozen forward benchmark evaluation.
  2. In Stage 4.2, to conduct the independent metric recalculation and the frozen feature ablation evaluation.
  At no point was any test evaluation score used iteratively to modify hyperparameters, select feature subsets, or adjust decision thresholds. The test partition remains an authentic, uncompromised forward out-of-sample evaluation.

---

## 2. Partition Arithmetic and Temporal Boundaries

The experimental dataset was partitioned chronologically to emulate retrospective operational hindcasting, ensuring that models never predict past events using future observations:

| Partition Name | Calendar Coverage | Duration (Days) | In-State IMD Grid Cells | Total Record Count | Primary Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training** | 2019-01-01 to 2022-12-31 | 1,461 | 252 | **364,644** | Model fitting, parameter estimation, scalers, encodings, Duan smearing |
| **Validation** | 2023-01-01 to 2023-12-31 | 365 | 252 | **91,980** | Early stopping, probability calibration, threshold optimization ($p^*$) |
| **Test** | 2024-01-01 to 2025-12-31 | 731 | 252 | **184,212** | Frozen forward evaluation, benchmark comparison, bootstrap CIs |
| **Total Full Dataset** | **2019-01-01 to 2025-12-31** | **2,557** | **252** | **640,836** | Complete harmonized research dataset |

*Note on Partition Arithmetic:* In Stage 3.4, the 14-day antecedent initialization window (2019-01-01 to 2019-01-14, 3,528 records) was dropped from Training to ensure all 14-day rolling features (`api_14d`, `target_rain_roll_14d_sum`) were 100% free of missing values, reducing Training from $368,172$ to exactly $364,644$ rows. Validation ($91,980$) and Test ($184,212$) maintain full calendar coverage without record loss.

---

## 3. Forensic Leakage Audit Across Preprocessing Components

We audited every data transformation to verify that no information from Validation or Test bled into Training:

### A. Target Encodings for Categorical Predictors
- **Variables Encoded:** `subdivision` (4 classes: Coastal, Malnad, North Interior, South Interior) and `zone_id` (4 classes).
- **Code Audit:**
  ```python
  subdiv_map = (
      df_train.groupby("subdivision")["target_rainfall_mm"].mean().to_dict()
  )
  zone_map = (
      df_train.groupby("zone_id")["target_rainfall_mm"].mean().to_dict()
  )
```
  *Audit Verification:* The dictionary mapping was derived strictly from `df_train`. Validation and Test datasets were encoded purely via `.map()` lookups, using constant global training means for any unseen fallbacks. **Status: LEAKAGE-FREE.**

### B. Feature Standardization for Linear Baselines (Ridge & ElasticNet)
- **Component:** `StandardScaler` applied to the 63 primary predictors.
- **Code Audit:**
  ```python
  scaler = StandardScaler()
  X_tr_scaled = scaler.fit_transform(X_tr)
  X_va_scaled = scaler.transform(X_va)
  X_te_scaled = scaler.transform(X_te)
```
  *Audit Verification:* `scaler.fit()` was called strictly on `X_tr`. Validation and Test matrices were transformed using the frozen training sample means and standard deviations. **Status: LEAKAGE-FREE.**

### C. Empirical Quantile Mapping (EQM)
- **Component:** 252 station-specific empirical cumulative distribution function (CDF) transfer functions.
- **Code Audit:**
  `run_baselines.py` fit the empirical quantiles strictly on 2019–2022 observed IMD and ERA5 values. Quantile transfer to 2023 and 2024–2025 utilized frozen linear interpolation between empirical knots. **Status: LEAKAGE-FREE.**

### D. Duan's Smearing Retransformation Factor
- **Component:** Non-parametric multiplier $S = \frac{1}{N} \sum \exp(e_i)$ for log-transformed regression.
- **Code Audit:**
  `train_regression_models.py` calculated residuals strictly from `z_tr - z_pred_tr`, yielding $S = 1.35112$. **Status: LEAKAGE-FREE.**

---

## 4. Hyperparameter and Decision Threshold Governance

A frequent failure mode in published machine learning research is optimizing decision thresholds ($p^*$) directly on the test set to inflate reported F1 or CSI scores.

### Audit of Decision Threshold Calibration:
We inspected the optimization routines across Tasks A, B, and D:
1. **Rain Occurrence Classifier (Task A, $\ge 0.1\text{ mm}$):**
   - The threshold grid $p \in [0.20, 0.60]$ was evaluated on the **2023 Validation set** to maximize Heidke Skill Score (HSS).
   - Optimal thresholds identified on Validation:
     - LightGBM: $p^* = 0.44$ (Validation $\text{HSS} = 0.6472$)
     - XGBoost: $p^* = 0.44$ (Validation $\text{HSS} = 0.6462$)
     - CatBoost: $p^* = 0.42$ (Validation $\text{HSS} = 0.6460$)
   - These thresholds were **frozen** and directly applied to the 2024–2025 Test partition. No test grid search was performed.
2. **Heavy Rain Classifier (Task B, $\ge 64.5\text{ mm}$):**
   - The threshold grid was evaluated on the **2023 Validation set** to maximize Critical Success Index (CSI).
   - Optimal thresholds identified on Validation:
     - LightGBM Focal Loss: $p^* = 0.53$ (Validation $\text{CSI} = 0.4327$)
     - LightGBM Class-Weighted: $p^* = 0.96$ (Validation $\text{CSI} = 0.4291$)
     - CatBoost Class-Weighted: $p^* = 0.98$ (Validation $\text{CSI} = 0.4258$)
     - LightGBM Unweighted: $p^* = 0.26$ (Validation $\text{CSI} = 0.4277$)
   - These exact thresholds were applied to the Test partition to produce the audited confusion matrices.
3. **Two-Stage Hurdle Architecture (Task D):**
   - The Stage 1 binary occurrence gate used the frozen threshold $p^* = 0.44$ calibrated in Task A.

### Audit of Early Stopping:
- All gradient-boosted models (LightGBM, XGBoost, CatBoost) were trained with `stopping_rounds = 50`.
- In every training script, the evaluation set passed to the early stopping callback was strictly `eval_set = [(X_va, y_va)]`.
- The test set was never passed as an evaluation set during training.

---

## 5. Transparent Disclosure of Test Partition Access and Protocol Governance

In compliance with rigorous scientific reproducibility standards, we disclose the exact timeline and nature of test set interactions:

1. **Stage 4.0 (Integrity Check):** Test partition was checked for missing values, shape ($184,212 \times 78$), date bounds (2024-01-01 to 2025-12-31), and station counts. No models were trained.
2. **Stage 4.1 (Model Development):** After models, scalers, and thresholds were frozen using Training and Validation data, forward evaluation was executed on the Test set to generate `baseline_results.csv`, `regression_model_results.csv`, `rain_occurrence_results.csv`, `heavy_rain_results.csv`, and `hurdle_model_results.csv`. **Crucially, no hyperparameter was modified, no feature was pruned, and no model was re-trained based on the test results.**
3. **Stage 4.2 (Independent Audit):**
   - In Task 1, test predictions were independently re-evaluated to verify mathematical accuracy.
   - In Task 4, 5,000 paired block bootstrap resamples were computed on test predictions to establish confidence intervals.
   - In Task 5 (Feature Ablation), all 6 ablation models were trained strictly on Training and evaluated on Validation. The frozen ablation models were then evaluated on the Test set strictly to document the out-of-sample impact of feature removal.

### Governance Statement:
The 2024–2025 Test partition was never used as a feedback mechanism for iterative design, model tuning, or feature selection. The experimental protocol strictly satisfies all conditions for an authentic out-of-sample scientific evaluation.

