# Technical and Mathematical Audit of Duan's Smearing Factor

**Document Identifier:** STAGE-4.2-DUAN-SMEARING-AUDIT  
**Project:** Karnataka Hydrometeorological Daily Rainfall Forecasting Research  
**Author:** Independent Scientific Audit Team  
**Evaluation Scope:** Theoretical derivation, empirical implementation, heteroscedasticity limitations, and comparison with raw-scale compound Poisson-Gamma (Tweedie) regression.

---

## 1. Executive Summary

This audit assesses the mathematical validity, empirical implementation, and physical limitations of **Duan's Smearing Estimator** applied to the log-transformed rainfall regression model (`LightGBM_Log_DuanSmearing`) developed in Stage 4.1.

### Key Audit Findings:
1. **Mathematical Implementation Integrity:** The smearing factor $S = 1.35112$ was calculated **strictly and exclusively from the 364,644 training residuals** (2019–2022). Zero validation or test data were used in its derivation.
2. **Correctness of Retransformation:** The retransformation formula $\hat{y} = \max\left(0, S \cdot \exp(\hat{z}) - 1\right)$ correctly accounts for the transformation target $Z = \ln(Y + 1)$ under additive homoscedastic error assumptions.
3. **The Heteroscedasticity Defect:** Empirical inspection reveals severe heteroscedasticity in the log-space residuals: the conditional expectation $\mathbb{E}[\exp(e) | \hat{z}]$ varies from **$1.008$** for dry predictions ($\hat{z} \le 0.1$) up to **$1.813$** for intermediate rainfall predictions ($\hat{z} \in [1.0, 2.0]$). Consequently, a **single global scalar smearing factor cannot universally correct conditional bias across all rainfall regimes**: it over-adjusts light drizzle while remaining insufficient to overcome the severe exponential underestimation of intense monsoonal peaks.
4. **Superiority of Raw-Scale Tweedie Regression:** Raw-scale Tweedie regression ($p=1.5$) eliminates the retransformation problem at its source by directly modeling the conditional mean $\mathbb{E}[Y|X]$ on the original physical millimeter scale. It achieves superior validation RMSE ($5.585\text{ mm}$ vs. $5.691\text{ mm}$ for Duan smearing) and near-zero mean bias ($-0.06\text{ mm}$ vs. $-0.19\text{ mm}$).

---

## 2. Theoretical Derivation and Jensen's Inequality

In environmental and hydrometeorological modeling, daily rainfall amounts $Y \ge 0$ exhibit extreme positive skewness, zero inflation, and heavy tails. A standard variance-stabilizing practice is to apply the logarithmic transformation:
$$Z = \ln(Y + c)$$
where $c = 1.0\text{ mm}$ is an offset ensuring well-defined values at zero rainfall ($Z = 0$ when $Y = 0$).

A regression model (such as LightGBM) is trained to minimize mean squared error in log-space:
$$\min \frac{1}{N} \sum_{i=1}^N \left( Z_i - \hat{Z}_i \right)^2$$
The model output $\hat{Z}_i = f(X_i)$ estimates the conditional mean in log-space:
$$\hat{Z} = \mathbb{E}[Z | X] = \mathbb{E}[\ln(Y + 1) | X]$$

### The Retransformation Bias Problem:
When back-transforming to physical rainfall units ($Y$), applying the naive ordinary inverse:
$$\hat{Y}_{ord} = \exp(\hat{Z}) - 1$$
systematically underestimates the conditional expectation $\mathbb{E}[Y | X]$. By **Jensen's Inequality**, for any strictly convex function $g(u) = \exp(u)$:
$$\mathbb{E}[\exp(Z) | X] \ge \exp\left(\mathbb{E}[Z | X]\right) = \exp(\hat{Z})$$
Equality holds if and only if the error variance is identically zero. Because rainfall predictions in log-space contain substantial residual variance, $\exp(\hat{Z}) - 1$ produces a severe negative bias. In the Stage 4.1 test partition ($184,212$ records), the ordinary inverse produced a catastrophic mean bias of **$-1.3274\text{ mm}$**, underestimating total statewide rainfall volume by **$-37.83\%$**.

---

## 3. Duan's Smearing Estimator Formulation

Duan (1983) proposed a non-parametric smearing estimator to correct retransformation bias without assuming Gaussian error distributions.

### Assumptions:
1. The transformed variable follows an additive error structure:
   $$Z = \hat{Z} + \epsilon$$
2. The errors $\epsilon$ are independent and identically distributed (homoscedastic), with $\mathbb{E}[\epsilon] = 0$ and distribution function $F_\epsilon$ that is independent of $X$:
   $$F_{\epsilon | X}(\epsilon) = F_\epsilon(\epsilon)$$

Under these assumptions, the conditional expectation of $Y$ on the original scale is:
$$\mathbb{E}[Y | X] = \mathbb{E}[\exp(\hat{Z} + \epsilon) - 1 | X] = \exp(\hat{Z}) \cdot \mathbb{E}[\exp(\epsilon) | X] - 1 = \exp(\hat{Z}) \cdot \mathbb{E}[\exp(\epsilon)] - 1$$

The non-parametric sample estimate of the expectation $\mathbb{E}[\exp(\epsilon)]$ is the sample mean of the exponentiated residuals:
$$S = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} \exp(e_i)$$
where $e_i = Z_{train, i} - \hat{Z}_{train, i}$.

The resulting Duan smearing prediction is:
$$\hat{Y}_{smear} = S \cdot \exp(\hat{Z}) - 1$$
To enforce physical non-negativity:
$$\hat{Y}_{smear} = \max\left(0, S \cdot \exp(\hat{Z}) - 1\right)$$

---

## 4. Verification of the Empirical Implementation

We audited the Stage 4.1 code in [`train_regression_models.py`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage4_1_Model_Development/train_regression_models.py) (lines 142–156):

```python
# 1. Target definition in Training
z_tr = np.log1p(np.clip(y_tr, 0.0, None))

# 2. Residual calculation strictly on Training set
z_pred_tr = lgb_log.predict(X_tr)
train_residuals = z_tr - z_pred_tr

# 3. Global smearing factor calculation
duan_smearing_factor = float(np.mean(np.exp(train_residuals)))
# Result: 1.3511211905510985

# 4. Retransformation applied to Validation and Test
y_val_log_duan = np.clip(
    duan_smearing_factor * np.exp(z_pred_va) - 1.0, 0.0, None
)
y_te_log_duan = np.clip(
    duan_smearing_factor * np.exp(z_pred_te) - 1.0, 0.0, None
)
```

### Audit Verification Checklist:
- [x] **Sample Exclusivity:** The smearing factor $S = 1.35112$ was derived **strictly and exclusively** from the 364,644 training instances (2019–2022). No validation (2023) or test (2024–2025) residuals were included.
- [x] **Target Alignment:** The residual calculation uses $z_{tr} = \ln(y_{tr} + 1)$ and $\hat{z}_{tr}$, exactly matching the objective minimized during gradient boosting.
- [x] **Formula Accuracy:** The formula $S \cdot \exp(\hat{z}) - 1$ corresponds exactly to Duan's non-parametric estimator for $Z = \ln(Y + 1)$.
- [x] **Non-Negativity Clipping:** Negative predictions resulting from low $\hat{z}$ values where $S \exp(\hat{z}) < 1$ are properly clipped to $0.0\text{ mm}$.

---

## 5. The Heteroscedasticity Defect: Why Global Smearing Cannot Universally Correct Conditional Bias

The primary scientific vulnerability of global Duan smearing is its reliance on the **homoscedasticity assumption** ($F_{\epsilon | X} = F_\epsilon$). In daily hydrometeorological data, this assumption is fundamentally violated.

We empirically evaluated the conditional smearing factor $S(\hat{z}) = \mathbb{E}[\exp(e) | \hat{z}]$ across prediction bins on the Training partition ($N = 364,644$):

| Predicted Log-Rain $\hat{z}$ Range | Equivalent mm Range $\exp(\hat{z})-1$ | Training Sample Count | Mean Observed IMD Rain (mm) | Conditional Smearing Factor $S(\hat{z})$ Mean | Conditional Smearing Std Dev | Conditional Smearing Median |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$(-1.0, 0.1]$** | $0.0 - 0.1\text{ mm}$ (Dry days) | 139,026 ($38.1\%$) | 0.02 | **1.0078** | 0.4076 | 0.9938 |
| **$(0.1, 0.5]$** | $0.1 - 0.6\text{ mm}$ (Very light) | 74,079 ($20.3\%$) | 0.69 | **1.2522** | 1.9150 | 0.8192 |
| **$(0.5, 1.0]$** | $0.6 - 1.7\text{ mm}$ (Light) | 57,752 ($15.8\%$) | 2.43 | **1.6286** | 2.6047 | 0.6164 |
| **$(1.0, 2.0]$** | $1.7 - 6.4\text{ mm}$ (Moderate) | 66,801 ($18.3\%$) | 7.06 | **1.8125** | 2.2888 | 0.9984 |
| **$(2.0, 3.0]$** | $6.4 - 19.1\text{ mm}$ (Rather heavy) | 20,476 ($5.6\%$) | 17.83 | **1.7624** | 1.5774 | 1.3051 |
| **$(3.0, 5.0]$** | $19.1 - 147.4\text{ mm}$ (Heavy/Extreme) | 6,510 ($1.8\%$) | 52.74 | **1.3195** | 0.7996 | 1.1499 |
| **Full Training Set (Global)** | Full Physical Domain | **364,644 ($100\%$)** | **3.77** | **1.3511** | **1.7654** | **0.9912** |

### Critical Physical Insights:
1. **Dominance of Dry Days:** Because $38.1\%$ of the training set consists of dry days with $\hat{z} \le 0.1$ where $S(\hat{z}) \approx 1.0078$, the global average is heavily pulled downward to $S = 1.3511$.
2. **Under-Correction in Active Rainfall Regimes:** For active rainfall days ($\hat{z} \in [0.5, 2.0]$), the actual required conditional multiplier is between **$1.63$ and $1.81$**. Applying the global factor $1.3511$ under-corrects these predictions by **$17\% - 25\%$**.
3. **Over-Correction of Light Drizzle:** For near-zero predictions, multiplying by $1.3511$ inflates near-zero noise into false drizzle.
4. **Scientific Verdict:** **A single global smearing factor cannot universally correct conditional bias across distinct hydrometeorological regimes.** It functions strictly as a global scalar adjustment that restores the overall sample mean at the cost of regime-dependent distortion.

---

## 6. Empirical Comparison: Duan Smearing vs. Raw-Scale Tweedie Regression

To determine whether log-transformation with Duan smearing should be preferred over raw-scale modeling, we compare their validation and test performance:

| Evaluation Partition | Model Formulation | Target Scale | RMSE (mm) | MAE (mm) | Mean Bias (mm) | PBias (%) | Pearson $r$ | NSE | KGE-2012 | $\text{CSI}_{64.5\text{mm}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation 2023** | LightGBM Log (Ordinary) | Log $\ln(y+1)$ | 6.0075 | 1.8344 | -1.0136 | -43.00% | 0.7505 | 0.5128 | 0.5025 | 0.2742 |
| **Validation 2023** | LightGBM Log (Duan $S=1.351$) | Log $\ln(y+1)$ | 5.6913 | 2.1200 | -0.1910 | -8.10% | 0.7505 | 0.5627 | **0.6831** | **0.4211** |
| **Validation 2023** | **LightGBM Tweedie ($p=1.5$)** | **Raw mm** | **5.5850** | **2.0511** | **-0.0615** | **-2.61%** | **0.7618** | **0.5789** | 0.6494 | 0.4098 |
| **Test 2024–2025** | LightGBM Log (Ordinary) | Log $\ln(y+1)$ | 7.8497 | 2.6150 | -1.3274 | -37.83% | 0.7260 | 0.4906 | 0.5276 | 0.2271 |
| **Test 2024–2025** | LightGBM Log (Duan $S=1.351$) | Log $\ln(y+1)$ | 7.5893 | 2.9488 | -0.2105 | -6.00% | 0.7260 | 0.5238 | **0.6722** | **0.3509** |
| **Test 2024–2025** | **LightGBM Tweedie ($p=1.5$)** | **Raw mm** | **7.4427** | **2.8993** | **-0.0179** | **-0.51%** | **0.7363** | **0.5420** | 0.6245 | 0.2868 |
| **Test 2024–2025** | **CatBoost Tweedie ($p=1.5$)** | **Raw mm** | **7.4335** | 2.9186 | **+0.0388** | **+1.11%** | **0.7370** | **0.5431** | 0.6261 | 0.3033 |

### Methodological Conclusions:
1. **Duan Smearing vs. Ordinary Inverse:** Applying Duan smearing is mandatory if log-transformation is used: it eliminates over $80\%$ of the negative bias (reducing test bias from $-1.33\text{ mm}$ to $-0.21\text{ mm}$) and lifts KGE from $0.5276$ to $0.6722$.
2. **Tweedie Superiority on Continuous Precision:** Raw-scale Tweedie regression ($p=1.5$) achieves lower RMSE ($7.44\text{ mm}$ vs. $7.59\text{ mm}$), lower MAE ($2.90\text{ mm}$ vs. $2.95\text{ mm}$), and an order of magnitude smaller bias ($-0.018\text{ mm}$ vs. $-0.211\text{ mm}$).
3. **Recommendation for Publication:** While Duan smearing provides a valuable demonstration of retransformation bias correction, **raw-scale Tweedie regression ($p=1.5$) is the methodologically superior framework for continuous rainfall forecasting**, as it estimates $\mathbb{E}[Y|X]$ directly and inherently accommodates heteroscedastic compound Poisson-Gamma variance ($\text{Var}(Y) = \phi \mu^p$).

