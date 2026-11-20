import re
from pathlib import Path

source_md = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage9_Final_Revised_Manuscript\Karnataka_Rainfall_Research_Paper_Revised.md")
target_md = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage9_1_Final_Validated_Manuscript\Karnataka_Rainfall_Research_Paper_Final.md")

with open(source_md, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Title / Header
# Ensure clean blinded header without any local machine paths
text = re.sub(r"\[Author Names Blinded.*?\]", "[Author Names Blinded for Peer Review / To Be Inserted by Author]", text)
text = re.sub(r"\[Department / Institutional Affiliation.*?\]", "[Department / Institutional Affiliation To Be Inserted by Author]", text)

# 2. Terminology replacements
text = text.replace("Station DOY Climatology", "Grid-cell DOY Climatology")
text = text.replace("Station DOY climatology", "Grid-cell DOY climatology")
text = text.replace("station DOY climatology", "grid-cell DOY climatology")
text = text.replace("Station IDs", "IMD grid-cell IDs")
text = text.replace("Station ID", "Grid-Cell ID")
text = text.replace("station ID", "grid-cell ID")
text = text.replace("Low-level jet moisture transport", "near-surface 10-meter wind circulation dynamics")
text = text.replace("low-level jet moisture transport", "near-surface 10-meter wind circulation dynamics")
text = text.replace("Numerical model precipitation forecast memory", "Antecedent reanalysis-derived precipitation and land-surface flux memory")

# 3. Citation closures in text
# Cite [4] in observational ground truth
text = text.replace(
    "gridded daily rainfall dataset serves as the observational ground truth [3].",
    "gridded daily rainfall dataset serves as the observational ground truth [3], [4]."
)

# Cite [14], [16], [20] in Model Formulations
text = text.replace(
    "- **LightGBM Tweedie Regressor ($p=1.5$):**",
    "- **LightGBM Tweedie Regressor ($p=1.5$) [14]:**"
)
text = text.replace(
    "- **XGBoost Tweedie Regressor:**",
    "- **XGBoost Tweedie Regressor [16]:**"
)
text = text.replace(
    "- **LightGBM Log-Duan Smearing:**",
    "- **LightGBM Log-Duan Smearing [20]:**"
)

# Cite [27], [28], [29] in Feature Attribution
text = text.replace(
    "Permutation feature importance was calculated",
    "Permutation feature importance [27], [28] was calculated"
)
text = text.replace(
    "SHapley Additive exPlanations (SHAP)",
    "SHapley Additive exPlanations (SHAP) [29]"
)

# 4. Insert Tier 3 (Linear Baselines: Ridge and ElasticNet) in Section 3.3
tier3_text = """#### Tier 3: Regularized Linear Baselines
- **Ridge Regression (L2):** Fits a linear model with $L_2$ Tikhonov regularization ($\alpha = 100.0$) on standardized features ($z$-score transformed strictly on 2019–2022 training data), establishing a regularized linear benchmark.
- **ElasticNet Regression (L1/L2):** Combines $L_1$ and $L_2$ penalties ($\alpha = 0.1$, $l_1\text{-ratio} = 0.5$) to evaluate sparse linear feature selection against non-linear tree ensembles.

#### Tier 4: Machine Learning Architectures"""

text = text.replace("#### Tier 4: Machine Learning Architectures", tier3_text)

# 5. Correct Heavy Rain Detection POD and False Alarm in Abstract, Section 4.2, and Section 5
# Replace erroneous 57.56% with authoritative 46.96%
text = re.sub(
    r"attains POD = 0\.5756, CSI = 0\.3072",
    "attains POD = 0.4696, CSI = 0.3512, Precision = 0.5822, FAR = 0.4178 (decision threshold p* = 0.53), missing 53.04% of observed heavy events",
    text
)
text = re.sub(
    r"POD of 57\.56%",
    "POD of 46.96% (missing 53.04% of observed events)",
    text
)
text = re.sub(
    r"POD = 57\.56%",
    "POD = 46.96% (missing 53.04% of observed heavy events)",
    text
)
text = text.replace("calibrated threshold", "decision threshold")
text = text.replace("calibrated decision threshold", "decision threshold")

# 6. Correct Permutation Importance for sro_lag_1d (+2.3545 mm)
text = re.sub(
    r"\+1\.42\s*mm\s*RMSE\s*increase",
    "+2.3545 mm RMSE increase (+42.16% degradation)",
    text
)
text = re.sub(
    r"\+1\.42\s*mm",
    "+2.3545 mm",
    text
)

# 7. Correct Bootstrap Replicate Count to 2,000 resamples
text = text.replace("1,000 block bootstrap", "2,000 block bootstrap")
text = text.replace("1,000 resamples", "2,000 resamples")
text = text.replace("1,000 bootstrap", "2,000 bootstrap")
text = text.replace("N = 1,000", "N = 2,000")
text = text.replace("p < 0.001", "p < 0.0005")

# 8. Correct sequential figure ordering in text and callouts:
# Swap Figure 8 and Figure 9 callouts and captions so that Figure 8 is Temporal Robustness (Section 4.5)
# and Figure 9 is Extreme Rainfall Intensity Limitations (Section 4.6).
# First replace existing Figure 9 in Section 4.5 to Temp_Fig_8
text = text.replace("Figure 9: Panel A shows annual RMSE consistency bar comparison", "Figure 8: Panel A shows annual RMSE consistency bar comparison")
text = text.replace("Figure 9 examines temporal generalisation", "Figure 8 examines temporal generalisation")
text = text.replace("[Figure 9: Panel A", "[Figure 8: Panel A")
text = text.replace("Figure 9 and Table 6", "Figure 8 and Table 6")
text = text.replace("Figure 9", "Figure 8") # in section 4.5

# Then replace existing Figure 8 in Section 4.6 to Figure 9
text = text.replace("Table 8 and Figure 8 provide an unvarnished audit", "Table 8 and Figure 9 provide an unvarnished audit")
text = text.replace("[Figure 8: Mean prediction bias and percentage bias across 7 IMD categories", "[Figure 9: Mean prediction bias and percentage bias across 7 IMD categories")
text = text.replace("Figure 8: Mean prediction bias and percentage bias across 7 IMD categories", "Figure 9: Mean prediction bias and percentage bias across 7 IMD categories")
text = text.replace("Figure 8", "Figure 9")

# 9. Table 3 Update in Markdown text
table3_md = """### Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; $N = 184,212$)
| Model Hierarchy Tier | Model Name | Loss Objective | RMSE (mm) | MAE (mm) | Mean Bias (mm) | PBias (%) | Pearson $r$ | NSE | KGE-2009 | KGE-2012 | Heavy CSI ($\ge 64.5\text{ mm}$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 0 (Naive Baselines)** | Rainfall Persistence ($y_{t-1}$) | Zero Learning | 10.3582 | 3.7508 | 0.0000 | 0.00 | 0.5565 | 0.1129 | 0.5565 | 0.5565 | 0.2361 |
| **Tier 0 (Naive Baselines)** | Grid-cell DOY Climatology | Zero Learning | 10.1155 | 4.3078 | +0.2296 | +6.54 | 0.4313 | 0.1540 | 0.3068 | 0.2851 | 0.0144 |
| **Tier 1 (Raw Reanalysis)** | Antecedent ERA5-Land Total Precip | Zero Learning | 10.4550 | 4.1416 | +0.5560 | +15.85 | 0.4911 | 0.0963 | 0.4510 | 0.4111 | 0.1233 |
| **Tier 2 (Statistical Downscaling)**| Empirical Quantile Mapping (EQM) | Non-Parametric Knots | 11.1970 | 4.2117 | +0.3867 | +11.02 | 0.4941 | -0.0366 | 0.4818 | 0.4763 | 0.1969 |
| **Tier 3 (Linear Baselines)** | Ridge Regression ($L_2$) | Squared Error + $L_2$ | 7.9750 | 3.3905 | +0.3308 | +9.43 | 0.6894 | 0.4742 | 0.5613 | 0.5184 | 0.2299 |
| **Tier 3 (Linear Baselines)** | ElasticNet Regression ($L_1/L_2$) | Squared Error + $L_1/L_2$ | 8.0156 | 3.3999 | +0.3034 | +8.65 | 0.6853 | 0.4688 | 0.5496 | 0.5101 | 0.2211 |
| **Tier 4 (Machine Learning)** | LightGBM RMSE (L2) | Squared Error | 7.4982 | 3.0806 | +0.2698 | +7.69 | 0.7320 | 0.5352 | 0.6190 | 0.5812 | 0.3016 |
| **Tier 4 (Machine Learning)** | LightGBM Log-Ordinary | Log-MSE | 7.8497 | 2.6150 | -1.3274 | -37.83 | 0.7260 | 0.4906 | 0.3703 | 0.5276 | 0.2271 |
| **Tier 4 (Machine Learning)** | LightGBM Log-Duan Smearing | Log-MSE + Smear | 7.5893 | 2.9488 | -0.2105 | -6.00 | 0.7260 | 0.5238 | 0.6439 | 0.6722 | 0.3509 |
| **Tier 4 (Machine Learning)** | **LightGBM Tweedie ($p=1.5$)** | **Compound Poisson-Gamma** | **7.4427** | **2.8993** | **-0.0179** | **-0.51** | **0.7363** | **0.5420** | **0.6218** | **0.6245** | **0.2868** |
| **Tier 4 (Machine Learning)** | **CatBoost Tweedie ($p=1.5$)** | **Compound Poisson-Gamma** | **7.4335** | **2.9186** | **+0.0388** | **+1.11** | **0.7370** | **0.5431** | **0.6318** | **0.6261** | **0.3033** |
| **Tier 4 (Machine Learning)** | **Hurdle Expected Value** | **Classifier + Conditional Reg.** | **7.4241** | **2.9300** | **+0.0342** | **+0.97** | **0.7380** | **0.5443** | **0.6165** | **0.6114** | **0.3072** |"""

text = re.sub(r"### Table 3:.*?### 4\.2", lambda m: table3_md + "\n\n### 4.2", text, flags=re.DOTALL)

# 10. Table 2 Update in Markdown text
table2_md = """### Table 2: Input Feature Groups, Predictor Schema, and Pipeline Gating
| Feature Group ID | Physical / Operational Category | Column Count | Gated Primary Predictors ($X$) | Physical Examples & Latency Window |
| :--- | :--- | :---: | :---: | :--- |
| **Group 1** | Target Labels & Ground Truth | 4 | 0 (Targets Only) | `target_rainfall_mm`, `target_is_rain`, `target_is_heavy_rain`, `target_rain_ge_35_5mm` |
| **Group 2** | Identifiers & Metadata | 6 | 0 (Audit & Indexing) | `date`, `station_id`, `district`, `zone_name`, `target_period`, `prediction_issue_time` |
| **Group 3** | Antecedent IMD Gauge Memory | 13 | 13 | `target_rain_lag_1d` to `_14d`, `target_rain_roll_3d/7d/14d`, `api_14d` (lagged $\ge 1\text{ day}$) |
| **Group 4** | ERA5-Land Precip & Hydrology | 14 | 14 | `era5_tp_lag_1d` to `_3d`, `ro_lag_1d`, `sro_lag_1d`, `src_lag_1d`, evaporation depths |
| **Group 5** | Atmospheric Thermodynamics & Wind | 12 | 12 | `t2m_mean/min/max_lag_1d`, `d2m_lag_1d`, `dewpoint_depression_lag_1d`, `sp_lag_1d`, `wind_speed_lag_1d` |
| **Group 6** | Soil Moisture & Energy Radiation Fluxes | 9 | 9 | `swvl1_lag_1d/3d/7d`, `swvl2_lag_1d`, `swvl3_lag_1d`, `ssrd_lag_1d`, `strd_lag_1d`, `slhf_lag_1d`, `sshf_lag_1d` |
| **Group 7** | Static Topography & Spatial Proximity | 6 | 6 | `elevation_dem_m`, `slope_dem_deg`, `latitude`, `longitude`, `dist_to_coast_km`, `dist_to_ghats_crest_km` |
| **Group 8** | Cyclic Temporal Ephemeris | 7 | 7 | `month`, `day_of_year`, `sin/cos_day_of_year`, `sin/cos_month`, `is_monsoon` |
| **Group 9** | Primary Regional Categorical Predictors | 2 | 2 | `subdivision` (4 regions), `zone_id` (10 zones) [target-encoded to `subdivision_enc`, `zone_id_enc`] |
| **Auxiliary**| Coastal Ocean-Boundary Fallback Flag | 1 | 0 (Auxiliary Quality Flag) | `is_coastal_fallback` (binary flag for Grid-Cell IDs 52, 68, 125 touching ocean cells) |
| **Excluded** | Pressure Proxies & Redundant Strings | 4 | 0 (Ablation / Excluded) | `elevation_pressure_proxy_m`, `slope_pressure_proxy_deg`, `interpolation_method`, `season` |
| **Total** | **Master Parquet Dataset** | **78** | **63 Eligible Features** | **Strict 24-hour antecedent causality enforced ($T_{\\text{issue}} = \\text{03:00 UTC day } t-1$)** |"""

text = re.sub(r"### Table 2:.*?### 3\.3", lambda m: table2_md + "\n\n### 3.3", text, flags=re.DOTALL)

with open(target_md, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved finalized manuscript text to {target_md} ({len(text)} characters).")
