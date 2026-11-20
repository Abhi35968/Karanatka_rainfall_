# Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)

**Authors:** [Author Names Blinded for Peer Review / To Be Inserted by Author]  
**Affiliation:** [Department / Institutional Affiliation To Be Inserted by Author]  
**Correspondence:** [Corresponding Author Email Address To Be Inserted by Author]  
**Target Layout:** Standard Two-Column Scientific Journal Article  
**Status:** Revised Research Manuscript (Pre-Submission Version)  

---

### Abstract
Accurate retrospective daily rainfall estimation over complex terrain is essential for catchment water balance analysis, agricultural planning, and flood hazard assessment. Over peninsular India, the sharp transition between the steep orographic barrier of the Western Ghats and the semi-arid interior Deccan plateau creates extreme spatial rainfall gradients that challenge global reanalysis and classical univariate downscaling. In this study, we evaluate a structured machine learning framework for retrospective daily rainfall reconstruction and heavy precipitation alerting across the 252 India Meteorological Department (IMD) 0.25° gridded land cells covering Karnataka State, India, over the 7-year period 2019–2025. Leveraging antecedent 24-hour land-surface fluxes and atmospheric thermodynamic variables from ECMWF ERA5-Land reanalysis, local antecedent IMD observations, and Copernicus GLO-90 DEM topography, we evaluate gradient-boosted decision trees using compound Poisson-Gamma (Tweedie, $p=1.5$) loss functions, two-stage hurdle models, and custom focal-loss classifiers. To prevent temporal information leakage across partitions, models are trained on 2019–2022 ($N = 364,644$), hyperparameter-tuned and threshold-calibrated strictly on the anomalous 2023 regional drought year ($N = 91,980$), and evaluated out-of-sample on a frozen 2-year forward test partition (2024–2025, $N = 184,212$). On continuous test evaluation, the three gradient-boosted architectures exhibit closely matched performance with distinct multi-criteria trade-offs: the two-stage Hurdle Expected Value model achieves the lowest Root Mean Squared Error ($\text{RMSE} = 7.4241\text{ mm}$), highest Pearson correlation ($r = 0.7380$), and highest Nash-Sutcliffe Efficiency ($\text{NSE} = 0.5443$); CatBoost Tweedie attains the highest Modified Kling-Gupta Efficiency ($\text{KGE-2012} = 0.6261$) with $\text{RMSE} = 7.4335\text{ mm}$; while LightGBM Tweedie achieves the lowest Mean Absolute Error ($\text{MAE} = 2.8993\text{ mm}$) and lowest mean bias ($-0.0179\text{ mm}$, or $-0.51\%$). All three architectures substantially outperform autoregressive Persistence ($\text{RMSE} = 10.3582\text{ mm}$), raw ERA5-Land reanalysis ($\text{RMSE} = 10.4550\text{ mm}$), and Empirical Quantile Mapping ($\text{EQM}$, $\text{RMSE} = 11.1970\text{ mm}$, $\text{NSE} = -0.0366$). Paired stationary block bootstrapping ($L = 7\text{d}, 14\text{d}$; $B = 2,000$ replicates) demonstrates that the $\approx 2.93\text{ mm}$ RMSE reduction over Persistence is statistically significant at an empirical test resolution of $p < 0.0005$. For heavy-rain alerting at the official IMD threshold ($\ge 64.5\text{ mm/day}$), a custom focal-loss classifier ($\gamma = 2.0$, $\alpha = 0.75$, validation-selected decision threshold $p^* = 0.53$) suppresses false alarms by $40.86\%$ ($427$ vs. $722$) relative to standard class weighting, achieving a Critical Success Index ($\text{CSI}$) of $0.3512$, Precision of $58.22\%$, and False Alarm Ratio ($\text{FAR}$) of $41.78\%$. Tree split gain analysis reveals that antecedent reanalysis runoff variables account for $54.52\%$ of total tree gain in the heavy-rain classifier, while feature ablation confirms that ERA5-Land predictors provide measurable incremental predictive value ($\Delta \text{RMSE} = +1.0084\text{ mm}$ upon ablation). However, a critical structural limitation is documented: continuous conditional expectation models underestimate $100\%$ of test observations exceeding $115.6\text{ mm}$ ($200$ events), plateauing near $70–90\text{ mm}$. We conclude that retrospective reanalysis downscaling provides strong predictive utility in orographically locked monsoonal regimes when land-surface runoff fluxes are incorporated, but operational hydrometeorological workflows must deploy paired asymmetric classifiers to mitigate conditional mean quantile shrinkage for extreme precipitation alerting.

**Keywords:** Hydrometeorological hindcasting, ERA5-Land reanalysis, IMD gridded rainfall, Tweedie regression, Hurdle model, Focal loss, Orographic precipitation, Western Ghats.

---

## 1. Introduction

Accurate estimation of daily precipitation at high spatial resolution is foundational to catchment hydrology, agricultural water allocation, reservoir dispatch, and extreme weather hazard mitigation [1], [2]. Over the Indian subcontinent, the spatial and temporal distribution of rainfall is heavily modulated by the Indian Summer Monsoon (June–September), which delivers more than $75\%$ of the annual precipitation budget [3], [6]. Karnataka State, situated in southwestern peninsular India, exemplifies the intense hydrometeorological diversity of the monsoon: it transitions abruptly from a hyper-humid coastal belt along the Arabian Sea to the rugged Western Ghats mountain range (elevations $1,000–1,800\text{ m}$), which triggers massive orographic precipitation, and further into the semi-arid leeward rain-shadow of the Deccan plateau [6], [7].

Despite the proliferation of global atmospheric reanalysis products—most notably the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 [1] and its high-resolution land component ERA5-Land [2]—reanalysis precipitation fields cannot be directly ingested into local hydrometeorological models without substantial post-processing. Hydrostatic atmospheric models operating at $0.1^\circ$ to $0.25^\circ$ grid spacings smooth steep topographic ridges, misplace the exact crest of orographic precipitation, and parameterize deep moist convection, frequently leading to systemic wet biases in light drizzle, diurnal phase errors, and severe underestimation of localized torrential downpours [7]–[9]. 

Classical statistical downscaling methods, such as Empirical Quantile Mapping (EQM) [10], adjust the marginal distribution of reanalysis precipitation against observed gauge records. However, EQM operates as a univariate transformation that assumes temporal stationarity, disrupts multi-variable physical consistency, and often degrades spatial field coherence during extreme convective episodes [11]. In recent years, machine learning (ML) architectures, particularly gradient-boosted decision trees (GBDT) and deep neural networks, have demonstrated substantial promise in learning complex, non-linear relationships between multi-level atmospheric predictors, surface energy fluxes, and observed precipitation [12], [13].

Nevertheless, daily precipitation modeling presents three intrinsic mathematical challenges that standard off-the-shelf machine learning formulations fail to resolve:
1. **Zero-Inflation and Semi-Continuous Support:** Daily rainfall observations exhibit a severe point mass at zero (representing dry days) coupled with a continuous, heavy-tailed positive distribution on wet days. Minimizing conventional Mean Squared Error (L2 loss) assumes Gaussian residuals, resulting in artificial background drizzle on dry days and severe underprediction of high-intensity events [17], [18].
2. **Extreme Imbalance in Hazard Detection:** Severe hydrometeorological impacts are concentrated in heavy rainfall events ($\ge 64.5\text{ mm/day}$ by IMD definition), which represent less than $1\%$ of daily observations. Standard cross-entropy classification suffers from overwhelming majority-class dominance, yielding either high miss rates or prohibitive false alarm ratios [21].
3. **Temporal Autocorrelation and Evaluation Integrity:** Atmospheric and hydrometeorological time series exhibit strong synoptic and intraseasonal persistence. Conventional random $k$-fold cross-validation leaks temporal information across training and testing partitions, leading to severely inflated skill estimates [25], [26].

In this study, we present a scientifically audited, leakage-free machine learning framework for retrospective daily rainfall reconstruction over Karnataka State from 2019 to 2025. By integrating antecedent ERA5-Land reanalysis fluxes [2], IMD 0.25° gridded daily rainfall observations [3], and Copernicus GLO-90 digital elevation data [5], we address three fundamental research questions:
- *First*, how do compound Poisson-Gamma (Tweedie) objective functions and two-stage hurdle models compare against naive baselines (Persistence, Climatology), raw ERA5-Land reanalysis, and classical EQM in continuous rainfall estimation?
- *Second*, can custom asymmetric focal-loss classifiers overcome extreme class imbalance to alert for heavy rainfall events ($\ge 64.5\text{ mm}$) without inducing unacceptable false alarm rates?
- *Third*, what is the empirical incremental predictive value of ERA5-Land surface and atmospheric variables, and which physical domains govern model skill across distinct orographic and convective regimes?

### Explicit Scope and Scientific Framing
To maintain strict scientific integrity, this work is explicitly framed as **retrospective reanalysis hindcasting and diagnostic downscaling**. It utilizes antecedent 24-hour accumulated reanalysis fields and previous-day IMD observations available at 03:00 UTC to estimate the daily accumulation cycle ending at 03:00 UTC (08:30 IST). It does **not** simulate an operational real-time Numerical Weather Prediction (NWP) forecasting harness. In operational forecasting, forward atmospheric model integrations would be required, and real-time gauge observations are subject to reporting latencies. Furthermore, the spatial units of analysis are the **252 IMD 0.25° gridded land cells** covering Karnataka [3]; they represent area-averaged daily accumulations rather than point observations from surface weather stations.

---

## 2. Study Area and Datasets

### 2.1 Geographical Domain & Meteorological Subdivisions
The study domain encompasses Karnataka State in southern India, bounded between latitudes $11.75^\circ\text{N}$ and $18.50^\circ\text{N}$ and longitudes $74.25^\circ\text{E}$ and $78.50^\circ\text{E}$, covering an area of approximately $191,791\text{ km}^2$. The state is officially divided by the IMD into four distinct meteorological subdivisions, encompassing 10 agroclimatic zones and 30 administrative districts:
1. **Coastal Karnataka (13 IMD cells):** A narrow lowland strip bordering the Arabian Sea that receives intense monsoonal precipitation ($3,000–4,500\text{ mm}$ annually) driven by low-level moisture advection.
2. **Malnad (35 IMD cells):** The windward slopes and crest of the Western Ghats mountain range (elevations up to $1,894\text{ m}$ at Kudremukh), characterized by steep orographic uplift and torrential monsoonal downpours.
3. **North Interior Karnataka (NIK; 118 IMD cells):** An expansive, semi-arid plateau characterized by black cotton soils, low annual rainfall ($500–700\text{ mm}$), and high convective variability.
4. **South Interior Karnataka (SIK; 86 IMD cells):** The southern Mysore plateau, receiving moderate rainfall from both the Southwest and Northeast (retreating) monsoons.

```text
+---------------------------------------------------------------------------------------------+
|                               FIGURE 1: STUDY AREA DOMAIN                                   |
| [Figure 1: High-resolution map of Karnataka showing topography from Copernicus GLO-90 DEM,  |
| Western Ghats crest line, 252 IMD 0.25° grid cells color-coded by meteorological           |
| subdivision, and explicit identification of the 3 coastal fallback cells (IDs 52, 68, 125)] |
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure1_study_area.png` and `.pdf`)*

### 2.2 Observational Target: IMD Gridded Rainfall
The ground-truth observational target is the IMD high-resolution daily gridded rainfall dataset ($0.25^\circ \times 0.25^\circ$, approximately $27\text{ km} \times 27\text{ km}$ at tropical latitudes) developed by Pai et al. [3] and Rajeevan et al. [4]. This dataset is interpolated via modified inverse distance weighting using daily records from 6,955 quality-controlled rain gauge stations across India. Each daily value corresponds to the 24-hour accumulation ending at 08:30 Indian Standard Time (IST; equivalent to 03:00 UTC). We extract all 252 in-state grid cells for the 2,556 calendar days spanning January 1, 2019 to December 31, 2025.

### 2.3 Reanalysis Predictors: ERA5-Land
Predictor features are derived from ECMWF ERA5-Land reanalysis [2], a global land-surface reanalysis dataset produced by running the HTESSEL land-surface model forced by ERA5 atmospheric fields at an enhanced spatial resolution of $0.1^\circ$ (approximately $9\text{ km}$) with hourly output. We extract 15 surface hydrometeorological fluxes and thermodynamic variables: total precipitation (`tp`), surface runoff (`sro`), total runoff (`ro`), 2-meter air temperature (`t2m`), 2-meter dewpoint temperature (`d2m`), surface pressure (`sp`), 10-meter wind components (`u10`, `v10`), surface solar radiation downwards (`ssrd`), surface thermal radiation downwards (`strd`), surface latent heat flux (`slhf`), surface sensible heat flux (`sshf`), evaporation (`e`), potential evaporation (`pev`), and volumetric soil water layer 1 (`swvl1`).

To strictly maintain temporal causality and avoid future-data leakage, hourly reanalysis fluxes are de-accumulated and aggregated over the 24-hour antecedent window ending at 03:00 UTC on day $t-1$. Therefore, all ERA5-Land predictors correspond to the antecedent 24-hour cycle preceding the issuance of the hindcast target.

### 2.4 Topography & Coastal Fallback Discretization
Topographic variables are derived from the Copernicus Global Digital Elevation Model (COP-DEM GLO-90) at 90-meter native resolution [5]. For each IMD grid cell centroid, we compute the area-averaged elevation (`elevation_dem_m`), local topographic slope (`slope_dem_deg`), distance to the Arabian Sea coastline (`dist_to_coast_km`), and orthogonal distance to the Western Ghats crest line (`dist_to_ghats_crest_km`).

**Treatment of Coastal Fallback Cells:** In the native ERA5-Land land-sea mask, grid pixels with more than $50\%$ ocean fraction are masked as water. For exactly **3 coastal IMD grid cells**—Grid-Cell ID 52 (Udupi/Malpe, lat $13.25^\circ$, lon $74.75^\circ$, coast dist $5.94\text{ km}$), Grid-Cell ID 68 (Kundapura, lat $13.50^\circ$, lon $74.75^\circ$, coast dist $15.11\text{ km}$), and Grid-Cell ID 125 (Karwar/Ankola, lat $14.75^\circ$, lon $74.25^\circ$, coast dist $6.98\text{ km}$)—the cell centroids fall over marine water pixels. Reanalysis predictors for these cells were extracted from the nearest-inland neighbor pixel. These 3 cells ($N = 2,193$ test instances) are explicitly tracked with a boolean flag (`is_coastal_fallback`) and evaluated alongside native-land cells.

### 2.5 Dataset Partitioning & Climatological Characteristics
The dataset is partitioned chronologically into three non-overlapping temporal periods:
- **Training Partition (2019–2022; 1,447 days):** $N = 364,644$ instances. Used exclusively for model parameter optimization, tree construction, and fitting all preprocessors (scalers, encoders, EQM transfer functions). A 14-day warmup quarantine (January 1–14, 2019; 3,528 instances) was purged to eliminate rolling lag initialization artifacts.
- **Validation Partition (2023; 365 days):** $N = 91,980$ instances. Climatologically, 2023 was an exceptional regional drought year across Karnataka, exhibiting only $27.39\%$ wet days ($>0.1\text{ mm}$) compared to $36.95\%$ in the training set. This anomalous period serves as a highly stringent benchmark for early stopping (patience $= 50$), hyperparameter tuning, and decision threshold calibration ($p^*$).
- **Test Partition (2024–2025; 731 days):** $N = 184,212$ instances (366 days in leap-year 2024, 365 days in 2025). This partition was held strictly frozen and accessed exclusively for final out-of-sample forward evaluation and descriptive ablation reporting. Zero test-based tuning was performed.

Table 1 summarizes the climatological and statistical characteristics across all partitions.

#### Table 1: Temporal Dataset Partitioning & Climatological Rainfall Distribution
| Partition | Calendar Years | Total Days | Total Instances | Wet Days ($\ge 0.1\text{ mm}$, %) | Mean Daily Rain (mm) | Std Dev (mm) | 50th Pct (mm) | 90th Pct (mm) | 99th Pct (mm) | Max Daily Rain (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | 2019–2022 | 1,447 | 364,644 | 36.95% | 3.7388 | 11.0264 | 0.0000 | 10.4200 | 54.8000 | 473.00 |
| **Validation** | 2023 | 365 | 91,980 | 27.39% | 2.3571 | 8.2325 | 0.0000 | 5.2500 | 38.3000 | 258.00 |
| **Test** | 2024–2025 | 731 | 184,212 | 35.04% | 3.5085 | 11.0006 | 0.0000 | 8.9000 | 54.2000 | 286.00 |
| **Full Dataset** | 2019–2025 | 2,543 | 640,836 | 35.03% | 3.4741 | 10.6627 | 0.0000 | 9.0700 | 52.4000 | 473.00 |

---

## 3. Methodology

```text
+---------------------------------------------------------------------------------------------+
|                          FIGURE 2: METHODOLOGICAL PIPELINE ARCHITECTURE                     |
| [Figure 2: End-to-end workflow diagram showing ERA5-Land de-accumulation, IMD gauge         |
| preparation, causal 24-h lag alignment, 63 primary feature engineering, chronological split, |
| Tweedie/Hurdle/Focal Loss model training, validation calibration, and bootstrap audit]      |
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure2_methodology.png` and `.pdf`)*

### 3.1 ERA5-Land De-accumulation Physics and Temporal Alignment
In raw ECMWF ERA5-Land records, accumulated surface variables (total precipitation `tp`, surface runoff `sro`, total runoff `ro`, evaporation `e`, downward radiation fluxes `ssrd`, `strd`, and turbulent heat fluxes `slhf`, `sshf`) are accumulated from the beginning of each forecast cycle starting at 00:00 UTC. At each hour $h \in \{1, \dots, 24\}$, the stored value represents the accumulated integral from 00:00 UTC to hour $h$. At 00:00 UTC on the subsequent day, the forecast cycle resets.

To recover physical hourly flux rates, we implement consecutive backward differencing:
$$\Delta x_h = x_h - x_{h-1}, \quad h \in \{2, \dots, 24\}$$
with $\Delta x_1 = x_1$ representing the first hour accumulation following the 00:00 UTC reset. Across all 84 months of data (2019–2025), zero negative increments ($< -10^{-5}\text{ mm}$) were detected, confirming physical conservation across reset boundaries.

**IMD Temporal Alignment:** Daily ground-truth rainfall from IMD is accumulated over the 24-hour window ending at 08:30 IST (03:00 UTC). To construct the IMD-aligned daily reanalysis total, de-accumulated hourly increments are integrated across 24 consecutive hours from 04:00 UTC on day $t-1$ through 03:00 UTC on day $t$:
$$X_{t}^{\text{IMD}} = \sum_{h=4}^{24} \Delta x_{t-1, h} + \sum_{h=1}^{3} \Delta x_{t, h}$$

**Physical Unit Conversions:**
- Precipitation (`tp`), surface runoff (`sro`), and total runoff (`ro`): Converted from raw model units of meters ($m$) to millimeters ($\text{mm}$) by multiplying by $1,000$.
- Downward shortwave (`ssrd`), longwave (`strd`), and latent/sensible heat fluxes (`slhf`, `sshf`): Converted from Joules per square meter ($\text{J/m}^2$) to daily accumulated energy flux densities in megajoules per square meter ($\text{MJ/m}^2$, dividing by $10^6$) or equivalent mean hourly flux rates in Watts per square meter ($\text{W/m}^2$, dividing by $86,400\text{ s}$).
- Temperatures (`t2m`, `d2m`): Converted from Kelvin ($K$) to Celsius ($^\circ\text{C}$) via $T(^\circ\text{C}) = T(K) - 273.15$. Dewpoint depression is calculated as $\Delta T_d = T_{2m} - T_{d2m}$.

**Causal Latency Window:** In our retrospective hindcasting framework, to estimate rainfall on day $t$ (accumulation ending at 03:00 UTC on day $t$), all predictor features are restricted to variables observed or integrated over the antecedent period ending at 03:00 UTC on day $t-1$. Thus, ERA5-Land fluxes are lagged by one full day (`_lag_1d`), representing the antecedent 24-hour cycle preceding the forecast target.

**Publication-Latency Assumption:** Local antecedent IMD observations ($y_{t-1}, y_{t-2}, \dots$) are incorporated under the retrospective hindcasting convention that previous-day gauge totals are known. In operational real-time forecasting, final quality-controlled IMD gridded products are subject to publication delays; therefore, this framework evaluates diagnostic reconstruction capacity rather than real-time operational deployment.

---

### 3.2 Feature Matrix Architecture
The final input matrix comprises 78 columns, of which **63 features** are eligible as primary explanatory variables ($X$) respecting the 24-hour causal latency constraint. As classified in Table 2, these 63 predictors span five physical and temporal domains:
1. **Cyclic Temporal Predictors (7 features):** Month, day of year, harmonic sine/cosine expansions ($\sin(2\pi \cdot \text{DOY}/366)$, $\cos(2\pi \cdot \text{DOY}/366)$, $\sin(2\pi \cdot \text{Month}/12)$, $\cos(2\pi \cdot \text{Month}/12)$), and Southwest monsoon indicator (`is_monsoon`).
2. **Static Topographic & Geographic Predictors (6 features):** `elevation_dem_m`, `slope_dem_deg`, `dist_to_coast_km`, `dist_to_ghats_crest_km`, latitude, and longitude.
3. **Antecedent Observed IMD Gauges (12 features):** Lagged local rainfall ($y_{t-1}, y_{t-2}, y_{t-3}, y_{t-7}, y_{t-14}$), rolling window means (3-day, 7-day, 14-day), rolling maximums, and Antecedent Precipitation Index ($\text{API}_{14} = \sum_{k=1}^{14} 0.85^k y_{t-k}$).
4. **Antecedent ERA5-Land Surface Fluxes (15 features):** Lagged precipitation (`tp_lag_1d`), surface runoff (`sro_lag_1d`), total runoff (`ro_lag_1d`), latent heat flux (`slhf_lag_1d`), sensible heat flux (`sshf_lag_1d`), evaporation (`e_lag_1d`), potential evaporation (`pev_lag_1d`), volumetric soil water layer 1 (`swvl1_lag_1d`), and corresponding 3-day rolling statistics.
5. **Antecedent ERA5-Land Thermodynamic Predictors (17 features):** Lagged temperatures (`t2m_lag_1d`, `d2m_lag_1d`), dewpoint depression (`dewpoint_depression_lag_1d`), surface pressure (`sp_lag_1d`), wind components (`u10_lag_1d`, `v10_lag_1d`, wind speed), solar radiation (`ssrd_lag_1d`), thermal radiation (`strd_lag_1d`), and 3-day rolling dynamics.

#### Table 2: Input Feature Groups, Predictor Schema, and Pipeline Gating
| Feature Group ID | Physical / Operational Category | Column Count | Gated Primary Predictors ($X$) | Physical Examples & Latency Window |
| :--- | :--- | :---: | :---: | :--- |
| **Group 1** | Target Labels & Ground Truth | 4 | 0 (Targets Only) | `target_rainfall_mm`, `target_is_rain`, `target_is_heavy_rain`, `target_rain_ge_35_5mm` |
| **Group 2** | Identifiers & Metadata | 6 | 0 (Audit & Indexing) | `date`, `station_id`, `district`, `zone_name`, `target_period`, `prediction_issue_time` |
| **Group 3** | Antecedent IMD Gauge Memory | 13 | 13 | `target_rain_lag_1d` to `_14d`, `target_rain_roll_3d/7d/14d`, `api_14d` (lagged $\ge 1	ext{ day}$) |
| **Group 4** | ERA5-Land Precip & Hydrology | 14 | 14 | `era5_tp_lag_1d` to `_3d`, `ro_lag_1d`, `sro_lag_1d`, `src_lag_1d`, evaporation depths |
| **Group 5** | Atmospheric Thermodynamics & Wind | 12 | 12 | `t2m_mean/min/max_lag_1d`, `d2m_lag_1d`, `dewpoint_depression_lag_1d`, `sp_lag_1d`, `wind_speed_lag_1d` |
| **Group 6** | Soil Moisture & Energy Radiation Fluxes | 9 | 9 | `swvl1_lag_1d/3d/7d`, `swvl2_lag_1d`, `swvl3_lag_1d`, `ssrd_lag_1d`, `strd_lag_1d`, `slhf_lag_1d`, `sshf_lag_1d` |
| **Group 7** | Static Topography & Spatial Proximity | 6 | 6 | `elevation_dem_m`, `slope_dem_deg`, `latitude`, `longitude`, `dist_to_coast_km`, `dist_to_ghats_crest_km` |
| **Group 8** | Cyclic Temporal Ephemeris | 7 | 7 | `month`, `day_of_year`, `sin/cos_day_of_year`, `sin/cos_month`, `is_monsoon` |
| **Group 9** | Primary Regional Categorical Predictors | 2 | 2 | `subdivision` (4 regions), `zone_id` (10 zones) [target-encoded to `subdivision_enc`, `zone_id_enc`] |
| **Auxiliary**| Coastal Ocean-Boundary Fallback Flag | 1 | 0 (Auxiliary Quality Flag) | `is_coastal_fallback` (binary flag for Grid-Cell IDs 52, 68, 125 touching ocean cells) |
| **Excluded** | Pressure Proxies & Redundant Strings | 4 | 0 (Ablation / Excluded) | `elevation_pressure_proxy_m`, `slope_pressure_proxy_deg`, `interpolation_method`, `season` |
| **Total** | **Master Parquet Dataset** | **78** | **63 Eligible Features** | **Strict 24-hour antecedent causality enforced ($T_{\text{issue}} = \text{03:00 UTC day } t-1$)** |

### 3.3 Model Formulations

#### Tier 0: Naive Baselines
- **Rainfall Persistence ($y_{t-1}$):** Assumes rainfall on day $t$ equals observed IMD rainfall on day $t-1$: $\hat{y}_t = y_{t-1}$.
- **DOY Station Climatology:** Predicts the historical day-of-year mean rainfall calculated across 2019–2022 for each respective grid cell.

#### Tier 1: Raw Reanalysis Benchmark
- **Antecedent ERA5-Land Total Precipitation (`tp_lag_1d`):** Directly evaluates uncalibrated 24-hour antecedent precipitation from ERA5-Land.

#### Tier 2: Statistical Downscaling Benchmark
- **Empirical Quantile Mapping (EQM):** Implements station-wise non-parametric quantile mapping [10], [11]. For each of the 252 IMD grid cells, empirical cumulative distribution functions (CDFs) of observed rainfall ($F_{\text{obs}}$) and ERA5-Land precipitation ($F_{\text{era5}}$) are fitted on 2019–2022 using 100 empirical quantile knots:
  $$\hat{y}_t = F_{\text{obs}}^{-1}\left(F_{\text{era5}}(x_t)\right)$$

#### Tier 3: Regularized Linear Baselines
- **Ridge Regression (L2):** Fits a linear model with $L_2$ Tikhonov regularization ($lpha = 100.0$) on standardized features ($z$-score transformed strictly on 2019–2022 training data), establishing a regularized linear benchmark.
- **ElasticNet Regression (L1/L2):** Combines $L_1$ and $L_2$ penalties ($lpha = 0.1$, $l_1	ext{-ratio} = 0.5$) to evaluate sparse linear feature selection against non-linear tree ensembles.

#### Tier 4: Machine Learning Architectures
- **LightGBM Tweedie Regressor ($p=1.5$) [14]:** Daily rainfall is modeled as an exponential dispersion model with compound Poisson-Gamma distribution [17], [18]. The variance relates to the mean via $\text{Var}(Y) = \phi \mu^p$ with variance power $p = 1.5$ and log-link function $\mu = \exp(f(x))$. The deviance loss optimized by gradient tree boosting is:
  $$\mathcal{L}_{\text{Tweedie}}(y, \mu) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$$
- **CatBoost Tweedie Regressor ($p=1.5$):** Implements symmetric oblivious decision trees optimizing Tweedie deviance ($p=1.5$) to evaluate architectural invariance [15].
- **XGBoost Tweedie Regressor:** Evaluates second-order gradient boosted trees optimizing Tweedie deviance [16].
- **LightGBM Log-Ordinary with Duan Smearing:** Models log-transformed rainfall $\log(y + 1)$ with mean squared error, applying Duan's non-parametric smearing retransformation factor [20] to correct for geometric mean retransformation bias.
- **Two-Stage Hurdle Model:** Decomposes rainfall into a binary rain occurrence classifier ($P(Y \ge 0.1 \mid X)$) using binary cross-entropy and a conditional regressor ($E[Y \mid Y \ge 0.1, X]$) trained exclusively on non-zero observations [19]. The final prediction is the expected value product:
  $$\hat{y} = \hat{P}(Y \ge 0.1 \mid X) \cdot \hat{E}[Y \mid Y \ge 0.1, X]$$

---

### 3.4 Heavy Rainfall Classification & Custom Focal Loss
To alert for extreme hydrometeorological hazards, we implement a dedicated binary classifier targeting the official IMD Heavy Rainfall threshold ($\ge 64.5\text{ mm/day}$), which represents a severe class imbalance ($0.6878\%$ prevalence in the test set). Standard binary cross-entropy is overwhelmed by negative instances. We implement a custom Focal Loss objective [21]:
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $p_t = p$ if $y = 1$ and $p_t = 1 - p$ if $y = 0$, with focusing parameter $\gamma = 2.0$ and weighting factor $\alpha = 0.75$.

**Surrogate Positive Hessian Formulation:** The exact analytical second-order derivative of focal loss is sign-indefinite in regions where predictions diverge sharply from ground truth, violating LightGBM's positive-semidefinite requirement for tree splitting. We employ a strictly positive-definite surrogate Hessian:
$$h_i = w_i \cdot p_i (1 - p_i)$$
where $w_i = \alpha_t (1 - p_t)^\gamma$. This guarantees numerical stability and smooth tree convergence.

**Decision Threshold Selection ($p^*$):** The operational decision threshold was calibrated strictly on the 2023 Validation partition to maximize the Critical Success Index ($\text{CSI}$), yielding $p^* = 0.53$ for Focal Loss, $p^* = 0.96$ for Class-Weighted LightGBM, and $p^* = 0.98$ for Class-Weighted CatBoost. These thresholds represent empirical decision boundaries optimized for contingency skill rather than calibrated posterior probabilities.

---

### 3.5 Evaluation Metrics & Statistical Verification

#### Continuous Hydrometeorological Metrics
Let $y_i$ denote observed rainfall and $\hat{y}_i$ denote predicted rainfall across $N$ instances:
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$
- **Mean Bias & Percentage Bias (PBias):** $\text{Bias} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i); \quad \text{PBias} = \frac{\sum (\hat{y}_i - y_i)}{\sum y_i} \times 100\%$
- **Pearson Correlation ($r$):** $r = \frac{\sum (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum (y_i - \bar{y})^2 \sum (\hat{y}_i - \bar{\hat{y}})^2}}$
- **Nash-Sutcliffe Efficiency (NSE) [22]:** $\text{NSE} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Classic Kling-Gupta Efficiency (KGE-2009) [23]:** $\text{KGE-2009} = 1 - \sqrt{(r - 1)^2 + (\alpha_k - 1)^2 + (\beta - 1)^2}$, where $\alpha_k = \sigma_{\hat{y}} / \sigma_y$ and $\beta = \bar{\hat{y}} / \bar{y}$.
- **Modified Kling-Gupta Efficiency (KGE-2012) [24]:**
  $$\text{KGE-2012} = 1 - \sqrt{(r - 1)^2 + (\gamma_k - 1)^2 + (\beta - 1)^2}$$
  where $\gamma_k = \text{CV}_{\hat{y}} / \text{CV}_y = (\sigma_{\hat{y}}/\bar{\hat{y}}) / (\sigma_y/\bar{y})$ represents the variability ratio, eliminating cross-correlation between bias and variability.

#### Categorical Alerting Metrics ($\ge 64.5\text{ mm}$)
From the $2 \times 2$ contingency table of Hits ($H$), Misses ($M$), False Alarms ($F$), and Correct Negatives ($C$):
- **Probability of Detection (POD / Recall):** $\text{POD} = \frac{H}{H + M}$
- **False Alarm Ratio (FAR):** $\text{FAR} = \frac{F}{H + F}; \quad \text{Precision} = 1 - \text{FAR} = \frac{H}{H + F}$
- **Critical Success Index (CSI / Threat Score):** $\text{CSI} = \frac{H}{H + M + F}$
- **Heidke Skill Score (HSS):** $\text{HSS} = \frac{2(HC - MF)}{(H + M)(M + C) + (H + F)(F + C)}$
- **Equitable Threat Score (ETS):** $\text{ETS} = \frac{H - H_{\text{exp}}}{H + M + F - H_{\text{exp}}}$, where $H_{\text{exp}} = \frac{(H + M)(H + F)}{N}$.

#### Paired Stationary Block Bootstrapping
Because daily rainfall displays spatial cross-correlation across the 252 grid cells and temporal autocorrelation across consecutive days, naive independent resampling produces severely over-optimistic standard errors. We implement a stationary paired block bootstrap protocol [25], [26]. Blocks of consecutive calendar dates of length $L = 7\text{ days}$ (synoptic weather scale) and $L = 14\text{ days}$ (intraseasonal Madden-Julian Oscillation scale) are resampled with replacement. All 252 spatial grid cells are preserved simultaneously for each sampled date, maintaining the exact spatial covariance structure. For each of $B = 2,000$ bootstrap iterations, paired differences in RMSE ($\Delta \text{RMSE}_b = \text{RMSE}_{\text{ML}, b} - \text{RMSE}_{\text{Benchmark}, b}$) are computed. Significance is reported at the empirical resolution threshold:
$$\text{Empirical } p = \frac{\sum_{b=1}^B \mathbb{I}(\Delta \text{RMSE}_b \ge 0) + 1}{B + 1} \implies p < 0.0005$$

---

## 4. Results

### 4.1 Continuous Rainfall Estimation & Multi-Criteria Model Benchmarking
Table 3 presents the audited benchmarking results on the frozen 2024–2025 forward test partition ($N = 184,212$; observed mean $= 3.5085\text{ mm}$, observed standard deviation $= 11.0006\text{ mm}$).

#### Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; $N = 184,212$)
| Model Hierarchy Tier | Model Name | Loss Objective | RMSE (mm) | MAE (mm) | Mean Bias (mm) | PBias (%) | Pearson $r$ | NSE | KGE-2009 | KGE-2012 | Heavy CSI ($\ge 64.5	ext{ mm}$) |
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
| **Tier 4 (Machine Learning)** | **Hurdle Expected Value** | **Classifier + Conditional Reg.** | **7.4241** | **2.9300** | **+0.0342** | **+0.97** | **0.7380** | **0.5443** | **0.6165** | **0.6114** | **0.3072** |

### 4.2 Heavy Rainfall Detection ($\ge 64.5\text{ mm}$)
Table 4 presents the frozen test contingency analysis and derived skill scores for detecting IMD Heavy Rainfall ($\ge 64.5\text{ mm}$; $1,267$ true events out of $184,212$ test records, prevalence $0.6878\%$).

#### Table 4: Heavy Rainfall Detection ($\ge 64.5\text{ mm}$) Contingency & Skill Metrics (Test Partition 2024–2025; $N = 184,212$)
| Model Architecture | Operating Threshold ($p^*$) | Calibration Source | Hits ($H$) | Misses ($M$) | False Alarms ($F$) | Correct Negatives ($C$) | Precision | POD / Recall | FAR | CSI | HSS | ETS | PR-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Focal Loss** | **0.53** | **Validation 2023 Max CSI** | **595** | **672** | **427** | **182,518** | **0.5822** | **0.4696** | **0.4178** | **0.3512** | **0.5169** | **0.3485** | **0.5084** |
| LightGBM Class-Weighted | 0.96 | Validation 2023 Max CSI | 688 | 579 | 722 | 182,223 | 0.4879 | 0.5430 | 0.5121 | 0.3459 | 0.5105 | 0.3427 | 0.4853 |
| CatBoost Class-Weighted | 0.98 | Validation 2023 Max CSI | 608 | 659 | 515 | 182,430 | 0.5414 | 0.4799 | 0.4586 | 0.3412 | 0.5056 | 0.3383 | 0.4739 |
| Rainfall Persistence | 64.5 mm | Direct Physical Lag | 484 | 783 | 783 | 182,162 | 0.3820 | 0.3820 | 0.6180 | 0.2361 | 0.3777 | 0.2328 | N/A |
| Raw ERA5-Land | 64.5 mm | Direct Antecedent Flux | 231 | 1,036 | 606 | 182,339 | 0.2760 | 0.1823 | 0.7240 | 0.1233 | 0.2153 | 0.1206 | N/A |
| Empirical Quantile Mapping | 64.5 mm | Station-Wise Knots | 372 | 895 | 622 | 182,323 | 0.3742 | 0.2936 | 0.6258 | 0.1969 | 0.3250 | 0.1940 | N/A |

```text
+---------------------------------------------------------------------------------------------+
|                           FIGURE 5: HEAVY RAIN DETECTION SKILL                              |
| [Figure 5: Panel A shows contingency breakdown (Hits, Misses, False Alarms); Panel B shows   |
| skill radar comparing Precision, Recall, CSI, and HSS across Focal Loss and baselines]      |
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure5_heavy_rain_detection.png` and `.pdf`)*

#### Heavy Rainfall Alerting Takeaways:
1. **False Alarm Suppression via Focal Loss:** While Class-Weighted LightGBM achieves a higher raw recall ($54.30\%$ vs. $46.96\%$), it incurs **$722$ false alarms**. In contrast, LightGBM with Focal Loss generates only **$427$ false alarms**—a **$40.86\%$ reduction in false alerts**. Consequently, Focal Loss achieves an operational Precision of **$58.22\%$** (FAR of $41.78\%$) and the highest overall Critical Success Index ($\text{CSI} = 0.3512$).
2. **Reanalysis Alerting Failure:** Raw ERA5-Land exhibits catastrophic failure for extreme alerting: it misses **$81.77\%$ of heavy events** ($1,036$ misses out of $1,267$) and produces a False Alarm Ratio of $72.40\%$ ($\text{CSI} = 0.1233$).
3. **Threshold Distinction:** The decision threshold $p^* = 0.53$ represents an operating decision cutoff selected on validation CSI; it does not constitute a formally calibrated posterior probability.

---

### 4.3 Feature Importance: Gain, Permutation, and Ablation
Figure 6 and Table 7 examine physical feature attribution and ablation dynamics. To avoid ambiguity, the mathematical basis of each importance metric is explicitly defined:

```text
+---------------------------------------------------------------------------------------------+
|                        FIGURE 6: FEATURE IMPORTANCE & ATTRIBUTION                           |
| [Figure 6: Horizontal bar chart showing top 15 predictors ranked by validation permutation  |
| RMSE degradation (Panel A) and donut chart showing relative gain contributions (Panel B)]   |
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure6_feature_interpretation.png` and `.pdf`)*

#### Distinct Feature Importance Perspectives:
1. **Tree Split Gain in Heavy Rain Classifier (Origin of 54.52%):**
   In the LightGBM Focal Loss heavy rainfall classifier, tree split gain measures the total reduction in focal loss objective attributed to splits on a given variable. Total runoff (`ro_lag_1d`) accounts for **$28.11\%$** of total tree gain, and surface runoff (`sro_lag_1d`) accounts for **$26.41\%$**. Together, these two simulated land-surface runoff variables account for **$54.52\%$ of total tree gain** in heavy-rain classification.
2. **Tree Split Gain in Continuous Tweedie Regressor:**
   In continuous LightGBM Tweedie regression, surface runoff (`sro_lag_1d`) accounts for **$34.61\%$** of total tree gain (Rank 1), skin reservoir content (`src_lag_1d`) accounts for **$28.28\%$** (Rank 2), downward thermal radiation (`strd_lag_1d`) accounts for **$12.26\%$** (Rank 3), and total runoff (`ro_lag_1d`) accounts for **$3.36\%$** (Rank 4). Combined surface moisture/runoff variables account for $66.25\%$ of continuous tree gain.
3. **Validation Permutation Importance (LightGBM Tweedie):**
   Permutation importance [27], [28] measures out-of-sample RMSE degradation ($\Delta \text{RMSE}$) when a single feature's values are randomly shuffled on the validation set. Shuffling surface runoff (`sro_lag_1d`) increases RMSE by **$+2.3545\text{ mm}$** ($+42.16\%$ increase, Rank 1), shuffling skin reservoir content (`src_lag_1d`) increases RMSE by **$+2.1241\text{ mm}$** ($+38.03\%$, Rank 2), and shuffling downward thermal radiation (`strd_lag_1d`) increases RMSE by **$+1.7960\text{ mm}$** ($+32.16\%$, Rank 3). Shuffling previous-day IMD rain (`target_rain_lag_1d`) increases RMSE by **$+0.8320\text{ mm}$** ($+14.90\%$, Rank 4).
4. **SHAP Attribution (TreeSHAP) [29]:**
   Mean absolute SHAP value analysis indicates that downward longwave radiation (`strd_lag_1d`) exerts the largest continuous magnitude impact on log-link predictions (mean $|\text{SHAP}| = 2.5745$, Rank 1), followed by surface runoff (`sro_lag_1d`, mean $|\text{SHAP}| = 0.8897$, Rank 2), serving as an atmospheric cloud-cover and thermal emission proxy.

#### Table 7: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)
| Scenario ID | Ablation Description | Features Retained | Val RMSE (mm) | Val KGE-2012 | Val Heavy CSI | Test RMSE (mm) | Test KGE-2012 | Test Heavy CSI | $\Delta$Test RMSE (mm) | Scientific Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scenario A** | All 63 Primary Predictors | 63 | 5.5850 | 0.6494 | 0.4098 | **7.4427** | **0.6245** | **0.2868** | 0.0000 | Baseline full configuration |
| **Scenario B** | Without Runoff & Hydrological Fluxes | 55 | 5.7149 | 0.6463 | 0.4129 | 7.6854 | 0.5925 | 0.2721 | +0.2427 | Simulated land fluxes provide critical moisture saturation proxies |
| **Scenario C** | Without Antecedent IMD Gauges | 51 | 5.7147 | 0.6235 | 0.3316 | 7.5828 | 0.6070 | 0.2693 | +0.1401 | Local gauge memory essential for extreme threshold alerting |
| **Scenario D** | Without Static Topography | 55 | 5.5809 | 0.6479 | 0.4097 | 7.4557 | 0.6252 | 0.2843 | +0.0130 | Terrain captures spatial orographic boundaries |
| **Scenario E** | Without ERA5 Predictors | 26 | 6.3909 | 0.4181 | 0.2652 | **8.4511** | **0.4833** | **0.1536** | **+1.0084** | ERA5 atmospheric predictors provide measurable incremental predictive value |
| **Scenario F** | Only Antecedent IMD and Temporal | 18 | 6.4289 | 0.4193 | 0.3043 | 8.5145 | 0.4686 | 0.1456 | +1.0718 | Baseline autoregressive floor |

#### Key Takeaway on Reanalysis Predictors:
Ablation Scenario E demonstrates that removing ERA5-Land predictors surges Test RMSE from $7.4427\text{ mm}$ to $8.4511\text{ mm}$ ($\Delta \text{RMSE} = \mathbf{+1.0084\text{ mm}}$), reduces KGE from $0.6245$ to $0.4833$, and collapses Heavy Rain CSI from $0.2868$ to $0.1536$. This establishes that ERA5-Land reanalysis predictors provide **measurable, statistically confirmed incremental predictive value** beyond local gauge autoregression.

---

### 4.4 Regional and Seasonal Disaggregation
Table 5 breaks down model performance across Karnataka's four meteorological subdivisions and four climatological seasons.

#### Table 5: Subdivisional & Seasonal Stratified Performance (LightGBM Tweedie vs. Baselines on Test Set)
| Stratum Type | Stratum Name | Sample Count | Observed Mean (mm) | ML RMSE (mm) | Persistence RMSE (mm) | Raw ERA5 RMSE (mm) | ML Bias (mm) | ML PBias (%) | ML Pearson $r$ | ML KGE-2012 | ML Heavy CSI ($\ge 64.5\text{ mm}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Subdivision** | **Coastal Karnataka (All 13 cells)** | 9,503 | 13.1855 | 14.9475 | 20.3843 | 19.3674 | -1.3244 | -10.04 | **0.8261** | **0.7407** | **0.3620** |
| Subdivision | -- Coastal Native Land (10 cells) | 7,310 | 12.9925 | 14.7751 | 20.0019 | 19.0503 | -1.4729 | -11.34 | 0.8258 | 0.7356 | 0.3516 |
| Subdivision | -- Coastal Fallback (3 cells: IDs 52, 68, 125) | 2,193 | 13.8286 | 15.5086 | 21.6100 | 20.3891 | -0.8294 | -6.00 | 0.8280 | 0.7504 | 0.3898 |
| **Subdivision** | **Malnad (35 cells)** | 25,585 | 6.3706 | 9.3018 | 12.7091 | 13.2843 | +0.1661 | +2.61 | **0.8043** | **0.7256** | **0.3441** |
| **Subdivision** | **North Interior Karnataka (118 cells)**| 86,258 | 2.3758 | 6.4317 | 9.0913 | 9.5698 | +0.0196 | +0.83 | **0.5274** | **0.3676** | **0.0296** |
| **Subdivision** | **South Interior Karnataka (86 cells)**| 62,866 | 2.4364 | 6.0472 | 8.5112 | 8.1312 | +0.0531 | +2.18 | **0.5513** | **0.3723** | **0.0000** |
| **Season** | **Southwest Monsoon (JJAS)** | 61,488 | 7.0215 | 10.1883 | 14.1842 | 14.9381 | +0.1431 | +2.04 | **0.7545** | **0.6451** | **0.3355** |
| **Season** | **Post-Monsoon Northeast (OND)** | 46,368 | 2.1379 | 6.1093 | 8.8290 | 7.9161 | +0.0612 | +2.86 | **0.5628** | **0.3823** | **0.0000** |
| **Season** | **Pre-Monsoon (MAM)** | 46,368 | 2.4463 | 6.6839 | 8.9736 | 8.6417 | -0.3207 | -13.11 | **0.6717** | **0.4859** | **0.0930** |
| **Season** | **Winter (JF)** | 29,988 | 0.0698 | 0.8060 | 1.2303 | 1.2443 | -0.0023 | -3.30 | **0.4881** | **0.3436** | **0.0000** |

```text
+---------------------------------------------------------------------------------------------+
|                        FIGURE 7: SUBDIVISIONAL & SEASONAL SKILL                             |
| [Figure 7: Regional breakdown of RMSE and KGE-2012 across subdivisions (Panel A) and         |
| seasonal progression from Southwest Monsoon to Winter (Panel B)]                            |
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure7_regional_seasonal.png` and `.pdf`)*

#### Regional & Seasonal Observations:
- **Orographic vs. Convective Divide:** Model performance splits sharply across geographical zones. In the orographic Western Ghats and Coastal plain, the model achieves high correlation ($r = 0.80–0.83$) and hydrological skill ($\text{KGE-2012} = 0.73–0.74$). In contrast, over the semi-arid interior plains (NIK and SIK), correlation encounters a distinct ceiling ($r \approx 0.53–0.55$; $\text{KGE-2012} \approx 0.37$).
- **Correlation vs. Error Magnitude:** While Coastal Karnataka exhibits the highest Pearson correlation ($r = 0.8261$), it also exhibits the highest absolute RMSE ($14.9475\text{ mm}$) due to extreme daily precipitation variance. Conversely, the hyper-arid Winter season exhibits the lowest RMSE ($0.8060\text{ mm}$) despite a modest correlation ($r = 0.4881$).
- **Coastal Fallback Verification:** The 3 coastal fallback cells (IDs 52, 68, 125; $N = 2,193$) achieve $\text{RMSE} = 15.5086\text{ mm}$ and $\text{KGE-2012} = 0.7504$, tracking closely with the 10 native-land coastal cells ($\text{RMSE} = 14.7751\text{ mm}$, $\text{KGE-2012} = 0.7356$). Nearest-inland extraction effectively preserves the orographic moisture advection signal.

---

### 4.5 Temporal Generalisation Across Forward Evaluation Years (2024 vs. 2025)
To evaluate out-of-sample temporal generalisation, model performance was audited separately for the two held-out forward calendar years in Table 6A (from Stage 6 verified outputs):

#### Table 6A: Annual Out-of-Sample Evaluation: 2024 (Leap Year) vs. 2025
| Calendar Year | Evaluated Model | Sample Count ($N$) | Observed Mean (mm) | Predicted Mean (mm) | MAE (mm) | RMSE (mm) | Bias (mm) | PBias (%) | Pearson $r$ | NSE | KGE-2012 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2024 (Full)** | **LightGBM Tweedie** | 92,232 | 3.3539 | 3.6981 | 3.0074 | 7.5094 | +0.3442 | +10.26 | 0.7320 | 0.5346 | 0.5676 |
| 2024 (Full) | CatBoost Tweedie | 92,232 | 3.3539 | 3.7431 | 3.0289 | 7.5253 | +0.3892 | +11.60 | 0.7309 | 0.5327 | 0.5577 |
| 2024 (Full) | Hurdle Expected Value | 92,232 | 3.3539 | 3.7534 | 3.0500 | 7.5210 | +0.3995 | +11.91 | 0.7311 | 0.5332 | 0.5436 |
| 2024 (Full) | Rainfall Persistence | 92,232 | 3.3539 | 3.3539 | 3.6498 | 10.3927 | -0.0000 | -0.00 | 0.5543 | 0.1087 | 0.5543 |
| 2024 (Full) | Raw ERA5-Land | 92,232 | 3.3539 | 4.1579 | 4.1904 | 10.6003 | +0.8041 | +23.97 | 0.4789 | 0.0727 | 0.3527 |
| 2024 (Full) | Empirical Quantile Mapping | 92,232 | 3.3539 | 3.9681 | 4.2069 | 11.1297 | +0.6142 | +18.31 | 0.4930 | -0.0222 | 0.4403 |
| **2025 (Full)** | **LightGBM Tweedie** | 91,980 | 3.6644 | 3.2834 | 2.7909 | 7.3752 | -0.3810 | -10.40 | 0.7426 | 0.5493 | 0.6541 |
| 2025 (Full) | CatBoost Tweedie | 91,980 | 3.6644 | 3.3519 | 2.8080 | 7.3403 | -0.3125 | -8.53 | 0.7446 | 0.5535 | 0.6689 |
| 2025 (Full) | Hurdle Expected Value | 91,980 | 3.6644 | 3.3324 | 2.8096 | 7.3257 | -0.3320 | -9.06 | 0.7466 | 0.5553 | 0.6538 |
| 2025 (Full) | Rainfall Persistence | 91,980 | 3.6644 | 3.6644 | 3.8522 | 10.3236 | +0.0000 | +0.00 | 0.5584 | 0.1168 | 0.5584 |
| 2025 (Full) | Raw ERA5-Land | 91,980 | 3.6644 | 3.9717 | 4.0926 | 10.3073 | +0.3073 | +8.39 | 0.5038 | 0.1196 | 0.4592 |
| 2025 (Full) | Empirical Quantile Mapping | 91,980 | 3.6644 | 3.8230 | 4.2165 | 11.2640 | +0.1586 | +4.33 | 0.4957 | -0.0514 | 0.4938 |

In leap-year 2024 ($N = 92,232$), LightGBM Tweedie achieved an RMSE of $7.5094\text{ mm}$ (vs. $10.3927\text{ mm}$ for Persistence and $10.6003\text{ mm}$ for Raw ERA5) and $\text{KGE-2012} = 0.5676$. In 2025 ($N = 91,980$), LightGBM Tweedie achieved an RMSE of $7.3752\text{ mm}$ (vs. $10.3236\text{ mm}$ for Persistence and $10.3073\text{ mm}$ for Raw ERA5) and $\text{KGE-2012} = 0.6541$. Across both years, all three machine learning formulations maintained consistent superiority over baseline and reanalysis benchmarks.

#### Table 6B: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)
| Model Pair Comparison | Block Length ($L$) | Physical Autocorrelation Scale | Mean $\Delta$RMSE (mm) | Standard Error (mm) | 95% Confidence Interval (mm) | Replicates $\Delta \ge 0$ | Empirical $p$-Value Resolution |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Tweedie vs. Persistence** | **7 days** | **Synoptic weather cycle** | **-2.9250** | **0.2046** | **[-3.3241, -2.5175]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. EQM** | **7 days** | **Synoptic weather cycle** | **-3.7542** | **0.3768** | **[-4.4626, -3.0039]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. Raw ERA5** | **7 days** | **Synoptic weather cycle** | **-3.0109** | **0.3482** | **[-3.7037, -2.3735]** | **0** | **$p < 0.0005$** |
| CatBoost Tweedie vs. Persistence | 7 days | Synoptic weather cycle | -2.9337 | 0.2077 | [-3.3415, -2.5276] | 0 | $p < 0.0005$ |
| CatBoost Tweedie vs. EQM | 7 days | Synoptic weather cycle | -3.7629 | 0.3848 | [-4.4936, -2.9954] | 0 | $p < 0.0005$ |
| **LightGBM Tweedie vs. Persistence** | **14 days** | **Intraseasonal / MJO scale** | **-2.9298** | **0.2271** | **[-3.3608, -2.4729]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. EQM** | **14 days** | **Intraseasonal / MJO scale** | **-3.7645** | **0.4243** | **[-4.5976, -2.9587]** | **0** | **$p < 0.0005$** |
| **LightGBM Tweedie vs. Raw ERA5** | **14 days** | **Intraseasonal / MJO scale** | **-3.0204** | **0.3785** | **[-3.7669, -2.3255]** | **0** | **$p < 0.0005$** |

```text
+---------------------------------------------------------------------------------------------+
|                          FIGURE 9: TEMPORAL STABILITY & BOOTSTRAP                           |
| [Figure 9: Panel A shows annual RMSE consistency bar comparison across 2024 and 2025;      |
| Panel B shows paired block bootstrap 95% confidence interval error bars for L=7d and L=14d] |
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure9_temporal_robustness.png` and `.pdf`)*

Across all 2,000 paired block bootstrap resamples under both 7-day and 14-day block lengths, zero replicates yielded $\Delta \text{RMSE} \ge 0$, confirming statistical significance at $p < 0.0005$ under temporal persistence.

---

## 5. Discussion

### 5.1 Hydrometeorological Role of Land-Surface Runoff Variables
A central finding is that modeled runoff variables (`sro_lag_1d` and `ro_lag_1d`) account for **$54.52\%$ of total tree split gain** in the heavy rain classifier and lead permutation importance in continuous regression. 

It is important to emphasize that ERA5-Land runoff variables represent **simulated land-surface water fluxes** from ECMWF's HTESSEL model, not direct in-situ streamflow gauge measurements. In tropical monsoonal regimes, raw precipitation in reanalysis is subject to convective parameterization smearing and spatial displacement. Modeled surface runoff, by contrast, is generated only when local infiltration capacity is exceeded or when the topsoil column approaches saturation. Runoff fluxes thus act as a dynamic physical integrator: they filter out transient atmospheric false alarms and retain only sustained, large-scale moisture convergence events.

### 5.2 Orographic Dynamics vs. The Interior Convective Ceiling
The sharp contrast between orographic skill ($r \ge 0.80$, $\text{KGE} \ge 0.72$) over the Western Ghats and the interior performance ceiling ($r \approx 0.53$, $\text{KGE} \approx 0.37$) reflects fundamental atmospheric physics [7], [9]. 

Over Coastal Karnataka and the Malnad Ghats, precipitation is predominantly forced by mechanical orographic lifting of the strong, moisture-laden southwesterly monsoonal low-level jet (LLJ) against the Western Ghats escarpment [6], [7]. Because this mechanical forcing is tightly coupled to resolved synoptic winds, boundary-layer moisture flux, and static terrain topography, reanalysis predictors and gradient-boosted trees capture this regime with high fidelity.

Conversely, precipitation over North and South Interior Karnataka is dominated by localized, unorganized convective thunderstorm cells during pre-monsoon and post-monsoon transition periods. These meso-$\beta$ and meso-$\gamma$ scale convective systems are triggered by localized boundary-layer thermal boundaries and boundary convergence lines that cannot be resolved on a hydrostatic $0.1^\circ$ grid [8], [9]. Reanalysis predictors provide only bulk column thermodynamics, establishing an inevitable predictability ceiling for 24-hour diagnostic reconstruction.

### 5.3 Extreme Tail Quantile Shrinkage & The Dual-Model Paradigm
Table 8 and Figure 9 provide an unvarnished audit of continuous model behavior across the seven official IMD rainfall intensity categories.

```text
+---------------------------------------------------------------------------------------------+
|                        FIGURE 8: INTENSITY ERROR SPREAD & UNDERESTIMATION                   |
| [Figure 9: Mean prediction bias and percentage bias across 7 IMD categories (Panel A) and    |
| fraction of underestimated events and RMSE error spread across intensity intervals (Panel B)]|
+---------------------------------------------------------------------------------------------+
```
*(Reference: `manuscript_figures/figure8_extreme_rainfall_errors.png` and `.pdf`)*

#### Table 8: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage (Test Set; $N = 184,212$)
| IMD Intensity Category | Rainfall Interval (mm) | Sample Count | % of Test Records | Observed Mean (mm) | Predicted Mean (mm) | MAE (mm) | RMSE (mm) | Mean Bias (mm) | PBias (%) | % Events Underestimated | Structural Limitation Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **No Rain** | $< 0.1$ | 119,669 | 64.96% | 0.00 | 0.77 | 0.77 | 2.10 | +0.77 | Undefined* | 0.0% | Background drizzle inflation artifact |
| **Very Light** | $0.1–2.4$ | 26,590 | 14.43% | 0.96 | 4.04 | 3.29 | 5.05 | +3.08 | +319.1% | 16.9% | Link-function exponential penalty overestimation |
| **Light** | $2.5–15.5$ | 25,854 | 14.03% | 7.03 | 7.46 | 4.67 | 6.58 | +0.43 | +6.1% | 53.8% | **Optimal continuous calibration sweet spot** |
| **Moderate** | $15.6–64.4$ | 10,832 | 5.88% | 29.81 | 17.22 | 16.55 | 20.21 | -12.59 | -42.2% | 83.5% | Progressive underestimation onset |
| **Heavy** | $64.5–115.5$ | 1,067 | 0.58% | 81.68 | 46.98 | 37.58 | 44.39 | -34.71 | -42.5% | 88.6% | Severe tail quantile shrinkage |
| **Very Heavy** | $115.6–204.4$ | 192 | 0.10% | 141.52 | 71.53 | 70.00 | 77.72 | -70.00 | -49.5% | **100.0%** | Severe tail quantile shrinkage (capped near $70–90\text{ mm}$) |
| **Extremely Heavy** | $\ge 204.5$ | 8 | 0.0043% | 225.02 | 73.66 | 151.36 | 154.76 | -151.36 | -67.3% | **100.0%** | Severe tail quantile shrinkage (capped near $70–90\text{ mm}$) |

*\*Note: Percentage bias for No Rain is mathematically undefined due to division by zero observed mean ($\bar{y} = 0.00\text{ mm}$).*

#### Theoretical Cause of Quantile Shrinkage:
In continuous regression optimizing convex loss functions (such as Tweedie deviance or L2 error), the model seeks the conditional expectation $E[Y \mid X]$. In highly skewed, zero-inflated distributions where extreme events are exceptionally sparse, predicting the true conditional mean inevitably pulls predictions toward the bulk distribution, compressing predictions into a ceiling around $70–90\text{ mm}$. As documented in Table 8, **$100\%$ of observations exceeding $115.6\text{ mm}$ (200 test events) are underestimated**.

This empirical reality demonstrates why single continuous regression models cannot be deployed in isolation for hydrometeorological hazard warning. An operational architecture must employ a **dual-model paradigm**: continuous regression (Tweedie or Hurdle) for catchment volumetric water balance, paired with dedicated asymmetric Focal Loss classification for threshold-based extreme alerting.

---

## 6. Limitations

To ensure transparency and scientific rigor, eight structural and experimental limitations are explicitly disclosed:
1. **Retrospective Hindcasting vs. Operational NWP:** This framework operates retrospectively using antecedent 24-hour accumulated ERA5-Land reanalysis fluxes and previous-day gauge observations available at 03:00 UTC. It does **not** provide operational real-time NWP forecasts, which require forward meteorological forecast fields.
2. **Absence of Spatial Generalisation:** All 252 IMD grid cells were present across training, validation, and testing partitions. While temporal generalisation is demonstrated across two forward out-of-sample years (2024–2025), **spatial generalisation to ungauged basins or unmonitored geographic regions is untested and not claimed**.
3. **Severe Extreme Quantile Shrinkage:** As detailed in Table 8, continuous regression models systematically underestimate extreme precipitation tails: $88.6\%$ of Heavy events ($64.5–115.5\text{ mm}$) and **$100\%$ of Very Heavy and Extremely Heavy events ($\ge 115.6\text{ mm}$)** are underestimated, plateauing near $70–90\text{ mm}$.
4. **Dry-Day Drizzle Inflation:** On true zero-rain days ($N = 119,669$, $64.96\%$ of test records), the continuous Tweedie model predicts a mean background drizzle of $+0.77\text{ mm}$ ($\text{MAE} = 0.77\text{ mm}$).
5. **Frozen Test-Set Access:** Frozen test partition results were accessed during final benchmarking and descriptive feature-ablation reporting. However, no model parameters, hyperparameters, feature subsets, or decision thresholds were tuned on test data.
6. **Coastal Fallback Discretization:** The 3 coastal fallback cells (IMD grid-cell IDs 52, 68, 125) utilize nearest-inland neighbor reanalysis extraction ($5.94\text{ to } 15.11\text{ km}$ displacement) due to marine water masking in ERA5-Land. While audited to match native coastal skill closely, they remain an extraction approximation.
7. **Predictive Association vs. Physical Causality:** Feature attribution rankings (permutation loss and tree gain) reflect predictive associations in the presence of correlated collinear predictors, not causal physical proof.
8. **Reanalysis Resolution Limits:** The $0.1^\circ$ spatial grid of ERA5-Land cannot resolve localized meso-$\gamma$ convective thunderstorm updrafts, imposing an intrinsic predictability floor over interior plateau plains ($r \approx 0.53$).

---

## 7. Conclusion

This study developed an empirical machine learning framework for retrospective daily rainfall reconstruction over Karnataka using ERA5-Land reanalysis and IMD gridded observations (2019–2025). The major findings and scientific contributions are:
1. **Multi-Criteria Baseline Outperformance:** Gradient tree boosting architectures (Hurdle Expected Value, CatBoost Tweedie, and LightGBM Tweedie) reduce test RMSE by $\approx 28.1–28.3\%$ relative to Persistence and by $\approx 33.5–33.7\%$ relative to Empirical Quantile Mapping. Hurdle Expected Value attains lowest RMSE ($7.4241\text{ mm}$) and highest NSE ($0.5443$), CatBoost attains highest KGE-2012 ($0.6261$), and LightGBM attains lowest MAE ($2.8993\text{ mm}$).
2. **Stationary Block Bootstrap Significance:** Resampling across synoptic ($7\text{d}$) and intraseasonal ($14\text{d}$) memory scales proves that observed RMSE reductions ($\approx 2.93\text{ mm}$) are statistically significant at an empirical resolution of $p < 0.0005$.
3. **Heavy Rain Alerting via Focal Loss:** In imbalanced heavy rainfall classification ($\ge 64.5\text{ mm}$), custom Focal Loss suppresses false alarms by $40.86\%$ relative to class weighting, achieving $\text{CSI} = 0.3512$, $\text{Precision} = 58.22\%$, and $\text{FAR} = 41.78\%$.
4. **Dominance of Land-Surface Hydrology:** Simulated surface and total runoff fluxes provide critical moisture saturation proxies, accounting for $54.52\%$ of total tree gain in the heavy rain classifier, while feature ablation confirms that ERA5 predictors provide measurable incremental predictive value ($\Delta \text{RMSE} = +1.0084\text{ mm}$).
5. **Clear Operational Boundaries:** Continuous conditional expectation models suffer from severe quantile shrinkage above $115.6\text{ mm}$ ($100\%$ underestimation), demonstrating the necessity of a paired classification architecture for extreme hydrometeorological hazard alerting.

---

### Data and Code Availability
All experimental configurations, processed data dictionaries, serialized models, publication tables, and cryptographic manifests are archived in the project repository. Code and processed data supporting this study are available from the corresponding author upon reasonable request.

---

## References

[1] H. Hersbach, B. Bell, P. Berrisford, S. Hirahara, A. Horányi, J. Muñoz-Sabater, J. Nicolas, C. Peubey, R. Radu, D. Schepers, et al., "The ERA5 global reanalysis," *Quarterly Journal of the Royal Meteorological Society*, vol. 146, no. 730, pp. 1999–2049, 2020. DOI: [10.1002/qj.3803](https://doi.org/10.1002/qj.3803).

[2] J. Muñoz-Sabater, E. Dutra, A. Agustí-Panareda, C. Albergel, G. Arduini, G. Balsamo, S. Boussetta, M. Choulga, S. Harrigan, H. Hersbach, et al., "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications," *Earth System Science Data*, vol. 13, no. 9, pp. 4349–4383, 2021. DOI: [10.5194/essd-13-4349-2021](https://doi.org/10.5194/essd-13-4349-2021).

[3] D. S. Pai, L. Sridhar, M. Rajeevan, O. P. Sreejith, N. S. Satbhai, and B. Mukhopadhyay, "Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region," *MAUSAM*, vol. 65, no. 1, pp. 1–18, 2014. DOI: [10.54302/mausam.v65i1.851](https://doi.org/10.54302/mausam.v65i1.851).

[4] M. Rajeevan, J. Bhate, J. D. Kale, and B. Lal, "High resolution daily gridded rainfall data for the Indian region: Analysis of break and active monsoon spells," *Current Science*, vol. 91, no. 3, pp. 296–306, 2006.

[5] Airbus Defence and Space and European Space Agency, "Copernicus Global Digital Elevation Model (COP-DEM GLO-90) User Guide," *European Space Agency Technical Documentation*, Issue 5.0, pp. 1–38, 2020. DOI: [10.5270/esa-c1qtxnn](https://doi.org/10.5270/esa-c1qtxnn).

[6] S. Gadgil, "The Indian monsoon and its variability," *Annual Review of Earth and Planetary Sciences*, vol. 31, no. 1, pp. 429–467, 2003. DOI: [10.1146/annurev.earth.31.100901.141251](https://doi.org/10.1146/annurev.earth.31.100901.141251).

[7] P. A. Francis and S. Gadgil, "Intense rainfall events over the west coast of India," *Meteorology and Atmospheric Physics*, vol. 94, no. 1–4, pp. 27–42, 2006. DOI: [10.1007/s00703-005-0167-2](https://doi.org/10.1007/s00703-005-0167-2).

[8] R. A. Houze Jr., D. C. Wilton, and B. F. Smull, "Monsoon convection in the Himalayan region as seen by the TRMM Precipitation Radar," *Quarterly Journal of the Royal Meteorological Society*, vol. 133, no. 626, pp. 1389–1411, 2007. DOI: [10.1002/qj.106](https://doi.org/10.1002/qj.106).

[9] J. A. Phadtare, J. K. Fletcher, A. N. Ross, A. G. Turner, and R. K. H. Schiemann, "Froude-number-based rainfall regimes over the Western Ghats mountains of India," *Quarterly Journal of the Royal Meteorological Society*, vol. 148, no. 749, pp. 3738–3755, 2022. DOI: [10.1002/qj.4367](https://doi.org/10.1002/qj.4367).

[10] M. J. Themeßl, A. Gobiet, and A. Leuprecht, "Empirical-statistical downscaling and error correction of daily precipitation from regional climate models," *International Journal of Climatology*, vol. 31, no. 10, pp. 1530–1544, 2011. DOI: [10.1002/joc.2168](https://doi.org/10.1002/joc.2168).

[11] A. J. Cannon, S. R. Sobie, and T. Q. Murdock, "Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes?" *Journal of Climate*, vol. 28, no. 17, pp. 6938–6959, 2015. DOI: [10.1175/JCLI-D-14-00754.1](https://doi.org/10.1175/JCLI-D-14-00754.1).

[12] J. Baño-Medina, R. Manzanas, and J. M. Gutiérrez, "Configuration and intercomparison of deep learning neural models for statistical downscaling," *Geoscientific Model Development*, vol. 13, no. 4, pp. 2109–2124, 2020. DOI: [10.5194/gmd-13-2109-2020](https://doi.org/10.5194/gmd-13-2109-2020).

[13] M. Reichstein, G. Camps-Valls, B. Stevens, M. Jung, J. Denzler, N. Carvalhais, and Prabhat, "Deep learning and process understanding for data-driven Earth system science," *Nature*, vol. 566, no. 7743, pp. 195–204, 2019. DOI: [10.1038/s41586-019-0912-1](https://doi.org/10.1038/s41586-019-0912-1).

[14] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, vol. 30, pp. 3146–3154, 2017.

[15] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems (NeurIPS 2018)*, vol. 31, pp. 6638–6648, 2018.

[16] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).

[17] M. C. K. Tweedie, "An index which distinguishes between some important exponential families," in *Statistics: Applications and New Directions, Proceedings of the Indian Statistical Institute Golden Jubilee International Conference*, J. K. Ghosh and J. Roy, Eds., Calcutta: Indian Statistical Institute, pp. 579–604, 1984.

[18] P. K. Dunn and G. K. Smyth, "Series evaluation of Tweedie exponential dispersion model densities," *Statistics and Computing*, vol. 15, no. 4, pp. 267–280, 2005. DOI: [10.1007/s11222-005-4070-y](https://doi.org/10.1007/s11222-005-4070-y).

[19] J. G. Cragg, "Some statistical models for limited dependent variables with application to the demand for durable goods," *Econometrica*, vol. 39, no. 5, pp. 829–844, 1971. DOI: [10.2307/1909582](https://doi.org/10.2307/1909582).

[20] N. Duan, "Smearing estimate: A nonparametric retransformation method," *Journal of the American Statistical Association*, vol. 78, no. 383, pp. 605–610, 1983. DOI: [10.1080/01621459.1983.10478017](https://doi.org/10.1080/01621459.1983.10478017).

[21] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017)*, pp. 2980–2988, 2017. DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324).

[22] J. E. Nash and J. V. Sutcliffe, "River flow forecasting through conceptual models part I — A discussion of principles," *Journal of Hydrology*, vol. 10, no. 3, pp. 282–290, 1970. DOI: [10.1016/0022-1694(70)90255-6](https://doi.org/10.1016/0022-1694(70)90255-6).

[23] H. V. Gupta, H. Kling, K. K. Yilmaz, and G. F. Martinez, "Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling," *Journal of Hydrology*, vol. 377, no. 1–2, pp. 80–91, 2009. DOI: [10.1016/j.jhydrol.2009.08.003](https://doi.org/10.1016/j.jhydrol.2009.08.003).

[24] H. Kling, M. Fuchs, and M. Paulin, "Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios," *Journal of Hydrology*, vol. 424–425, pp. 264–277, 2012. DOI: [10.1016/j.jhydrol.2012.01.011](https://doi.org/10.1016/j.jhydrol.2012.01.011).

[25] H. R. Künsch, "The jackknife and the bootstrap for general stationary observations," *The Annals of Statistics*, vol. 17, no. 3, pp. 1217–1241, 1989. DOI: [10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265).

[26] D. N. Politis and J. P. Romano, "The stationary bootstrap," *Journal of the American Statistical Association*, vol. 89, no. 428, pp. 1303–1313, 1994. DOI: [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870).

[27] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

[28] A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously," *Journal of Machine Learning Research*, vol. 20, no. 177, pp. 1–81, 2019.

[29] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS 2017)*, vol. 30, pp. 4765–4774, 2017.

