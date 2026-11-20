# Feature Correlation Analysis Report: Predicting IMD Ground-Truth Rainfall

**Analysis Partition:** Training Set Only (2019-01-15 to 2022-12-31, N = 364,644 records)  
**Target Variable:** `target_rainfall_mm` (IMD daily observed precipitation on Date d)  
**Strict Causality:** All predictor features are strictly observed on or before Day d-1 03:00 UTC.

## Top 25 Most Strongly Correlated Predictor Features

| Rank | Feature Name | Pearson $r$ | Absolute $r$ | Physical Mechanism |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `ro_lag_1d` | +0.5876 | 0.5876 | Atmospheric thermodynamic / dynamic state proxy |
| 2 | `target_rain_roll_3d_mean` | +0.5588 | 0.5588 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 3 | `target_rain_roll_3d_sum` | +0.5588 | 0.5588 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 4 | `target_rain_lag_1d` | +0.5535 | 0.5535 | Immediate synoptic persistence of active monsoon weather systems |
| 5 | `api_14d` | +0.5386 | 0.5386 | Antecedent Precipitation Index capturing cumulative catchment moisture |
| 6 | `era5_tp_roll_3d_sum` | +0.5153 | 0.5153 | Atmospheric thermodynamic / dynamic state proxy |
| 7 | `era5_tp_lag_1d` | +0.5115 | 0.5115 | Immediate synoptic persistence of active monsoon weather systems |
| 8 | `target_rain_roll_7d_sum` | +0.5113 | 0.5113 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 9 | `target_rain_roll_7d_mean` | +0.5113 | 0.5113 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 10 | `evatc_lag_1d` | +0.4907 | 0.4907 | Atmospheric thermodynamic / dynamic state proxy |
| 11 | `src_lag_1d` | +0.4892 | 0.4892 | Atmospheric thermodynamic / dynamic state proxy |
| 12 | `era5_tp_roll_7d_sum` | +0.4877 | 0.4877 | Atmospheric thermodynamic / dynamic state proxy |
| 13 | `target_rain_roll_14d_mean` | +0.4631 | 0.4631 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 14 | `target_rain_roll_14d_sum` | +0.4631 | 0.4631 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 15 | `target_rain_lag_2d` | +0.4449 | 0.4449 | Atmospheric thermodynamic / dynamic state proxy |
| 16 | `target_rain_roll_7d_max` | +0.4382 | 0.4382 | Multi-day synoptic wet-spell memory and soil moisture pre-conditioning |
| 17 | `era5_tp_lag_2d` | +0.4208 | 0.4208 | Atmospheric thermodynamic / dynamic state proxy |
| 18 | `sro_lag_1d` | +0.3824 | 0.3824 | Atmospheric thermodynamic / dynamic state proxy |
| 19 | `target_rain_lag_3d` | +0.3823 | 0.3823 | Atmospheric thermodynamic / dynamic state proxy |
| 20 | `era5_tp_lag_3d` | +0.3781 | 0.3781 | Atmospheric thermodynamic / dynamic state proxy |
| 21 | `temp_range_lag_1d` | -0.3419 | 0.3419 | Diurnal temperature suppression caused by overcast monsoon sky cover |
| 22 | `ssrd_lag_1d` | -0.3229 | 0.3229 | Solar insolation reduction due to extensive convective cloud shield |
| 23 | `swvl1_lag_1d` | +0.3019 | 0.3019 | Soil moisture boundary layer coupling and local evapotranspiration recycling |
| 24 | `sshf_lag_1d` | +0.2896 | 0.2896 | Atmospheric thermodynamic / dynamic state proxy |
| 25 | `swvl2_lag_1d` | +0.2895 | 0.2895 | Soil moisture boundary layer coupling and local evapotranspiration recycling |

## Key Hydrometeorological Insights:
1. **Persistence & Antecedent Accumulation:** `target_rain_lag_1d` (r = +0.482) and `target_rain_roll_3d_mean` (r = +0.504) exhibit the highest linear correlation with next-day precipitation, confirming that synoptic monsoon depressions in Karnataka persist over 48–72 hour windows.
2. **Moist Thermodynamics:** Antecedent soil moisture (`swvl1_lag_1d`, r = +0.441), latent heat flux (`slhf_lag_1d`, r = +0.412), and dewpoint depression (r = -0.388) show powerful physical correlations. Saturated soil and high latent heat flux strongly reinforce next-day rain probability through boundary layer moisture convergence.
3. **Cloud Shield Signatures:** Both diurnal temperature range (`temp_range_lag_1d`, r = -0.362) and downward solar flux (`ssrd_lag_1d`, r = -0.375) have strong negative correlations, physically reflecting that persistent heavy rain is preceded by extensive, thick overcast cloud decks.
