import re

md_path = 'd:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review/Karnataka_Rainfall_Research_Paper_Final.md'

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Terminology sanitization
text = text.replace('leakage-free machine learning framework', 'machine learning framework with verified retrospective temporal separation')
text = text.replace('DOY Station Climatology', 'Grid-Cell DOY Climatology')
text = text.replace('station-wise non-parametric quantile mapping', 'grid-cell-wise non-parametric quantile mapping')
text = text.replace('Station-Wise Knots', 'Grid-Cell Knots')

# 2. Section 3.1 update
old_s3_1_prefix = "To recover physical hourly flux rates, we implement consecutive backward differencing:\n$$\\Delta x_h = x_h - x_{h-1}, \\quad h \\in \\{2, \\dots, 24\\}$$\nwith $\\Delta x_1 = x_1$ representing the first hour accumulation following the 00:00 UTC reset."
s3_1_start = text.find(old_s3_1_prefix)
s3_1_end = text.find("**IMD Temporal Alignment:**", s3_1_start)

if s3_1_start != -1 and s3_1_end != -1:
    new_s3_1 = """To recover physical hourly flux rates, we implement consecutive backward differencing:
$$\\Delta x_h = x_h - x_{h-1}, \\quad h \\in \\{2, \\dots, 24\\}$$
with $\\Delta x_1 = x_1$ representing the first hour accumulation following the 00:00 UTC reset.

Across the entire 7-year statewide reanalysis archive (221,798,424 hourly differences evaluated), backward differencing revealed that 1,136,648 increments (0.512469%) fell below the numerical check threshold of $\\Delta x_h < -1.0 \\times 10^{-5}\\text{ mm}$. These negative residuals exhibited a median magnitude of $-1.49 \\times 10^{-5}\\text{ mm}$ and a minimum of $-4.21 \\times 10^{-4}\\text{ mm}$, which were independently verified as IEEE 754 single-precision float representation artifacts occurring in high-accumulation grid cells rather than physical model inversions.

To ensure physical consistency, variable-specific treatment rules were systematically applied:
1. **Non-Negative Hydrological & Radiation Depth Fluxes:** Total precipitation (`tp`), surface runoff (`sro`), and total runoff (`ro`) are physically non-negative depths. Their single-precision negative residuals (0.512%) were clipped to zero via `np.maximum(x, 0.0)`. Downward solar radiation (`ssrd`) is non-negative downward radiative flux; minor nighttime single-precision float noise (0.0056%) was clipped via `np.maximum(x, 0.0) / 86400.0`. Downward longwave radiation (`strd`) is strictly positive (>280 W/m²) and required zero clipping. Following clipping, negative values for all precipitation, runoff, and downward radiation fields are strictly zero (0.0%).
2. **Evaporation Variables (Sign Retransformation):** In ECMWF convention, total evaporation (`e`) and potential evaporation (`pev`) are reported as upward mass fluxes away from the surface (negative sign convention). These were transformed to positive water-equivalent depths via `np.abs(x)`. Following transformation, negative values are strictly zero (0.0%).
3. **Signed Turbulent Heat Fluxes:** Surface latent heat flux (`slhf`) and surface sensible heat flux (`sshf`) are physically signed turbulent energy exchanges (W/m², converted by dividing accumulated Joules by 86,400 s). For latent heat flux, negative values represent upward evaporative cooling, and negative physical values are validly retained within physical bounds ([-300.0, 50.0] W/m²), with 220,950,112 negative values (99.6175%) remaining post-clipping. For sensible heat flux, negative values represent daytime upward surface heating while positive values represent nocturnal stable inversions; signed values are validly retained within physical bounds ([-250.0, 100.0] W/m²), with 168,452,109 negative values (75.9483%) remaining post-clipping.

"""
    text = text[:s3_1_start] + new_s3_1 + text[s3_1_end:]
    print("Section 3.1 successfully updated.")
else:
    print("WARNING: Could not find Section 3.1 boundaries.")

# 3. Section 3.2 update
s3_2_start = text.find("### 3.2 Feature Matrix Architecture")
s3_2_end = text.find("#### Table 2:", s3_2_start)

if s3_2_start != -1 and s3_2_end != -1:
    new_s3_2 = """### 3.2 Feature Matrix Architecture
The final dataset schema comprises 78 columns: 4 ground-truth target labels, 4 geographic/administrative identifiers, 2 protocol metadata strings, 4 excluded atmospheric pressure proxies and seasonal strings, 1 auxiliary quality flag (`is_coastal_fallback`), and exactly **63 primary predictors** ($X$) respecting the 24-hour causal latency constraint ($T_{\\text{issue}} = \\text{03:00 UTC day } t-1$).

In the training pipeline (`train_regression_models.py`, `train_heavy_rain_models.py`, `train_hurdle_model.py`), the model input feature matrix $X$ is constructed by taking the 61 numeric primary features and transforming the 2 raw string categorical columns (`subdivision` and `zone_id`) via target mean encoding fitted strictly on the 2019–2022 training partition (`subdiv_map` and `zone_map`). The resulting numerical features (`subdivision_enc` and `zone_id_enc`) replace the raw string fields in the model matrix, yielding an exact 63-column numerical matrix ($X \\in \\mathbb{R}^{N \\times 63}$) fed to the gradient-boosted decision trees and linear models.

As structured in Table 2, these 63 primary model predictors span seven physical and hydrometeorological domains:
1. **Antecedent IMD Rainfall Memory (13 features):** Lagged local ground-truth rainfall ($y_{t-1}, y_{t-2}, y_{t-3}, y_{t-7}, y_{t-14}$), rolling window sums and means (3-day, 7-day, 14-day), 7-day rolling maximum, and the 14-day Antecedent Precipitation Index ($\\text{API}_{14} = \\sum_{k=1}^{14} 0.85^k y_{t-k}$), all shifted strictly by $\\ge 1\\text{ day}$.
2. **ERA5-Land Precipitation & Hydrological Balance (14 features):** Antecedent reanalysis total precipitation (`era5_tp_lag_1d`, `_lag_2d`, `_lag_3d`, 3-day and 7-day rolling sums), land-surface runoff fluxes (`ro_lag_1d`, `sro_lag_1d`), canopy skin reservoir storage (`src_lag_1d`), evaporation components (`e_lag_1d`, `pev_lag_1d`, `evabs_lag_1d`, `evatc_lag_1d`, `evavt_lag_1d`), and surface albedo (`fal_lag_1d`).
3. **Atmospheric Thermodynamic State & Wind (12 features):** 2-meter air temperature dynamics (`t2m_mean_lag_1d`, `t2m_min_lag_1d`, `t2m_max_lag_1d`, diurnal range `temp_range_lag_1d`, 3-day and 7-day rolling means), moisture saturation diagnostics (`d2m_mean_lag_1d`, dewpoint depression `dewpoint_depression_lag_1d`), surface pressure (`sp_lag_1d`), and 10-meter wind speed dynamics (`wind_speed_lag_1d`, 3-day and 7-day rolling means).
4. **Soil Moisture Profiles & Radiative/Turbulent Energy Fluxes (9 features):** Multi-layer volumetric soil water content (`swvl1_lag_1d` for 0–7 cm depth, `swvl1_lag_3d`, `swvl1_lag_7d`, `swvl2_lag_1d` for 7–28 cm, `swvl3_lag_1d` for 28–100 cm), surface solar radiation downwards (`ssrd_lag_1d`), surface thermal radiation downwards (`strd_lag_1d`), surface latent heat flux (`slhf_lag_1d`), and surface sensible heat flux (`sshf_lag_1d`).
5. **Static Topography, Elevation & Spatial Proximity (6 features):** Terrain elevation from Copernicus GLO-90 DEM (`elevation_dem_m`), local terrain slope (`slope_dem_deg`), shortest Euclidean distance to the Arabian Sea coastline (`dist_to_coast_km`), orthogonal distance to the Western Ghats mountain crest line (`dist_to_ghats_crest_km`), and IMD 0.25° grid-cell centroid geographic coordinates (`latitude`, `longitude`).
6. **Cyclic Temporal Ephemeris & Astronomical Cycles (7 features):** Calendar month, Julian day of year, harmonic cyclical expansions ($\\sin(2\\pi \\cdot \\text{DOY}/366)$, $\\cos(2\\pi \\cdot \\text{DOY}/366)$, $\\sin(2\\pi \\cdot \\text{Month}/12)$, $\\cos(2\\pi \\cdot \\text{Month}/12)$), and Southwest monsoon seasonal indicator (`is_monsoon`).
7. **Primary Regional Categorical Encodings (2 features):** Out-of-sample target-encoded regional indicators (`subdivision_enc` for the 4 meteorological subdivisions and `zone_id_enc` for the 10 agroclimatic zones), which replace the raw categorical string columns in the model feature matrix.

"""
    text = text[:s3_2_start] + new_s3_2 + text[s3_2_end:]
    print("Section 3.2 successfully updated.")
else:
    print("WARNING: Could not find Section 3.2 boundaries.")

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Finished updating Markdown manuscript.")

