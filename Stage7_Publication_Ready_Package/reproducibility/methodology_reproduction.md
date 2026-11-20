# Methodology and Protocol Reproduction Guide

**Project:** Karnataka Hydrometeorological Daily Rainfall Hindcasting  
**Repository:** `d:\PROJECT\karnataka_rainfall_cds\`  
**Purpose:** End-to-end procedural guide detailing data pipelines, de-accumulation, spatial collocation, model training, evaluation, and statistical uncertainty estimation.

---

## 1. Raw Data Requirements and Sourcing

1. **ERA5-Land Reanalysis (ECMWF / C3S):**
   - Downloaded via `cdsapi` from the Climate Data Store.
   - Domain: Latitude 11.0°N to 19.0°N, Longitude 73.5°E to 79.0°E (covering Karnataka and maritime buffers).
   - Variables: 30 parameters including total precipitation (`tp`), 2m temperature (`t2m`), 2m dewpoint (`d2m`), surface pressure (`sp`), 10m wind speed (`u10`, `v10`), soil moisture (`swvl1` to `swvl4`), surface runoff (`sro`), total runoff (`ro`), skin reservoir content (`src`), and energy fluxes (`ssrd`, `strd`, `slhf`, `sshf`).
   - Temporal range: 84 monthly files from 2019-01 to 2025-12.

2. **IMD Gridded Rainfall Data (IMD Pune):**
   - Source: National Climate Centre, India Meteorological Department, Pune.
   - Format: 0.25° × 0.25° binary gridded daily rainfall (`.grd` files).
   - Extracted using `imdlib` for the Karnataka bounding box.
   - Exactly 252 in-state land grid cells identified via official Karnataka administrative boundary shapefile masking.

3. **Copernicus Digital Elevation Model (GLO-90):**
   - Source: European Space Agency (ESA) Copernicus Programme.
   - Native 90m point elevation extracted at the 252 IMD centroids.
   - Macro-slope (degrees) computed across a 27.75 km baseline (matching the 0.25° IMD grid resolution).

---

## 2. Temporal Alignment and Precipitation De-accumulation

A critical scientific pitfall in reanalysis studies is misaligning the reporting cycle.
1. **The Alignment Problem:** IMD daily rainfall accumulations represent rain falling between **08:30 IST on Day d-1 and 08:30 IST on Day d**, corresponding exactly to **03:00 UTC Day d-1 to 03:00 UTC Day d**.
2. **De-accumulation Protocol:** Raw ERA5-Land precipitation accumulates from 00:00 UTC. To prevent leakage and mismatch, hourly fluxes are de-accumulated:
   $$\Delta P(t) = P(t) - P(t-1)$$
   Hourly values are then aggregated over the 24 hours spanning 03:00 UTC (d-1) to 03:00 UTC (d).
3. **Forensic Verification:** All temporal splits and cumulative counters were audited in Stage 2.1 to guarantee that no cumulative counter resets bleed across calendar days.

---

## 3. Spatial Collocation and the Coastal Fallback Protocol

1. **Native Land Cells:** 249 of the 252 IMD grid cells fell directly over native land cells in the ERA5-Land land-sea mask.
2. **Coastal Fallback Protocol:** Exactly 3 grid cells had centroids located over open Arabian Sea water:
   - Station ID 52 (Lat 13.25°N, Lon 74.75°E, Udupi / Malpe)
   - Station ID 68 (Lat 13.50°N, Lon 74.75°E, Kundapura)
   - Station ID 125 (Lat 14.75°N, Lon 74.25°E, Karwar / Ankola)
3. For these 3 cells, nearest-neighbor inland extraction was applied, and the indicator `is_coastal_fallback = 1` was permanently attached. The remaining 249 cells carry `is_coastal_fallback = 0`.

---

## 4. Leakage-Free Chronological Splitting

1. **Quarantine Buffer:** The first 14 days of 2019 (2019-01-01 to 2019-01-14, 3,528 instances) were quarantined to ensure all 14-day antecedent lags (`api_14d`, `target_rain_lag_14d`, `target_rain_roll_14d_sum`) had 100% complete historical records without imputation.
2. **Training Partition (2019-01-15 to 2022-12-31):** $N = 364,644$ records across 1,447 days.
3. **Validation Partition (2023-01-01 to 2023-12-31):** $N = 91,980$ records across 365 days.
4. **Test Partition [FROZEN] (2024-01-01 to 2025-12-31):** $N = 184,212$ records across 731 days (366 days in 2024 leap year + 365 days in 2025).
5. **Leakage Gate:** Target encodings for `subdivision` and `zone_id` were fitted strictly on Training data.

---

## 5. Model Architecture and Training

1. **Champion Regressor (LightGBM Tweedie):**
   - Objective: Compound Poisson-Gamma (`tweedie`, variance power $p = 1.5$).
   - Log-link: $\ln(\mu) = \sum f_k(X)$.
   - Learning rate: $0.05$, max leaves: $31$, stopping patience: $50$ iterations on Validation RMSE.
2. **Two-Stage Hurdle Architecture:**
   - Stage 1: LightGBM binary classifier ($p^* = 0.44$, calibrated on Validation HSS).
   - Stage 2: LightGBM conditional regressor trained strictly on wet instances ($y > 0.1\text{ mm}$).
   - Formulation: $\hat{y}_{EV} = P(\text{Rain} \ge 0.1) \times \hat{y}_{cond}$.
3. **Extreme Event Classifier (LightGBM Focal Loss):**
   - Custom focal loss objective: $\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$ with $\gamma = 2.0$, $\alpha = 0.75$.
   - Stabilized using positive-definite surrogate Hessian: $h_i = w_i p (1 - p)$.
   - Decision threshold: $p^* = 0.53$, calibrated strictly on 2023 Validation CSI.

---

## 6. Statistical Uncertainty: Paired Block Bootstrap Protocol

To account for spatial cross-correlation and multi-day temporal autocorrelation:
1. Resample blocks of consecutive calendar days with replacement.
2. **Crucial Rule:** When date $d$ is selected, all 252 statewide grid cells for date $d$ are resampled together, perfectly preserving the statewide spatial correlation matrix.
3. Block lengths tested: $L = 7$ days (synoptic memory) and $L = 14$ days (bi-weekly / MJO intraseasonal memory).
4. Replicates: $B = 2,000$ (and $B = 5,000$ in master audit).
5. Paired difference: $\Delta \text{RMSE}^{(b)} = \text{RMSE}_{ML}^{(b)} - \text{RMSE}_{Baseline}^{(b)}$.
6. 95% Confidence Intervals: computed from empirical 2.5th and 97.5th percentiles.
7. Resolution limit: Reported as $p < \frac{1}{B+1}$.
