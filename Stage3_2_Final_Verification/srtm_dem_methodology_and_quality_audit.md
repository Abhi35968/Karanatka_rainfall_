# SRTM DEM TOPOGRAPHIC EXTRACTION, SLOPE CALCULATION, AND DATA QUALITY AUDIT

**Project:** Machine-Learning-Guided Daily Rainfall Estimation and Forecasting for Karnataka Using High-Resolution ERA5-Land Reanalysis and IMD Observations  
**Document Code:** AUDIT-DEM-SLOPE-VER-1.0  
**Date:** November 2026  
**Auditor:** Senior Hydrometeorological Research Scientist and Geospatial Data Engineer  
**Status:** COMPLETE & CERTIFIED  

---

## 1. Executive Summary & Objective

In Stage 3 feature engineering, initial terrain elevation (`elevation_m`) and regional topographic slope (`slope_deg`) were computed using a barometric hypsometric proxy derived from multi-year ERA5-Land surface pressure (`sp_mean_hpa`). 

This document provides a thorough audit of the actual Digital Elevation Model (DEM) data sources, coordinate extraction mechanics, slope estimation algorithms, edge-case handlings, and quality control checks for all 252 IMD grid-cell coordinates spanning the state of Karnataka. Furthermore, it addresses the scientific distinction between broad hypsometric correlation and derivative slope accuracy, establishing the explicit requirements for Stage 4 modeling.

---

## 2. DEM Data Source, Provenance, and Spatial Specifications

### 2.1. Dataset Specifications
* **Dataset / Product Name:** NASA / USGS Shuttle Radar Topography Mission (SRTM) Global 3-arc-second Digital Elevation Model (SRTM3 v4.1 / SRTM Non-Void Filled v3.0).
* **Provider / Custodian:** National Aeronautics and Space Administration (NASA), Jet Propulsion Laboratory (JPL), United States Geological Survey (USGS), and CGIAR Consortium for Spatial Information (CGIAR-CSI).
* **API Providers & Source URLs:**
  1. *Primary Query Service:* Open-Elevation Global API (`https://api.open-elevation.com/api/v1/lookup`), serving the processed SRTM3 v4.1 GeoTIFF tile collection.
  2. *Secondary Fallback Service:* Open-Meteo Global Elevation API (`https://api.open-meteo.com/v1/elevation`), which queries Copernicus DEM (GLO-90, 90 m) integrated with NASA SRTM 90 m.
* **Native Spatial Resolution:** 3 arc-seconds ($\approx 0.0008333^\circ$), corresponding to approximately $90\text{ meters} \times 90\text{ meters}$ at the equator and $\sim 87\text{ meters}$ meridionally at $15^\circ\text{N}$.
* **Coordinate Reference System (CRS):**
  * *Horizontal Datum:* World Geodetic System 1984 (WGS 84, EPSG:4326), geographic latitude/longitude in decimal degrees.
  * *Vertical Datum:* Earth Gravitational Model 1996 (EGM96) geoid, orthometric height above mean sea level in meters ($m$).

---

## 3. Sampling Mechanics & Coastline / Nodata Handling

### 3.1. Coordinate-to-Pixel Sampling Methodology
* **Target Coordinates:** 252 regular grid cell centroids corresponding to the IMD $0.25^\circ \times 0.25^\circ$ daily gridded rainfall network covering Karnataka (latitudes: $11.75^\circ\text{N}$ to $18.50^\circ\text{N}$; longitudes: $74.25^\circ\text{E}$ to $77.75^\circ\text{E}$).
* **Sampling Scheme:** Exact centroid point query. The query coordinates $(lat_i, lon_i)$ were submitted directly to the API endpoints without pre-resampling or raster decimation. The underlying service performs bilinear interpolation between adjacent 90-meter DEM raster cells to estimate point elevations.
* **Area vs. Point Sampling:** It is critical to note that the returned DEM elevation represents the **point terrain elevation at the $0.25^\circ$ cell centroid**, rather than an area-averaged integral over the full $27.75\text{ km} \times 27.75\text{ km}$ grid footprint.

### 3.2. Nodata, Coastline, and Ocean-Cell Handling
* **Missing Value Inspection:** Across all 252 station coordinates, exactly **0 nodata values** (such as $-9999$ or `NaN`) were encountered.
* **Coastal Boundary Cells:** Station 104 located on the immediate Arabian Sea coastline in Uttara Kannada ($14.25^\circ\text{N}, 74.50^\circ\text{E}$) returned an elevation of **$1.12\text{ m}$**. All remaining coastal stations (Karwar, Kumta, Honnavar, Bhatkal, Kundapura, Udupi, Mangalore) returned physically valid elevations between $1.12\text{ m}$ and $108.5\text{ m}$.
* **Inland Water Bodies:** In the SRTM3 v4.1 product, major inland water surfaces (reservoirs such as KRS, Tungabhadra, Almatti) are flattened and void-filled using auxiliary cartographic vectors (SRTM Water Body Data / SWBD). No anomalous negative land elevations were detected across the inland coordinates.

---

## 4. Topographic Slope Calculation Methodology

### 4.1. Mathematical Formulation
Topographic slope represents the magnitude of the spatial elevation gradient vector $\nabla z = \left( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial y} \right)$.

In `audit_dem_elevation_slope.py`, slope is evaluated across the $0.25^\circ$ regional grid using a central finite-difference operator on 4-connected cardinal neighbours:

$$\Delta y = 0.25^\circ \times 111,000\text{ m/deg} = 27,750\text{ m}$$
$$\Delta x_i = 0.25^\circ \times 111,000\text{ m/deg} \times \cos(\text{latitude}_i)$$

1. **Meridional Gradient ($\partial z / \partial y$):**
   $$\frac{\partial z}{\partial y} = \frac{z(\text{lat}_i + 0.25^\circ, \text{lon}_i) - z(\text{lat}_i - 0.25^\circ, \text{lon}_i)}{2 \Delta y}$$
2. **Zonal Gradient ($\partial z / \partial x$):**
   $$\frac{\partial z}{\partial x} = \frac{z(\text{lat}_i, \text{lon}_i + 0.25^\circ) - z(\text{lat}_i, \text{lon}_i - 0.25^\circ)}{2 \Delta x_i}$$
3. **Slope Magnitude:**
   $$\text{Slope Magnitude } S = \sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} \quad [\text{dimensionless, m/m}]$$
   $$\text{Slope in Degrees } \theta = \arctan(S) \times \frac{180^\circ}{\pi} \quad [^\circ]$$

* **Units:** Explicitly expressed in **degrees** ($^\circ$), NOT percent.
* **Boundary Condition:** For peripheral stations where an adjacent cardinal neighbor lies outside the 252 in-state station mask (e.g., along the western maritime boundary or state borders), boundary values are clamped using one-sided differences. For completely isolated single-cell boundary corners where two opposite cardinal cells are absent, the gradient clamps to $0.000^\circ$ (which occurs at Station 104).

### 4.2. Critical Physical Clarification: Regional Macro-Gradient vs. Micro-Scale DEM Slope
It is vital to distinguish between:
1. **Regional Macro-Scale Slope ($27.75\text{ km}$ spacing):** The slope computed here measures the **broad regional orographic incline** across a $28\text{ km}$ baseline. It quantifies the macro-scale ramp of the Western Ghats rising from the Arabian Sea up to the Deccan plateau, which is the primary driver of synoptic-scale orographic uplift and adiabatic cooling during the Southwest Monsoon.
2. **Local Micro-Scale DEM Slope ($90\text{ m}$ spacing):** If slope were calculated directly on the native 90m SRTM raster using Horn's or Zevenbergen-Thorne's algorithm over a $3 \times 3$ pixel window ($270\text{ m}$ baseline), local slopes in the Western Ghats canyons and cliffs would frequently reach $25^\circ\text{–}60^\circ$.
* **Conclusion:** The feature must be formally documented as the **Regional Macro-Topographic Gradient on 0.25° Grid**, to avoid misrepresenting it as high-resolution hillside steepness.

---

## 5. Quantitative Audit of All 252 IMD Stations

### 5.1. DEM Elevation Distribution
* **Total In-State Stations:** 252
* **Missing / NaN Values:** 0
* **Minimum Elevation:** $1.12\text{ m}$ (Station 104, Bhatkal coast, $14.25^\circ\text{N}, 74.50^\circ\text{E}$)
* **5th Percentile:** $108.52\text{ m}$
* **Median (50th Percentile):** $615.85\text{ m}$
* **Mean Elevation:** $610.01\text{ m} \pm 209.80\text{ m}$
* **95th Percentile:** $911.51\text{ m}$
* **Maximum Elevation:** $1,599.54\text{ m}$ (Station 72, Bababudangiri/Kudremukh range in Chikkamagaluru, $13.50^\circ\text{N}, 75.75^\circ\text{E}$)

### 5.2. DEM Regional Slope Distribution
* **Minimum Slope:** $0.000^\circ$ (boundary clamped cell)
* **Median Slope:** $0.114^\circ$
* **Mean Slope:** $0.186^\circ \pm 0.217^\circ$
* **90th Percentile:** $0.387^\circ$
* **95th Percentile:** $0.811^\circ$
* **Maximum Slope:** $1.084^\circ$ (Station 106, Western Ghats escarpment in Uttara Kannada, $14.25^\circ\text{N}, 75.00^\circ\text{E}$)

### 5.3. Comparison Across Meteorological Subdivisions

| Meteorological Subdivision | Station Count | DEM Elevation Mean (m) | Pressure Proxy Mean (m) | DEM Slope Mean (deg) | Proxy Slope Mean (deg) | DEM Slope Max (deg) | Proxy Slope Max (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coastal Karnataka** | 13 | 109.59 | 196.90 | **0.532°** | 0.449° | **1.084°** | 0.943° |
| **South Interior - Malnad** | 35 | 653.36 | 633.95 | **0.402°** | 0.307° | **0.887°** | 0.779° |
| **South Interior - Plain** | 86 | 756.98 | 756.79 | **0.148°** | 0.121° | 0.330° | 0.330° |
| **North Interior Karnataka**| 118 | 545.18 | 553.21 | **0.110°** | 0.083° | 0.286° | 0.231° |

---

## 6. Evaluation of the Pressure-Derived Elevation Proxy

### 6.1. The Scientific Fallacy of "High Correlation Proves Slope Accuracy"
In Stage 3.1, a high Pearson correlation ($r = 0.9604$) was observed between the surface-pressure hypsometric proxy and true DEM elevation. However, **a high correlation in zeroth-order elevation does NOT imply accuracy in spatial derivatives (slope)**:

1. **Derivative Sensitivity to High-Frequency Truncation:**
   $$\nabla z_{\text{proxy}} - \nabla z_{\text{dem}} = \nabla (z_{\text{proxy}} - z_{\text{dem}})$$
   Even if the absolute elevation difference is bounded within $5\%$, high-frequency spatial discrepancies between adjacent cells will induce large errors in finite differences $(\partial z / \partial x, \partial z / \partial y)$.
2. **Empirical Evidence of Distortion:**
   * In the steep Western Ghats (Malnad), the true DEM slope has a standard deviation of $0.334^\circ$, whereas the pressure proxy slope compresses the standard deviation to $0.201^\circ$ (a $40\%$ underestimation of topographic dispersion).
   * In Coastal Karnataka, the maximum true DEM slope reaches $1.084^\circ$, whereas the proxy truncates at $0.943^\circ$.
   * At mountain peaks (Station 72), the pressure proxy underestimates true elevation by $-527.1\text{ m}$ ($1,072.4\text{ m}$ vs $1,599.5\text{ m}$) because ERA5-Land's spectral truncations filter out sub-grid topographic pinnacles.

### 6.2. Status in Stage 3 Parquet Dataset vs. Requirements for Stage 4
* **Current Parquet Status:** In `stage3_dataset.parquet`, the column named `elevation_m` currently contains the **barometric pressure-derived proxy**, and `slope_deg` was derived from this proxy.
* **Mandatory Requirement for Stage 4:** 
  1. True DEM elevation must be explicitly introduced as `elevation_dem_m` or replace `elevation_m`.
  2. True DEM regional slope must be explicitly introduced as `slope_dem_deg` or replace `slope_deg`.
  3. If the barometric proxy is retained in feature sets, it must be renamed to `elevation_pressure_proxy_m` and justified as a dynamic proxy for multi-year surface pressure rather than true terrain elevation.
  4. Machine learning models in Stage 4 will train on `elevation_dem_m` and `slope_dem_deg` as the verified geophysical predictors.

---

## 7. Quality Certification

* **All 252 IMD coordinates verified:** YES
* **Zero missing values in SRTM DEM elevations:** YES
* **Physical plausibility confirmed:** YES (Elevations: $1.1\text{ m}$ to $1,599.5\text{ m}$; Slopes: $0.0^\circ$ to $1.08^\circ$)
* **Scientific caveat on macro-gradient documented:** YES
* **Data file generated:** [`Stage3_2_Final_Verification/data/station_topography_verified.csv`](file:///d:/PROJECT/karnataka_rainfall_cds/Stage3_2_Final_Verification/data/station_topography_verified.csv) containing both DEM and proxy metrics for all 252 stations.

