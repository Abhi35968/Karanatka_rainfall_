# TOPOGRAPHIC SLOPE FORMULATION AND MATHEMATICAL VERIFICATION REPORT

**Project:** Machine-Learning-Guided Daily Rainfall Estimation and Forecasting for Karnataka Using High-Resolution ERA5-Land Reanalysis and IMD Observations  
**Document Code:** SLOPE-FORMULA-VERIFY-1.0  
**Date:** November 2026  
**Auditor:** Senior Hydrometeorological Research Scientist and Geospatial Data Engineer  
**Status:** MATHEMATICALLY AUDITED & CERTIFIED  

---

## 1. Executive Summary & Purpose

In Stage 3 and Stage 3.1, regional topographic slope was introduced as a static geospatial feature (`slope_dem_deg`). This report provides an independent mathematical verification of the finite-difference gradient operators, coordinate distance calculations, boundary condition handling, and angular unit conversions.

### Key Conclusions:
1. **Physical Meaning:** The slope evaluated across the $0.25^\circ \times 0.25^\circ$ IMD grid represents a **regional macro-topographic inclination across a $27.75\text{ km}$ baseline**, NOT the micro-scale hillside slope of local valleys or cliffs.
2. **Formula Soundness:** Central difference denominators correctly incorporate the factor of $2.0 \times \Delta y$ and $2.0 \times \Delta x$ with latitude-dependent longitudinal convergence.
3. **Boundary Condition Rectification:** A critical defect in the previous Stage 3.1 clamping code was identified: clamping missing boundary cells to `c_elev` while dividing by $2 \Delta$ halved the boundary gradient. The mathematically rigorous one-sided difference ($\Delta z / \Delta$) has been codified and implemented.

---

## 2. Mathematical Formulation of the Regional Gradient Operator

### 2.1. Coordinate Metric & Longitudinal Convergence
Let an IMD grid cell centroid be located at geographic coordinates $(\phi_i, \lambda_j)$, where $\phi_i$ is latitude and $\lambda_j$ is longitude in decimal degrees.

On a spherical Earth approximation (WGS 84 mean radius $R = 6,371\text{ km}$, yielding $1^\circ \approx 111.0\text{ km}$):
* **Meridional Grid Spacing ($\Delta y$):** Constant across all latitudes:
  $$\Delta y = 0.25^\circ \times 111,000\text{ m/deg} = 27,750\text{ meters}$$
* **Zonal Grid Spacing ($\Delta x_i$):** Latitude-dependent due to meridian convergence:
  $$\Delta x_i = 0.25^\circ \times 111,000\text{ m/deg} \times \cos(\phi_i) = 27,750 \cos(\phi_i)\text{ meters}$$

At the southern boundary of Karnataka ($\phi \approx 11.75^\circ\text{N}$):
$$\Delta x = 27,750 \cos(11.75^\circ) = 27,750 \times 0.97904 = 27,168.4\text{ m}$$
At the northern boundary of Karnataka ($\phi \approx 18.50^\circ\text{N}$):
$$\Delta x = 27,750 \cos(18.50^\circ) = 27,750 \times 0.94832 = 26,316.0\text{ m}$$
The longitudinal distance shrinks by $852.4\text{ m}$ ($3.1\%$) from south to north, confirming the necessity of latitude-dependent metric scaling.

---

### 2.2. Finite Difference Operators

#### A. Interior Grid Cells (4 Cardinal Neighbors Present)
When both adjacent neighbors along a cardinal axis belong to the 252 in-state station mask:
* **Zonal Gradient ($\partial z / \partial x$):**
  $$\left(\frac{\partial z}{\partial x}\right)_i = \frac{z(\phi_i, \lambda_j + 0.25^\circ) - z(\phi_i, \lambda_j - 0.25^\circ)}{2 \Delta x_i} = \frac{z_{E} - z_{W}}{2 \Delta x_i}$$
* **Meridional Gradient ($\partial z / \partial y$):**
  $$\left(\frac{\partial z}{\partial y}\right)_i = \frac{z(\phi_i + 0.25^\circ, \lambda_j) - z(\phi_i - 0.25^\circ, \lambda_j)}{2 \Delta y} = \frac{z_{N} - z_{S}}{2 \Delta y}$$

*Factor of 2 Check:* The distance from $(\lambda_j - 0.25^\circ)$ to $(\lambda_j + 0.25^\circ)$ spans two grid intervals ($2 \Delta x_i$). The denominator $2 \Delta x_i$ is mathematically exact.

#### B. Boundary Cells (One Cardinal Neighbor Present)
For peripheral cells along state borders or the Arabian Sea coastline:
* If East neighbor is missing (e.g. eastern state border):
  $$\left(\frac{\partial z}{\partial x}\right)_i = \frac{z_{C} - z_{W}}{\Delta x_i} \quad (\text{One-sided backward difference over } 1 \text{ interval})$$
* If West neighbor is missing (e.g. coastal stations facing Arabian Sea):
  $$\left(\frac{\partial z}{\partial x}\right)_i = \frac{z_{E} - z_{C}}{\Delta x_i} \quad (\text{One-sided forward difference over } 1 \text{ interval})$$
* If North neighbor is missing:
  $$\left(\frac{\partial z}{\partial y}\right)_i = \frac{z_{C} - z_{S}}{\Delta y}$$
* If South neighbor is missing:
  $$\left(\frac{\partial z}{\partial y}\right)_i = \frac{z_{N} - z_{C}}{\Delta y}$$
* If both neighbors on an axis are absent (isolated single-cell horn):
  $$\frac{\partial z}{\partial x} = 0 \quad \text{or} \quad \frac{\partial z}{\partial y} = 0$$

*Audit of Prior Code Defect:* In `audit_dem_elevation_slope.py`, when a neighbor was missing, the code performed `(z_E - z_C) / (2 * dx)`. This inadvertently divided a 1-interval distance by a 2-interval denominator, cutting boundary gradients in half. The harmonized code corrects this to $\Delta x_i$ or $\Delta y$.

---

### 2.3. Vector Magnitude and Angular Conversion
The total gradient magnitude $S$ is the Euclidean norm of the directional derivative vector:

$$S = \sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} \quad \left[\frac{\text{meters rise}}{\text{meters run}}, \text{ dimensionless}\right]$$

The angular slope $\theta$ in degrees is:

$$\theta = \arctan(S) \times \frac{180^\circ}{\pi} \quad [^\circ]$$

*Dimensional Consistency:*
* $[z] = \text{meters}$
* $[x] = [y] = \text{meters}$
* $[\partial z / \partial x] = \text{m} / \text{m} = 1$ (dimensionless)
* $[S] = \sqrt{1^2 + 1^2} = 1$ (dimensionless)
* $\arctan(S) \in [0, \pi/2)\text{ radians}$
* $\theta = \arctan(S) \times \frac{180^\circ}{\pi} \in [0^\circ, 90^\circ)$

The calculation is physically consistent and strictly in **degrees ($^\circ$)**, not percent or radians.

---

## 3. Independent Numerical Traceability Example

To confirm algorithmic precision, we trace a controlled manual test case:

### Test Case Parameters:
* Centroid Coordinate: $\phi = 15.00^\circ\text{N}, \lambda = 75.00^\circ\text{E}$
* Center Elevation: $z_C = 600.00\text{ m}$
* North Neighbor $(15.25^\circ\text{N}, 75.00^\circ\text{E})$: $z_N = 650.00\text{ m}$
* South Neighbor $(14.75^\circ\text{N}, 75.00^\circ\text{E})$: $z_S = 550.00\text{ m}$
* East Neighbor $(15.00^\circ\text{N}, 75.25^\circ\text{E})$: $z_E = 620.00\text{ m}$
* West Neighbor $(15.00^\circ\text{N}, 74.75^\circ\text{E})$: $z_W = 580.00\text{ m}$

### Manual Calculation:
1. **Grid Spacings:**
   $$\Delta y = 27,750.00\text{ m} \implies 2 \Delta y = 55,500.00\text{ m}$$
   $$\cos(15^\circ) = 0.965925826$$
   $$\Delta x = 27,750 \times 0.965925826 = 26,804.44\text{ m} \implies 2 \Delta x = 53,608.88\text{ m}$$

2. **Partial Derivatives:**
   $$\frac{\partial z}{\partial y} = \frac{650.00 - 550.00}{55,500.00} = \frac{100.00}{55,500.00} = 0.001801802\text{ m/m}$$
   $$\frac{\partial z}{\partial x} = \frac{620.00 - 580.00}{53,608.88} = \frac{40.00}{53,608.88} = 0.000746145\text{ m/m}$$

3. **Slope Vector Magnitude:**
   $$S = \sqrt{(0.000746145)^2 + (0.001801802)^2} = \sqrt{5.56732 \times 10^{-7} + 3.24649 \times 10^{-6}} = \sqrt{3.80322 \times 10^{-6}} = 0.001950185$$

4. **Slope in Degrees:**
   $$\theta = \arctan(0.001950185) \times \frac{180^\circ}{\pi} = 0.001950182 \times 57.2957795 = \mathbf{0.111736^\circ}$$

### Python Implementation Verification:
Running the identical inputs through `build_terrain_audit.py`:
$$\theta_{\text{code}} = \mathbf{0.1117^\circ}$$
The automated routine and manual derivation agree to 4 decimal places.

---

## 4. Boundary Sensitivity Trace: Coastal Escarpment

Consider Station 104 on the Bhatkal coast ($14.25^\circ\text{N}, 74.50^\circ\text{E}, z_C = 0.0\text{ m}$ Copernicus / $1.12\text{ m}$ OE):
* Western Neighbor: Arabian Sea (outside state mask, non-existent station).
* Eastern Neighbor (Station 105, $14.25^\circ\text{N}, 74.75^\circ\text{E}$): Foothills of Western Ghats ($z_E = 485.0\text{ m}$).
* Grid Distance: $\Delta x = 27,750 \cos(14.25^\circ) = 26,903.5\text{ m}$.

* **Correct One-Sided Formulation:**
  $$\frac{\partial z}{\partial x} = \frac{485.0 - 0.0}{26,903.5} = 0.018027 \implies \theta_x = \arctan(0.018027) \times \frac{180^\circ}{\pi} = \mathbf{1.032^\circ}$$
* **Old Buggy Clamp Formulation:**
  $$\frac{\partial z}{\partial x} = \frac{485.0 - 0.0}{2 \times 26,903.5} = 0.009014 \implies \theta_x = \mathbf{0.516^\circ}$$

The corrected formulation accurately captures the steep $1.03^\circ$ regional orographic incline facing incoming monsoonal Arabian Sea winds.

---

## 5. Physical Scale Representation

It is a non-negotiable reporting standard that this feature must be titled:
> **"Regional Macro-Topographic Gradient on 0.25° Grid"**

* It reflects synoptic-scale terrain elevation change averaged over $28\text{ kilometers}$.
* It captures regional orographic barriers, adiabatic cooling ramps, and rain-shadow leeward subsidence.
* It does **not** reflect micro-scale hillside slope ($20^\circ\text{–}60^\circ$) or ravine steepness.

