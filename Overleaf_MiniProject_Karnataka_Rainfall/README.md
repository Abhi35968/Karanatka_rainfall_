# Karnataka Daily Rainfall Reconstruction: Overleaf Project Package

This folder contains the complete, self-contained LaTeX source code, figures, bibliography, and institutional assets for the Mini Project Report:

**Title:** Predictive Modeling, Numerical Reanalysis Verification, and Agro-Climatic Bias Correction of Daily Rainfall Across Karnataka, India  
**Authors:** Abhinandan Kumar (23BDS001), Mihir (23BCS080), Rishabh Verma (23BCS107)  
**Supervisor:** Dr. Sunil C K, Assistant Professor  
**Department:** Department of Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology Dharwad  
**Date:** November 2026

---

## How to Open in Overleaf

1. Download the file `Overleaf_MiniProject_Karnataka_Rainfall.zip` (located in this directory or in your Downloads folder).
2. Go to [Overleaf](https://www.overleaf.com) and log in.
3. Click **New Project** -> **Upload Project**.
4. Select or drag-and-drop `Overleaf_MiniProject_Karnataka_Rainfall.zip`.
5. Overleaf will automatically unpack all files and open `main.tex`.
6. Ensure the compiler is set to **pdfLaTeX** and TeX Live version is **2022** or newer (default).
7. Click **Recompile** to generate the complete PDF report.

---

## Package Contents

- `main.tex`: Master LaTeX document formatted strictly according to the institutional IIIT Dharwad Mini Project Template.
- `Myreferences.bib`: Complete BibTeX database containing all 29 cited peer-reviewed articles and dataset publications.
- `iiIT-Dharwad-Logo-horizontal-500x154[1].png`: Official IIIT Dharwad horizontal logo embedded in the title page.
- `figures/`: High-resolution figures (both PNG and PDF vector formats):
  - `figure1_study_area`: Physiographic map of Karnataka with DEM elevation and 252 IMD grid cells.
  - `figure2_methodology`: End-to-end retrospective causal architecture and feature engineering workflow.
  - `figure3_observed_predicted`: Density parity hexbin of observed vs. predicted daily rainfall.
  - `figure4_model_benchmarking`: Multi-metric radar and bar comparisons across Tier 0 to Tier 4 models.
  - `figure5_heavy_rain_detection`: Precision-Recall curves and contingency breakdown for heavy rain alerting.
  - `figure6_feature_interpretation`: Tree gain importance, SHAP values, and permutation degradation rankings.
  - `figure7_regional_seasonal`: Subdivisional and seasonal hydrometeorological performance stratification.
  - `figure8_temporal_robustness`: Out-of-sample annual consistency (2024 vs 2025) and stationary block bootstrap error bars.
  - `figure9_extreme_rainfall_errors`: Diagnostic quantile error profiles and conditional mean ceiling analysis.
