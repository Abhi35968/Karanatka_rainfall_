import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pandas as pd
from pathlib import Path
import os

out_dir = Path("d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review_Final")
out_dir.mkdir(parents=True, exist_ok=True)
out_docx = out_dir / "Karnataka_Rainfall_Research_Paper_Final.docx"
fig_dir = out_dir / "manuscript_figures"
tbl_dir = out_dir / "manuscript_tables"
stage6_dir = Path("d:/PROJECT/karnataka_rainfall_cds/Stage6_Validation_Sensitivity/results")

print("--- Step 1: Loading updated tables ---")
df_t1 = pd.read_csv(tbl_dir / "table1_dataset_characteristics.csv")
df_t2 = pd.read_csv(tbl_dir / "table2_feature_groups.csv")
df_t3 = pd.read_csv(tbl_dir / "table3_model_benchmarking.csv")
df_t4 = pd.read_csv(tbl_dir / "table4_heavy_rain_classification.csv")
df_t5_ablation = pd.read_csv(tbl_dir / "table7_feature_ablation.csv")
df_t6_regional = pd.read_csv(tbl_dir / "table5_regional_seasonal_performance.csv")
df_t7_bootstrap = pd.read_csv(tbl_dir / "table6_statistical_robustness.csv")
df_t8_intensity = pd.read_csv(tbl_dir / "table8_rainfall_intensity_limitations.csv")

# Annual performance table
y24_clean = pd.read_csv(tbl_dir / "table6a_annual_performance.csv")

print("--- Step 2: Generating Document Object ---")
doc = Document()

# Set Standard Page Margins (1 inch all around)
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

def set_font(run, font_name="Calibri", size_pt=10.5, bold=False, italic=False, color_rgb=(0, 0, 0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_title(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, "Calibri", 18, bold=True, color_rgb=(24, 43, 73))

def add_meta(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    set_font(run, "Calibri", 9.5, italic=True, color_rgb=(80, 80, 80))

def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_font(run, "Calibri", 13.5, bold=True, color_rgb=(24, 43, 73))

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_font(run, "Calibri", 11.5, bold=True, color_rgb=(45, 75, 115))

def add_p(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_font(run, "Calibri", 10.0, color_rgb=(30, 30, 30))
    return p

def add_figure(img_path, caption):
    if not img_path.exists():
        print(f"WARNING: Image not found: {img_path}")
        return
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(img_path), width=Inches(6.5))
    
    cp = doc.add_paragraph()
    cp.paragraph_format.space_before = Pt(2)
    cp.paragraph_format.space_after = Pt(8)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c_run = cp.add_run(caption)
    set_font(c_run, "Calibri", 9.0, italic=True, color_rgb=(70, 70, 70))

def add_table_from_df(df, caption):
    cp = doc.add_paragraph()
    cp.paragraph_format.space_before = Pt(8)
    cp.paragraph_format.space_after = Pt(3)
    c_run = cp.add_run(caption)
    set_font(c_run, "Calibri", 9.5, bold=True, color_rgb=(24, 43, 73))
    
    table = doc.add_table(rows=len(df)+1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    
    # Header row
    hdr_cells = table.rows[0].cells
    for c_idx, col_name in enumerate(df.columns):
        clean_name = str(col_name).replace("_", " ")
        cell = hdr_cells[c_idx]
        cell.text = clean_name
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shading = OxmlElement('w:shd')
        shading.set(qn('w:val'), 'clear')
        shading.set(qn('w:color'), 'auto')
        shading.set(qn('w:fill'), 'E8EEF5')
        cell._tc.get_or_add_tcPr().append(shading)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                set_font(r, "Calibri", 8.0, bold=True, color_rgb=(24, 43, 73))
                
    # Data rows
    for r_idx, row in df.iterrows():
        row_cells = table.rows[r_idx+1].cells
        for c_idx, val in enumerate(row):
            cell = row_cells[c_idx]
            if pd.isna(val):
                cell.text = "N/A"
            elif isinstance(val, float):
                cell.text = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                cell.text = str(val)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1.5)
                p.paragraph_format.space_after = Pt(1.5)
                p.paragraph_format.line_spacing = 1.05
                if c_idx > 1 and not isinstance(val, str):
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    set_font(r, "Calibri", 8.0, color_rgb=(30, 30, 30))
                    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

print("--- Step 3: Assembling Manuscript Content ---")

# TITLE & METADATA
add_title("Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)")
add_meta("Author List: [Author Names Blinded for Peer Review / To Be Inserted by Author]")
add_meta("Affiliations: [Department / Institutional Affiliation To Be Inserted by Author]")
add_meta("Correspondence: [Corresponding Author Email Address To Be Inserted Upon Acceptance]")
add_meta("Publication Target: High-Impact Peer-Reviewed Hydrometeorological / Atmospheric Science Journal")

# ABSTRACT
add_h1("Abstract")
add_p(
    "Accurate retrospective reconstruction of daily precipitation across topographically complex tropical regions is fundamental "
    "for catchment water balance accounting, agricultural monitoring, and hydrological gap-filling. Reanalysis products such as ECMWF ERA5-Land "
    "provide continuous atmospheric and land-surface fields but exhibit substantial localized biases over complex orography. Here, we evaluate a "
    "retrospective daily rainfall reconstruction and hindcasting framework across the 252 India Meteorological Department (IMD) 0.25° gridded "
    "land cells covering Karnataka State, India, over the 7-year period 2019–2025 (N = 640,836 grid-cell-days). Ingesting 63 primary antecedent "
    "predictors derived from ERA5-Land surface fluxes, atmospheric thermodynamic states, soil moisture profiles, DEM orography, and antecedent IMD observations "
    "under strict causal latency constraints (hindcast issue boundary T_issue = 03:00 UTC on day t-1; target window = 03:00 UTC day t-1 to 03:00 UTC day t), "
    "we evaluate regularized linear baselines, gradient-boosted decision trees, and two-stage hurdle architectures. Models are trained on 2019–2022 (N = 364,644), "
    "hyperparameter-tuned and threshold-selected strictly on the 2023 validation partition (N = 91,980, characterized by lower mean daily rainfall during the validation year: "
    "daily mean = 2.3571 mm/day over 365 calendar days vs. 3.7388 mm/day in training over 1,447 calendar days, alongside 27.39% wet days vs. 36.95% in training), and evaluated out-of-sample on a frozen "
    "2-year forward test partition (2024–2025, N = 184,212). On continuous test evaluation, machine learning models achieve strong predictive skill with "
    "distinct multi-criteria trade-offs: Hurdle Expected Value attains the lowest root-mean-square error (RMSE = 7.4241 mm), highest Pearson correlation (r = 0.7380), "
    "and highest Nash-Sutcliffe Efficiency (NSE = 0.5443); CatBoost Tweedie attains the highest Modified Kling-Gupta Efficiency (KGE-2012 = 0.6261) with "
    "RMSE = 7.4335 mm; and LightGBM Tweedie achieves the lowest mean absolute error (MAE = 2.8993 mm) and lowest mean bias (-0.0179 mm; -0.51%). "
    "All three architectures substantially outperform autoregressive Persistence (RMSE = 10.3582 mm), raw antecedent ERA5-Land reanalysis (RMSE = 10.4550 mm), and "
    "grid-cell-wise Empirical Quantile Mapping (EQM, RMSE = 11.1970 mm, NSE = -0.0366). Paired stationary block bootstrapping (B = 2,000 resamples; L = 7d, 14d) "
    "provides strong statistical evidence of lower RMSE over Persistence (-2.925 mm; p < 0.0005) and raw ERA5-Land (-3.011 mm; p < 0.0005) under temporal persistence. "
    "Feature attribution reveals that antecedent reanalysis runoff fluxes account for 54.52% of split gain in heavy rain classification and +2.3545 mm validation "
    "permutation RMSE degradation in continuous regression, acting as a model-derived hydrometeorological memory proxy for antecedent soil saturation. "
    "However, critical limitations are documented: continuous models underestimate 100.0% of events exceeding 115.6 mm (200 test events) due to conditional expectation "
    "shrinkage, plateauing at 70–90 mm. For heavy-rain alerting (>=64.5 mm), an experimental focal-loss classifier achieves a test probability of detection of 46.96% "
    "(precision = 58.22%, CSI = 0.3512 at validation-selected decision threshold p* = 0.53, with 427 false alarms), missing 53.04% of observed heavy events (672 misses). "
    "Crucially, the study does not establish reliable detection of very-heavy (>=115.6 mm) or extremely-heavy (>=204.5 mm) precipitation. These findings demonstrate "
    "that while machine learning effectively reconstructs regional daily water budgets, it represents a diagnostic retrospective reconstruction engine rather than "
    "an operational forecasting or extreme-hazard warning system, requiring further independent validation and extreme-event calibration for operational deployment."
)
add_p("Keywords: Retrospective hindcasting, ERA5-Land reanalysis, IMD gridded rainfall, Tweedie regression, Two-stage hurdle model, Focal loss, Orographic precipitation, Western Ghats.")

# 1. INTRODUCTION
add_h1("1. Introduction")
add_p(
    "High-resolution retrospective precipitation reconstruction is essential for water resource planning, drought monitoring, hydrological water balance modeling, "
    "and agricultural risk assessment. Over tropical peninsular India, the Western Ghats mountain range creates an abrupt orographic barrier that produces one "
    "of the world's sharpest climatic gradients. Within an orthogonal distance of less than 60 km, annual rainfall drops precipitously from over 4,000 mm along "
    "the windward coastal-crest escarpment to under 600 mm across the leeward semi-arid Deccan plateau. Capturing this extreme spatial transition poses severe "
    "challenges for both global reanalysis products and traditional statistical downscaling techniques."
)
add_p(
    "While ECMWF ERA5-Land reanalysis provides hourly surface hydrological and energy flux estimates at 0.1° resolution globally, its precipitation fields "
    "suffer from systematic local biases, including drizzle inflation on dry days, underestimation of convective peaks, and smoothing of sharp orographic crests. "
    "Classical statistical downscaling, such as Empirical Quantile Mapping (EQM), calibrates the marginal distribution against observed gauge records but "
    "operates univariately, assuming temporal stationarity and disrupting physical consistency across atmospheric variables."
)
add_p(
    "In this study, we evaluate a scientifically audited machine learning framework with verified retrospective temporal separation for daily rainfall "
    "reconstruction across Karnataka from 2019 to 2025. Crucially, this research is explicitly framed as a retrospective rainfall reconstruction and hindcasting "
    "investigation using ERA5-Land reanalysis and antecedent IMD observations, rather than an operational real-time forecasting or flood warning harness. "
    "Operating under a 24-hour antecedent causal latency window, the pipeline evaluates whether machine learning can exploit land-surface flux memory to enhance "
    "retrospective precipitation estimation beyond classical downscaling, while rigorously documenting the structural limitations of conditional mean models during extreme events."
)

# 2. STUDY AREA AND DATASETS
add_h1("2. Study Area and Datasets")
add_h2("2.1 Geographical Domain & Meteorological Subdivisions")
add_p(
    "The study area comprises Karnataka State (11.75°N–18.50°N, 74.25°E–78.50°E; ~191,791 km²), encompassing 30 administrative districts organized into "
    "four official IMD meteorological subdivisions: Coastal Karnataka (13 IMD grid cells; maritime, heavy monsoonal rainfall), Malnad (35 grid cells; "
    "Western Ghats ridge and windward slopes), North Interior Karnataka (NIK; 118 grid cells; semi-arid plateau, black cotton soils), and South Interior "
    "Karnataka (SIK; 86 grid cells; southern plateau receiving both Southwest and Northeast monsoons). Analysis is conducted across the 252 IMD 0.25° gridded land cells covering the state."
)

add_figure(fig_dir / "figure1_study_area.png", 
           "Figure 1: High-resolution physiographic map of Karnataka depicting Copernicus GLO-90 DEM elevation, Western Ghats crest line, 252 IMD 0.25° gridded cells color-coded by meteorological subdivision, and the 3 coastal fallback cells (IDs 52, 68, 125).")

add_h2("2.2 Observational Target: IMD Gridded Rainfall")
add_p(
    "The ground-truth target is the daily 0.25° gridded rainfall product from the India Meteorological Department (IMD) [Pai et al., 2014], interpolated via "
    "modified inverse distance weighting from 6,955 quality-controlled rain gauges across India. Each daily record represents the 24-hour accumulation ending at "
    "08:30 Indian Standard Time (IST; equivalent to 03:00 UTC). We extract all 252 in-state grid cells across the 2,556 calendar days spanning January 1, 2019 "
    "to December 31, 2025. These gridded cells represent area-averaged daily accumulations rather than point observations from surface weather stations."
)

add_h2("2.3 Reanalysis Features: ECMWF ERA5-Land")
add_p(
    "Atmospheric predictors and land-surface fluxes are extracted from ECMWF ERA5-Land at 0.1° (~9 km) hourly resolution across 2019–2025. Extracted variables "
    "include total precipitation (tp), surface runoff (sro), sub-surface/total runoff (ro), 2-meter air temperature (t2m), 2-meter dewpoint temperature (d2m), "
    "surface pressure (sp), 10-meter wind components (u10, v10), surface solar radiation downwards (ssrd), surface thermal radiation downwards (strd), surface "
    "latent heat flux (slhf), surface sensible heat flux (sshf), total evaporation (e), potential evaporation (pev), volumetric soil water content across three layers "
    "(swvl1: 0–7 cm, swvl2: 7–28 cm, swvl3: 28–100 cm), skin reservoir content (src), and forecast albedo (fal)."
)

add_h2("2.4 Static Topography & Coastal Proximity")
add_p(
    "Terrain elevation and slope were derived from the Copernicus GLO-90 Digital Elevation Model (90-meter resolution). Orthogonal distance to the Arabian Sea "
    "coastline and shortest distance to the Western Ghats crest line were computed for each grid cell centroid to capture low-level maritime moisture advection "
    "and mechanical barrier effects."
)

add_h2("2.5 Chronological Partitioning & Climatological Characteristics")
add_p(
    "To respect temporal autocorrelation and prevent future-data leakage across splits, the 7-year dataset is partitioned chronologically:\n"
    "• Training Partition (2019–2022; 1,447 calendar days; N = 364,644 instances; 36.95% wet days; mean = 3.7388 mm/day, std = 11.0264 mm).\n"
    "• Validation Partition (2023; 365 calendar days; N = 91,980 instances; characterized by lower mean daily rainfall during the validation year: 27.39% wet days vs. 36.95% in training; mean = 2.3571 mm/day vs. 3.7388 mm/day in training across 1,447 calendar days, std = 8.2325 mm).\n"
    "• Out-of-Sample Forward Test Partition (2024–2025; 731 calendar days; N = 184,212 instances; 35.04% wet days; mean = 3.5085 mm/day, std = 11.0006 mm).\n"
    "All model hyperparameter tuning and decision threshold selections were performed strictly on the 2023 validation partition; the test partition remained frozen until final evaluation."
)

add_table_from_df(df_t1, "Table 1: Temporal Dataset Partitioning & Climatological Characteristics (2019–2025)")

# 3. METHODOLOGY
add_h1("3. Methodology")
add_h2("3.1 ERA5-Land De-accumulation Physics and Temporal Alignment")
add_p(
    "In raw ECMWF ERA5-Land records, accumulated surface variables (total precipitation tp, surface runoff sro, total runoff ro, evaporation e, potential evaporation pev, "
    "downward radiation fluxes ssrd, strd, and turbulent heat fluxes slhf, sshf) are stored as running integrals reset to zero at 00:00 UTC each day. Hourly physical "
    "increments are recovered via consecutive backward differencing: Delta x_1 = x_1 at 01:00 UTC, and Delta x_h = x_h - x_{h-1} for h in {2, ..., 24}."
)
add_p(
    "Across the statewide reanalysis archive evaluated by the quality control instrument across 2,543 calendar days (15 January 2019 to 31 December 2025, excluding the "
    "14-day warm-up period from 1 to 14 January 2019 utilized for antecedent lag construction), backward differencing was evaluated across 221,798,424 valid (non-NaN) hourly "
    "forecast increments specifically instrumented for total precipitation (tp) across the 81 x 51 spatial grid (3,737 terrestrial land cells across 2,473 evaluated days). "
    "This contrasts with the theoretical maximum of 252,123,192 increments (4,131 grid cells x 24 hours x 2,543 days), with the excluded increments comprising 23,384,688 "
    "marine increments masked out over the Arabian Sea (394 invariant ocean cells) and uncounted increments at month-boundary forecast cycle transitions. In tp, 1,136,648 increments "
    "(0.512469%) fell below the numerical check threshold of Delta x_h < -1.0 x 10^-5 mm, exhibiting a median magnitude of -1.49 x 10^-5 mm and a minimum of -4.21 x 10^-4 mm. "
    "These residuals were verified as IEEE 754 single-precision float representation artifacts occurring in high-accumulation grid cells rather than physical model inversions."
)
add_p(
    "To ensure physical consistency, variable-specific treatment rules were systematically applied:\n"
    "1. Non-Negative Hydrological & Radiation Depth Fluxes: Total precipitation (tp), surface runoff (sro), and total runoff (ro) are physically non-negative depths. "
    "Their single-precision negative residuals were clipped to zero via np.maximum(x, 0.0). For tp, post-clipping negatives are strictly 0 (0.0%). For sro and ro, numerical "
    "clipping was verified in code, but dedicated pre-clip negative counters were not independently instrumented in Stage 2 logs. Downward solar radiation (ssrd) is "
    "non-negative downward radiative flux; minor nighttime float noise (0.0056%) was clipped via np.maximum(x, 0.0) / 86400.0. Downward longwave radiation (strd) is strictly "
    "positive (>280 W/m²) and required zero clipping. Following clipping, negative values for all precipitation, runoff, and downward radiation fields are strictly zero (0.0%).\n"
    "2. Evaporation Variables (Sign Retransformation): In ECMWF convention, total evaporation (e) and potential evaporation (pev) are reported as upward mass fluxes away "
    "from the surface (negative sign convention). These were transformed to positive water-equivalent depths via np.abs(x). Following transformation, negative values are strictly zero (0.0%).\n"
    "3. Signed Turbulent Heat Fluxes: Surface latent heat flux (slhf) and surface sensible heat flux (sshf) are physically signed turbulent energy exchanges (W/m², converted "
    "by dividing accumulated Joules by 86,400 s). For latent heat flux, negative values represent upward evaporative cooling, and negative physical values are validly retained "
    "within physical bounds ([-300.0, 50.0] W/m²), with 220,950,112 negative values (99.6175%) remaining post-clipping. For sensible heat flux (sshf), physically signed "
    "values are validly retained within physical bounds ([-250.0, 100.0] W/m²), with 168,452,109 negative values (75.9483%) and positive values preserved according to "
    "ECMWF convention without arbitrary zero clipping.\n"
    "Crucially, numerical clipping eliminates truncation artifacts to stabilize machine-learning optimization, but it does not prove mathematical conservation or physical correctness of the underlying model."
)
add_p(
    "To establish an unlagged baseline benchmark (Raw ERA5-Land and EQM), hourly increments are integrated across the 24-hour window from 04:00 UTC on day t-1 through "
    "03:00 UTC on day t: X_t^{IMD} = sum_{h=4}^{24} Delta x_{t-1, h} + sum_{h=1}^3 Delta x_{t, h}. This allows direct comparison against ground-truth IMD daily rainfall y_t "
    "over the exact same accumulation window."
)
add_p(
    "Crucially, in our retrospective machine-learning pipeline, the hindcast issue boundary is fixed at T_issue = 03:00 UTC on day t-1. All predictor features are strictly "
    "restricted to variables completed on or before T_issue. Therefore, the dynamic machine learning feature is lagged by one full day (era5_tp_lag_1d = X_{t-1}^{IMD}), "
    "which spans 04:00 UTC day t-2 to 03:00 UTC day t-1. All rolling features apply an explicit shift(1). Contemporaneous target-day reanalysis columns are purged prior to training, "
    "ensuring verified retrospective temporal separation with identically zero temporal overlap."
)

add_figure(fig_dir / "figure2_methodology.png", 
           "Figure 2: End-to-end retrospective machine learning architecture depicting ERA5-Land flux de-accumulation, IMD gauge preparation, 24-h causal lag alignment, 63 primary feature engineering, temporal partitioning, model training, and multi-criteria evaluation.")

add_h2("3.2 Feature Matrix Architecture")
add_p(
    "The final dataset schema comprises 78 columns: 4 ground-truth target labels, 4 geographic/administrative identifiers, 2 protocol metadata strings, 4 excluded "
    "atmospheric pressure proxies and seasonal strings, 1 auxiliary quality flag (is_coastal_fallback), and exactly 63 primary predictors (X) respecting the 24-hour "
    "causal latency constraint (T_issue = 03:00 UTC on day t-1)."
)
add_p(
    "In the training pipeline (train_regression_models.py, train_heavy_rain_models.py, train_hurdle_model.py), the model input feature matrix X is constructed by "
    "taking the 61 numeric primary features and transforming the 2 raw string categorical columns (subdivision and zone_id) via target mean encoding fitted strictly "
    "on the 2019–2022 training partition (subdiv_map and zone_map). The resulting numerical features (subdivision_enc and zone_id_enc) replace the raw string fields "
    "in the model matrix, yielding an exact 63-column numerical matrix (X in R^{N x 63}) fed to the gradient-boosted decision trees and linear models without future target leakage."
)
add_p(
    "As structured in Table 2, these 63 primary model predictors span seven physical and hydrometeorological domains:\n"
    "1. Antecedent IMD Rainfall Memory (13 features): Lagged local ground-truth rainfall (y_{t-1}, y_{t-2}, y_{t-3}, y_{t-7}, y_{t-14}), rolling window sums and means "
    "(3-day, 7-day, 14-day), 7-day rolling maximum, and the 14-day Antecedent Precipitation Index (API_14 = sum_{k=1}^{14} 0.85^k y_{t-k}), all shifted strictly by >=1 day.\n"
    "2. ERA5-Land Precipitation & Hydrological Balance (14 features): Antecedent reanalysis total precipitation (era5_tp_lag_1d, _lag_2d, _lag_3d, 3-day and 7-day rolling sums), "
    "land-surface runoff fluxes (ro_lag_1d, sro_lag_1d), canopy skin reservoir storage (src_lag_1d), evaporation components (e_lag_1d, pev_lag_1d, evabs_lag_1d, evatc_lag_1d, evavt_lag_1d), "
    "and surface albedo (fal_lag_1d).\n"
    "3. Atmospheric Thermodynamic State & Wind (12 features): 2-meter air temperature dynamics (t2m_mean_lag_1d, t2m_min_lag_1d, t2m_max_lag_1d, diurnal range temp_range_lag_1d, "
    "3-day and 7-day rolling means), moisture saturation diagnostics (d2m_mean_lag_1d, dewpoint depression dewpoint_depression_lag_1d), surface pressure (sp_lag_1d), and 10-meter "
    "wind speed dynamics (wind_speed_lag_1d, 3-day and 7-day rolling means).\n"
    "4. Soil Moisture Profiles & Radiative/Turbulent Energy Fluxes (9 features): Multi-layer volumetric soil water content (swvl1_lag_1d for 0–7 cm depth, swvl1_lag_3d, swvl1_lag_7d, "
    "swvl2_lag_1d for 7–28 cm, swvl3_lag_1d for 28–100 cm), surface solar radiation downwards (ssrd_lag_1d), surface thermal radiation downwards (strd_lag_1d), surface latent heat "
    "flux (slhf_lag_1d), and surface sensible heat flux (sshf_lag_1d).\n"
    "5. Static Topography, Elevation & Spatial Proximity (6 features): Terrain elevation from Copernicus GLO-90 DEM (elevation_dem_m), local terrain slope (slope_dem_deg), "
    "shortest Euclidean distance to the Arabian Sea coastline (dist_to_coast_km), orthogonal distance to the Western Ghats mountain crest line (dist_to_ghats_crest_km), and "
    "IMD 0.25° grid-cell centroid geographic coordinates (latitude, longitude).\n"
    "6. Cyclic Temporal Ephemeris & Astronomical Cycles (7 features): Calendar month, Julian day of year, harmonic cyclical expansions (sin/cos of day of year and month), "
    "and Southwest monsoon seasonal indicator (is_monsoon).\n"
    "7. Primary Regional Categorical Encodings (2 features): Out-of-sample target-encoded regional indicators (subdivision_enc for the 4 meteorological subdivisions and "
    "zone_id_enc for the 10 agroclimatic zones), which replace the raw categorical string columns in the model feature matrix."
)

add_table_from_df(df_t2, "Table 2: Input Feature Groups, Predictor Schema, and Pipeline Gating")

add_h2("3.3 Model Formulations Across Tiers")
add_p(
    "We implement a hierarchical modeling framework across five structural tiers:\n"
    "• Tier 0 (Naive Baselines): Persistence (y_t = y_{t-1}) and Grid-Cell DOY Climatology (historical day-of-year mean per grid cell).\n"
    "• Tier 1 (Raw Reanalysis Benchmark): Uncalibrated antecedent ERA5-Land precipitation.\n"
    "• Tier 2 (Statistical Downscaling): Empirical Quantile Mapping (EQM) using 100 empirical quantile knots fitted per IMD grid cell on 2019–2022.\n"
    "• Tier 3 (Regularized Linear Baselines): Ridge Regression (L2 regularization, alpha = 100.0) and ElasticNet Regression (alpha = 0.1, l1_ratio = 0.5) fitted on standardized features.\n"
    "• Tier 4 (Machine Learning Architectures): LightGBM Tweedie (p=1.5) [14], CatBoost Tweedie (p=1.5) [15], XGBoost Tweedie [16], LightGBM Log-Ordinary with Duan Smearing [20], "
    "and a Two-Stage Hurdle Model combining binary rain occurrence classification (LGBM logistic) with truncated regression (LGBM Gamma GLM) [19]."
)

add_h2("3.4 Dedicated Heavy Rainfall Classification & Custom Asymmetric Focal Loss")
add_p(
    "Because extreme rainfall events (>=64.5 mm/day) constitute only 0.6878% of test records, standard binary cross-entropy produces prohibitive miss rates. "
    "We implement an experimental custom Lin et al. [21] Focal Loss objective in LightGBM with focusing parameter gamma = 2.0 and alpha-balancing = 0.75:\n"
    "FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)\n"
    "The first- and second-order custom gradients were analytically derived. The validation-selected decision threshold p* = 0.53 was chosen strictly on the 2023 "
    "validation partition to maximize the Critical Success Index (CSI). It represents an empirical decision boundary rather than a calibrated posterior probability."
)

add_h2("3.5 Evaluation Metrics & Paired Stationary Block Bootstrapping")
add_p(
    "Continuous performance is evaluated using RMSE, MAE, Mean Bias, Percentage Bias (PBias), Pearson correlation (r), Nash-Sutcliffe Efficiency (NSE) [22], "
    "Classic KGE-2009 [23], and Modified KGE-2012 [24]. Categorical heavy rain alerting is evaluated via Hits (H), Misses (M), False Alarms (F), Correct Negatives (C), "
    "Probability of Detection (POD / Recall), False Alarm Ratio (FAR), Precision, Critical Success Index (CSI / Threat Score), Heidke Skill Score (HSS), and Equitable Threat Score (ETS)."
)
add_p(
    "To account for spatial cross-correlation across the 252 grid cells and temporal autocorrelation across consecutive days, we implement a stationary paired block "
    "bootstrap protocol [25], [26] with B = 2,000 resamples under block lengths L = 7 days (synoptic weather cycle) and L = 14 days (intraseasonal scale). "
    "The paired stationary block-bootstrap analysis provides strong statistical evidence of lower RMSE under the evaluated resampling assumptions (empirical resolution p < 0.0005)."
)

# 4. RESULTS
add_h1("4. Results")
add_h2("4.1 Continuous Rainfall Estimation & Multi-Criteria Benchmarking")
add_p(
    "Table 3 presents out-of-sample forward evaluation on the frozen 2024–2025 test partition (N = 184,212 instances). Rather than demonstrating a single universally "
    "dominant architecture, the three gradient-boosted models exhibit competitive performance characterized by distinct multi-criteria trade-offs:\n"
    "• Hurdle Expected Value achieves the lowest overall RMSE (7.4241 mm), highest Pearson correlation (r = 0.7380), and highest Nash-Sutcliffe Efficiency (NSE = 0.5443).\n"
    "• CatBoost Tweedie attains the highest Modified Kling-Gupta Efficiency (KGE-2012 = 0.6261) with RMSE = 7.4335 mm and MAE = 2.9186 mm.\n"
    "• LightGBM Tweedie achieves the lowest Mean Absolute Error (MAE = 2.8993 mm) and lowest mean bias (-0.0179 mm; -0.51% relative bias).\n"
    "All three gradient-boosted architectures substantially outperform naive Persistence (RMSE = 10.3582 mm; -2.93 mm reduction), raw uncalibrated ERA5-Land (RMSE = 10.4550 mm; "
    "-3.03 mm reduction), and grid-cell-wise Empirical Quantile Mapping (EQM, RMSE = 11.1970 mm, NSE = -0.0366)."
)

add_table_from_df(df_t3, "Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; N = 184,212)")

add_figure(fig_dir / "figure3_observed_predicted.png", 
           "Figure 3: Density parity hexbin between observed and predicted daily rainfall on the 2024–2025 test partition. Color bar represents log-scaled instance density.")

add_figure(fig_dir / "figure4_model_benchmarking.png", 
           "Figure 4: Comparative bar panels showing RMSE, MAE, Pearson correlation, and KGE-2012 across all 11 evaluated models on the frozen test partition.")

add_h2("4.2 Heavy Rainfall Detection and Retrospective Classifier Evaluation")
add_p(
    "Table 4 summarizes the performance of the dedicated heavy rainfall classification models for events >= 64.5 mm (1,267 true events out of 184,212 test records, prevalence = 0.6878%)."
)

add_table_from_df(df_t4, "Table 4: Dedicated Heavy Rainfall (>=64.5 mm) Contingency & Skill Score Evaluation (Test 2024–2025)")

add_figure(fig_dir / "figure5_heavy_rain_detection.png", 
           "Figure 5: Synoptic heavy-rain detection performance. Panel A shows contingency breakdown (Hits, Misses, False Alarms); Panel B shows Precision-Recall curves across classification architectures.")

add_p(
    "At the validation-selected decision threshold p* = 0.53, the LightGBM Focal Loss classifier achieves a test POD of 46.96% (595 hits), Precision of "
    "58.22%, CSI of 0.3512, and FAR of 41.78% (427 false alarms), missing 53.04% of observed heavy events (672 misses out of 1,267). Relative to standard "
    "class weighting, Focal Loss suppresses false alarms by 40.86% (427 vs. 722). However, because the model misses more than half of observed heavy events, "
    "it represents an experimental retrospective classification component rather than an operational flood warning system. Furthermore, this study does not "
    "establish reliable detection of very-heavy (>=115.6 mm) or extremely-heavy (>=204.5 mm) precipitation."
)

add_h2("4.3 Feature Attribution: Gain, Permutation, and Ablation Dynamics")
add_p(
    "Figure 6 and Table 5 examine physical feature attribution and ablation dynamics across the 63 primary predictors. To avoid overconfident interpretation, "
    "we emphasize that tree split gain and permutation importance quantify empirical predictive sensitivity and variance reduction within the learned decision-tree "
    "ensemble, rather than deterministic physical causality. Furthermore, the ablation scenarios reported on the 2024–2025 test partition are descriptive comparisons "
    "on the frozen test set without additional hypothesis testing."
)

add_figure(fig_dir / "figure6_feature_interpretation.png", 
           "Figure 6: Feature attribution analysis. Panel A shows top 15 predictors ranked by validation permutation RMSE degradation; Panel B shows relative gain contribution by physical predictor domain.")

add_table_from_df(df_t5_ablation, "Table 5: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)")

add_p(
    "In the LightGBM Focal Loss heavy rainfall classifier, sub-surface runoff (ro_lag_1d, 28.11%) and surface runoff (sro_lag_1d, 26.41%) together account for "
    "54.52% of total tree split gain. In continuous LightGBM Tweedie regression, surface runoff accounts for 34.61% of tree gain and skin reservoir content accounts "
    "for 28.28%. In validation permutation importance (N = 91,980, 3 random seeds), surface runoff ranks first (+2.3545 mm RMSE degradation, +42.16% increase), "
    "followed by skin reservoir content (+2.1241 mm) and downward longwave flux (+1.7960 mm). Ablation Scenario E confirms that excluding ERA5-Land predictors degrades "
    "Test RMSE from 7.4427 mm to 8.4511 mm (+1.0084 mm), demonstrating their measurable incremental predictive value beyond local gauge autoregression."
)

add_h2("4.4 Regional and Seasonal Disaggregation")
add_p(
    "Table 6 details model performance across Karnataka's four meteorological subdivisions and four climatological seasons."
)

add_table_from_df(df_t6_regional, "Table 6: Subdivisional & Seasonal Stratified Performance on Test Partition")

add_figure(fig_dir / "figure7_regional_seasonal.png", 
           "Figure 7: Subdivisional and seasonal hydrometeorological stratification. Panel A shows RMSE and KGE-2012 across subdivisions; Panel B shows seasonal progression across Southwest Monsoon, Post-Monsoon, Pre-Monsoon, and Winter.")

add_p(
    "In orographic Coastal Karnataka and Malnad, the model achieves high correlation (r = 0.80–0.83) and hydrological skill (KGE-2012 = 0.73–0.74). Over the "
    "semi-arid interior plains (NIK and SIK), correlation encounters a distinct ceiling (r ~ 0.53–0.55; KGE ~ 0.37). The 3 coastal fallback cells "
    "(Grid-Cell IDs 52, 68, 125; N = 2,193) achieve RMSE = 15.51 mm and KGE = 0.7504, tracking closely with the 10 native-land coastal cells (RMSE = 14.78 mm, "
    "KGE = 0.7356), confirming that nearest-inland neighbor extraction effectively preserves the orographic moisture advection signal."
)

add_h2("4.5 Temporal Generalisation Across Forward Evaluation Years (2024 vs. 2025) & Block Bootstrap")
add_p(
    "To evaluate out-of-sample temporal generalisation, model performance was audited separately for the two held-out forward calendar years in Table 7 (Panel A). "
    "In 2024 (N = 92,232; active monsoon convective year), LightGBM Tweedie achieved RMSE = 7.5094 mm, MAE = 3.0074 mm, Bias = +0.3442 mm (+10.26%), "
    "r = 0.7320, NSE = 0.5346, and KGE-2012 = 0.5676. In 2025 (N = 91,980; subdued monsoon year), LightGBM Tweedie achieved RMSE = 7.3752 mm, "
    "MAE = 2.7909 mm, Bias = -0.3810 mm (-10.40%), r = 0.7426, NSE = 0.5493, and KGE-2012 = 0.6541. CatBoost Tweedie achieved RMSE of 7.5253 mm in 2024 "
    "and 7.3403 mm in 2025; Hurdle Expected Value achieved RMSE of 7.5210 mm in 2024 and 7.3257 mm in 2025."
)

add_table_from_df(y24_clean, "Table 7A: Annual Out-of-Sample Performance Breakdown: 2024 (Leap Year) vs. 2025")
add_table_from_df(df_t7_bootstrap, "Table 7B: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)")

add_figure(fig_dir / "figure8_temporal_robustness.png", 
           "Figure 8: Temporal robustness and block bootstrap confidence intervals. Panel A shows annual RMSE consistency bar comparison across 2024 and 2025 forward evaluation years; Panel B shows paired block bootstrap 95% confidence interval error bars (B = 2,000 replicates, empirical resolution p < 0.0005) for L=7d and L=14d block lengths.")

add_p(
    "Across all 2,000 block bootstrap resamples under both 7-day and 14-day block lengths, zero replicates yielded Delta RMSE >= 0, establishing that the paired "
    "stationary block-bootstrap analysis provides strong statistical evidence of lower RMSE under the evaluated resampling assumptions (empirical resolution p < 0.0005)."
)

add_h2("4.6 Rainfall Intensity Limitations & Extreme Quantile Shrinkage")
add_p(
    "Table 8 and Figure 9 provide an audit of continuous model behavior across the seven official IMD rainfall intensity categories."
)

add_table_from_df(df_t8_intensity, "Table 8: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage (Test Set; N = 184,212)")

add_figure(fig_dir / "figure9_extreme_rainfall_errors.png", 
           "Figure 9: Rainfall intensity error profiles and quantile shrinkage. Panel A shows Mean Bias (mm) and Percentage Bias (%) across the 7 IMD categories; Panel B shows underestimation frequency (%) and sample counts.")

add_p(
    "A critical structural limitation is documented: continuous conditional expectation models underestimate 100.0% of test observations exceeding 115.6 mm "
    "(200 events, comprising 192 Very Heavy and 8 Extremely Heavy events), with predictions plateauing near 70–90 mm despite observed extremes up to 286.0 mm. "
    "Furthermore, on true dry days (<0.1 mm; N = 119,669; 64.96% of test records), the model exhibits background drizzle inflation (+0.77 mm mean bias; "
    "Percentage Bias is undefined due to zero denominator). These structural boundaries prove that continuous regression alone cannot capture localized convective bursts."
)

# 5. DISCUSSION
add_h1("5. Discussion")
add_p(
    "A primary physical finding is that land-surface runoff fluxes account for 54.52% of total tree gain in heavy rain classification and lead "
    "permutation importance in continuous regression. Crucially, ERA5-Land runoff variables represent simulated land-surface water fluxes from ECMWF's "
    "HTESSEL land model, rather than direct streamflow observations. In tropical monsoonal regimes, modeled surface runoff is generated only when local "
    "infiltration capacity is exceeded or topsoil approaches saturation. Runoff fluxes act as a physical integrator, filtering out transient atmospheric "
    "false alarms and retaining sustained moisture convergence."
)
add_p(
    "The performance dichotomy between orographic Western Ghats (r >= 0.80) and interior Deccan plateau (r ~ 0.53) reflects atmospheric scale limits. "
    "Western Ghats precipitation is mechanically forced by low-level moisture advection perpendicular to the terrain ridge, which reanalysis resolves with "
    "high skill. Conversely, interior precipitation is governed by unorganized, localized mesoscale convective thunderstorm cells unresolvable on a 0.1° grid."
)
add_p(
    "Furthermore, we explicitly emphasize that because ERA5-Land has an operational release latency of 2 to 3 months, this framework represents a "
    "retrospective hydrometeorological reconstruction and gap-filling pipeline, rather than an operational real-time nowcasting system. The heavy-rain "
    "classifier serves as an experimental retrospective classification component (CSI = 0.3512, precision = 58.22%), but misses 53.04% of observed heavy events. "
    "Operational hazard warning deployment would require extensive forward NWP integration, independent real-time validation, and specialized extreme-event calibration."
)

# 6. LIMITATIONS
add_h1("6. Limitations")
add_p(
    "Key structural limitations must be acknowledged:\n"
    "1. Retrospective Hindcasting vs. Real-Time Forecasting: Reanalysis release latency (2–3 months) precludes real-time flood nowcasting.\n"
    "2. Absence of Spatial Generalisation: All 252 IMD grid cells were present across partitions; transferability to ungauged basins is untested.\n"
    "3. Extreme Quantile Shrinkage: Continuous models underestimate 100.0% of events exceeding 115.6 mm (200 test events), plateauing at 70–90 mm.\n"
    "4. Heavy-Rain Miss Rate & Unreliable Extreme Alerting: The focal-loss classifier misses 53.04% of observed heavy events (>=64.5 mm) and does not establish reliable detection of very-heavy or extremely-heavy precipitation.\n"
    "5. Dry-Day Drizzle Inflation: On true zero-rain days (N = 119,669; 64.96% of test records), the continuous model predicts a mean drizzle of +0.77 mm.\n"
    "6. Frozen Test-Set Access: Test data was accessed for frozen evaluation and descriptive ablation reporting; zero parameter tuning was performed.\n"
    "7. Coastal Fallback Strategy: The 3 coastal margin cells (IDs 52, 68, 125) rely on nearest-inland neighbor extraction due to marine water masking.\n"
    "8. Predictive Sensitivity vs. Physical Causality: Feature attribution metrics reflect empirical predictive associations rather than deterministic causal proof.\n"
    "9. Reanalysis Resolution Limits: The 0.1° grid spacing cannot resolve meso-gamma convective thunderstorms over interior plains."
)

# 7. DATA AVAILABILITY & ACKNOWLEDGMENTS
add_h1("7. Data and Code Availability")
add_p(
    "All modeling workflows, feature extraction scripts, trained model joblib files, and validation routines are available in the project reproducibility package. "
    "Gridded daily IMD precipitation data are maintained by the National Data Centre (NDC), India Meteorological Department, Pune (accessible upon user registration). "
    "ERA5-Land reanalysis datasets are publicly available from the Copernicus Climate Data Store under an open license."
)

add_h1("8. Acknowledgments")
add_p(
    "The authors acknowledge the India Meteorological Department (IMD), Ministry of Earth Sciences, Government of India, for providing daily gridded precipitation data, "
    "and the European Centre for Medium-Range Weather Forecasts (ECMWF) and Copernicus Climate Change Service (C3S) for making ERA5-Land reanalysis freely accessible. "
    "This research was supported by [Grant / Funding Agency To Be Inserted by Author]."
)

# 9. REFERENCES
add_h1("References")
refs = [
    "[1] V. Masson-Delmotte et al., \"IPCC, 2021: Summary for Policymakers,\" in Climate Change 2021: The Physical Science Basis, Cambridge University Press, 2021.",
    "[2] J. Muñoz-Sabater et al., \"ERA5-Land: A state-of-the-art global reanalysis dataset for land applications,\" Earth System Science Data, vol. 13, no. 9, pp. 4349–4383, 2021. DOI: 10.5194/essd-13-4349-2021.",
    "[3] D. S. Pai et al., \"Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets,\" Mausam, vol. 65, no. 1, pp. 1–18, 2014. DOI: 10.54302/mausam.v65i1.851.",
    "[4] M. Rajeevan, J. Bhate, and A. K. Jaswal, \"Analysis of variability and extremes of daily rainfall over India using a new high resolution gridded data set: 1951–2003,\" Current Science, vol. 91, no. 3, pp. 296–308, 2006.",
    "[5] European Space Agency, \"Copernicus Global Digital Elevation Model (GLO-90),\" 2020. DOI: 10.5270/esa-c1qtxnn.",
    "[6] B. N. Goswami, V. Venugopal, D. Sengupta, M. S. Madhusoodanan, and P. Xavier, \"Increasing trend of extreme rain events over India in a warming environment,\" Science, vol. 314, no. 5804, pp. 1442–1445, 2006. DOI: 10.1126/science.1132027.",
    "[7] R. Houze, \"Orographic precipitation,\" Annual Review of Earth and Planetary Sciences, vol. 31, pp. 153–185, 2003. DOI: 10.1146/annurev.earth.31.100901.141251.",
    "[8] U. C. Mohanty et al., \"Simulation of heavy precipitation over steep orography of Western Ghats in India using high-resolution mesoscale models,\" Meteorology and Atmospheric Physics, vol. 89, pp. 161–178, 2005. DOI: 10.1007/s00703-005-0167-2.",
    "[9] H. Hersbach et al., \"The ERA5 global reanalysis,\" Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730, pp. 1999–2049, 2020. DOI: 10.1002/qj.3803.",
    "[10] C. Piani et al., \"Statistical bias correction of global simulated daily precipitation and temperature using empirical quantile mapping,\" Journal of Climatology, vol. 31, no. 10, pp. 1530–1544, 2011. DOI: 10.1002/joc.2168.",
    "[11] A. J. Cannon, S. R. Sobie, and T. Q. Murdock, \"Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes?\" Journal of Climate, vol. 28, no. 17, pp. 6938–6959, 2015. DOI: 10.1175/JCLI-D-14-00754.1.",
    "[12] J. Baño-Medina, R. Manzanas, and J. M. Gutiérrez, \"Configuration and intercomparison of deep learning neural models for statistical downscaling,\" Geoscientific Model Development, vol. 13, no. 4, pp. 2109–2124, 2020. DOI: 10.5194/gmd-13-2109-2020.",
    "[13] M. Reichstein, G. Camps-Valls, B. Stevens, M. Jung, J. Denzler, N. Carvalhais, and Prabhat, \"Deep learning and process understanding for data-driven Earth system science,\" Nature, vol. 566, no. 7743, pp. 195–204, 2019. DOI: 10.1038/s41586-019-0912-1.",
    "[14] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, \"LightGBM: A highly efficient gradient boosting decision tree,\" in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, pp. 3146–3154, 2017.",
    "[15] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, \"CatBoost: unbiased boosting with categorical features,\" in Advances in Neural Information Processing Systems (NeurIPS 2018), vol. 31, pp. 6638–6648, 2018.",
    "[16] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016. DOI: 10.1145/2939672.2939785.",
    "[17] M. C. K. Tweedie, \"An index which distinguishes between some important exponential families,\" in Statistics: Applications and New Directions, Proceedings of the Indian Statistical Institute Golden Jubilee International Conference, J. K. Ghosh and J. Roy, Eds., Calcutta: Indian Statistical Institute, pp. 579–604, 1984.",
    "[18] P. K. Dunn and G. K. Smyth, \"Series evaluation of Tweedie exponential dispersion model densities,\" Statistics and Computing, vol. 15, no. 4, pp. 267–280, 2005. DOI: 10.1007/s11222-005-4070-y.",
    "[19] J. G. Cragg, \"Some statistical models for limited dependent variables with application to the demand for durable goods,\" Econometrica, vol. 39, no. 5, pp. 829–844, 1971. DOI: 10.2307/1909582.",
    "[20] N. Duan, \"Smearing estimate: A nonparametric retransformation method,\" Journal of the American Statistical Association, vol. 78, no. 383, pp. 605–610, 1983. DOI: 10.1080/01621459.1983.10478017.",
    "[21] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" in Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017), pp. 2980–2988, 2017. DOI: 10.1109/ICCV.2017.324.",
    "[22] J. E. Nash and J. V. Sutcliffe, \"River flow forecasting through conceptual models part I — A discussion of principles,\" Journal of Hydrology, vol. 10, no. 3, pp. 282–290, 1970. DOI: 10.1016/0022-1694(70)90255-6.",
    "[23] H. V. Gupta, H. Kling, K. K. Yilmaz, and G. F. Martinez, \"Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling,\" Journal of Hydrology, vol. 377, no. 1–2, pp. 80–91, 2009. DOI: 10.1016/j.jhydrol.2009.08.003.",
    "[24] H. Kling, M. Fuchs, and M. Paulin, \"Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios,\" Journal of Hydrology, vol. 424–425, pp. 264–277, 2012. DOI: 10.1016/j.jhydrol.2012.01.011.",
    "[25] H. R. Künsch, \"The jackknife and the bootstrap for general stationary observations,\" The Annals of Statistics, vol. 17, no. 3, pp. 1217–1241, 1989. DOI: 10.1214/aos/1176347265.",
    "[26] D. N. Politis and J. P. Romano, \"The stationary bootstrap,\" Journal of the American Statistical Association, vol. 89, no. 428, pp. 1303–1313, 1994. DOI: 10.1080/01621459.1994.10476870.",
    "[27] L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001. DOI: 10.1023/A:1010933404324.",
    "[28] A. Fisher, C. Rudin, and F. Dominici, \"All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously,\" Journal of Machine Learning Research, vol. 20, no. 177, pp. 1–81, 2019.",
    "[29] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, pp. 4765–4774, 2017."
]

for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(ref)
    set_font(run, "Calibri", 8.5, color_rgb=(40, 40, 40))

doc.save(out_docx)
print(f"Saved finalized manuscript DOCX to {out_docx} ({out_docx.stat().st_size / 1e6:.2f} MB)")
