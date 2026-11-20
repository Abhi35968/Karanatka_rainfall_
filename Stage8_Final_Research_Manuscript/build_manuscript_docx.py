import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import pandas as pd
from pathlib import Path

stage8_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage8_Final_Research_Manuscript")
fig_dir = stage8_dir / "manuscript_figures"
tbl_dir = stage8_dir / "manuscript_tables"

doc = Document()

# Set standard 0.75-inch margins for publication layout
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

# Styling helpers
def set_font(run, font_name="Calibri", size_pt=10, bold=False, italic=False, color_rgb=(0, 0, 0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_font(run, "Calibri", 18, bold=True, color_rgb=(15, 32, 67))

def add_authors(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    set_font(run, "Calibri", 10.5, bold=True, color_rgb=(50, 50, 50))

def add_meta(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    set_font(run, "Calibri", 9, italic=True, color_rgb=(100, 100, 100))

def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_font(run, "Calibri", 13, bold=True, color_rgb=(15, 32, 67))

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_font(run, "Calibri", 11, bold=True, color_rgb=(30, 60, 114))

def add_p(text, space_after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_font(run, "Calibri", 10, color_rgb=(30, 30, 30))
    return p

def add_equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_font(run, "Cambria Math", 10, italic=True, color_rgb=(20, 20, 20))

def add_figure(img_path, caption):
    if not img_path.exists():
        print(f"Warning: Figure {img_path} not found")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(img_path), width=Inches(6.5))
    
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_after = Pt(10)
    run_cap = cap.add_run(caption)
    set_font(run_cap, "Calibri", 8.5, italic=True, color_rgb=(60, 60, 60))

def add_table_from_df(df, caption, col_widths=None):
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    run_cap = cap.add_run(caption)
    set_font(run_cap, "Calibri", 9, bold=True, color_rgb=(15, 32, 67))

    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Format Header Row
    hdr_cells = t.rows[0].cells
    for i, col_name in enumerate(df.columns):
        hdr_cells[i].text = str(col_name)
        # Background color #1E3C72
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1E3C72"/>')
        hdr_cells[i]._tc.get_or_add_tcPr().append(shading)
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                set_font(run, "Calibri", 8, bold=True, color_rgb=(255, 255, 255))
    
    # Format Data Rows
    for r_idx, row in df.iterrows():
        row_cells = t.rows[r_idx + 1].cells
        bg_color = "F4F6F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            # Format number if float
            if isinstance(val, float):
                val_str = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            row_cells[c_idx].text = val_str
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
            row_cells[c_idx]._tc.get_or_add_tcPr().append(shading)
            for p in row_cells[c_idx].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or not any(char.isdigit() for char in val_str) else WD_ALIGN_PARAGRAPH.RIGHT
                for run in p.runs:
                    set_font(run, "Calibri", 7.5, color_rgb=(40, 40, 40))

    # Add space after table
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)

print("--- Building Document Body ---")

# Title and Meta
add_title("Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)")
add_authors("Senior Atmospheric and Machine Learning Research Consortium")
add_meta("Hydrometeorological Data Science Research Initiative | Working Manuscript (Submission Ready)")

# Abstract Box
p_abs_lbl = doc.add_paragraph()
p_abs_lbl.paragraph_format.space_before = Pt(6)
p_abs_lbl.paragraph_format.space_after = Pt(2)
run_abs_lbl = p_abs_lbl.add_run("ABSTRACT")
set_font(run_abs_lbl, "Calibri", 10.5, bold=True, color_rgb=(15, 32, 67))

abstract_text = (
    "Accurate retrospective daily rainfall estimation over complex terrain is vital for water resource management, "
    "agricultural planning, and flood risk assessment. Over South Asia, the sharp contrast between steep orographic barriers "
    "and semi-arid interior plateaus poses severe challenges for numerical reanalysis and classical statistical downscaling. "
    "In this study, we establish a leakage-free machine learning framework for retrospective daily rainfall reconstruction "
    "and extreme precipitation alerting across the 252 India Meteorological Department (IMD) 0.25° gridded land cells covering "
    "Karnataka State, India, over the 7-year period 2019–2025. Leveraging antecedent 24-hour surface fluxes and atmospheric "
    "thermodynamic variables from ECMWF ERA5-Land reanalysis, local antecedent IMD observations, and Copernicus GLO-90 DEM "
    "topography, we formulate and compare gradient-boosted decision trees using compound Poisson-Gamma (Tweedie, p=1.5) objective "
    "functions, two-stage hurdle models, and custom focal-loss classifiers. Models are trained on 2019–2022 (N = 364,644), "
    "hyperparameter-tuned and threshold-calibrated strictly on the anomalous 2023 regional drought year (N = 91,980), and evaluated "
    "out-of-sample on a frozen 2-year forward test partition (2024–2025, N = 184,212). On continuous test evaluation, the champion "
    "LightGBM Tweedie model achieves a Root Mean Squared Error (RMSE) of 7.4427 mm, Mean Absolute Error (MAE) of 2.8993 mm, Mean Bias "
    "of -0.0179 mm (-0.51%), Pearson correlation r = 0.7363, Nash-Sutcliffe Efficiency (NSE) of 0.5420, and Modified Kling-Gupta "
    "Efficiency (KGE-2012) of 0.6245, while CatBoost Tweedie demonstrates equivalent skill (RMSE = 7.4335 mm, MAE = 2.9186 mm, "
    "KGE-2012 = 0.6261). Both decisively outperform autoregressive Persistence (RMSE = 10.3582 mm), raw ERA5-Land (RMSE = 10.4550 mm), "
    "and Empirical Quantile Mapping (EQM, RMSE = 11.1970 mm). Paired stationary block bootstrapping (L = 7d, 14d; B = 2,000) proves "
    "that the ~2.93 mm RMSE reduction over Persistence is statistically significant at an empirical resolution of p < 0.0005. "
    "For operational heavy-rain detection (>= 64.5 mm), the custom focal-loss classifier (gamma = 2.0, alpha = 0.75, calibrated p* = 0.53) "
    "suppresses false alarms by 40.86% (427 vs. 722) relative to standard class weighting, achieving a Critical Success Index (CSI) "
    "of 0.3512, Precision of 58.22%, and False Alarm Ratio (FAR) of 41.78%. Permutation attribution reveals that antecedent land-surface "
    "runoff fluxes account for 54.52% of total tree gain, while feature ablation proves that ERA5 predictors provide measurable incremental "
    "predictive value (Delta RMSE = +1.0084 mm upon ablation). However, a critical structural limitation is documented: continuous "
    "conditional expectation models underestimate 100% of test events exceeding 115.6 mm (200 events), plateauing near 70–90 mm. "
    "We conclude that retrospective reanalysis hindcasting is exceptionally effective in orographically locked monsoonal regimes when "
    "land-surface runoff fluxes are incorporated, but operational hydrometeorological workflows must deploy paired asymmetric classifiers "
    "to overcome conditional mean quantile shrinkage for extreme precipitation alerting."
)
p_abs = doc.add_paragraph()
p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_abs.paragraph_format.left_indent = Inches(0.25)
p_abs.paragraph_format.right_indent = Inches(0.25)
p_abs.paragraph_format.space_after = Pt(6)
run_abs = p_abs.add_run(abstract_text)
set_font(run_abs, "Calibri", 9, italic=True, color_rgb=(40, 40, 40))

p_kw = doc.add_paragraph()
p_kw.paragraph_format.left_indent = Inches(0.25)
p_kw.paragraph_format.right_indent = Inches(0.25)
p_kw.paragraph_format.space_after = Pt(14)
run_kw_lbl = p_kw.add_run("Keywords: ")
set_font(run_kw_lbl, "Calibri", 9, bold=True, color_rgb=(15, 32, 67))
run_kw = p_kw.add_run("Hydrometeorological hindcasting, ERA5-Land reanalysis, IMD gridded rainfall, Tweedie regression, Focal loss, Orographic precipitation, Western Ghats.")
set_font(run_kw, "Calibri", 9, color_rgb=(60, 60, 60))

# SECTION 1: INTRODUCTION
add_h1("1. Introduction")
add_p(
    "Accurate estimation of daily precipitation at high spatial resolution is foundational to catchment hydrology, agricultural "
    "water allocation, reservoir dispatch, and extreme weather hazard mitigation [1], [2]. Over the Indian subcontinent, the spatial "
    "and temporal distribution of rainfall is heavily modulated by the Indian Summer Monsoon (June–September), which delivers more than "
    "75% of the annual precipitation budget [3], [6]. Karnataka State, situated in southwestern peninsular India, exemplifies the intense "
    "hydrometeorological diversity of the monsoon: it transitions abruptly from a hyper-humid coastal belt along the Arabian Sea to the "
    "rugged Western Ghats mountain range (elevations 1,000–1,800 m), which triggers massive orographic precipitation, and further into the "
    "semi-arid leeward rain-shadow of the Deccan plateau [6], [7]."
)
add_p(
    "Despite the proliferation of global atmospheric reanalysis products—most notably the European Centre for Medium-Range Weather "
    "Forecasts (ECMWF) ERA5 [1] and its high-resolution land component ERA5-Land [2]—reanalysis precipitation fields cannot be directly "
    "ingested into local hydrometeorological models without substantial post-processing. Hydrostatic atmospheric models operating at "
    "0.1° to 0.25° grid spacings smooth steep topographic ridges, misplace the exact crest of orographic precipitation, and parameterize "
    "deep moist convection, frequently leading to systemic wet biases in light drizzle, diurnal phase errors, and severe underestimation "
    "of localized torrential downpours [7]–[9]."
)
add_p(
    "Classical statistical downscaling methods, such as Empirical Quantile Mapping (EQM) [10], adjust the marginal distribution of "
    "reanalysis precipitation against observed point or gridded gauge records. However, EQM operates as a univariate transformation that "
    "assumes temporal stationarity, disrupts multi-variable physical consistency, and often degrades spatial field coherence during extreme "
    "convective episodes [11]. In recent years, machine learning (ML) architectures, particularly gradient-boosted decision trees (GBDT) "
    "and deep neural networks, have demonstrated substantial promise in learning complex, non-linear relationships between multi-level "
    "atmospheric predictors, surface energy fluxes, and observed precipitation [12], [13]."
)
add_p(
    "Nevertheless, daily precipitation modeling presents three intrinsic mathematical challenges that standard off-the-shelf machine "
    "learning formulations fail to resolve: (1) Zero-inflation and semi-continuous support, where standard Gaussian MSE penalizes errors "
    "symmetrically and inflates artificial drizzle; (2) Extreme class imbalance in hydrometeorological hazard alerting (events >= 64.5 mm "
    "comprise less than 1% of instances); and (3) Temporal autocorrelation, which causes severe information leakage under naive k-fold "
    "cross-validation [25], [26]."
)
add_p(
    "This study presents a scientifically audited, leakage-free machine learning framework for retrospective daily rainfall reconstruction "
    "over Karnataka from 2019 to 2025. This work is explicitly framed as retrospective reanalysis hindcasting and diagnostic downscaling "
    "using antecedent 24-hour accumulated ERA5-Land fluxes and previous-day IMD observations available at 03:00 UTC (08:30 IST); it does "
    "not simulate an operational real-time NWP forecast. All spatial units correspond to the 252 IMD 0.25° gridded land cells covering Karnataka."
)

# SECTION 2: STUDY AREA AND DATASETS
add_h1("2. Study Area and Datasets")
add_p(
    "The study domain encompasses Karnataka State in southern India (11.75°–18.50°N, 74.25°–78.50°E; 191,791 km²), comprising 30 administrative "
    "districts and 10 agroclimatic zones across four meteorological subdivisions: Coastal Karnataka (13 cells), Malnad (35 cells), North Interior "
    "Karnataka (NIK; 118 cells), and South Interior Karnataka (SIK; 86 cells)."
)
add_figure(fig_dir / "figure1_study_area.png", 
           "Figure 1: Geographic domain and spatial discretization of Karnataka showing elevation contours from Copernicus GLO-90 DEM, Western Ghats crest line, 252 IMD 0.25° grid cells color-coded by meteorological subdivision, and explicit identification of the 3 coastal fallback cells (Station IDs 52, 68, 125).")

add_p(
    "The ground-truth observational target is the IMD high-resolution daily gridded rainfall dataset (0.25° x 0.25°) developed by Pai et al. [3], "
    "interpolated from 6,955 quality-controlled rain-gauge stations across India. Reanalysis predictors are derived from ECMWF ERA5-Land (0.1° grid) "
    "[2], de-accumulated and aggregated over the 24-hour antecedent window ending at 03:00 UTC on day d-1. Topographic features are extracted from "
    "the Copernicus Global Digital Elevation Model (COP-DEM GLO-90) [5]."
)
add_p(
    "In the native ERA5-Land land-sea mask, exactly 3 coastal IMD grid cells—Station ID 52 (Udupi/Malpe), Station ID 68 (Kundapura), and Station ID 125 "
    "(Karwar/Ankola)—fall over Arabian Sea marine pixels. Predictors for these 3 cells were extracted via nearest-inland neighbor mapping (5.94 to "
    "15.11 km displacement). These 3 fallback cells (N = 2,193 test instances) are explicitly tracked with an indicator flag and evaluated separately."
)

df_t1 = pd.read_csv(tbl_dir / "table1_dataset_characteristics.csv")
add_table_from_df(df_t1, "Table 1: Dataset Partitioning, Spatial Extent, and Climatological Characteristics across Partitions")

# SECTION 3: METHODOLOGY
add_h1("3. Methodology")
add_figure(fig_dir / "figure2_methodology.png", 
           "Figure 2: End-to-end retrospective machine learning architecture depicting ERA5-Land flux de-accumulation, IMD gauge preparation, 24-h causal lag alignment, 63 primary feature engineering, temporal partitioning, model training, validation calibration, and block bootstrap verification.")

add_p(
    "The input matrix comprises 78 columns, of which 63 features are eligible primary predictors respecting the 24-hour antecedent causality constraint. "
    "Features are classified into Cyclic Temporal (7), Static Orography (6), Antecedent IMD Gauge Memory (12), ERA5-Land Surface Fluxes (15), and "
    "ERA5-Land Thermodynamics (17)."
)

df_t2 = pd.read_csv(tbl_dir / "table2_feature_groups.csv")
add_table_from_df(df_t2, "Table 2: Input Feature Matrix Architecture and Physical Domain Groupings")

add_h2("3.1 Model Formulations")
add_p(
    "We evaluate four model tiers: Naive Baselines (Persistence, Station DOY Climatology), Raw Reanalysis (ERA5-Land total precipitation), "
    "Statistical Downscaling (Empirical Quantile Mapping; EQM), and Machine Learning architectures. The primary machine learning regressor "
    "is formulated using the Tweedie compound Poisson-Gamma distribution [17], [18], parameterized with variance power p = 1.5:"
)
add_equation("Var(Y) = phi * mu^p,   1 < p < 2;   L_Tweedie(y, mu) = 2 * [ y^(2-p)/((1-p)(2-p)) - y * mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]")

add_p(
    "For extreme heavy rainfall detection (>= 64.5 mm), we implement a custom asymmetric Focal Loss classifier [21]:"
)
add_equation("L_Focal(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t);   h_i = w_i * p_i * (1 - p_i)")
add_p(
    "with gamma = 2.0, alpha = 0.75, and a strictly positive-definite surrogate Hessian h_i ensuring numerical stability during gradient tree splitting. "
    "The operational decision threshold was calibrated on the 2023 Validation partition to maximize CSI, yielding p* = 0.53."
)

# SECTION 4: RESULTS
add_h1("4. Results")
add_h2("4.1 Continuous Rainfall Estimation")
add_p(
    "Table 3 summarizes the out-of-sample forward evaluation on the frozen 2024–2025 test partition (N = 184,212; observed mean = 3.5085 mm, "
    "standard deviation = 11.0006 mm)."
)

df_t3 = pd.read_csv(tbl_dir / "table3_model_benchmarking.csv")
add_table_from_df(df_t3, "Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; N = 184,212)")

add_figure(fig_dir / "figure3_observed_predicted.png", 
           "Figure 3: Observed versus predicted daily rainfall. Panel A displays logarithmic hexbin density parity for LightGBM Tweedie; Panel B displays extreme quantile-quantile (Q-Q) alignment from the 90th to 99.9th percentiles.")
add_figure(fig_dir / "figure4_model_benchmarking.png", 
           "Figure 4: Multidimensional model benchmarking comparing RMSE (mm), MAE (mm), Pearson correlation (r), and Modified Kling-Gupta Efficiency (KGE-2012) across the 5-tier evaluation portfolio.")

add_p(
    "LightGBM Tweedie achieves an RMSE of 7.4427 mm, MAE of 2.8993 mm, Mean Bias of -0.0179 mm (-0.51%), Pearson r = 0.7363, NSE = 0.5420, and "
    "KGE-2012 = 0.6245. CatBoost Tweedie demonstrates equivalent skill (RMSE = 7.4335 mm, MAE = 2.9186 mm, KGE-2012 = 0.6261). CatBoost achieves a "
    "marginally lower RMSE by 0.0092 mm, while LightGBM achieves a lower MAE by 0.0193 mm and lower absolute percentage bias (0.51% vs. 1.11%). Both "
    "decisively outperform Persistence (RMSE 10.3582 mm), Raw ERA5-Land (RMSE 10.4550 mm), and EQM (RMSE 11.1970 mm)."
)

add_h2("4.2 Heavy Rainfall Detection (>= 64.5 mm)")
add_p(
    "Table 4 presents the frozen test contingency analysis for IMD Heavy Rainfall (1,267 events out of 184,212 records, prevalence 0.6878%)."
)

df_t4 = pd.read_csv(tbl_dir / "table4_heavy_rain_classification.csv")
add_table_from_df(df_t4, "Table 4: Heavy Rainfall Detection (>= 64.5 mm) Contingency & Skill Scores under Frozen Thresholds")

add_figure(fig_dir / "figure5_heavy_rain_detection.png", 
           "Figure 5: Heavy rainfall detection skill at the official 64.5 mm threshold. Panel A shows grouped contingency counts (Hits, Misses, False Alarms); Panel B shows skill radar comparing Precision, Recall, CSI, and HSS.")

add_p(
    "Focal Loss (p* = 0.53) achieves 595 Hits, 672 Misses, and 427 False Alarms, yielding Precision = 58.22%, Recall = 46.96%, FAR = 41.78%, CSI = 0.3512, "
    "and HSS = 0.5169. Relative to Class-Weighted LightGBM (722 false alarms), Focal Loss cuts false alarms by 40.86%, raising operational precision "
    "from 48.79% to 58.22% while maintaining comparable CSI."
)

add_h2("4.3 Feature Contribution & Systematic Ablation")
add_p(
    "Figure 6 and Table 7 examine physical feature attribution and ablation dynamics. Reanalysis surface runoff fluxes account for 54.52% of total tree gain, "
    "followed by antecedent IMD gauges (17.92%), atmospheric column variables (16.48%), and topography (11.08%)."
)

add_figure(fig_dir / "figure6_feature_interpretation.png", 
           "Figure 6: Feature attribution analysis. Panel A shows top 15 predictors ranked by validation permutation RMSE degradation; Panel B shows relative gain contribution by physical predictor domain.")

df_t7 = pd.read_csv(tbl_dir / "table7_feature_ablation.csv")
add_table_from_df(df_t7, "Table 7: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)")

add_p(
    "Ablation Scenario E confirms that removing ERA5-Land predictors surges Test RMSE from 7.4427 mm to 8.4511 mm (Delta RMSE = +1.0084 mm), drops KGE "
    "from 0.6245 to 0.4833, and collapses Heavy CSI from 0.2868 to 0.1536, proving their measurable incremental predictive value."
)

add_h2("4.4 Regional and Seasonal Disaggregation")
add_p(
    "Table 5 details model performance across Karnataka's 4 meteorological subdivisions and 4 climatological seasons."
)

df_t5 = pd.read_csv(tbl_dir / "table5_regional_seasonal_performance.csv")
add_table_from_df(df_t5, "Table 5: Subdivisional & Seasonal Stratified Performance on Test Partition")

add_figure(fig_dir / "figure7_regional_seasonal.png", 
           "Figure 7: Subdivisional and seasonal hydrometeorological stratification. Panel A shows RMSE and KGE-2012 across subdivisions; Panel B shows seasonal progression across Southwest Monsoon, Post-Monsoon, Pre-Monsoon, and Winter.")

add_p(
    "In orographic Coastal Karnataka and Malnad, the model achieves high correlation (r = 0.80–0.83) and hydrological skill (KGE = 0.73–0.74). Over the "
    "semi-arid interior plains (NIK and SIK), correlation encounters a distinct ceiling (r ~ 0.53–0.55; KGE ~ 0.37). The 3 coastal fallback cells "
    "(IDs 52, 68, 125; N = 2,193) achieve RMSE = 15.51 mm and KGE = 0.7504, tracking closely with the 10 native-land coastal cells (RMSE = 14.78 mm, "
    "KGE = 0.7356)."
)

add_h2("4.5 Temporal Robustness & Block Bootstrap Uncertainty")
add_p(
    "Table 6 summarizes the paired stationary block bootstrap resampling distributions across 2,000 resamples for L = 7d and L = 14d block lengths."
)

df_t6 = pd.read_csv(tbl_dir / "table6_statistical_robustness.csv")
add_table_from_df(df_t6, "Table 6: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)")

add_figure(fig_dir / "figure9_temporal_robustness.png", 
           "Figure 9: Temporal robustness and empirical bootstrap distributions. Panel A compares out-of-sample skill across 2024 and 2025; Panel B shows empirical probability density functions of paired Delta RMSE for L=7d and L=14d block lengths.")

add_p(
    "In leap-year 2024, LightGBM Tweedie achieves RMSE = 8.0416 mm and KGE = 0.6275; in 2025, it achieves RMSE = 6.7972 mm and KGE = 0.6186. Across all "
    "2,000 block bootstrap resamples under both 7-day and 14-day block lengths, zero replicates yielded Delta RMSE >= 0, establishing significance "
    "at p < 0.0005."
)

# SECTION 5: DISCUSSION
add_h1("5. Discussion")
add_p(
    "A primary physical insight is that land-surface runoff fluxes (sro_lag_1d, ro_lag_1d) contribute over 54% of total tree gain, outranking raw precipitation. "
    "In tropical monsoonal regimes, modeled surface runoff is generated only when local infiltration capacity is exceeded or topsoil approaches saturation. "
    "Runoff fluxes act as a physical integrator, filtering out transient atmospheric false alarms and retaining sustained moisture convergence."
)
add_p(
    "The performance dichotomy between orographic Western Ghats (r >= 0.80) and interior Deccan plateau (r ~ 0.53) reflects atmospheric scale limits. "
    "Ghats precipitation is mechanically forced by low-level jet advection perpendicular to the terrain ridge, which hydrostatic reanalysis resolves with "
    "high skill. Conversely, interior precipitation is governed by unorganized, localized mesoscale convective thunderstorm cells unresolvable by a 0.1° grid."
)
add_p(
    "Table 8 and Figure 8 provide an audit of continuous model behavior across the 7 official IMD rainfall intensity categories."
)

df_t8 = pd.read_csv(tbl_dir / "table8_rainfall_intensity_limitations.csv")
add_table_from_df(df_t8, "Table 8: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage (Test Set; N = 184,212)")

add_figure(fig_dir / "figure8_extreme_rainfall_errors.png", 
           "Figure 8: Rainfall intensity error profiles and quantile shrinkage. Panel A shows Mean Bias (mm) and Percentage Bias (%) across the 7 IMD categories; Panel B shows percentage of underestimated events and RMSE error spread.")

add_p(
    "Continuous models optimizing convex losses seek the conditional expectation E[Y|X]. In highly skewed distributions, predicting the conditional mean "
    "inevitably pulls predictions toward the bulk distribution, compressing predictions into a ceiling around 70–90 mm. Exactly 100% of observations "
    "exceeding 115.6 mm (200 test events) are underestimated, proving that continuous regression cannot be deployed in isolation for extreme hazard alerting."
)

# SECTION 6: LIMITATIONS
add_h1("6. Limitations")
add_p(
    "Eight structural limitations are explicitly disclosed: (1) Retrospective reanalysis hindcasting, not real-time operational NWP forecasting; "
    "(2) Temporal partition fixed on the same 252 IMD grid cells; spatial transferability to ungauged basins is untested and not claimed; "
    "(3) Systematic extreme rainfall underestimation: 100% of the 200 events >= 115.6 mm were underestimated in continuous evaluation; "
    "(4) Minor background drizzle inflation (+0.77 mm average on dry days); (5) Frozen test set was accessed for descriptive benchmarking and ablation "
    "without test-based tuning; (6) Nearest-inland extraction for the 3 coastal fallback cells (IDs 52, 68, 125); (7) Feature attribution reflects "
    "predictive association, not causal proof; and (8) The 0.1° resolution of ERA5-Land cannot resolve localized meso-gamma convective thunderstorms."
)

# SECTION 7: CONCLUSION
add_h1("7. Conclusion")
add_p(
    "This study developed a leakage-free machine learning framework for retrospective daily rainfall reconstruction over Karnataka using ERA5-Land reanalysis "
    "and IMD gridded observations (2019–2025). Gradient-boosted Tweedie decision trees reduce test RMSE by 28.15% relative to Persistence and by 33.53% "
    "relative to Empirical Quantile Mapping. Stationary paired block bootstrapping confirms significance at p < 0.0005. Reanalysis surface runoff fluxes "
    "provide dominant predictive utility, while feature ablation proves that ERA5 predictors provide measurable incremental predictive value. "
    "However, severe extreme tail quantile shrinkage above 115.6 mm demonstrates that operational workflows must deploy a dual-model paradigm pairing "
    "continuous Tweedie regression for volumetric water balance with asymmetric focal-loss classification for extreme hazard alerting."
)

# REFERENCES
add_h1("References")
df_refs = pd.read_csv(stage8_dir / "reference_verification.csv")
for _, r in df_refs.iterrows():
    p_ref = doc.add_paragraph()
    p_ref.paragraph_format.left_indent = Inches(0.3)
    p_ref.paragraph_format.first_line_indent = Inches(-0.3)
    p_ref.paragraph_format.space_after = Pt(4)
    p_ref.paragraph_format.line_spacing = 1.1
    ref_text = f"[{r['ref_num']}] {r['authors']}, \"{r['title']},\" {r['journal_or_conf']}, {r['volume_issue_pages']}, {r['year']}. DOI: {r['doi_or_url']}."
    run_ref = p_ref.add_run(ref_text)
    set_font(run_ref, "Calibri", 8.5, color_rgb=(40, 40, 40))

docx_path = stage8_dir / "Karnataka_Rainfall_Research_Paper.docx"
doc.save(docx_path)
print(f"Saved complete manuscript DOCX to {docx_path}")

