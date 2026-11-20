import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import pandas as pd
from pathlib import Path

stage9_1_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage9_1_Final_Validated_Manuscript")
stage6_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage6_Robustness_Generalisation")
fig_dir = stage9_1_dir / "manuscript_figures"
tbl_dir = stage9_1_dir / "manuscript_tables"
out_docx = stage9_1_dir / "Karnataka_Rainfall_Research_Paper_Final.docx"

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
    set_font(run, "Calibri", 17, bold=True, color_rgb=(15, 32, 67))

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
    p.paragraph_format.space_before = Pt(13)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_font(run, "Calibri", 12.5, bold=True, color_rgb=(15, 32, 67))

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_font(run, "Calibri", 10.5, bold=True, color_rgb=(30, 60, 114))

def add_p(text, space_after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_font(run, "Calibri", 9.5, color_rgb=(30, 30, 30))
    return p

def add_equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_font(run, "Cambria Math", 9.5, italic=True, color_rgb=(20, 20, 20))

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
    cap.paragraph_format.space_after = Pt(9)
    run_cap = cap.add_run(caption)
    set_font(run_cap, "Calibri", 8.5, italic=True, color_rgb=(60, 60, 60))

def add_table_from_df(df, caption):
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.keep_with_next = True
    run_cap = cap.add_run(caption)
    set_font(run_cap, "Calibri", 9, bold=True, color_rgb=(15, 32, 67))

    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header Row
    hdr_cells = t.rows[0].cells
    for i, col_name in enumerate(df.columns):
        hdr_cells[i].text = str(col_name)
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1E3C72"/>')
        hdr_cells[i]._tc.get_or_add_tcPr().append(shading)
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                set_font(r, "Calibri", 8, bold=True, color_rgb=(255, 255, 255))
    
    # Data Rows
    for r_idx, row in df.iterrows():
        row_cells = t.rows[r_idx + 1].cells
        bg_color = "F4F6F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val) if pd.notna(val) else "N/A"
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
            row_cells[c_idx]._tc.get_or_add_tcPr().append(shading)
            for p in row_cells[c_idx].paragraphs:
                p.paragraph_format.space_before = Pt(1.5)
                p.paragraph_format.space_after = Pt(1.5)
                for r in p.runs:
                    set_font(r, "Calibri", 8, color_rgb=(30, 30, 30))

print("--- Building Final Validated DOCX Manuscript ---")

# TITLE & FRONT MATTER
add_title("Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)")
add_authors("[Author Names Blinded for Peer Review / To Be Inserted by Author]")
add_meta("[Department / Institutional Affiliation To Be Inserted by Author] | Correspondence Email To Be Inserted Upon Acceptance")

# ABSTRACT
add_h1("Abstract")
add_p(
    "Accurate retrospective reconstruction of daily precipitation across topographically complex tropical regions is fundamental "
    "for regional water resource accounting, agricultural monitoring, and hydrological gap-filling. Reanalysis products such as ECMWF ERA5-Land "
    "provide continuous atmospheric and land-surface fields but exhibit substantial localized biases over heterogeneous terrain. Here, we present a "
    "machine-learning-based daily rainfall hindcasting framework that reconstructs ground-truth precipitation across 252 India Meteorological Department "
    "(IMD) 0.25° grid cells covering Karnataka, India, over the 7-year period 2019–2025 (N = 640,836 grid-cell-days). Ingesting 63 primary antecedent "
    "predictors derived from ERA5-Land surface fluxes, thermal states, soil moisture profiles, DEM orography, and antecedent IMD gauge observations under "
    "strict causal latency constraints (T_issue = 03:00 UTC on day t-1; target window = 03:00 UTC day t-1 to 03:00 UTC day t), we evaluate regularized "
    "linear models, gradient-boosted decision trees, and two-stage hurdle architectures. On the strictly held-out forward test partition (2024–2025; "
    "N = 184,212), machine learning models achieve strong performance with distinct multi-criteria trade-offs: Hurdle Expected Value attains the lowest "
    "overall root-mean-square error (RMSE = 7.4241 mm) and highest Nash-Sutcliffe Efficiency (NSE = 0.5443); CatBoost Tweedie attains the highest "
    "Modified Kling-Gupta Efficiency (KGE-2012 = 0.6261); and LightGBM Tweedie achieves the lowest mean absolute error (MAE = 2.8993 mm) and lowest "
    "mean bias (-0.0179 mm; -0.51%). Paired stationary block bootstrapping (B = 2,000 resamples) demonstrates significant RMSE reductions over persistence "
    "(-2.925 mm; p < 0.0005) and raw ERA5-Land reanalysis (-3.011 mm; p < 0.0005). Feature attribution reveals that antecedent land-surface runoff fluxes "
    "account for 54.52% of split gain in heavy rain classification and +2.3545 mm validation permutation RMSE degradation in continuous regression, acting "
    "as a model-derived hydrometeorological memory proxy for antecedent soil saturation. However, severe conditional expectation shrinkage causes "
    "continuous models to underestimate 100.0% of events exceeding 115.6 mm (200 test events), and the dedicated heavy-rain focal classifier (>=64.5 mm) "
    "achieves a test probability of detection of 46.96% (precision = 58.22%, CSI = 0.3512 at decision threshold p* = 0.53), missing 53.04% of observed events. "
    "These findings prove that while machine learning effectively reconstructs regional daily water budgets, it represents a diagnostic retrospective "
    "reconstruction engine rather than an operational flood warning system, requiring extreme-value post-processing for localized convective extremes."
)

# 1. INTRODUCTION
add_h1("1. Introduction")
add_p(
    "Accurate spatio-temporal quantification of daily precipitation is vital for water resource management, agricultural planning, "
    "and hydrometeorological hazard assessment [1], [2]. In tropical monsoon environments, rainfall exhibits pronounced intermittency, high dynamic "
    "range, zero-inflation, and complex orographic modulation [3], [6]. In southwestern Peninsular India, the state of Karnataka exemplifies these "
    "hydrometeorological complexities, spanning from the steep windward escarpments of the Western Ghats (annual precipitation > 3,500 mm) to the "
    "semi-arid rain-shadow plains of the Deccan plateau (< 500 mm) [7], [9]."
)
add_p(
    "While atmospheric reanalyses such as ECMWF ERA5-Land provide continuous global hydrometeorological fluxes at ~9 km resolution [2], their operational "
    "precipitation estimates exhibit substantial systematic errors, orographic displacement, and convective drizzle biases [10], [11]. In this study, "
    "we formulate an end-to-end retrospective daily rainfall reconstruction framework combining ERA5-Land reanalysis and the IMD 0.25° gridded observation "
    "network across Karnataka from 2019 through 2025 [3], [4]. Using 63 primary antecedent physical predictors, we conduct a multi-criteria investigation "
    "evaluating regularized linear baselines, tree-based machine learning architectures (LightGBM [14], CatBoost [15], XGBoost [16]), and two-stage hurdle "
    "models under strict out-of-sample temporal partitioning."
)

# 2. STUDY AREA & DATASETS
add_h1("2. Study Area and Hydrometeorological Datasets")
add_p(
    "Karnataka encompasses 191,791 km² across 10 agroclimatic zones and four official IMD meteorological subdivisions: Coastal Karnataka, Malnad (Western Ghats), "
    "North Interior Karnataka (NIK), and South Interior Karnataka (SIK). The state terrain ranges from sea level to 1,925 m above mean sea level."
)

add_figure(fig_dir / "figure1_study_area.png", 
           "Figure 1: High-resolution physiographic map of Karnataka depicting elevation from the Copernicus GLO-90 DEM, Western Ghats crest line, 10 agroclimatic zones, and the 252 IMD 0.25° gridded land cells across the state. Inset shows India-wide geographical context.")

df_t1 = pd.read_csv(tbl_dir / "table1_dataset_characteristics.csv")
add_table_from_df(df_t1, "Table 1: Spatiotemporal Dataset Architecture and Chronological Partitions (2019–2025)")

add_p(
    "Observational ground truth is provided by the IMD high-resolution (0.25° x 0.25°) daily gridded rainfall dataset [3], [4]. Daily values correspond to the "
    "24-hour accumulation ending at 08:30 Indian Standard Time (IST; equivalent to 03:00 UTC). Reanalysis predictors are derived from ECMWF ERA5-Land [2] "
    "and spatially interpolated to the 252 IMD grid cells using bilinear interpolation, with nearest-land neighbor fallback implemented for exactly 3 coastal "
    "boundary cells (Grid-Cell IDs 52, 68, 125 out of 13 coastal cells) that touch open ocean cells."
)

# 3. METHODOLOGY
add_h1("3. Methodology and Experimental Formulation")
add_h2("3.1 ERA5-Land De-accumulation Physics and Temporal Alignment")
add_p(
    "In raw ECMWF ERA5-Land records, accumulated surface variables (total precipitation tp, surface runoff sro, total runoff ro, evaporation e, solar and "
    "thermal radiation ssrd, strd, and turbulent heat fluxes slhf, sshf) are stored as running integrals reset to zero at 00:00 UTC each day. Hourly physical "
    "increments are recovered via backward differencing: Delta x_1 = x_1 at 01:00 UTC, and Delta x_h = x_h - x_{h-1} for h in {2, ..., 24}."
)
add_p(
    "To establish an unlagged baseline benchmark product (Product B, Raw ERA5-Land / EQM), hourly increments are integrated across the 24-hour window from "
    "04:00 UTC on day t-1 through 03:00 UTC on day t: X_t^{IMD} = sum_{h=4}^{24} Delta x_{t-1, h} + sum_{h=1}^3 Delta x_{t, h}. This allows direct physical "
    "comparison against ground-truth IMD daily rainfall y_t over the exact same accumulation window."
)
add_p(
    "Crucially, to predict rainfall on day t in our retrospective machine-learning pipeline, the prediction issue time is fixed at T_issue = 03:00 UTC on day t-1. "
    "All predictor features are strictly restricted to variables completed on or before T_issue. Therefore, the machine learning input feature is lagged by "
    "one full day (era5_tp_lag_1d = X_{t-1}^{IMD}), which spans 04:00 UTC day t-2 to 03:00 UTC day t-1. All rolling features apply an explicit shift(1). "
    "Contemporaneous target-day reanalysis columns are purged prior to training, ensuring zero future data leakage."
)

add_figure(fig_dir / "figure2_methodology.png", 
           "Figure 2: End-to-end retrospective machine learning architecture depicting ERA5-Land flux de-accumulation, IMD gauge preparation, 24-h causal lag alignment, 63 primary feature engineering, temporal partitioning, model training, and multi-criteria evaluation.")

add_h2("3.2 Feature Matrix Architecture")
add_p(
    "Table 2 summarizes the 63 primary predictors and auxiliary data-quality indicators engineered across seven physical domains."
)

df_t2 = pd.read_csv(tbl_dir / "table2_feature_groups.csv")
add_table_from_df(df_t2, "Table 2: Input Feature Groups, Predictor Schema, and Pipeline Gating")

add_h2("3.3 Model Formulations Across Tiers")
add_p(
    "We implement a hierarchical modeling framework across five structural tiers:"
)
add_p(
    "• Tier 0 (Naive Baselines): Persistence (y_t = y_{t-1}) and Grid-cell DOY Climatology (historical day-of-year mean).\n"
    "• Tier 1 (Raw Reanalysis Benchmark): Uncalibrated antecedent ERA5-Land precipitation.\n"
    "• Tier 2 (Statistical Downscaling): Empirical Quantile Mapping (EQM) using 100 empirical quantile knots fitted per grid cell on 2019–2022.\n"
    "• Tier 3 (Regularized Linear Baselines): Ridge Regression (L2 regularization, alpha = 100.0) and ElasticNet Regression (alpha = 0.1, l1_ratio = 0.5) "
    "fitted on standardized features.\n"
    "• Tier 4 (Advanced Machine Learning): LightGBM Tweedie (p=1.5) [14], CatBoost Tweedie (p=1.5) [15], XGBoost Tweedie [16], LightGBM Log-Ordinary with "
    "Duan Smearing [20], and Two-Stage Hurdle Expected Value models [19]."
)
add_p(
    "For heavy-rain hazard detection (>= 64.5 mm), we train a dedicated LightGBM classifier optimizing custom Focal Loss [21] with a positive-definite "
    "surrogate Hessian approximation: h_i = w_i * p_i * (1 - p_i), with operational decision threshold p* = 0.53 selected on 2023 validation data to maximize CSI."
)

# 4. RESULTS
add_h1("4. Results")
add_h2("4.1 Continuous Rainfall Estimation & Multi-Criteria Benchmarking")
add_p(
    "Table 3 presents out-of-sample forward evaluation on the frozen 2024–2025 test partition (N = 184,212; statewide observed mean = 3.507 mm/day)."
)

df_t3 = pd.read_csv(tbl_dir / "table3_model_benchmarking.csv")
add_table_from_df(df_t3, "Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; N = 184,212)")

add_figure(fig_dir / "figure3_observed_predicted.png", 
           "Figure 3: Density parity hexbin between observed and LightGBM Tweedie predictions (Panel A) and cumulative distribution functions across models (Panel B).")

add_figure(fig_dir / "figure4_model_benchmarking.png", 
           "Figure 4: Comparative bar panels showing RMSE, MAE, Pearson r, and KGE-2012 across tiers on the frozen test partition.")

add_p(
    "The evaluation reveals nuanced multi-criteria trade-offs among the top three ML architectures: Hurdle Expected Value achieves the lowest overall RMSE "
    "(7.4241 mm), highest Pearson correlation (r = 0.7380), and highest NSE (0.5443); CatBoost Tweedie achieves the highest Modified KGE-2012 (0.6261) "
    "and second-lowest RMSE (7.4335 mm); and LightGBM Tweedie achieves the lowest MAE (2.8993 mm) and lowest absolute bias (-0.0179 mm; -0.51%). "
    "All three architectures substantially outperform regularized linear models (Ridge RMSE = 7.9750 mm; ElasticNet RMSE = 8.0156 mm), raw ERA5-Land "
    "(RMSE = 10.4550 mm), persistence (RMSE = 10.3582 mm), and EQM (RMSE = 11.1970 mm)."
)

add_h2("4.2 Heavy Rainfall Detection and Operational Classifier Bounding")
add_p(
    "Table 4 summarizes the performance of the dedicated heavy rainfall classification models for events >= 64.5 mm (prevalence = 0.6878%)."
)

df_t4 = pd.read_csv(tbl_dir / "table4_heavy_rain_classification.csv")
add_table_from_df(df_t4, "Table 4: Dedicated Heavy Rainfall (>=64.5 mm) Contingency & Skill Score Evaluation (Test 2024–2025)")

add_figure(fig_dir / "figure5_heavy_rain_detection.png", 
           "Figure 5: Synoptic heavy-rain detection performance. Panel A shows contingency breakdown (Hits, Misses, False Alarms); Panel B shows Precision-Recall curves across classification architectures.")

add_p(
    "At the validation-selected decision threshold p* = 0.53, the LightGBM Focal Loss classifier achieves a test POD of 46.96% (595 hits), Precision of "
    "58.22%, CSI of 0.3512, and FAR of 41.78% (427 false alarms), missing 53.04% (672 misses out of 1,267 observed heavy events). While dramatically "
    "superior to uncalibrated reanalysis (POD = 18.23%, CSI = 0.1233) and persistence (POD = 38.20%, CSI = 0.2361), this confirms that the classifier "
    "misses more than half of observed heavy events and cannot serve as an autonomous real-time flood warning system."
)

add_h2("4.3 Feature Attribution: Gain, Permutation, and Ablation Dynamics")
add_p(
    "Figure 6 and Table 7 examine physical feature attribution and ablation dynamics across the 63 primary predictors."
)

add_figure(fig_dir / "figure6_feature_interpretation.png", 
           "Figure 6: Feature attribution analysis. Panel A shows top 15 predictors ranked by validation permutation RMSE degradation; Panel B shows relative gain contribution by physical predictor domain.")

df_t7 = pd.read_csv(tbl_dir / "table7_feature_ablation.csv")
add_table_from_df(df_t7, "Table 7: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)")

add_p(
    "In the LightGBM Focal Loss heavy rainfall classifier, sub-surface runoff (ro_lag_1d, 28.11%) and surface runoff (sro_lag_1d, 26.41%) together account for "
    "54.52% of total tree split gain. In continuous LightGBM Tweedie regression, surface runoff accounts for 34.61% of tree gain and skin reservoir "
    "content accounts for 28.28%. In validation permutation importance (N = 91,980), surface runoff ranks first (+2.3545 mm RMSE degradation, +42.16% increase), "
    "followed by skin reservoir content (+2.1241 mm) and downward longwave flux (+1.7960 mm). Ablation Scenario E confirms that excluding ERA5-Land predictors "
    "degrades Test RMSE from 7.4427 mm to 8.4511 mm (+1.0084 mm), demonstrating their measurable incremental predictive value."
)

add_h2("4.4 Regional and Seasonal Disaggregation")
add_p(
    "Table 5 details model performance across Karnataka's four meteorological subdivisions and four climatological seasons."
)

df_t5 = pd.read_csv(tbl_dir / "table5_regional_seasonal_performance.csv")
add_table_from_df(df_t5, "Table 5: Subdivisional & Seasonal Stratified Performance on Test Partition")

add_figure(fig_dir / "figure7_regional_seasonal.png", 
           "Figure 7: Subdivisional and seasonal hydrometeorological stratification. Panel A shows RMSE and KGE-2012 across subdivisions; Panel B shows seasonal progression across Southwest Monsoon, Post-Monsoon, Pre-Monsoon, and Winter.")

add_p(
    "In orographic Coastal Karnataka and Malnad, the model achieves high correlation (r = 0.80–0.83) and hydrological skill (KGE = 0.73–0.74). Over the "
    "semi-arid interior plains (NIK and SIK), correlation encounters a distinct ceiling (r ~ 0.53–0.55; KGE ~ 0.37). The 3 coastal fallback cells "
    "(Grid-Cell IDs 52, 68, 125; N = 2,193) achieve RMSE = 15.51 mm and KGE = 0.7504, tracking closely with the 10 native-land coastal cells (RMSE = 14.78 mm, "
    "KGE = 0.7356)."
)

add_h2("4.5 Temporal Generalisation Across Forward Evaluation Years (2024 vs. 2025)")
add_p(
    "To evaluate out-of-sample temporal generalisation, model performance was audited separately for the two held-out forward calendar years in Table 6A. "
    "In 2024 (N = 92,232; active monsoon convective year), LightGBM Tweedie achieved RMSE = 7.5094 mm, MAE = 3.0074 mm, Bias = +0.3442 mm (+10.26%), "
    "r = 0.7320, NSE = 0.5346, and KGE-2012 = 0.5676. In 2025 (N = 91,980; subdued monsoon year), LightGBM Tweedie achieved RMSE = 7.3752 mm, "
    "MAE = 2.7909 mm, Bias = -0.3810 mm (-10.40%), r = 0.7426, NSE = 0.5493, and KGE-2012 = 0.6541. CatBoost Tweedie achieved RMSE of 7.5253 mm in 2024 "
    "and 7.3403 mm in 2025; Hurdle Expected Value achieved RMSE of 7.5210 mm in 2024 and 7.3257 mm in 2025."
)

df_yearly = pd.read_csv(stage6_dir / "yearly_performance_results.csv")
y24_rows = df_yearly[(df_yearly["temporal_stratum"].isin(["Year_2024_Full", "Year_2025_Full"])) & 
                     (df_yearly["model"].isin(["LightGBM_Tweedie", "CatBoost_Tweedie", "Hurdle_ExpectedValue", "Persistence", "Raw_ERA5", "EQM"]))].copy()
y24_clean = y24_rows[["temporal_stratum", "model", "sample_count", "obs_mean", "pred_mean", "MAE", "RMSE", "Bias", "PBias_pct", "Pearson_r", "NSE", "KGE_2012"]].copy()
y24_clean.columns = ["Strata", "Model", "N", "Obs_Mean", "Pred_Mean", "MAE_mm", "RMSE_mm", "Bias_mm", "PBias_pct", "Pearson_r", "NSE", "KGE_2012"]
add_table_from_df(y24_clean, "Table 6A: Annual Out-of-Sample Performance Breakdown: 2024 (Leap Year) vs. 2025")

df_t6 = pd.read_csv(tbl_dir / "table6_statistical_robustness.csv")
add_table_from_df(df_t6, "Table 6B: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)")

add_figure(fig_dir / "figure8_temporal_robustness_seq.png", 
           "Figure 8: Temporal robustness and block bootstrap confidence intervals. Panel A shows annual RMSE consistency bar comparison across 2024 and 2025 forward evaluation years; Panel B shows paired block bootstrap 95% confidence interval error bars (B = 2,000 replicates, p < 0.0005) for L=7d and L=14d block lengths.")

add_p(
    "Across all 2,000 block bootstrap resamples under both 7-day and 14-day block lengths, zero replicates yielded Delta RMSE >= 0, establishing "
    "significance at p < 0.0005 under temporal persistence."
)

add_h2("4.6 Rainfall Intensity Limitations & Extreme Quantile Shrinkage")
add_p(
    "Table 8 and Figure 9 provide an audit of continuous model behavior across the seven official IMD rainfall intensity categories."
)

df_t8 = pd.read_csv(tbl_dir / "table8_rainfall_intensity_limitations.csv")
add_table_from_df(df_t8, "Table 8: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage (Test Set; N = 184,212)")

add_figure(fig_dir / "figure9_extreme_rainfall_errors_seq.png", 
           "Figure 9: Rainfall intensity error profiles and quantile shrinkage. Panel A shows Mean Bias (mm) and Percentage Bias (%) across the 7 IMD categories; Panel B shows percentage of underestimated events and RMSE error spread across intensity intervals.")

add_p(
    "Continuous models optimizing convex losses seek the conditional expectation E[Y|X]. In highly skewed distributions, predicting the conditional mean "
    "inevitably pulls predictions toward the bulk distribution, compressing predictions into an empirical ceiling around 70–90 mm. Exactly 100.0% of "
    "observations exceeding 115.6 mm (200 test events, comprising 192 Very Heavy and 8 Extremely Heavy events) are underestimated, demonstrating that "
    "continuous regression cannot be deployed in isolation for extreme hydraulic infrastructure sizing."
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
    "classifier serves as a regional synoptic alerting filter (CSI = 0.3512, precision = 58.22%), but misses 53.04% of observed heavy events."
)

# 6. LIMITATIONS
add_h1("6. Limitations")
add_p(
    "Key structural limitations must be acknowledged: (1) Reanalysis latency precludes real-time forecasting; (2) Tail quantile shrinkage causes severe "
    "underestimation for rainfall exceeding 115.6 mm; (3) Spatial generalisation is bounded to the 252 Karnataka grid cells and should not be transferred "
    "to differing climatic zones without localized retuning; and (4) The 3 coastal fallback cells (Grid-Cell IDs 52, 68, 125) rely on nearest-inland neighbor "
    "reanalysis values due to land-sea boundary masking."
)

# 7. DATA AVAILABILITY & ACKNOWLEDGMENTS
add_h1("7. Data and Code Availability")
add_p(
    "All modeling workflows, feature extraction scripts, trained model joblib files, and validation routines are available in the project reproducibility "
    "package. Gridded daily IMD precipitation data are maintained by the National Data Centre (NDC), India Meteorological Department, Pune (accessible upon "
    "registration). ERA5-Land reanalysis datasets are publicly available from the Copernicus Climate Data Store under an open license."
)

add_h1("8. Acknowledgments")
add_p(
    "The authors acknowledge the India Meteorological Department (IMD) for providing high-resolution gridded rainfall observations and ECMWF/Copernicus "
    "Climate Change Service (C3S) for ERA5-Land reanalysis data. [Grant numbers, funding agencies, and computational facility acknowledgments to be inserted "
    "by author prior to publication]."
)

# REFERENCES
add_h1("References")
refs = [
    "[1] H. Hersbach, B. Bell, P. Berrisford, S. Hirahara, A. Horányi, J. Muñoz-Sabater, J. Nicolas, C. Peubey, R. Radu, D. Schepers, et al., \"The ERA5 global reanalysis,\" Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730, pp. 1999–2049, 2020. DOI: 10.1002/qj.3803.",
    "[2] J. Muñoz-Sabater, E. Dutra, A. Agustí-Panareda, C. Albergel, G. Arduini, G. Balsamo, S. Boussetta, M. Choulga, S. Harrigan, H. Hersbach, et al., \"ERA5-Land: A state-of-the-art global reanalysis dataset for land applications,\" Earth System Science Data, vol. 13, no. 9, pp. 4349–4383, 2021. DOI: 10.5194/essd-13-4349-2021.",
    "[3] D. S. Pai, L. Sridhar, M. Rajeevan, O. P. Sreejith, N. S. Satbhai, and B. Mukhopadhyay, \"Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region,\" MAUSAM, vol. 65, no. 1, pp. 1–18, 2014. DOI: 10.54302/mausam.v65i1.851.",
    "[4] M. Rajeevan, J. Bhate, J. D. Kale, and B. Lal, \"High resolution daily gridded rainfall data for the Indian region: Analysis of break and active monsoon spells,\" Current Science, vol. 91, no. 3, pp. 296–306, 2006.",
    "[5] Airbus Defence and Space and European Space Agency, \"Copernicus Global Digital Elevation Model (COP-DEM GLO-90) User Guide,\" European Space Agency Technical Documentation, Issue 5.0, pp. 1–38, 2020. DOI: 10.5270/esa-c1qtxnn.",
    "[6] S. Gadgil, \"The Indian monsoon and its variability,\" Annual Review of Earth and Planetary Sciences, vol. 31, no. 1, pp. 429–467, 2003. DOI: 10.1146/annurev.earth.31.100901.141251.",
    "[7] P. A. Francis and S. Gadgil, \"Intense rainfall events over the west coast of India,\" Meteorology and Atmospheric Physics, vol. 94, no. 1–4, pp. 27–42, 2006. DOI: 10.1007/s00703-005-0167-2.",
    "[8] R. A. Houze Jr., D. C. Wilton, and B. F. Smull, \"Monsoon convection in the Himalayan region as seen by the TRMM Precipitation Radar,\" Quarterly Journal of the Royal Meteorological Society, vol. 133, no. 626, pp. 1389–1411, 2007. DOI: 10.1002/qj.106.",
    "[9] J. A. Phadtare, J. K. Fletcher, A. N. Ross, A. G. Turner, and R. K. H. Schiemann, \"Froude-number-based rainfall regimes over the Western Ghats mountains of India,\" Quarterly Journal of the Royal Meteorological Society, vol. 148, no. 749, pp. 3738–3755, 2022. DOI: 10.1002/qj.4367.",
    "[10] M. J. Themeßl, A. Gobiet, and A. Leuprecht, \"Empirical-statistical downscaling and error correction of daily precipitation from regional climate models,\" International Journal of Climatology, vol. 31, no. 10, pp. 1530–1544, 2011. DOI: 10.1002/joc.2168.",
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

