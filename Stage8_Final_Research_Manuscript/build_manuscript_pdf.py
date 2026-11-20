import os
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

stage8_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage8_Final_Research_Manuscript")
fig_dir = stage8_dir / "manuscript_figures"
tbl_dir = stage8_dir / "manuscript_tables"
pdf_path = stage8_dir / "Karnataka_Rainfall_Research_Paper.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 762, "Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka (2019–2025)")
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)
        
        # Running footer (all pages)
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(36, 40, 576, 40)
        footer_text = f"Scientific Research Paper | Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 28, footer_text)
        self.drawString(36, 28, "Hydrometeorological Data Science Research Initiative | Peer-Review Manuscript")
        self.restoreState()

# Initialize Document
doc = SimpleDocTemplate(
    str(pdf_path),
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=46,
    bottomMargin=46
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    alignment=1, # Center
    textColor=colors.HexColor("#0F2043"),
    spaceAfter=6
)

author_style = ParagraphStyle(
    "DocAuthor",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    alignment=1,
    textColor=colors.HexColor("#333333"),
    spaceAfter=3
)

meta_style = ParagraphStyle(
    "DocMeta",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    leading=11,
    alignment=1,
    textColor=colors.HexColor("#666666"),
    spaceAfter=10
)

h1_style = ParagraphStyle(
    "Heading1_Custom",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#0F2043"),
    spaceBefore=12,
    spaceAfter=4,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "Heading2_Custom",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    textColor=colors.HexColor("#1E3C72"),
    spaceBefore=8,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "Body_Custom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    alignment=4, # Justify
    textColor=colors.HexColor("#222222"),
    spaceAfter=5
)

abstract_style = ParagraphStyle(
    "Abstract_Custom",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8,
    leading=11,
    alignment=4,
    textColor=colors.HexColor("#333333")
)

caption_style = ParagraphStyle(
    "Caption_Custom",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=7.5,
    leading=10,
    alignment=4,
    textColor=colors.HexColor("#444444"),
    spaceBefore=3,
    spaceAfter=8
)

eq_style = ParagraphStyle(
    "Equation_Custom",
    parent=styles["Normal"],
    fontName="Courier-Oblique",
    fontSize=8.5,
    leading=12,
    alignment=1,
    textColor=colors.HexColor("#111111"),
    spaceBefore=4,
    spaceAfter=4
)

ref_style = ParagraphStyle(
    "Ref_Custom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=9.5,
    alignment=4,
    leftIndent=15,
    firstLineIndent=-15,
    textColor=colors.HexColor("#333333"),
    spaceAfter=3
)

story = []

# Title & Meta
story.append(Paragraph("Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)", title_style))
story.append(Paragraph("Senior Atmospheric and Machine Learning Research Consortium", author_style))
story.append(Paragraph("Hydrometeorological Data Science Research Initiative | Technical Working Group<br/>Correspondence: <code>d:\\PROJECT\\karnataka_rainfall_cds\\</code>", meta_style))

# Abstract Box
abs_text = (
    "<b>ABSTRACT — </b>Accurate retrospective daily rainfall estimation over complex terrain is vital for water resource "
    "management, agricultural planning, and flood risk assessment. Over South Asia, the sharp contrast between steep orographic barriers "
    "and semi-arid interior plateaus poses severe challenges for numerical reanalysis and classical statistical downscaling. "
    "In this study, we establish a leakage-free machine learning framework for retrospective daily rainfall reconstruction and extreme "
    "precipitation alerting across the 252 India Meteorological Department (IMD) 0.25° gridded land cells covering Karnataka State, India, "
    "over the 7-year period 2019–2025. Leveraging antecedent 24-hour surface fluxes and atmospheric thermodynamic variables from ECMWF "
    "ERA5-Land reanalysis, local antecedent IMD observations, and Copernicus GLO-90 DEM topography, we formulate and compare gradient-boosted "
    "decision trees using compound Poisson-Gamma (Tweedie, p=1.5) objective functions, two-stage hurdle models, and custom focal-loss "
    "classifiers. Models are trained on 2019–2022 (N = 364,644), hyperparameter-tuned and threshold-calibrated strictly on the anomalous 2023 "
    "regional drought year (N = 91,980), and evaluated out-of-sample on a frozen 2-year forward test partition (2024–2025, N = 184,212). "
    "On continuous test evaluation, the champion LightGBM Tweedie model achieves an RMSE of 7.4427 mm, MAE of 2.8993 mm, Mean Bias of -0.0179 mm "
    "(-0.51%), Pearson correlation r = 0.7363, Nash-Sutcliffe Efficiency (NSE) of 0.5420, and Modified Kling-Gupta Efficiency (KGE-2012) of 0.6245, "
    "while CatBoost Tweedie demonstrates equivalent skill (RMSE = 7.4335 mm, MAE = 2.9186 mm, KGE-2012 = 0.6261). Both decisively outperform "
    "autoregressive Persistence (RMSE = 10.3582 mm), raw ERA5-Land (RMSE = 10.4550 mm), and Empirical Quantile Mapping (EQM, RMSE = 11.1970 mm). "
    "Paired stationary block bootstrapping (L = 7d, 14d; B = 2,000) proves that the ~2.93 mm RMSE reduction over Persistence is statistically "
    "significant at an empirical resolution of p < 0.0005. For operational heavy-rain detection (>= 64.5 mm), the custom focal-loss classifier "
    "(gamma = 2.0, alpha = 0.75, calibrated p* = 0.53) suppresses false alarms by 40.86% (427 vs. 722) relative to standard class weighting, "
    "achieving a Critical Success Index (CSI) of 0.3512, Precision of 58.22%, and False Alarm Ratio (FAR) of 41.78%. Permutation attribution reveals "
    "that antecedent land-surface runoff fluxes account for 54.52% of total tree gain, while feature ablation proves that ERA5 predictors provide "
    "measurable incremental predictive value (Delta RMSE = +1.0084 mm upon ablation). However, a critical structural limitation is documented: "
    "continuous conditional expectation models underestimate 100% of test events exceeding 115.6 mm (200 events), plateauing near 70–90 mm. "
    "We conclude that retrospective reanalysis hindcasting is exceptionally effective in orographically locked monsoonal regimes when land-surface "
    "runoff fluxes are incorporated, but operational hydrometeorological workflows must deploy paired asymmetric classifiers to overcome "
    "conditional mean quantile shrinkage for extreme precipitation alerting.<br/><br/>"
    "<b>Keywords:</b> Hydrometeorological hindcasting, ERA5-Land reanalysis, IMD gridded rainfall, Tweedie regression, Focal loss, Orographic precipitation, Western Ghats."
)

abs_table = Table([[Paragraph(abs_text, abstract_style)]], colWidths=[540])
abs_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F9")),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(abs_table)
story.append(Spacer(1, 8))

# Helper to add image
def add_pdf_figure(img_path, caption_text, width=520, height=220):
    if Path(img_path).exists():
        img = Image(str(img_path), width=width, height=height)
        cap = Paragraph(caption_text, caption_style)
        story.append(KeepTogether([img, cap]))
        story.append(Spacer(1, 4))

# Helper to add table
def add_pdf_table(csv_path, caption_text, col_widths=None):
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    cap = Paragraph(caption_text, ParagraphStyle("TblCap", parent=caption_style, fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#0F2043"), spaceBefore=4, spaceAfter=2))
    
    table_data = []
    # Header
    hdr = [Paragraph(f"<b>{c}</b>", ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=6, leading=7, alignment=1, textColor=colors.white)) for c in df.columns]
    table_data.append(hdr)
    
    # Rows
    for _, row in df.iterrows():
        r_cells = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, float):
                val_str = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            r_cells.append(Paragraph(val_str, ParagraphStyle("TD", fontName="Helvetica", fontSize=5.5, leading=7, alignment=1, textColor=colors.HexColor("#222222"))))
        table_data.append(r_cells)
    
    if col_widths is None:
        cw = 540 / len(df.columns)
        col_widths = [cw] * len(df.columns)
        
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3C72")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D7DE")),
    ]))
    story.append(KeepTogether([cap, t]))
    story.append(Spacer(1, 6))

# SECTION 1: INTRODUCTION
story.append(Paragraph("1. Introduction", h1_style))
story.append(Paragraph(
    "Accurate estimation of daily precipitation at high spatial resolution is foundational to catchment hydrology, agricultural water allocation, reservoir dispatch, and extreme weather hazard mitigation [1], [2]. Over the Indian subcontinent, the spatial and temporal distribution of rainfall is heavily modulated by the Indian Summer Monsoon (June–September), which delivers more than 75% of the annual precipitation budget [3], [6]. Karnataka State, situated in southwestern peninsular India, exemplifies the intense hydrometeorological diversity of the monsoon: it transitions abruptly from a hyper-humid coastal belt along the Arabian Sea to the rugged Western Ghats mountain range (elevations 1,000–1,800 m), which triggers massive orographic precipitation, and further into the semi-arid leeward rain-shadow of the Deccan plateau [6], [7].",
    body_style
))
story.append(Paragraph(
    "Despite the proliferation of global atmospheric reanalysis products—most notably the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 [1] and its high-resolution land component ERA5-Land [2]—reanalysis precipitation fields cannot be directly ingested into local hydrometeorological models without substantial post-processing. Hydrostatic atmospheric models operating at 0.1° to 0.25° grid spacings smooth steep topographic ridges, misplace the exact crest of orographic precipitation, and parameterize deep moist convection, frequently leading to systemic wet biases in light drizzle, diurnal phase errors, and severe underestimation of localized torrential downpours [7]–[9]. Classical statistical downscaling methods, such as Empirical Quantile Mapping (EQM) [10], adjust the marginal distribution of reanalysis precipitation against observed gauge records, but assume temporal stationarity, disrupt multi-variable physical consistency, and often degrade spatial field coherence during extreme convective episodes [11]. Machine learning architectures have demonstrated substantial promise in learning complex, non-linear relationships between multi-level atmospheric predictors, surface fluxes, and observed precipitation [12], [13].",
    body_style
))
story.append(Paragraph(
    "Nevertheless, daily precipitation modeling presents three intrinsic mathematical challenges: (1) Zero-inflation and semi-continuous support; (2) Extreme class imbalance in hazard detection (events >= 64.5 mm comprise <1% of records); and (3) Temporal autocorrelation, which inflates performance under naive cross-validation [25], [26]. This study presents a scientifically audited, leakage-free machine learning framework for retrospective daily rainfall reconstruction over Karnataka from 2019 to 2025. This work is explicitly framed as retrospective reanalysis hindcasting and diagnostic downscaling using antecedent 24-hour accumulated ERA5-Land fluxes and previous-day IMD observations available at 03:00 UTC (08:30 IST); it does not simulate an operational real-time NWP forecast. All spatial units correspond to the 252 IMD 0.25° gridded land cells covering Karnataka.",
    body_style
))

# SECTION 2: STUDY AREA AND DATASETS
story.append(Paragraph("2. Study Area and Datasets", h1_style))
story.append(Paragraph(
    "The study domain encompasses Karnataka State in southern India (11.75°–18.50°N, 74.25°–78.50°E; 191,791 km²), comprising 30 administrative districts and 10 agroclimatic zones across four meteorological subdivisions: Coastal Karnataka (13 cells), Malnad (35 cells), North Interior Karnataka (NIK; 118 cells), and South Interior Karnataka (SIK; 86 cells).",
    body_style
))
add_pdf_figure(fig_dir / "figure1_study_area.png", 
               "Figure 1: Geographic domain and spatial discretization of Karnataka showing topography from Copernicus GLO-90 DEM, Western Ghats crest line, 252 IMD 0.25° grid cells color-coded by meteorological subdivision, and explicit identification of the 3 coastal fallback cells (Station IDs 52, 68, 125).",
               width=500, height=200)

story.append(Paragraph(
    "The ground-truth target is the IMD 0.25° gridded daily rainfall dataset (Pai et al. [3]), interpolated from 6,955 rain-gauge stations across India. Reanalysis predictors are derived from ECMWF ERA5-Land (0.1° grid) [2], de-accumulated over the 24-hour antecedent window ending at 03:00 UTC on day d-1. Topographic features are extracted from the Copernicus GLO-90 DEM [5]. In the native ERA5-Land land-sea mask, exactly 3 coastal cells (Station IDs 52, 68, 125) fall over Arabian Sea marine pixels; predictors for these cells were extracted from the nearest-inland neighbor pixel (5.94 to 15.11 km displacement) and are explicitly flagged.",
    body_style
))

add_pdf_table(tbl_dir / "table1_dataset_characteristics.csv", 
              "Table 1: Dataset Partitioning, Spatial Extent, and Climatological Characteristics across Partitions")

# SECTION 3: METHODOLOGY
story.append(Paragraph("3. Methodology", h1_style))
add_pdf_figure(fig_dir / "figure2_methodology.png", 
               "Figure 2: End-to-end retrospective machine learning architecture depicting ERA5-Land flux de-accumulation, IMD gauge preparation, 24-h causal lag alignment, 63 primary feature engineering, temporal partitioning, model training, validation calibration, and block bootstrap verification.",
               width=510, height=210)

story.append(Paragraph(
    "The master matrix contains 78 columns, with 63 features eligible as primary predictors respecting the 24-hour antecedent causality constraint (Cyclic Temporal: 7, Static Orography: 6, Antecedent IMD Gauge Memory: 12, ERA5-Land Surface Fluxes: 15, ERA5-Land Thermodynamics: 17).",
    body_style
))

add_pdf_table(tbl_dir / "table2_feature_groups.csv", 
              "Table 2: Input Feature Matrix Architecture and Physical Domain Groupings",
              col_widths=[60, 110, 50, 70, 250])

story.append(Paragraph("3.1 Model Formulations and Optimization", h2_style))
story.append(Paragraph(
    "We evaluate four model tiers: Naive Baselines (Persistence, Station DOY Climatology), Raw Reanalysis (ERA5-Land precipitation), Statistical Downscaling (Empirical Quantile Mapping; EQM), and Machine Learning architectures. The primary machine learning regressor is formulated using the Tweedie compound Poisson-Gamma distribution [17], [18], parameterized with variance power p = 1.5:",
    body_style
))
story.append(Paragraph("Var(Y) = &phi;&mu;<sup>p</sup>,   1 < p < 2;   L<sub>Tweedie</sub>(y, &mu;) = 2 &times; [ y<sup>2-p</sup>/((1-p)(2-p)) - y&mu;<sup>1-p</sup>/(1-p) + &mu;<sup>2-p</sup>/(2-p) ]", eq_style))

story.append(Paragraph(
    "For extreme heavy rainfall detection (>= 64.5 mm), we implement a custom asymmetric Focal Loss classifier [21]:",
    body_style
))
story.append(Paragraph("L<sub>Focal</sub>(p<sub>t</sub>) = -&alpha;<sub>t</sub>(1 - p<sub>t</sub>)<sup>&gamma;</sup> log(p<sub>t</sub>);   h<sub>i</sub> = w<sub>i</sub> p<sub>i</sub>(1 - p<sub>i</sub>)", eq_style))
story.append(Paragraph(
    "with gamma = 2.0, alpha = 0.75, and a strictly positive-definite surrogate Hessian h_i ensuring numerical stability during gradient tree splitting. The decision threshold was calibrated on the 2023 Validation partition to maximize CSI, yielding p* = 0.53.",
    body_style
))

# SECTION 4: RESULTS
story.append(Paragraph("4. Results", h1_style))
story.append(Paragraph("4.1 Continuous Rainfall Estimation", h2_style))
story.append(Paragraph(
    "Table 3 summarizes the out-of-sample forward evaluation on the frozen 2024–2025 test partition (N = 184,212; observed mean = 3.5085 mm, standard deviation = 11.0006 mm).",
    body_style
))

add_pdf_table(tbl_dir / "table3_model_benchmarking.csv", 
              "Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; N = 184,212)")

add_pdf_figure(fig_dir / "figure3_observed_predicted.png", 
               "Figure 3: Observed versus predicted daily rainfall. Panel A displays logarithmic hexbin density parity for LightGBM Tweedie; Panel B displays extreme quantile-quantile (Q-Q) alignment from the 90th to 99.9th percentiles.",
               width=510, height=195)
add_pdf_figure(fig_dir / "figure4_model_benchmarking.png", 
               "Figure 4: Multidimensional model benchmarking comparing RMSE (mm), MAE (mm), Pearson correlation (r), and Modified Kling-Gupta Efficiency (KGE-2012) across the 5-tier evaluation portfolio.",
               width=510, height=195)

story.append(Paragraph(
    "LightGBM Tweedie achieves an RMSE of 7.4427 mm, MAE of 2.8993 mm, Mean Bias of -0.0179 mm (-0.51%), Pearson r = 0.7363, NSE = 0.5420, and KGE-2012 = 0.6245. CatBoost Tweedie demonstrates equivalent skill (RMSE = 7.4335 mm, MAE = 2.9186 mm, KGE-2012 = 0.6261). CatBoost achieves a marginally lower RMSE by 0.0092 mm, while LightGBM achieves a lower MAE by 0.0193 mm and lower absolute percentage bias (0.51% vs. 1.11%). Both decisively outperform Persistence (RMSE 10.3582 mm), Raw ERA5-Land (RMSE 10.4550 mm), and EQM (RMSE 11.1970 mm).",
    body_style
))

story.append(Paragraph("4.2 Heavy Rainfall Detection (>= 64.5 mm)", h2_style))
story.append(Paragraph(
    "Table 4 presents the frozen test contingency analysis for IMD Heavy Rainfall (1,267 events out of 184,212 records, prevalence 0.6878%).",
    body_style
))

add_pdf_table(tbl_dir / "table4_heavy_rain_classification.csv", 
              "Table 4: Heavy Rainfall Detection (>= 64.5 mm) Contingency & Skill Scores under Frozen Thresholds")

add_pdf_figure(fig_dir / "figure5_heavy_rain_detection.png", 
               "Figure 5: Heavy rainfall detection skill at the official 64.5 mm threshold. Panel A shows grouped contingency counts (Hits, Misses, False Alarms); Panel B shows skill radar comparing Precision, Recall, CSI, and HSS.",
               width=510, height=185)

story.append(Paragraph(
    "Focal Loss (p* = 0.53) achieves 595 Hits, 672 Misses, and 427 False Alarms, yielding Precision = 58.22%, Recall = 46.96%, FAR = 41.78%, CSI = 0.3512, and HSS = 0.5169. Relative to Class-Weighted LightGBM (722 false alarms), Focal Loss cuts false alarms by 40.86%, raising operational precision from 48.79% to 58.22% while maintaining comparable CSI.",
    body_style
))

story.append(Paragraph("4.3 Feature Contribution & Systematic Ablation", h2_style))
story.append(Paragraph(
    "Figure 6 and Table 7 examine physical feature attribution and ablation dynamics. Reanalysis surface runoff fluxes account for 54.52% of total tree gain, followed by antecedent IMD gauges (17.92%), atmospheric column variables (16.48%), and topography (11.08%).",
    body_style
))

add_pdf_figure(fig_dir / "figure6_feature_interpretation.png", 
               "Figure 6: Feature attribution analysis. Panel A shows top 15 predictors ranked by validation permutation RMSE degradation; Panel B shows relative gain contribution by physical predictor domain.",
               width=510, height=185)

add_pdf_table(tbl_dir / "table7_feature_ablation.csv", 
              "Table 7: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)")

story.append(Paragraph(
    "Ablation Scenario E confirms that removing ERA5-Land predictors surges Test RMSE from 7.4427 mm to 8.4511 mm (Delta RMSE = +1.0084 mm), drops KGE from 0.6245 to 0.4833, and collapses Heavy CSI from 0.2868 to 0.1536, proving their measurable incremental predictive value.",
    body_style
))

story.append(Paragraph("4.4 Regional and Seasonal Disaggregation", h2_style))
story.append(Paragraph(
    "Table 5 details model performance across Karnataka's 4 meteorological subdivisions and 4 climatological seasons.",
    body_style
))

add_pdf_table(tbl_dir / "table5_regional_seasonal_performance.csv", 
              "Table 5: Subdivisional & Seasonal Stratified Performance on Test Partition")

add_pdf_figure(fig_dir / "figure7_regional_seasonal.png", 
               "Figure 7: Subdivisional and seasonal hydrometeorological stratification. Panel A shows RMSE and KGE-2012 across subdivisions; Panel B shows seasonal progression across Southwest Monsoon, Post-Monsoon, Pre-Monsoon, and Winter.",
               width=510, height=185)

story.append(Paragraph(
    "In orographic Coastal Karnataka and Malnad, the model achieves high correlation (r = 0.80–0.83) and hydrological skill (KGE = 0.73–0.74). Over the semi-arid interior plains (NIK and SIK), correlation encounters a distinct ceiling (r ~ 0.53–0.55; KGE ~ 0.37). The 3 coastal fallback cells (IDs 52, 68, 125; N = 2,193) achieve RMSE = 15.51 mm and KGE = 0.7504, tracking closely with the 10 native-land coastal cells (RMSE = 14.78 mm, KGE = 0.7356).",
    body_style
))

story.append(Paragraph("4.5 Temporal Robustness & Block Bootstrap Uncertainty", h2_style))
story.append(Paragraph(
    "Table 6 summarizes the paired stationary block bootstrap resampling distributions across 2,000 resamples for L = 7d and L = 14d block lengths.",
    body_style
))

add_pdf_table(tbl_dir / "table6_statistical_robustness.csv", 
              "Table 6: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)")

add_pdf_figure(fig_dir / "figure9_temporal_robustness.png", 
               "Figure 9: Temporal robustness and empirical bootstrap distributions. Panel A compares out-of-sample skill across 2024 and 2025; Panel B shows empirical probability density functions of paired Delta RMSE for L=7d and L=14d block lengths.",
               width=510, height=185)

story.append(Paragraph(
    "In leap-year 2024, LightGBM Tweedie achieves RMSE = 8.0416 mm and KGE = 0.6275; in 2025, it achieves RMSE = 6.7972 mm and KGE = 0.6186. Across all 2,000 block bootstrap resamples under both 7-day and 14-day block lengths, zero replicates yielded Delta RMSE >= 0, establishing significance at p < 0.0005.",
    body_style
))

# SECTION 5: DISCUSSION
story.append(Paragraph("5. Discussion", h1_style))
story.append(Paragraph(
    "A primary physical insight is that land-surface runoff fluxes (sro_lag_1d, ro_lag_1d) contribute over 54% of total tree gain, outranking raw precipitation. In tropical monsoonal regimes, modeled surface runoff is generated only when local infiltration capacity is exceeded or topsoil approaches saturation. Runoff fluxes act as a physical integrator, filtering out transient atmospheric false alarms and retaining sustained moisture convergence.",
    body_style
))
story.append(Paragraph(
    "The performance dichotomy between orographic Western Ghats (r >= 0.80) and interior Deccan plateau (r ~ 0.53) reflects atmospheric scale limits. Ghats precipitation is mechanically forced by low-level jet advection perpendicular to the terrain ridge, which hydrostatic reanalysis resolves with high skill. Conversely, interior precipitation is governed by unorganized, localized mesoscale convective thunderstorm cells unresolvable by a 0.1° grid.",
    body_style
))
story.append(Paragraph(
    "Table 8 and Figure 8 provide an audit of continuous model behavior across the 7 official IMD rainfall intensity categories.",
    body_style
))

add_pdf_table(tbl_dir / "table8_rainfall_intensity_limitations.csv", 
              "Table 8: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage (Test Set; N = 184,212)")

add_pdf_figure(fig_dir / "figure8_extreme_rainfall_errors.png", 
               "Figure 8: Rainfall intensity error profiles and quantile shrinkage. Panel A shows Mean Bias (mm) and Percentage Bias (%) across the 7 IMD categories; Panel B shows percentage of underestimated events and RMSE error spread.",
               width=510, height=185)

story.append(Paragraph(
    "Continuous models optimizing convex losses seek the conditional expectation E[Y|X]. In highly skewed distributions, predicting the conditional mean inevitably pulls predictions toward the bulk distribution, compressing predictions into a ceiling around 70–90 mm. Exactly 100% of observations exceeding 115.6 mm (200 test events) are underestimated, proving that continuous regression cannot be deployed in isolation for extreme hazard alerting.",
    body_style
))

# SECTION 6: LIMITATIONS
story.append(Paragraph("6. Limitations", h1_style))
story.append(Paragraph(
    "Eight structural limitations are explicitly disclosed: (1) Retrospective reanalysis hindcasting, not real-time operational NWP forecasting; "
    "(2) Temporal partition fixed on the same 252 IMD grid cells; spatial transferability to ungauged basins is untested and not claimed; "
    "(3) Systematic extreme rainfall underestimation: 100% of the 200 events >= 115.6 mm were underestimated in continuous evaluation; "
    "(4) Minor background drizzle inflation (+0.77 mm average on dry days); (5) Frozen test set was accessed for descriptive benchmarking and ablation "
    "without test-based tuning; (6) Nearest-inland extraction for the 3 coastal fallback cells (IDs 52, 68, 125); (7) Feature attribution reflects "
    "predictive association, not causal proof; and (8) The 0.1° resolution of ERA5-Land cannot resolve localized meso-gamma convective thunderstorms.",
    body_style
))

# SECTION 7: CONCLUSION
story.append(Paragraph("7. Conclusion", h1_style))
story.append(Paragraph(
    "This study developed a leakage-free machine learning framework for retrospective daily rainfall reconstruction over Karnataka using ERA5-Land reanalysis and IMD gridded observations (2019–2025). Gradient-boosted Tweedie decision trees reduce test RMSE by 28.15% relative to Persistence and by 33.53% relative to Empirical Quantile Mapping. Stationary paired block bootstrapping confirms significance at p < 0.0005. Reanalysis surface runoff fluxes provide dominant predictive utility, while feature ablation proves that ERA5 predictors provide measurable incremental predictive value. However, severe extreme tail quantile shrinkage above 115.6 mm demonstrates that operational workflows must deploy a dual-model paradigm pairing continuous Tweedie regression for volumetric water balance with asymmetric focal-loss classification for extreme hazard alerting.",
    body_style
))

# REFERENCES
story.append(Paragraph("References", h1_style))
df_refs = pd.read_csv(stage8_dir / "reference_verification.csv")
for _, r in df_refs.iterrows():
    ref_text = f"<b>[{r['ref_num']}]</b> {r['authors']}, \"{r['title']},\" <i>{r['journal_or_conf']}</i>, {r['volume_issue_pages']}, {r['year']}. DOI: {r['doi_or_url']}."
    story.append(Paragraph(ref_text, ref_style))

print("--- Building PDF Document ---")
doc.build(story, canvasmaker=NumberedCanvas)
print(f"Saved complete manuscript PDF to {pdf_path}")

