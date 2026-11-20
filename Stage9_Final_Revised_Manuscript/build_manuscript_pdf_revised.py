import os
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

stage9_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage9_Final_Revised_Manuscript")
stage6_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage6_Robustness_Generalisation")
fig_dir = stage9_dir / "manuscript_figures"
tbl_dir = stage9_dir / "manuscript_tables"
pdf_path = stage9_dir / "Karnataka_Rainfall_Research_Paper_Revised.pdf"

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
        footer_text = f"Research Paper (Pre-Submission Revision) | Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 28, footer_text)
        self.drawString(36, 28, "Scientific Manuscript Revision | Karnataka Daily Rainfall Hindcasting")
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

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=19,
    alignment=1,
    textColor=colors.HexColor("#0F2043"),
    spaceAfter=6
)

author_style = ParagraphStyle(
    "DocAuthor",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=12,
    alignment=1,
    textColor=colors.HexColor("#333333"),
    spaceAfter=3
)

meta_style = ParagraphStyle(
    "DocMeta",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8,
    leading=11,
    alignment=1,
    textColor=colors.HexColor("#666666"),
    spaceAfter=10
)

h1_style = ParagraphStyle(
    "Heading1_Custom",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=14,
    textColor=colors.HexColor("#0F2043"),
    spaceBefore=11,
    spaceAfter=4,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "Heading2_Custom",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=9.5,
    leading=12,
    textColor=colors.HexColor("#1E3C72"),
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "Body_Custom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    alignment=4,
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
    spaceAfter=7
)

eq_style = ParagraphStyle(
    "Equation_Custom",
    parent=styles["Normal"],
    fontName="Courier-Oblique",
    fontSize=8,
    leading=11,
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

# Title & Meta (Clean author placeholders)
story.append(Paragraph("Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)", title_style))
story.append(Paragraph("[Author Names Blinded for Peer Review / To Be Inserted by Author]", author_style))
story.append(Paragraph("[Department / Institutional Affiliation To Be Inserted by Author] | Correspondence: [Corresponding Author Email]", meta_style))

# Abstract Box
abs_text = (
    "<b>ABSTRACT — </b>Accurate retrospective daily rainfall estimation over complex terrain is essential for catchment water "
    "balance analysis, agricultural planning, and flood hazard assessment. Over peninsular India, the sharp transition between the "
    "steep orographic barrier of the Western Ghats and the semi-arid interior Deccan plateau creates extreme spatial rainfall gradients "
    "that challenge global reanalysis and classical univariate downscaling. In this study, we evaluate a structured machine learning "
    "framework for retrospective daily rainfall reconstruction and heavy precipitation alerting across the 252 India Meteorological "
    "Department (IMD) 0.25° gridded land cells covering Karnataka State, India, over the 7-year period 2019–2025. Leveraging antecedent "
    "24-hour land-surface fluxes and atmospheric thermodynamic variables from ECMWF ERA5-Land reanalysis, local antecedent IMD observations, "
    "and Copernicus GLO-90 DEM topography, we evaluate gradient-boosted decision trees using compound Poisson-Gamma (Tweedie, p=1.5) loss "
    "functions, two-stage hurdle models, and custom focal-loss classifiers. To prevent temporal information leakage across partitions, "
    "models are trained on 2019–2022 (N = 364,644), hyperparameter-tuned and threshold-calibrated strictly on the anomalous 2023 regional "
    "drought year (N = 91,980), and evaluated out-of-sample on a frozen 2-year forward test partition (2024–2025, N = 184,212). "
    "On continuous test evaluation, the three gradient-boosted architectures exhibit closely matched performance with distinct multi-criteria "
    "trade-offs: the two-stage Hurdle Expected Value model achieves the lowest Root Mean Squared Error (RMSE = 7.4241 mm), highest Pearson "
    "correlation (r = 0.7380), and highest Nash-Sutcliffe Efficiency (NSE = 0.5443); CatBoost Tweedie attains the highest Modified Kling-Gupta "
    "Efficiency (KGE-2012 = 0.6261) with RMSE = 7.4335 mm; while LightGBM Tweedie achieves the lowest Mean Absolute Error (MAE = 2.8993 mm) "
    "and lowest mean bias (-0.0179 mm, or -0.51%). All three architectures substantially outperform autoregressive Persistence (RMSE = 10.3582 mm), "
    "raw ERA5-Land reanalysis (RMSE = 10.4550 mm), and Empirical Quantile Mapping (EQM, RMSE = 11.1970 mm, NSE = -0.0366). Paired stationary "
    "block bootstrapping (L = 7d, 14d; B = 2,000 replicates) demonstrates that the ~2.93 mm RMSE reduction over Persistence is statistically "
    "significant at an empirical test resolution of p < 0.0005. For heavy-rain alerting at the official IMD threshold (>= 64.5 mm/day), "
    "a custom focal-loss classifier (gamma = 2.0, alpha = 0.75, validation-selected decision threshold p* = 0.53) suppresses false alarms by "
    "40.86% (427 vs. 722) relative to standard class weighting, achieving a Critical Success Index (CSI) of 0.3512, Precision of 58.22%, "
    "and False Alarm Ratio (FAR) of 41.78%. Tree split gain analysis reveals that antecedent reanalysis runoff variables account for 54.52% "
    "of total tree gain in the heavy-rain classifier, while feature ablation confirms that ERA5-Land predictors provide measurable incremental "
    "predictive value (Delta RMSE = +1.0084 mm upon ablation). However, a critical structural limitation is documented: continuous conditional "
    "expectation models underestimate 100% of test observations exceeding 115.6 mm (200 events), plateauing near 70–90 mm. We conclude that "
    "retrospective reanalysis downscaling provides strong predictive utility in orographically locked monsoonal regimes when land-surface runoff "
    "fluxes are incorporated, but operational hydrometeorological workflows must deploy paired asymmetric classifiers to mitigate conditional "
    "mean quantile shrinkage for extreme precipitation alerting.<br/><br/>"
    "<b>Keywords:</b> Hydrometeorological hindcasting, ERA5-Land reanalysis, IMD gridded rainfall, Tweedie regression, Hurdle model, Focal loss, Orographic precipitation, Western Ghats."
)

abs_table = Table([[Paragraph(abs_text, abstract_style)]], colWidths=[540])
abs_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F9")),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(abs_table)
story.append(Spacer(1, 8))

def add_pdf_figure(img_path, caption_text, width=510, height=205):
    if Path(img_path).exists():
        img = Image(str(img_path), width=width, height=height)
        cap = Paragraph(caption_text, caption_style)
        story.append(KeepTogether([img, cap]))
        story.append(Spacer(1, 4))

def add_pdf_table(csv_path, caption_text, col_widths=None):
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    cap = Paragraph(caption_text, ParagraphStyle("TblCap", parent=caption_style, fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#0F2043"), spaceBefore=4, spaceAfter=2))
    
    table_data = []
    hdr = [Paragraph(f"<b>{c}</b>", ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=5.5, leading=7, alignment=1, textColor=colors.white)) for c in df.columns]
    table_data.append(hdr)
    
    for _, row in df.iterrows():
        r_cells = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, float):
                val_str = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                val_str = str(val)
            r_cells.append(Paragraph(val_str, ParagraphStyle("TD", fontName="Helvetica", fontSize=5, leading=6.5, alignment=1, textColor=colors.HexColor("#222222"))))
        table_data.append(r_cells)
    
    if col_widths is None:
        cw = 540 / len(df.columns)
        col_widths = [cw] * len(df.columns)
        
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3C72")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D7DE")),
    ]))
    story.append(KeepTogether([cap, t]))
    story.append(Spacer(1, 5))

# 1. INTRODUCTION
story.append(Paragraph("1. Introduction", h1_style))
story.append(Paragraph(
    "Accurate estimation of daily precipitation at high spatial resolution is foundational to catchment hydrology, agricultural water allocation, "
    "reservoir dispatch, and extreme weather hazard mitigation [1], [2]. Over the Indian subcontinent, the spatial and temporal distribution of rainfall "
    "is heavily modulated by the Indian Summer Monsoon (June–September), which delivers more than 75% of the annual precipitation budget [3], [6]. "
    "Karnataka State, situated in southwestern peninsular India, exemplifies the intense hydrometeorological diversity of the monsoon: it transitions "
    "abruptly from a hyper-humid coastal belt along the Arabian Sea to the rugged Western Ghats mountain range (elevations 1,000–1,800 m), which triggers "
    "massive orographic precipitation, and further into the semi-arid leeward rain-shadow of the Deccan plateau [6], [7].",
    body_style
))
story.append(Paragraph(
    "Despite the proliferation of global atmospheric reanalysis products—most notably the European Centre for Medium-Range Weather Forecasts (ECMWF) "
    "ERA5 [1] and its high-resolution land component ERA5-Land [2]—reanalysis precipitation fields cannot be directly ingested into local hydrometeorological "
    "models without substantial post-processing. Hydrostatic atmospheric models operating at 0.1° to 0.25° grid spacings smooth steep topographic ridges, "
    "misplace the exact crest of orographic precipitation, and parameterize deep moist convection, frequently leading to systemic wet biases in light drizzle, "
    "diurnal phase errors, and severe underestimation of localized torrential downpours [7]–[9]. Classical statistical downscaling methods, such as Empirical "
    "Quantile Mapping (EQM) [10], adjust the marginal distribution of reanalysis precipitation against observed gauge records, but assume temporal "
    "stationarity, disrupt multi-variable physical consistency, and often degrade spatial field coherence during extreme convective episodes [11]. "
    "Machine learning architectures have demonstrated substantial promise in learning complex, non-linear relationships between multi-level atmospheric "
    "predictors, surface fluxes, and observed precipitation [12], [13].",
    body_style
))
story.append(Paragraph(
    "Nevertheless, daily precipitation modeling presents three intrinsic mathematical challenges: (1) Zero-inflation and semi-continuous support; "
    "(2) Extreme class imbalance in hazard detection (events >= 64.5 mm comprise <1% of records); and (3) Temporal autocorrelation, which inflates performance "
    "under naive cross-validation [25], [26]. This study presents a structured machine learning framework for retrospective daily rainfall reconstruction "
    "over Karnataka from 2019 to 2025. This work is explicitly framed as retrospective reanalysis hindcasting and diagnostic downscaling using antecedent "
    "24-hour accumulated ERA5-Land fluxes and previous-day IMD observations available at 03:00 UTC (08:30 IST); it does not simulate an operational "
    "real-time NWP forecast. All spatial units correspond to the 252 IMD 0.25° gridded land cells covering Karnataka.",
    body_style
))

# 2. STUDY AREA AND DATASETS
story.append(Paragraph("2. Study Area and Datasets", h1_style))
story.append(Paragraph(
    "The study domain encompasses Karnataka State in southern India (11.75°–18.50°N, 74.25°–78.50°E; 191,791 km²), comprising 30 administrative districts "
    "and 10 agroclimatic zones across four meteorological subdivisions: Coastal Karnataka (13 cells), Malnad (35 cells), North Interior Karnataka "
    "(NIK; 118 cells), and South Interior Karnataka (SIK; 86 cells).",
    body_style
))
add_pdf_figure(fig_dir / "figure1_study_area.png", 
               "Figure 1: Geographic domain and spatial discretization of Karnataka showing topography from Copernicus GLO-90 DEM, Western Ghats crest line, 252 IMD 0.25° grid cells color-coded by meteorological subdivision, and explicit identification of the 3 coastal fallback cells (Station IDs 52, 68, 125).",
               width=500, height=195)

story.append(Paragraph(
    "The ground-truth target is the IMD 0.25° gridded daily rainfall dataset (Pai et al. [3]), interpolated from 6,955 rain-gauge stations across India. "
    "Reanalysis predictors are derived from ECMWF ERA5-Land (0.1° grid) [2], de-accumulated over the 24-hour antecedent window ending at 03:00 UTC on day t-1. "
    "Topographic features are extracted from the Copernicus GLO-90 DEM [5]. In the native ERA5-Land land-sea mask, exactly 3 coastal cells (Station IDs 52, 68, 125) "
    "fall over Arabian Sea marine pixels; predictors for these cells were extracted from the nearest-inland neighbor pixel (5.94 to 15.11 km displacement) "
    "and are explicitly tracked.",
    body_style
))

add_pdf_table(tbl_dir / "table1_dataset_characteristics.csv", 
              "Table 1: Dataset Partitioning, Spatial Extent, and Climatological Characteristics across Partitions")

# 3. METHODOLOGY
story.append(Paragraph("3. Methodology", h1_style))
add_pdf_figure(fig_dir / "figure2_methodology.png", 
               "Figure 2: End-to-end retrospective machine learning architecture depicting ERA5-Land flux de-accumulation, IMD gauge preparation, 24-h causal lag alignment, 63 primary feature engineering, temporal partitioning, model training, validation calibration, and block bootstrap verification.",
               width=510, height=205)

story.append(Paragraph("3.1 ERA5-Land De-accumulation Physics and Temporal Alignment", h2_style))
story.append(Paragraph(
    "In raw ECMWF ERA5-Land records, accumulated surface variables are accumulated from the beginning of each forecast cycle starting at 00:00 UTC. "
    "At each hour h in {1, ..., 24}, the stored value represents the accumulated integral from 00:00 UTC to hour h. At 00:00 UTC on the subsequent day, "
    "the forecast cycle resets. To recover physical hourly flux rates, we implement backward differencing: Delta x_h = x_h - x_{h-1} for h in {2, ..., 24}, "
    "with Delta x_1 = x_1. Across all 84 months (2019–2025), zero negative increments (< -10^-5 mm) were detected. De-accumulated hourly increments "
    "are integrated across 24 consecutive hours from 04:00 UTC on day t-1 through 03:00 UTC on day t to match the IMD accumulation window. "
    "All predictors are restricted to the antecedent cycle ending at 03:00 UTC on day t-1. Local antecedent IMD observations (y_{t-1}, y_{t-2}, ...) "
    "are incorporated under the retrospective hindcasting convention that previous-day gauge totals are known.",
    body_style
))

add_pdf_table(tbl_dir / "table2_feature_groups.csv", 
              "Table 2: Input Feature Matrix Architecture and Physical Domain Groupings",
              col_widths=[60, 110, 50, 70, 250])

story.append(Paragraph("3.2 Model Formulations", h2_style))
story.append(Paragraph(
    "We evaluate four model tiers: Naive Baselines (Persistence, Station DOY Climatology), Raw Reanalysis (ERA5-Land precipitation), "
    "Statistical Downscaling (Empirical Quantile Mapping; EQM), and Machine Learning architectures. The primary machine learning regressors include "
    "the Tweedie compound Poisson-Gamma distribution [17], [18] (variance power p = 1.5, log link mu = exp(f(x))) and two-stage Hurdle Expected "
    "Value regression (P(y >= 0.1 | x) * E[y | y >= 0.1, x]) [19]:",
    body_style
))
story.append(Paragraph("Var(Y) = &phi;&mu;<sup>p</sup>,   1 < p < 2;   L<sub>Tweedie</sub>(y, &mu;) = 2 &times; [ y<sup>2-p</sup>/((1-p)(2-p)) - y&mu;<sup>1-p</sup>/(1-p) + &mu;<sup>2-p</sup>/(2-p) ]", eq_style))
story.append(Paragraph("E[Y | X] = P(Y &ge; 0.1 | X) &times; E[Y | Y &ge; 0.1, X]", eq_style))

story.append(Paragraph(
    "For extreme heavy rainfall detection (>= 64.5 mm), we implement a custom asymmetric Focal Loss classifier [21]:",
    body_style
))
story.append(Paragraph("L<sub>Focal</sub>(p<sub>t</sub>) = -&alpha;<sub>t</sub>(1 - p<sub>t</sub>)<sup>&gamma;</sup> log(p<sub>t</sub>);   h<sub>i</sub> = w<sub>i</sub> p<sub>i</sub>(1 - p<sub>i</sub>)", eq_style))
story.append(Paragraph(
    "with gamma = 2.0, alpha = 0.75, and a strictly positive-definite surrogate Hessian h_i ensuring numerical stability during gradient tree splitting. "
    "The operational decision threshold was calibrated on the 2023 Validation partition to maximize CSI, yielding p* = 0.53.",
    body_style
))

# 4. RESULTS
story.append(Paragraph("4. Results", h1_style))
story.append(Paragraph("4.1 Continuous Rainfall Estimation & Multi-Criteria Benchmarking", h2_style))
story.append(Paragraph(
    "Table 3 summarizes the out-of-sample forward evaluation on the frozen 2024–2025 test partition (N = 184,212; observed mean = 3.5085 mm, "
    "standard deviation = 11.0006 mm).",
    body_style
))

add_pdf_table(tbl_dir / "table3_model_benchmarking.csv", 
              "Table 3: Comprehensive Out-of-Sample Model Benchmarking Portfolio (Test Partition 2024–2025; N = 184,212)")

add_pdf_figure(fig_dir / "figure3_observed_predicted.png", 
               "Figure 3: Observed versus predicted daily rainfall. Panel A displays logarithmic hexbin density parity for LightGBM Tweedie; Panel B displays extreme quantile-quantile (Q-Q) alignment from the 90th to 99.9th percentiles.",
               width=510, height=190)
add_pdf_figure(fig_dir / "figure4_model_benchmarking.png", 
               "Figure 4: Multidimensional model benchmarking comparing RMSE (mm), MAE (mm), Pearson correlation (r), and Modified Kling-Gupta Efficiency (KGE-2012) across the 5-tier evaluation portfolio.",
               width=510, height=190)

story.append(Paragraph(
    "The three advanced machine learning architectures demonstrate closely matched predictive skill with distinct multi-criteria trade-offs: "
    "Hurdle Expected Value achieves the lowest overall RMSE (7.4241 mm), highest Pearson r (0.7380), and highest NSE (0.5443); CatBoost Tweedie "
    "attains the highest Modified KGE-2012 (0.6261) with RMSE = 7.4335 mm; while LightGBM Tweedie achieves the lowest MAE (2.8993 mm) and lowest "
    "mean bias (-0.0179 mm, or -0.51%). All three models decisively outperform Persistence (RMSE 10.3582 mm), Raw ERA5-Land (RMSE 10.4550 mm), "
    "and EQM (RMSE 11.1970 mm).",
    body_style
))

story.append(Paragraph("4.2 Heavy Rainfall Detection (>= 64.5 mm)", h2_style))
story.append(Paragraph(
    "Table 4 presents the frozen test contingency analysis for IMD Heavy Rainfall (1,267 events out of 184,212 records, prevalence 0.6878%).",
    body_style
))

add_pdf_table(tbl_dir / "table4_heavy_rain_classification.csv", 
              "Table 4: Heavy Rainfall Detection (>= 64.5 mm) Contingency & Skill Scores under Frozen Decision Thresholds")

add_pdf_figure(fig_dir / "figure5_heavy_rain_detection.png", 
               "Figure 5: Heavy rainfall detection skill at the official 64.5 mm threshold. Panel A shows grouped contingency counts (Hits, Misses, False Alarms); Panel B shows skill radar comparing Precision, Recall, CSI, and HSS.",
               width=510, height=180)

story.append(Paragraph(
    "Focal Loss (p* = 0.53) achieves 595 Hits, 672 Misses, and 427 False Alarms, yielding Precision = 58.22%, Recall = 46.96%, FAR = 41.78%, CSI = 0.3512, "
    "and HSS = 0.5169. Relative to Class-Weighted LightGBM (722 false alarms), Focal Loss cuts false alarms by 40.86%, raising operational precision "
    "from 48.79% to 58.22% while maintaining comparable CSI.",
    body_style
))

story.append(Paragraph("4.3 Feature Importance: Gain, Permutation, and Ablation", h2_style))
story.append(Paragraph(
    "Figure 6 and Table 7 examine physical feature attribution and ablation dynamics across the 63 primary model predictors. "
    "In the LightGBM Focal Loss heavy rainfall classifier, total runoff (ro_lag_1d, 28.11%) and surface runoff (sro_lag_1d, 26.41%) together account for "
    "54.52% of total tree split gain. In continuous LightGBM Tweedie regression, surface runoff accounts for 34.61% of tree gain and skin reservoir "
    "content accounts for 28.28%. In validation permutation importance, surface runoff ranks first (+2.3545 mm RMSE degradation, +42.16% increase). "
    "Ablation Scenario E confirms that removing ERA5-Land predictors surges Test RMSE from 7.4427 mm to 8.4511 mm (Delta RMSE = +1.0084 mm), "
    "proving their measurable incremental predictive value.",
    body_style
))

add_pdf_figure(fig_dir / "figure6_feature_interpretation.png", 
               "Figure 6: Feature attribution analysis. Panel A shows top 15 predictors ranked by validation permutation RMSE degradation; Panel B shows relative gain contribution by physical predictor domain.",
               width=510, height=180)

add_pdf_table(tbl_dir / "table7_feature_ablation.csv", 
              "Table 7: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)")

story.append(Paragraph("4.4 Regional and Seasonal Disaggregation", h2_style))
story.append(Paragraph(
    "Table 5 details model performance across Karnataka's 4 meteorological subdivisions and 4 climatological seasons. "
    "In orographic Coastal Karnataka and Malnad, the model achieves high correlation (r = 0.80–0.83) and hydrological skill (KGE = 0.73–0.74). Over the "
    "semi-arid interior plains (NIK and SIK), correlation encounters a distinct ceiling (r ~ 0.53–0.55; KGE ~ 0.37). The 3 coastal fallback cells "
    "(IDs 52, 68, 125; N = 2,193) achieve RMSE = 15.51 mm and KGE = 0.7504, tracking closely with the 10 native-land coastal cells (RMSE = 14.78 mm, "
    "KGE = 0.7356).",
    body_style
))

add_pdf_table(tbl_dir / "table5_regional_seasonal_performance.csv", 
              "Table 5: Subdivisional & Seasonal Stratified Performance on Test Partition")

add_pdf_figure(fig_dir / "figure7_regional_seasonal.png", 
               "Figure 7: Subdivisional and seasonal hydrometeorological stratification. Panel A shows RMSE and KGE-2012 across subdivisions; Panel B shows seasonal progression across Southwest Monsoon, Post-Monsoon, Pre-Monsoon, and Winter.",
               width=510, height=180)

story.append(Paragraph("4.5 Temporal Generalisation Across Forward Evaluation Years (2024 vs. 2025)", h2_style))
story.append(Paragraph(
    "To evaluate out-of-sample temporal generalisation, model performance was audited separately for the two held-out forward calendar years in Table 6A "
    "(from Stage 6 verified outputs). In leap-year 2024 (N = 92,232), LightGBM Tweedie achieved an RMSE of 7.5094 mm, MAE of 3.0074 mm, Bias of +0.3442 mm "
    "(+10.26%), Pearson r = 0.7320, NSE = 0.5346, and KGE-2012 = 0.5676. In 2025 (N = 91,980), LightGBM Tweedie achieved an RMSE of 7.3752 mm, "
    "MAE of 2.7909 mm, Bias of -0.3810 mm (-10.40%), Pearson r = 0.7426, NSE = 0.5493, and KGE-2012 = 0.6541. CatBoost Tweedie achieved RMSE of "
    "7.5253 mm in 2024 and 7.3403 mm in 2025; Hurdle Expected Value achieved RMSE of 7.5210 mm in 2024 and 7.3257 mm in 2025.",
    body_style
))

# Build Table 6A DataFrame
df_yearly = pd.read_csv(stage6_dir / "yearly_performance_results.csv")
y24_rows = df_yearly[(df_yearly["temporal_stratum"].isin(["Year_2024_Full", "Year_2025_Full"])) & 
                     (df_yearly["model"].isin(["LightGBM_Tweedie", "CatBoost_Tweedie", "Hurdle_ExpectedValue", "Persistence", "Raw_ERA5", "EQM"]))].copy()
y24_clean = y24_rows[["temporal_stratum", "model", "sample_count", "obs_mean", "pred_mean", "MAE", "RMSE", "Bias", "PBias_pct", "Pearson_r", "NSE", "KGE_2012"]].copy()
y24_clean.columns = ["Strata", "Model", "N", "Obs_Mean", "Pred_Mean", "MAE_mm", "RMSE_mm", "Bias_mm", "PBias_pct", "Pearson_r", "NSE", "KGE_2012"]
# Save to stage9 manuscript_tables
y24_clean.to_csv(tbl_dir / "table6a_annual_performance.csv", index=False)

add_pdf_table(tbl_dir / "table6a_annual_performance.csv", 
              "Table 6A: Annual Out-of-Sample Performance Breakdown: 2024 (Leap Year) vs. 2025",
              col_widths=[65, 80, 40, 45, 45, 40, 45, 45, 45, 45, 45, 45])

add_pdf_table(tbl_dir / "table6_statistical_robustness.csv", 
              "Table 6B: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)")

add_pdf_figure(fig_dir / "figure9_temporal_robustness.png", 
               "Figure 9: Temporal robustness and block bootstrap confidence intervals. Panel A shows annual RMSE consistency bar comparison across 2024 and 2025 forward evaluation years; Panel B shows paired block bootstrap 95% confidence interval error bars (B=2,000 replicates) for L=7d and L=14d block lengths.",
               width=510, height=180)

story.append(Paragraph(
    "Across all 2,000 block bootstrap resamples under both 7-day and 14-day block lengths, zero replicates yielded Delta RMSE >= 0, establishing "
    "significance at p < 0.0005 under temporal persistence.",
    body_style
))

# 5. DISCUSSION
story.append(Paragraph("5. Discussion", h1_style))
story.append(Paragraph(
    "A primary physical insight is that land-surface runoff fluxes contribute 54.52% of total tree gain in heavy rain classification and lead "
    "permutation importance in continuous regression. ERA5-Land runoff variables represent simulated land-surface water fluxes from ECMWF's HTESSEL "
    "model, not direct streamflow measurements. In tropical monsoonal regimes, modeled surface runoff is generated only when local infiltration "
    "capacity is exceeded or topsoil approaches saturation. Runoff fluxes act as a physical integrator, filtering out transient atmospheric false "
    "alarms and retaining sustained moisture convergence.",
    body_style
))
story.append(Paragraph(
    "The performance dichotomy between orographic Western Ghats (r >= 0.80) and interior Deccan plateau (r ~ 0.53) reflects atmospheric scale limits. "
    "Ghats precipitation is mechanically forced by low-level jet advection perpendicular to the terrain ridge, which hydrostatic reanalysis resolves with "
    "high skill. Conversely, interior precipitation is governed by unorganized, localized mesoscale convective thunderstorm cells unresolvable by a 0.1° grid.",
    body_style
))
story.append(Paragraph(
    "Table 8 and Figure 8 provide an audit of continuous model behavior across the 7 official IMD rainfall intensity categories.",
    body_style
))

add_pdf_table(tbl_dir / "table8_rainfall_intensity_limitations.csv", 
              "Table 8: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage (Test Set; N = 184,212)")

add_pdf_figure(fig_dir / "figure8_extreme_rainfall_errors.png", 
               "Figure 8: Rainfall intensity error profiles and quantile shrinkage. Panel A shows Mean Bias (mm) and Percentage Bias (%) across the 7 IMD categories; Panel B shows percentage of underestimated events and RMSE error spread across intensity intervals.",
               width=510, height=180)

story.append(Paragraph(
    "Continuous models optimizing convex losses seek the conditional expectation E[Y|X]. In highly skewed distributions, predicting the conditional mean "
    "inevitably pulls predictions toward the bulk distribution, compressing predictions into a ceiling around 70–90 mm. Exactly 100% of observations "
    "exceeding 115.6 mm (200 test events) are underestimated, proving that continuous regression cannot be deployed in isolation for extreme hazard alerting.",
    body_style
))

# 6. LIMITATIONS
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

# 7. CONCLUSION
story.append(Paragraph("7. Conclusion", h1_style))
story.append(Paragraph(
    "This study evaluated a structured machine learning framework for retrospective daily rainfall reconstruction over Karnataka using ERA5-Land "
    "reanalysis and IMD gridded observations (2019–2025). Advanced gradient-boosted decision trees (Hurdle Expected Value, CatBoost Tweedie, "
    "and LightGBM Tweedie) reduce test RMSE by ~28.1–28.3% relative to Persistence and by ~33.5–33.7% relative to Empirical Quantile Mapping. "
    "Stationary paired block bootstrapping confirms significance at p < 0.0005. Simulated surface runoff fluxes provide dominant predictive "
    "utility, while feature ablation proves that ERA5 predictors provide measurable incremental predictive value. However, severe extreme tail "
    "quantile shrinkage above 115.6 mm demonstrates that operational workflows must deploy a dual-model paradigm pairing continuous regression "
    "for volumetric water balance with asymmetric focal-loss classification for extreme hazard alerting.",
    body_style
))

# REFERENCES
story.append(Paragraph("References", h1_style))
df_refs = pd.read_csv(Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage8_Final_Research_Manuscript\reference_verification.csv"))
for _, r in df_refs.iterrows():
    ref_text = f"<b>[{r['ref_num']}]</b> {r['authors']}, \"{r['title']},\" <i>{r['journal_or_conf']}</i>, {r['volume_issue_pages']}, {r['year']}. DOI: {r['doi_or_url']}."
    story.append(Paragraph(ref_text, ref_style))

print("--- Building Revised PDF Document ---")
doc.build(story, canvasmaker=NumberedCanvas)
print(f"Saved revised manuscript PDF to {pdf_path}")

