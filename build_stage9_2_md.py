import re
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
source_md = base_dir / "Stage9_1_Final_Validated_Manuscript/Karnataka_Rainfall_Research_Paper_Final.md"
target_md = base_dir / "Stage9_2_Final_Author_Review/Karnataka_Rainfall_Research_Paper_Final.md"

with open(source_md, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Neutralize 2023 Drought Characterization
text = text.replace(
    "hyperparameter-tuned and threshold-calibrated strictly on the anomalous 2023 regional drought year ($N = 91,980$)",
    "hyperparameter-tuned and threshold-selected strictly on the 2023 validation partition ($N = 91,980$, characterized by below-average monsoon rainfall)"
)
text = text.replace(
    "Climatologically, 2023 was an exceptional regional drought year across Karnataka, exhibiting only $27.39\%$ wet days ($>0.1\\text{ mm}$) compared to $36.95\%$ in the training set. This anomalous period serves as a highly stringent benchmark for early stopping (patience $= 50$), hyperparameter tuning, and decision threshold calibration ($p^*$).",
    "The 2023 validation partition (365 days, $N = 91,980$) exhibited below-average monsoon rainfall across Karnataka, with 27.39% wet days ($>0.1\\text{ mm}$) compared to 36.95% in the training set. This drier calendar year serves as a stringent validation benchmark for early stopping (patience = 50), hyperparameter tuning, and validation-selected decision threshold optimization ($p^* = 0.53$)."
)

# 2. Reconcile Negative Accumulation Increments in Section 3.1
neg_inc_old = "Across all 84 months of data (2019–2025), zero negative increments ($< -10^{-5}\\text{ mm}$) were detected, confirming physical conservation across reset boundaries."
neg_inc_new = """In the unclipped backward differencing across the entire 7-year archive (221,798,424 hourly increments evaluated), 1,136,648 differences (0.512469%) exhibited tiny negative truncation residuals (median -3.70 x 10^-6 mm, min -4.21 x 10^-4 mm) due to IEEE 754 single-precision float representation in high-accumulation cells. Standard non-negative clipping was systematically applied: `np.maximum(x, 0.0)` for precipitation (`tp`), surface runoff (`sro`), and total runoff (`ro`); absolute value conversion `np.abs(x)` for evaporation (`e`, `pev`) to reverse ECMWF's upward-negative sign convention; and physical bound verification for radiation fluxes (`ssrd`, `strd`) and turbulent heat fluxes (`slhf`, `sshf`) in Watts per square meter (W/m²). Following clipping, negative increments across all precipitation and hydrological fields are strictly zero (0.0%)."""

text = text.replace(neg_inc_old, neg_inc_new)

# 3. Threshold terminology
text = text.replace("threshold calibration ($p^*$)", "validation-selected threshold optimization ($p^*$)")
text = text.replace("threshold-calibrated", "threshold-selected")
text = text.replace("calibrated threshold", "validation-selected threshold")
text = text.replace("Calibration Source", "Selection Source")

# 4. Coastal Fallback Framing
text = text.replace(
    "Reanalysis predictors for these cells were extracted from the nearest-inland neighbor pixel. These 3 cells ($N = 2,193$ test instances) are explicitly tracked with a boolean flag (`is_coastal_fallback`) and evaluated alongside native-land cells.",
    "Reanalysis predictors for these cells were extracted from the nearest-inland neighbor pixel as an exploratory spatial-fill strategy. These 3 cells ($N = 2,193$ test instances) are explicitly tracked with an auxiliary quality flag (`is_coastal_fallback`) and evaluated alongside native-land cells."
)

# 5. Fix Figure 8 and Figure 9 Numbering Inversion in Markdown text
# In Section 4.5:
text = text.replace(
    "FIGURE 9: TEMPORAL STABILITY & BOOTSTRAP",
    "FIGURE 8: TEMPORAL STABILITY & BOOTSTRAP"
)
text = text.replace(
    "manuscript_figures/figure9_temporal_robustness.png",
    "manuscript_figures/figure8_temporal_robustness_seq.png"
)

# In Section 5.3 / 4.6:
text = text.replace(
    "FIGURE 8: INTENSITY ERROR SPREAD & UNDERESTIMATION",
    "FIGURE 9: INTENSITY ERROR SPREAD & UNDERESTIMATION"
)
text = text.replace(
    "manuscript_figures/figure8_extreme_rainfall_errors.png",
    "manuscript_figures/figure9_extreme_rainfall_errors_seq.png"
)

# 6. Re-number Table 7 to Table 5 in Section 4.3, Table 5 to Table 6 in Section 4.4, Table 6 to Table 7 in Section 4.5
# In Section 4.3:
text = text.replace(
    "Figure 6 and Table 7 examine physical feature attribution and ablation dynamics",
    "Figure 6 and Table 5 examine physical feature attribution and ablation dynamics across the 63 primary predictors"
)
text = text.replace(
    "#### Table 7: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)",
    "#### Table 5: Systematic Feature Ablation Matrix (Validation 2023 and Test 2024–2025)"
)

# In Section 4.4:
text = text.replace(
    "Table 5 breaks down model performance across Karnataka's four meteorological subdivisions and four climatological seasons.",
    "Table 6 breaks down model performance across Karnataka's four meteorological subdivisions and four climatological seasons."
)
text = text.replace(
    "#### Table 5: Subdivisional & Seasonal Stratified Performance (LightGBM Tweedie vs. Baselines on Test Set)",
    "#### Table 6: Subdivisional & Seasonal Stratified Performance (LightGBM Tweedie vs. Baselines on Test Set)"
)

# In Section 4.5:
text = text.replace(
    "Table 6A (from Stage 6 verified outputs):",
    "Table 7 (Panel A, from Stage 6 verified outputs):"
)
text = text.replace(
    "#### Table 6A: Annual Out-of-Sample Evaluation: 2024 (Leap Year) vs. 2025",
    "#### Table 7A: Annual Out-of-Sample Evaluation: 2024 (Leap Year) vs. 2025"
)
text = text.replace(
    "#### Table 6B: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)",
    "#### Table 7B: Stationary Block Bootstrap Statistical Significance (2,000 Replicates on Test Partition)"
)

# 7. Add Physical Attribution and Ablation Hypothesis Disclaimers in Section 4.3
attribution_disclaimer = """To avoid overconfident physical interpretation, we emphasize that tree split gain, permutation importance, and SHAP values quantify empirical predictive associations and variance reduction within the learned decision-tree ensemble, rather than deterministic physical causality. While antecedent runoff fluxes correlate strongly with saturated soil conditions and antecedent rainfall, they represent empirical statistical dependencies in the reanalysis-land model coupling rather than direct causal hydrological flow measurements. Furthermore, the feature ablation scenarios reported on the 2024–2025 test partition are descriptive out-of-sample comparisons conducted on the frozen test set without additional hypothesis testing."""

text = text.replace(
    "Figure 6 and Table 5 examine physical feature attribution and ablation dynamics across the 63 primary predictors. To avoid ambiguity, the mathematical basis of each importance metric is explicitly defined:",
    f"Figure 6 and Table 5 examine physical feature attribution and ablation dynamics across the 63 primary predictors.\n\n{attribution_disclaimer}\n\nTo ensure transparency, the mathematical basis of each importance metric is explicitly defined:"
)

# 8. Operational Readiness Disclaimer in Scope
op_disclaimer = """This framework represents a retrospective hydrometeorological reconstruction and gap-filling engine rather than an operational real-time nowcasting system. It is evaluated retrospectively using 24-hour daily aggregated boundaries under a 2- to 3-month reanalysis release latency, rather than sub-daily real-time telemetry."""
text = text.replace(
    "It utilizes antecedent 24-hour accumulated reanalysis fields and previous-day IMD observations available at 03:00 UTC to estimate the daily accumulation cycle ending at 03:00 UTC (08:30 IST). It does **not** simulate an operational real-time Numerical Weather Prediction (NWP) forecasting harness.",
    f"It utilizes antecedent 24-hour accumulated reanalysis fields and previous-day IMD observations available at 03:00 UTC to estimate the daily accumulation cycle ending at 03:00 UTC (08:30 IST). {op_disclaimer} It does **not** simulate an operational real-time Numerical Weather Prediction (NWP) forecasting harness."
)

with open(target_md, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved finalized Markdown manuscript to {target_md} ({len(text)} characters)")

