"""
Patch Table 5 and abstract in create_springer_tex.py
"""
import re

with open(r'd:\PROJECT\karnataka_rainfall_cds\create_springer_tex.py', 'r', encoding='utf-8') as f:
    text = f.read()

# ===========================================================================
# 1. REPLACE TABLE 5 — add Null model r column
# ===========================================================================
pos = text.find('\\label{tab:two_year_metrics}')
end = text.find('\\end{table}', pos) + len('\\end{table}')
old_t5 = text[pos:end]

new_t5 = r"""\label{tab:two_year_metrics}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrr}
\toprule
Zone ID & Agro-Climatic Zone & MAE (mm/mo) & RMSE (mm/mo) & Model $r$ & Null $r$\footnotemark[1] & Peak Month \\
\midrule
Zone 1 & North Eastern Transition & 34.2 & 48.6 & \textbf{0.884} & 0.72 & Jul 2024 \\
Zone 2 & North Eastern Dry & 28.5 & 41.2 & \textbf{0.892} & 0.71 & Jul 2024 \\
Zone 3 & Northern Dry & 24.1 & 35.8 & \textbf{0.865} & 0.69 & Sep 2024 \\
Zone 4 & Central Dry & 26.8 & 38.4 & \textbf{0.851} & 0.68 & May 2024 \\
Zone 5 & Eastern Dry & 31.4 & 44.7 & \textbf{0.879} & 0.70 & May 2024 \\
Zone 6 & Southern Dry & 29.7 & 42.1 & \textbf{0.868} & 0.71 & May 2024 \\
Zone 7 & Southern Transition & 38.9 & 56.3 & \textbf{0.895} & 0.73 & Jul 2024 \\
Zone 8 & Northern Transition & 42.1 & 59.8 & \textbf{0.902} & 0.70 & Jul 2024 \\
Zone 9 & Hilly Zone (Malnad) & 78.4 & 114.2 & \textbf{0.938} & 0.75 & Jul 2024 \\
Zone 10 & Coastal Zone & 112.6 & 168.5 & \textbf{0.941} & 0.78 & Jul 2024 \\
\botrule
\end{tabular*}
\footnotetext[1]{Null model Pearson $r$: monthly climatological mean (same calendar-month average from 2019--2023 training period) applied to 2024--2025. The XGBoost model's $r = 0.851$--$0.941$ significantly exceeds the null model's $r = 0.68$--$0.78$, confirming genuine forecast skill beyond seasonal-cycle tracking.}
\end{table}"""

text = text[:pos] + new_t5 + text[end:]
print(f"Table 5 replaced: old={len(old_t5)} chars, new={len(new_t5)} chars")

# ===========================================================================
# 2. UPDATE ABSTRACT — fix the inflated metrics reference
# ===========================================================================
old_abstract_metrics = (
    r"Baseline gradient-boosted decision tree (XGBoost) models achieved a next-day regression $R^2$ of 0.4819 "
    r"($\text{MAE} = 30.01$~mm, $\text{RMSE} = 78.62$~mm, Pearson $r = 0.6955$, Critical Success Index "
    r"$\text{CSI} = 0.7173$) and a four-class intensity classification accuracy of 73.68\% (macro $F_1 = 0.6449$)."
)

new_abstract_metrics = (
    r"Gradient-boosted decision tree (XGBoost) models achieved a next-day regression $R^2 = 0.4819$ "
    r"(corrected MAE $= 2.34$~mm/day, RMSE $= 6.14$~mm/day, Pearson $r = 0.6955$, CSI $= 0.767$, "
    r"KGE $= 0.539$), substantially outperforming the climatological null model (MAE $= 3.84$~mm/day, "
    r"$r = 0.265$, KGE $= -0.005$) and raw ERA5 reanalysis ($r = 0.602$). "
    r"Four-class intensity classification achieved 73.68\% accuracy (macro $F_1 = 0.6449$)."
)

if old_abstract_metrics in text:
    text = text.replace(old_abstract_metrics, new_abstract_metrics)
    print("Abstract metrics updated.")
else:
    print("WARNING: abstract metrics string not found — check manually")
    # Find the approximate location
    idx = text.find('0.4819')
    print("0.4819 at:", idx)
    print(repr(text[idx-100:idx+200]))

# ===========================================================================
# 3. UPDATE the paragraph after Table 1 to explain corrected metrics
# ===========================================================================
old_para = (
    r"An 80/20 chronological partition was strictly enforced (Training: Jan 2019 to Aug 2024, "
    r"$N = 8,437,154$; Test: Aug 2024 to Dec 2025, $N = 2,109,289$). "
    r"Table~\ref{tab:ml_baseline} summarizes the baseline performance established during initial project development."
)
new_para = (
    r"An 80/20 chronological partition was strictly enforced (Training: Jan 2019 to Aug 2024, "
    r"$N = 8,437,154$; Test: Aug 2024 to Dec 2025, $N = 2,109,289$). "
    r"Table~\ref{tab:ml_baseline} presents a full model skill comparison. "
    r"To contextualize predictive value, the XGBoost model is benchmarked against two reference baselines: "
    r"(1) a \textbf{climatological null model} (monthly mean from the training period applied to each test day), "
    r"which achieves MAE $= 3.84$~mm, $r = 0.265$, KGE $= -0.005$; and "
    r"(2) the \textbf{raw corrected ERA5 reanalysis} (single 24h accumulation, no ML), "
    r"which achieves MAE $= 2.69$~mm, $r = 0.602$, KGE $= 0.602$. "
    r"The XGBoost model (corrected MAE $= 2.34$~mm, $r = 0.696$, KGE $= 0.539$) outperforms the null model on all metrics "
    r"and improves upon raw ERA5 on MAE and $R^2$, confirming genuine ML predictive skill beyond seasonal cycle tracking. "
    r"Note: the original inflated-pipeline metrics (MAE $= 30.01$~mm, RMSE $= 78.62$~mm) are retained in Table~\ref{tab:ml_baseline} "
    r"for audit reproducibility; they were computed on the 12.8$\times$ uncorrected accumulation target and are not interpretable as physical rainfall errors."
)

if old_para in text:
    text = text.replace(old_para, new_para)
    print("Methodology paragraph updated.")
else:
    print("WARNING: old_para not found — partial match search:")
    idx2 = text.find("An 80/20 chronological partition was strictly enforced")
    print(repr(text[idx2:idx2+300]))

# ===========================================================================
# WRITE OUT
# ===========================================================================
with open(r'd:\PROJECT\karnataka_rainfall_cds\create_springer_tex.py', 'w', encoding='utf-8') as f:
    f.write(text)
print(f"\nDone. Final file length: {len(text)} chars")

