import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

src_fig_dir = Path("d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review/manuscript_figures")
out_base = Path("d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review_Final")
out_fig_dir = out_base / "manuscript_figures"
out_fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Copy Figures 1 to 7 directly
for i in range(1, 8):
    for ext in [".png", ".pdf"]:
        src_file = src_fig_dir / f"figure{i}_"
        # Find match
        matches = list(src_fig_dir.glob(f"figure{i}_*{ext}"))
        if matches:
            # Pick the shortest or standard name
            target_name = out_fig_dir / f"figure{i}_{matches[0].name.split('_', 1)[1]}"
            shutil.copy2(matches[0], target_name)
            print(f"Copied Figure {i}: {matches[0].name} -> {target_name.name}")

# 2. Copy Figure 8 (Temporal Robustness)
# In src, figure8_temporal_robustness_seq.png is temporal robustness
src_fig8_png = src_fig_dir / "figure8_temporal_robustness_seq.png"
if not src_fig8_png.exists():
    src_fig8_png = src_fig_dir / "figure9_temporal_robustness.png"
shutil.copy2(src_fig8_png, out_fig_dir / "figure8_temporal_robustness.png")

src_fig8_pdf = src_fig_dir / "figure9_temporal_robustness.pdf"
if src_fig8_pdf.exists():
    shutil.copy2(src_fig8_pdf, out_fig_dir / "figure8_temporal_robustness.pdf")
print("Standardized Figure 8: figure8_temporal_robustness.png / .pdf")

# 3. Generate Improved Figure 9: Rainfall Intensity Error Profiles & Extreme Quantile Shrinkage
# Let's read table 8 data
tbl_path = Path("d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/publication_tables/table8_rainfall_intensity_limitations.csv")
df_t8 = pd.read_csv(tbl_path)

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)

clean_cats = [
    "No Rain\n(<0.1 mm)",
    "Very Light\n(0.1–2.4 mm)",
    "Light\n(2.5–15.5 mm)",
    "Moderate\n(15.6–64.4 mm)",
    "Heavy\n(64.5–115.5 mm)",
    "Very Heavy\n(115.6–204.4 mm)",
    "Extremely Heavy\n(≥204.5 mm)"
]

# Panel A: Mean Bias
colors_a = ["#2b5c8f" if b >= 0 else "#c93b3b" for b in df_t8["Mean_Bias_mm"]]
bars_a = axes[0].bar(range(len(df_t8)), df_t8["Mean_Bias_mm"], color=colors_a, width=0.55, edgecolor="black", linewidth=0.8)
axes[0].axhline(0, color="black", linestyle="-", linewidth=1.0)
axes[0].set_xticks(range(len(df_t8)))
axes[0].set_xticklabels(clean_cats, fontsize=9.5, fontweight="medium")
axes[0].set_title("A. Conditional Mean Bias (mm) Across 7 IMD Intensity Categories", fontsize=12.5, fontweight="bold", pad=12)
axes[0].set_ylabel("Mean Bias (Predicted − Observed, mm)", fontsize=11, fontweight="bold")
axes[0].set_ylim(-195, 28)
axes[0].grid(axis="y", linestyle=":", alpha=0.6)

# Improve label readability in Panel A
for idx, r in df_t8.iterrows():
    b_val = r["Mean_Bias_mm"]
    pct_val = r["Percentage_Bias_pct"]
    
    if idx == 0:
        val_str = f"+{b_val:.2f} mm\n(Undefined*)"
        y_pos = b_val + 4.5
        va = "bottom"
    elif b_val >= 0:
        val_str = f"+{b_val:.2f} mm\n(+{pct_val:.0f}%)"
        y_pos = b_val + 4.5
        va = "bottom"
    else:
        val_str = f"{b_val:.1f} mm\n({pct_val:.0f}%)"
        y_pos = b_val - 6.0
        va = "top"
    
    axes[0].text(idx, y_pos, val_str, ha="center", va=va, fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

# Panel B: Underestimation Percentage
bars_b = axes[1].bar(range(len(df_t8)), df_t8["Percent_Underestimated"], color="#b22222", width=0.55, edgecolor="black", linewidth=0.8)
axes[1].axhline(50, color="#4a4a4a", linestyle="--", linewidth=1.2, label="50% Median Parity Reference Line")
axes[1].axhline(100, color="#8b0000", linestyle=":", linewidth=1.0, alpha=0.8)
axes[1].set_xticks(range(len(df_t8)))
axes[1].set_xticklabels(clean_cats, fontsize=9.5, fontweight="medium")
axes[1].set_title("B. Proportion of Instances Underestimated (%) by Intensity Category", fontsize=12.5, fontweight="bold", pad=12)
axes[1].set_ylabel("Underestimated Observations (%)", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 118)
axes[1].grid(axis="y", linestyle=":", alpha=0.6)
axes[1].legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9)

for idx, r in df_t8.iterrows():
    u_val = r["Percent_Underestimated"]
    count_val = r["Sample_Count"]
    if idx == 0:
        label_text = f"{u_val:.1f}%\n(N={count_val:,})"
    elif idx >= 5:
        label_text = f"{u_val:.1f}%\n(N={count_val:,})\n[100% Miss]"
    else:
        label_text = f"{u_val:.1f}%\n(N={count_val:,})"
    
    axes[1].text(idx, u_val + 3.0, label_text, ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

plt.tight_layout()
fig9_png = out_fig_dir / "figure9_extreme_rainfall_errors.png"
fig9_pdf = out_fig_dir / "figure9_extreme_rainfall_errors.pdf"
fig.savefig(fig9_png, dpi=300, bbox_inches="tight")
fig.savefig(fig9_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Generated improved Figure 9: {fig9_png.name} and {fig9_pdf.name}")
