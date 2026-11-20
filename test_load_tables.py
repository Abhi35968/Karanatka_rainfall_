import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
stage9_2_dir = base_dir / "Stage9_2_Final_Author_Review"
fig_dir = stage9_2_dir / "manuscript_figures"
tbl_dir = stage9_2_dir / "manuscript_tables"
stage6_dir = base_dir / "Stage6_Robustness_Generalisation"

out_md = stage9_2_dir / "Karnataka_Rainfall_Research_Paper_Final.md"
out_docx = stage9_2_dir / "Karnataka_Rainfall_Research_Paper_Final.docx"

print("Step 1: Reading tables...")
df_t1 = pd.read_csv(tbl_dir / "table1_dataset_characteristics.csv")
df_t2 = pd.read_csv(tbl_dir / "table2_feature_groups.csv")
df_t3 = pd.read_csv(tbl_dir / "table3_model_benchmarking.csv")
df_t4 = pd.read_csv(tbl_dir / "table4_heavy_rain_classification.csv")
df_t5_ablation = pd.read_csv(tbl_dir / "table7_feature_ablation.csv")
df_t6_regional = pd.read_csv(tbl_dir / "table5_regional_seasonal_performance.csv")
df_t7_bootstrap = pd.read_csv(tbl_dir / "table6_statistical_robustness.csv")
df_t8_intensity = pd.read_csv(tbl_dir / "table8_rainfall_intensity_limitations.csv")

# Annual generalisation dataframe
df_yearly = pd.read_csv(stage6_dir / "yearly_performance_results.csv")
y24_rows = df_yearly[(df_yearly["temporal_stratum"].isin(["Year_2024_Full", "Year_2025_Full"])) & 
                     (df_yearly["model"].isin(["LightGBM_Tweedie", "CatBoost_Tweedie", "Hurdle_ExpectedValue", "Persistence", "Raw_ERA5", "EQM"]))].copy()
y24_clean = y24_rows[["temporal_stratum", "model", "sample_count", "obs_mean", "pred_mean", "MAE", "RMSE", "Bias", "PBias_pct", "Pearson_r", "NSE", "KGE_2012"]].copy()
y24_clean.columns = ["Strata", "Model", "N", "Obs_Mean", "Pred_Mean", "MAE_mm", "RMSE_mm", "Bias_mm", "PBias_pct", "Pearson_r", "NSE", "KGE_2012"]

print("Tables loaded successfully.")

