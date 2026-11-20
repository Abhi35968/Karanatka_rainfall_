import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

base = Path(r"d:\PROJECT\karnataka_rainfall_cds")
plots_dir = base / "Stage2_1_Audit/figures"
plots_dir.mkdir(parents=True, exist_ok=True)

df_rain = pd.read_csv(base / "Stage2_Corrected_Processing/data/corrected_daily_rainfall.csv")
df_rain["date"] = pd.to_datetime(df_rain["date"])

# -----------------------------------------------------------------------------
# Figure 1: Temporal Split Timeline & Continuity
# -----------------------------------------------------------------------------
print("Generating Fig2_1_1_temporal_split_timeline.png...")
df_daily_agg = df_rain.groupby("date").agg({
    "station_id": "count",
    "imd_observed_mm": "mean",
    "corrected_era5_imd_aligned_mm": "mean"
}).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), dpi=300, sharex=True)

# Station completeness timeline
ax1.plot(df_daily_agg["date"], df_daily_agg["station_id"], color="#1f77b4", linewidth=1.5)
ax1.axvline(pd.Timestamp("2023-01-01"), color="#d62728", linestyle="--", linewidth=1.8, label="Train / Val Boundary (2022-12-31 | 2023-01-01)")
ax1.axvline(pd.Timestamp("2024-01-01"), color="#9467bd", linestyle="--", linewidth=1.8, label="Val / Test Boundary (2023-12-31 | 2024-01-01)")

# Shade splits
ax1.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="#1f77b4", alpha=0.08, label="Train Split (1,461 days, 368,172 samples)")
ax1.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), color="#ff7f0e", alpha=0.12, label="Validation Split (365 days, 91,980 samples)")
ax1.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"), color="#2ca02c", alpha=0.08, label="Test Split (731 days, 184,212 samples)")

ax1.set_ylabel("Reporting Grid Cells", fontsize=11, fontweight="bold")
ax1.set_ylim(240, 260)
ax1.set_yticks([250, 252])
ax1.set_title("(a) Temporal Split Allocation & Daily Spatial Completeness across All 2,557 Days (Zero Missing)", fontsize=12, fontweight="bold")
ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax1.grid(True, linestyle="--", alpha=0.5)

# Statewide rainfall timeline
ax2.plot(df_daily_agg["date"], df_daily_agg["imd_observed_mm"], color="#1f77b4", linewidth=1.2, label="IMD Observed Ground Truth")
ax2.plot(df_daily_agg["date"], df_daily_agg["corrected_era5_imd_aligned_mm"], color="#2ca02c", linewidth=1.0, alpha=0.8, label="Corrected ERA5 (IMD-Aligned)")
ax2.axvline(pd.Timestamp("2023-01-01"), color="#d62728", linestyle="--", linewidth=1.8)
ax2.axvline(pd.Timestamp("2024-01-01"), color="#9467bd", linestyle="--", linewidth=1.8)
ax2.set_ylabel("Statewide Mean Rain (mm/day)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Calendar Date (2019–2025)", fontsize=11, fontweight="bold")
ax2.set_title("(b) Statewide Daily Precipitation Continuum across Partition Boundaries", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.tight_layout()
fig1_p = plots_dir / "Fig2_1_1_temporal_split_timeline.png"
plt.savefig(fig1_p, dpi=300)
plt.close()
print("Saved:", fig1_p.name)

# -----------------------------------------------------------------------------
# Figure 2: UTC vs IMD-Aligned Daily Differences & Mass Conservation
# -----------------------------------------------------------------------------
print("Generating Fig2_1_2_utc_vs_imd_daily_differences.png...")
df_diff = pd.read_csv(base / "Stage2_1_Audit/data/utc_vs_imd_daily_difference.csv")
diff_vals = df_diff["diff_imd_minus_utc_mm"].values

fig = plt.figure(figsize=(14, 10), dpi=300)
gs = fig.add_gridspec(2, 2)

# Panel A: Difference Histogram (Zoomed to [-30, +30] mm)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(diff_vals, bins=120, range=(-30, 30), color="#1f77b4", edgecolor="#333333", alpha=0.75)
ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label=f"Mean Diff = {diff_vals.mean():+.5f} mm")
ax1.set_xlabel("Daily Precipitation Difference (P_IMD - P_UTC, mm/day)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Frequency (Station-Days)", fontsize=10, fontweight="bold")
ax1.set_title("(a) Distribution of Daily Differences (N = 644,364)", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel B: Cumulative Multi-Year Mass Conservation
ax2 = fig.add_subplot(gs[0, 1])
df_dates_mean = df_rain.groupby("date").agg({
    "imd_observed_mm": "mean",
    "corrected_era5_utc_mm": "mean",
    "corrected_era5_imd_aligned_mm": "mean"
}).reset_index()
cum_obs = df_dates_mean["imd_observed_mm"].cumsum()
cum_utc = df_dates_mean["corrected_era5_utc_mm"].cumsum()
cum_imd = df_dates_mean["corrected_era5_imd_aligned_mm"].cumsum()

ax2.plot(df_dates_mean["date"], cum_obs, color="#1f77b4", linewidth=1.8, label="IMD Ground Truth Observed")
ax2.plot(df_dates_mean["date"], cum_utc, color="#ff7f0e", linestyle="--", linewidth=1.8, label="Corrected UTC (00–24h)")
ax2.plot(df_dates_mean["date"], cum_imd, color="#2ca02c", linestyle=":", linewidth=2.0, label="Corrected IMD-Aligned (03–03h)")
ax2.set_xlabel("Date (2019–2025)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Cumulative Mean Precipitation (mm)", fontsize=10, fontweight="bold")
ax2.set_title("(b) 7-Year Cumulative Mass Conservation Continuum", fontsize=11, fontweight="bold")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.5)

# Panel C: Storm Event Phase Shifts (July 2024 zoom)
ax3 = fig.add_subplot(gs[1, :])
zoom_sub = df_diff[(df_diff["date"] >= "2024-07-01") & (df_diff["date"] <= "2024-07-31") & (df_diff["station_id"] == 52)]
ax3.plot(pd.to_datetime(zoom_sub["date"]), zoom_sub["imd_observed_mm"], color="#1f77b4", marker="o", linewidth=2.0, label="IMD Gauge Observed")
ax3.plot(pd.to_datetime(zoom_sub["date"]), zoom_sub["corrected_era5_imd_aligned_mm"], color="#2ca02c", marker="s", linewidth=1.8, label="ERA5 IMD-Aligned (03–03h UTC)")
ax3.plot(pd.to_datetime(zoom_sub["date"]), zoom_sub["corrected_era5_utc_mm"], color="#ff7f0e", linestyle="--", marker="^", linewidth=1.5, label="ERA5 Standard UTC (00–24h UTC)")
ax3.set_xlabel("Date (July 2024 - Udupi Coastal Station 52)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Daily Precipitation (mm/day)", fontsize=11, fontweight="bold")
ax3.set_title("(c) Synoptic Storm Phase Shifting Demonstration: IMD Window Tracks Heavy Peak Timing Accurately", fontsize=12, fontweight="bold")
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

plt.tight_layout()
fig2_p = plots_dir / "Fig2_1_2_utc_vs_imd_daily_differences.png"
plt.savefig(fig2_p, dpi=300)
plt.close()
print("Saved:", fig2_p.name)

# -----------------------------------------------------------------------------
# Figure 3: Coastal Fallback Analysis
# -----------------------------------------------------------------------------
print("Generating Fig2_1_3_coastal_fallback_analysis.png...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
coastal_ids = [52, 68, 125]
titles = ["Station 52 (Udupi)", "Station 68 (Udupi)", "Station 125 (Uttara Kannada)"]

for idx, st_id in enumerate(coastal_ids):
    st_sub = df_rain[df_rain["station_id"] == st_id]
    y_o = st_sub["imd_observed_mm"].values
    y_f = st_sub["corrected_era5_imd_aligned_mm"].values
    r_val = np.corrcoef(y_o, y_f)[0, 1]
    
    axes[idx].scatter(y_o, y_f, alpha=0.25, s=12, color="#2ca02c")
    axes[idx].plot([0, 200], [0, 200], "r--", linewidth=1.5, label="1:1 Agreement")
    axes[idx].set_xlim(0, 180)
    axes[idx].set_ylim(0, 180)
    axes[idx].set_xlabel("IMD Observed Rain (mm/day)", fontsize=10, fontweight="bold")
    axes[idx].set_ylabel("Nearest-Land Fallback Rain (mm/day)", fontsize=10, fontweight="bold")
    axes[idx].set_title(f"{titles[idx]}\n(r = {r_val:.3f}, MAE = {np.mean(np.abs(y_f - y_o)):.2f} mm)", fontsize=11, fontweight="bold")
    axes[idx].grid(True, linestyle="--", alpha=0.5)
    axes[idx].legend(loc="upper left")

plt.tight_layout()
fig3_p = plots_dir / "Fig2_1_3_coastal_fallback_analysis.png"
plt.savefig(fig3_p, dpi=300)
plt.close()
print("Saved:", fig3_p.name)
print("All supporting figures successfully created!")

