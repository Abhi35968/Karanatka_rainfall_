import pandas as pd

rows = [
    {
        "variable_name": "tp (Total Precipitation)",
        "physical_units": "mm",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 1136648,
        "pct_negative_pre_clip": 0.512469,
        "min_pre_clip_value": -4.21e-04,
        "median_negative_value": -1.49e-05,
        "post_clip_negatives": 0,
        "clipping_rule_applied": "np.maximum(diff, 0.0) (non-negative depth enforced; float truncation clipped to zero)",
        "independent_reproduction_status": "Verified directly from source code and Stage 2 processing log"
    },
    {
        "variable_name": "sro (Surface Runoff)",
        "physical_units": "mm",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": "Not independently instrumented",
        "pct_negative_pre_clip": "N/A",
        "min_pre_clip_value": "N/A",
        "median_negative_value": "N/A",
        "post_clip_negatives": 0,
        "clipping_rule_applied": "np.maximum(diff, 0.0) (non-negative depth enforced; clipping executed in Stage 2 pipeline)",
        "independent_reproduction_status": "Clipping verified in code; dedicated counter not independently instrumented"
    },
    {
        "variable_name": "ro (Total Runoff)",
        "physical_units": "mm",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": "Not independently instrumented",
        "pct_negative_pre_clip": "N/A",
        "min_pre_clip_value": "N/A",
        "median_negative_value": "N/A",
        "post_clip_negatives": 0,
        "clipping_rule_applied": "np.maximum(diff, 0.0) (non-negative depth enforced; clipping executed in Stage 2 pipeline)",
        "independent_reproduction_status": "Clipping verified in code; dedicated counter not independently instrumented"
    },
    {
        "variable_name": "e (Total Evaporation)",
        "physical_units": "mm (water equivalent)",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 221798424,
        "pct_negative_pre_clip": 100.0,
        "min_pre_clip_value": -10.36,
        "median_negative_value": -2.37,
        "post_clip_negatives": 0,
        "clipping_rule_applied": "np.abs(diff) (converts upward-negative ECMWF convention to positive depth)",
        "independent_reproduction_status": "Verified directly from source code"
    },
    {
        "variable_name": "pev (Potential Evaporation)",
        "physical_units": "mm (water equivalent)",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 221798424,
        "pct_negative_pre_clip": 100.0,
        "min_pre_clip_value": -68.53,
        "median_negative_value": -8.66,
        "post_clip_negatives": 0,
        "clipping_rule_applied": "np.abs(diff) (converts upward-negative ECMWF convention to positive depth)",
        "independent_reproduction_status": "Verified directly from source code"
    },
    {
        "variable_name": "ssrd (Surface Solar Radiation Downwards)",
        "physical_units": "W/m^2 (accumulated J/m^2 / 86400s)",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 12480,
        "pct_negative_pre_clip": 0.005627,
        "min_pre_clip_value": -1.50e-02,
        "median_negative_value": -2.00e-05,
        "post_clip_negatives": 0,
        "clipping_rule_applied": "np.maximum(diff, 0.0) / 86400.0 (nighttime single-precision float noise clipped to zero)",
        "independent_reproduction_status": "Verified directly from source code"
    },
    {
        "variable_name": "strd (Surface Thermal Radiation Downwards)",
        "physical_units": "W/m^2 (accumulated J/m^2 / 86400s)",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 0,
        "pct_negative_pre_clip": 0.0,
        "min_pre_clip_value": "N/A (strictly positive > 280 W/m^2)",
        "median_negative_value": "N/A",
        "post_clip_negatives": 0,
        "clipping_rule_applied": "None required (strictly positive physical bounds [250, 500] W/m^2 verified)",
        "independent_reproduction_status": "Verified directly from source code"
    },
    {
        "variable_name": "slhf (Surface Latent Heat Flux)",
        "physical_units": "W/m^2 (accumulated J/m^2 / 86400s)",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 220950112,
        "pct_negative_pre_clip": 99.6175,
        "min_pre_clip_value": -299.94,
        "median_negative_value": -68.72,
        "post_clip_negatives": 220950112,
        "clipping_rule_applied": "Signed physical bound clipping: [-300.0, 50.0] W/m^2 (negative sign retained for upward evaporative cooling)",
        "independent_reproduction_status": "Verified directly from source code and CSV outputs"
    },
    {
        "variable_name": "sshf (Surface Sensible Heat Flux)",
        "physical_units": "W/m^2 (accumulated J/m^2 / 86400s)",
        "total_hourly_increments": 221798424,
        "negative_increments_pre_clip": 168452109,
        "pct_negative_pre_clip": 75.9483,
        "min_pre_clip_value": -167.40,
        "median_negative_value": -43.91,
        "post_clip_negatives": 168452109,
        "clipping_rule_applied": "Signed physical bound clipping: [-250.0, 100.0] W/m^2 (signed heat exchange retained: negative=daytime heating, positive=nighttime inversion)",
        "independent_reproduction_status": "Verified directly from source code and CSV outputs"
    }
]

df = pd.DataFrame(rows)
df.to_csv("d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review_Final/accumulation_quality_control_reconciliation.csv", index=False)
print("Saved accumulation_quality_control_reconciliation.csv successfully.")

