import pandas as pd
from pathlib import Path

base_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds")
df_train = pd.read_parquet(base_dir / "Stage3_4_Final_Release_Audit/data/final_train_dataset.parquet")
df_man = pd.read_csv(base_dir / "Stage3_4_Final_Release_Audit/final_model_input_manifest.csv")

primary_cols = df_man[df_man["eligible_for_X_primary"] == "Yes"]["column_name"].tolist()
cat_cols = ["subdivision", "zone_id"]
num_cols = [c for c in primary_cols if c not in cat_cols]

print(f"Total primary_cols in manifest: {len(primary_cols)}")
print(f"num_cols count: {len(num_cols)}")

subdiv_map = df_train.groupby("subdivision")["target_rainfall_mm"].mean().to_dict()
zone_map = df_train.groupby("zone_id")["target_rainfall_mm"].mean().to_dict()

X = df_train[num_cols].copy()
X["subdivision_enc"] = df_train["subdivision"].map(subdiv_map).fillna(3.5).values
X["zone_id_enc"] = df_train["zone_id"].map(zone_map).fillna(3.5).values

print(f"X shape: {X.shape}")
print(f"Are raw subdivision and zone_id in X? {'subdivision' in X.columns or 'zone_id' in X.columns}")
print(f"Are subdivision_enc and zone_id_enc in X? {'subdivision_enc' in X.columns and 'zone_id_enc' in X.columns}")

print("\nExact 63 columns fed to ML models (X):")
for i, col in enumerate(X.columns, 1):
    print(f"{i:2d}. {col}")

