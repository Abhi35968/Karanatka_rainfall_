import pandas as pd
df = pd.read_csv(r'd:\PROJECT\karnataka_rainfall_cds\Stage9_2_Final_Author_Review\complete_63_feature_manifest.csv')
for idx, row in df.iterrows():
    for col in df.columns:
        if 'station' in str(row[col]).lower():
            print(f"Row {idx} ({row['feature_name']}): col [{col}] -> {row[col]}")

