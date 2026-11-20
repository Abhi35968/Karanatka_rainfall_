import pandas as pd
df = pd.read_csv('d:/PROJECT/karnataka_rainfall_cds/Stage5_Model_Interpretation/permutation_importance.csv')
for i, r in df.head(15).iterrows():
    print(f"{r['permutation_rank']}. {r['feature']}: delta_rmse_mean={r['delta_rmse_mean']:.4f}, delta_rmse_std={r['delta_rmse_std']:.4f}, pct_increase={r['percent_rmse_increase']:.2f}%")
