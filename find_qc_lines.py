with open('d:/PROJECT/karnataka_rainfall_cds/corrected_era5_processing.py', 'r') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if 'neg_tp_increments' in line or 'total_tp_increments' in line:
        print(f"Line {idx+1}: {line.strip()}")
