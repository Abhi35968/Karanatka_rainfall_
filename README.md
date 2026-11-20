# Karnataka Rainfall ERA5-Land Machine Learning Pipeline

An end-to-end pipeline to download **7 years of continuous ERA5-Land hourly climate data (2019–2025)** from the Copernicus Climate Data Store (CDS) for the **Karnataka region**, preprocess the dataset (unit conversion, missing values, precipitation accumulation, quality checks), engineer spatio-temporal ML features, and train Machine Learning models for rainfall forecasting.

---

## 📐 Pipeline Architecture

```
                 Copernicus CDS
                       │
                       │  ERA5-Land Hourly Data (2019-2025)
                       ▼
               Karnataka Bounding Box
            [19.0°N, 74.0°E, 11.0°S, 79.0°E]
                       │
                       ▼
             30 Selected Variables
                       │
                       ▼
                Monthly NetCDF Files
                       │
                       ▼
                 Preprocessing
            ├── Unit Conversion (K -> °C, m -> mm, Pa -> hPa)
            ├── Missing-Value Handling (Interpolation & Imputation)
            ├── Precipitation Processing (Non-negativity clipping & Accumulation)
            └── Quality Checks (Physical bounds validation & QC reporting)
                       │
                       ▼
               ML-Ready Dataset (10.5M+ rows)
            ├── Spatial & Temporal Aggregations
            ├── Feature Engineering (Lags, Dewpoint depression, Temp range, Cyclical time)
            └── ML Targets (Next-day rainfall depth & intensity class)
                       │
                       ▼
             Machine Learning Model
            ├── XGBoost / Random Forest Regressor (Rainfall depth prediction)
            └── XGBoost Classifier (Rainfall occurrence & intensity category)
```

---

## 🚀 Quick Start & Usage

### 1. Installation & Environment Setup
Make sure you have installed the required dependencies:
```bash
pip install cdsapi xarray netCDF4 pandas numpy scikit-learn xgboost matplotlib scipy
```

---

### 2. Run Complete Pipeline (One-Line Orchestrator)

#### 🧪 Demo Mode (Instant Verification with Sample Data)
Runs the entire pipeline end-to-end immediately without waiting for API download queues:
```bash
python run_pipeline.py --mode demo --start-year 2019 --end-year 2025
```

#### 🌐 Production Download Mode (Real Copernicus CDS Data)
Downloads 7 years (2019–2025) of hourly ERA5-Land data across all 12 months for Karnataka and processes it:
```bash
python run_pipeline.py --mode download --start-year 2019 --end-year 2025
```

---

## 🛠️ Step-by-Step Script Reference

### Step 1: Downloading 7 Years of ERA5-Land Data from CDS
- **Script**: [`download_era5_karnataka.py`](file:///d:/PROJECT/karnataka_rainfall_cds/download_era5_karnataka.py)
- **Features**:
  - Automatically splits 30 weather variables into smaller request chunks to respect CDS's 12,000-field limit.
  - Downloads NetCDF files per chunk and merges them into monthly files.
  - Saved under `ERA5_Karnataka/YYYY/MM/era5land_karnataka_YYYY_MM.nc`.

**Example:**
```bash
python download_era5_karnataka.py --start-year 2019 --end-year 2025 --months 1 2 3 4 5 6 7 8 9 10 11 12
```


---

### Step 2: Preprocessing NetCDF Datasets
- **Script**: [`preprocess_era5.py`](file:///d:/PROJECT/karnataka_rainfall_cds/preprocess_era5.py)
- **Features**:
  - **Unit Conversion**: Converts Kelvin to °C, meters of water to mm, Pa to hPa, and vector components $(u_{10}, v_{10})$ to Wind Speed ($m/s$).
  - **Missing-Value Handling**: Temporal linear interpolation and spatial fill.
  - **Precipitation Processing**: Non-negative clipping and accumulation handling.
  - **Quality Checks**: Physical range checks & JSON validation report generation (`qc_report_*.json`).

**Example:**
```bash
python preprocess_era5.py --input-dir ERA5_Karnataka --output-dir Preprocessed_Karnataka
```

---

### Step 3: Feature Engineering & ML Dataset Creation
- **Script**: [`create_ml_dataset.py`](file:///d:/PROJECT/karnataka_rainfall_cds/create_ml_dataset.py)
- **Engineered Features**:
  - Daily summary statistics (`precip_daily_mm`, `temp_mean_c`, `temp_max_c`, `temp_min_c`, `pressure_hpa`, `wind_speed_ms`, `soil_moisture`).
  - Derived atmospheric indices (`temp_range_c`, `dewpoint_depression_c`).
  - Cyclical seasonal encoding (`sin_day_of_year`, `cos_day_of_year`).
  - Spatial-temporal lags (`precip_lag_1d`, `precip_lag_2d`, `precip_lag_3d`, `temp_lag_1d`).
  - Cumulative rolling windows (`precip_roll_3d_sum`, `precip_roll_7d_sum`).
  - Targets: Next-day rainfall amount (`target_next_day_precip_mm`) & Intensity category (`target_rain_class`).

**Example:**
```bash
python create_ml_dataset.py --input-dir Preprocessed_Karnataka --output-file ML_Ready_Karnataka_Rainfall.csv
```

---

### Step 4: Machine Learning Model Training & Evaluation
- **Script**: [`train_rainfall_model.py`](file:///d:/PROJECT/karnataka_rainfall_cds/train_rainfall_model.py)
- **Models & Evaluation**:
  - **Regression**: XGBoost / Random Forest Regressor evaluated with RMSE, MAE, $R^2$, Pearson $r$, and Critical Success Index (CSI).
  - **Classification**: XGBoost Classifier evaluated with Accuracy, Macro F1, and Confusion Matrix.
  - **Output Plots**: Saved in `Model_Outputs/` (`regression_actual_vs_predicted.png`, `regression_feature_importance.png`, `classification_confusion_matrix.png`).

**Example:**
```bash
python train_rainfall_model.py --input-file ML_Ready_Karnataka_Rainfall.csv --output-dir Model_Outputs
```

---

## 🔑 Copernicus CDS Credentials Setup
To download actual data from Copernicus Climate Data Store:
1. Register at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
2. Copy your API Personal Access Token from your CDS profile.
3. Save it to `%USERPROFILE%\.cdsapirc` (Windows) or `~/.cdsapirc` (Linux/Mac):
   ```text
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR-UID:YOUR-API-KEY
   ```
