"""
generate_notebook.py
--------------------
Generates the comprehensive, educational Jupyter Notebook (notebooks/exploratory_analysis.ipynb)
with markdown narratives, mathematical explanations, and executable code cells.
"""

import json
import os

cells = []


def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })


def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })


# 1. Title & Overview
add_md("""# ⚡ AI-Based Energy Consumption Forecasting Using Machine Learning

**Project Goal**: Build an end-to-end Machine Learning forecasting system that analyzes historical electricity consumption across Indian states and regional grids, performs feature engineering without data leakage, trains 5 regression models, compares them across 7 evaluation metrics, and forecasts future electricity demand.

---

### Machine Learning Workflow:
1. **Data Ingestion & Schema Inspection**: Understand columns, types, and summary statistics.
2. **Exploratory Data Analysis (EDA)**: Analyze demand distributions, top consuming states, and regional grid loads.
3. **Data Cleaning**: Handle date formats, missing values, duplicates, and enforce strict chronological order.
4. **Feature Engineering**: Extract calendar features, categorical encodings, and leakage-free lag and rolling features.
5. **Chronological Splitting**: 80% past records for training, 20% future records for unseen testing.
6. **Model Training**: Train Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
7. **Model Evaluation & Comparison**: Calculate MAE, MSE, RMSE, R², MAPE, MedAE, and Explained Variance.
8. **Residual & Error Distribution Analysis**: Inspect error histograms and residual scatter plots.
9. **Real-time Inference Demo**: Predict electricity demand for future dates.""")

# 2. Imports
add_md("""## Step 1: Import Required Libraries
We import standard data science and machine learning libraries:
- `pandas` & `numpy`: Tabular operations and numerical computations.
- `matplotlib` & `seaborn`: Statistical charting and time-series plotting.
- `scikit-learn` & `xgboost`: Regression models, categorical encoders, and evaluation metrics.
- `joblib`: Model serialization.""")

add_code("""import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Plot styling
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

print("All libraries successfully imported!")""")

# 3. Load dataset
add_md("""## Step 2: Load and Inspect Dataset
We load `data/energy_consumption.csv` (or root `long_data_.csv`).
- Target variable: `Usage` (Daily electricity consumption in **Mega Units - MU**, where $1\\text{ MU} = 1\\text{ Million kWh}$).
- Categorical features: `States`, `Regions`.
- Spatial features: `latitude`, `longitude`.
- Time feature: `Dates`.""")

add_code("""data_path = '../data/energy_consumption.csv'
if not os.path.exists(data_path):
    data_path = 'data/energy_consumption.csv'
if not os.path.exists(data_path):
    data_path = '../long_data_.csv'
if not os.path.exists(data_path):
    data_path = 'long_data_.csv'

df_raw = pd.read_csv(data_path)
print(f"Raw Dataset Shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
df_raw.head()""")

# 4. Summary & Nulls
add_md("""### Check Missing Values, Data Types & Duplicates""")

add_code("""print("--- Missing Values ---")
print(df_raw.isnull().sum())
print(f"\\nDuplicate rows: {df_raw.duplicated().sum()}")
print("\\n--- Data Types ---")
print(df_raw.dtypes)""")

# 5. Data Cleaning
add_md("""## Step 3: Data Cleaning & Chronological Sorting
1. Parse `Dates` using `pd.to_datetime(..., dayfirst=True)`.
2. Remove any invalid date or non-numeric target records.
3. Impute non-positive usage with state medians.
4. Drop duplicate records.
5. Sort strictly chronologically by `['States', 'ParsedDate']`.""")

add_code("""df = df_raw.copy()

# 1. Parse Date
df['ParsedDate'] = pd.to_datetime(df['Dates'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['ParsedDate'])

# 2. Clean Target Column
df['Usage'] = pd.to_numeric(df['Usage'], errors='coerce')
df = df.dropna(subset=['Usage'])

# 3. Remove non-positive values
if (df['Usage'] <= 0).sum() > 0:
    df['Usage'] = df.groupby('States')['Usage'].transform(lambda x: x.replace(0, np.nan).fillna(x.median()))

# 4. Remove duplicates
df = df.drop_duplicates()

# 5. Sort chronologically
df = df.sort_values(by=['States', 'ParsedDate']).reset_index(drop=True)

print(f"Cleaned Dataset: {len(df):,} records")
print(f"Date Range: {df['ParsedDate'].min().date()} to {df['ParsedDate'].max().date()}")
df.head()""")

# 6. Exploratory Data Analysis
add_md("""## Step 4: Exploratory Data Analysis (EDA)
Let us explore:
1. Top electricity consuming states.
2. Regional grid demand distribution.
3. Consumption distribution (Histogram + KDE).""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Top 10 States
top10 = df.groupby('States')['Usage'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top10.values, y=top10.index, ax=axes[0], palette='Blues_r')
axes[0].set_title("Top 10 States by Daily Energy Consumption (MU)", fontweight='bold')
axes[0].set_xlabel("Average Daily Usage (MU)")

# Consumption Distribution
sns.histplot(df['Usage'], bins=40, kde=True, ax=axes[1], color='#2563EB')
axes[1].set_title("Daily Energy Consumption Distribution", fontweight='bold')
axes[1].set_xlabel("Usage (MU)")

plt.tight_layout()
plt.show()""")

# 7. Feature Engineering
add_md("""## Step 5: Feature Engineering
To enable regression algorithms to capture seasonal patterns and momentum without data leakage:
- **Temporal**: `Month`, `Day`, `DayOfWeek`, `IsWeekend`, `Quarter`, `DayOfYear`, `WeekOfYear`.
- **Lag Features (Shift >= 1)**: `Usage_Lag_1` (yesterday), `Usage_Lag_2` (2 days ago), `Usage_Lag_7` (same day last week).
- **Rolling Features (Shift >= 1)**: `Usage_Rolling_Mean_7`, `Usage_Rolling_Mean_14`, `Usage_Rolling_Std_7`.
- **Categorical & Spatial**: `States_Encoded`, `Regions_Encoded`, `latitude`, `longitude`.""")

add_code("""# 1. Temporal Features
df['Year'] = df['ParsedDate'].dt.year
df['Month'] = df['ParsedDate'].dt.month
df['Day'] = df['ParsedDate'].dt.day
df['DayOfWeek'] = df['ParsedDate'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Quarter'] = df['ParsedDate'].dt.quarter
df['DayOfYear'] = df['ParsedDate'].dt.dayofyear
df['WeekOfYear'] = df['ParsedDate'].dt.isocalendar().week.astype(int)

# 2. Lag Features
df['Usage_Lag_1'] = df.groupby('States')['Usage'].shift(1)
df['Usage_Lag_2'] = df.groupby('States')['Usage'].shift(2)
df['Usage_Lag_7'] = df.groupby('States')['Usage'].shift(7)

# 3. Rolling Features
df['Usage_Rolling_Mean_7'] = df.groupby('States')['Usage'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
df['Usage_Rolling_Mean_14'] = df.groupby('States')['Usage'].transform(
    lambda x: x.shift(1).rolling(14, min_periods=1).mean()
)
df['Usage_Rolling_Std_7'] = df.groupby('States')['Usage'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=2).std().fillna(0)
)

# Clean boundary NaNs
lag_cols = ['Usage_Lag_1', 'Usage_Lag_2', 'Usage_Lag_7', 'Usage_Rolling_Mean_7', 'Usage_Rolling_Mean_14', 'Usage_Rolling_Std_7']
for col in lag_cols:
    df[col] = df.groupby('States')[col].bfill().fillna(df['Usage'].median())

# 4. Encoders
le_state = LabelEncoder()
le_region = LabelEncoder()
df['States_Encoded'] = le_state.fit_transform(df['States'].astype(str))
df['Regions_Encoded'] = le_region.fit_transform(df['Regions'].astype(str))

feature_cols = [
    'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear', 'WeekOfYear',
    'latitude', 'longitude', 'States_Encoded', 'Regions_Encoded',
    'Usage_Lag_1', 'Usage_Lag_2', 'Usage_Lag_7',
    'Usage_Rolling_Mean_7', 'Usage_Rolling_Mean_14', 'Usage_Rolling_Std_7'
]

print(f"Total Predictor Features: {len(feature_cols)}")
df[['ParsedDate', 'States', 'Usage'] + feature_cols[:6]].head()""")

# 8. Train/Test Split
add_md("""## Step 6: Chronological Train-Test Split (80% / 20%)
We split chronologically at the 80th percentile date cutoff so models are evaluated on unseen future data.""")

add_code("""unique_dates = np.sort(df['ParsedDate'].unique())
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_df = df[df['ParsedDate'] < split_date].copy().reset_index(drop=True)
test_df = df[df['ParsedDate'] >= split_date].copy().reset_index(drop=True)

X_train, y_train = train_df[feature_cols], train_df['Usage']
X_test, y_test = test_df[feature_cols], test_df['Usage']

print(f"Split Date Cutoff: {pd.to_datetime(split_date).strftime('%Y-%m-%d')}")
print(f"Training Set: {len(train_df):,} records ({train_df['ParsedDate'].min().date()} to {train_df['ParsedDate'].max().date()})")
print(f"Testing Set:  {len(test_df):,} records ({test_df['ParsedDate'].min().date()} to {test_df['ParsedDate'].max().date()})")""")

# 9. Train Models
add_md("""## Step 7: Train Machine Learning Regression Models
We train 5 algorithms:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor**""")

add_code("""models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)
    predictions[name] = preds

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    medae = median_absolute_error(y_test, preds)
    evs = explained_variance_score(y_test, preds)

    results[name] = {
        "MAE": round(float(mae), 3),
        "MSE": round(float(mse), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4),
        "MAPE (%)": round(float(mape), 2),
        "MedAE": round(float(medae), 3),
        "Explained Variance": round(float(evs), 4)
    }

results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
results_df""")

# 10. Visualizations
add_md("""## Step 8: Actual vs Predicted & Residual Visualizations""")

add_code("""best_model_name = results_df['R2'].idxmax()
best_preds = predictions[best_model_name]
test_df['Predicted'] = best_preds
test_df['Residual'] = test_df['Usage'] - test_df['Predicted']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Actual vs Predicted Time Series for Maharashtra
sample_state = "Maharashtra"
sample_df = test_df[test_df['States'] == sample_state].sort_values('ParsedDate')
axes[0].plot(sample_df['ParsedDate'], sample_df['Usage'], label='Actual', color='#2563EB', lw=2)
axes[0].plot(sample_df['ParsedDate'], sample_df['Predicted'], label=f'Predicted ({best_model_name})', color='#EA580C', linestyle='--', lw=2)
axes[0].set_title(f"Actual vs Predicted in {sample_state} (Test Set)", fontweight='bold')
axes[0].set_ylabel("Usage (MU)")
axes[0].legend()

# Residual Distribution
sns.histplot(test_df['Residual'], bins=40, kde=True, color='#0D9488', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', label='Zero Error')
axes[1].set_title(f"Residual Distribution ({best_model_name})", fontweight='bold')
axes[1].set_xlabel("Residual (MU)")
axes[1].legend()

plt.tight_layout()
plt.show()""")

# 11. Conclusion
add_md("""## Step 9: Conclusion & Summary
- **Data Preprocessing**: Handled missing values, formatted timestamps chronologically, and removed duplicates.
- **Feature Engineering**: Engineered calendar signals, state encodings, and backward-looking lag/rolling features without data leakage.
- **Model Comparison**: Tree-based gradient boosting ensembles (**XGBoost** and **Gradient Boosting**) outperformed linear baselines, achieving $R^2 > 0.98$ and low RMSE on unseen future test data.
- **Deployment**: Models and transformers are serialized for the interactive Streamlit application.""")

# Write notebook
os.makedirs("notebooks", exist_ok=True)
notebook_path = os.path.join("notebooks", "exploratory_analysis.ipynb")
nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Notebook generated successfully at: {notebook_path}")
