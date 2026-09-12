# ⚡ AI-Based Energy Consumption Forecasting Using Machine Learning

An end-to-end, production-grade Machine Learning forecasting system designed to predict daily electricity consumption across Indian states and regional power grids using historical time-series data.

---

## 1. Project Title
**AI-Based Energy Consumption Forecasting Using Machine Learning**

---

## 2. Project Objective
The primary objective of this project is to build an accurate, leakage-free time-series forecasting pipeline that:
1. Ingests and preprocesses multi-region historical power consumption records.
2. Performs time-series feature engineering including calendar dynamics, spatial coordinates, regional grid encodings, and backward-looking lag/rolling statistics.
3. Evaluates 5 diverse Machine Learning regression algorithms (**Linear Regression**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, and **XGBoost**) using a strict chronological train/test split.
4. Assesses forecasting performance across 7 evaluation metrics ($R^2$, RMSE, MAE, MSE, MAPE, MedAE, and Explained Variance).
5. Provides an interactive, professional **Streamlit Web Application** with live forecasting, downloadable CSV reports, and residual diagnostics.

---

## 3. Problem Statement
Electricity cannot be stored efficiently or cost-effectively at massive grid scale. Grid operators must continuously balance electricity generation with real-time consumer and industrial demand. 

Electricity demand exhibits complex dynamics influenced by:
- **Temporal & Seasonal Factors**: Month of the year (summer cooling vs. winter heating), day of the week, and weekend vs. weekday industrial loads.
- **Geographic & Spatial Factors**: State population density, industrialization, and regional grid topologies.
- **Inertial Momentum**: Previous-day demand (Lag-1) and weekly cyclical demand (Lag-7).

Imprecise forecasting leads to:
- **Under-generation**: Blackouts, voltage drops, and severe grid failure risks.
- **Over-generation**: Excess fuel burning, elevated operational costs, and unnecessary carbon emissions.

---

## 4. Dataset Overview
- **Dataset Source**: Daily Power Consumption in India (2019–2020)
- **Primary Data File**: `data/energy_consumption.csv` (16,587 clean records across 33 States/UTs and 5 Regional Grids)
- **Target Variable ($y$)**: `Usage` — Daily Electricity Consumption in **Mega Units (MU)**  
  *(Note: $1\text{ MU} = 1\text{ Million kWh} = 1\text{ GWh}$)*
- **Core Dataset Schema**:
  - `Dates`: Date and timestamp (`DD/MM/YYYY 00:00:00`)
  - `States`: 33 Indian States / Union Territories (e.g., Maharashtra, Gujarat, Tamil Nadu, Delhi, Punjab)
  - `Regions`: 5 Regional Grids (`NR`, `WR`, `SR`, `ER`, `NER`)
  - `latitude`, `longitude`: Geographic spatial coordinates
  - `Usage`: Daily power consumed in Mega Units (MU)

---

## 5. Data Preprocessing
The preprocessing pipeline (`src/data_preprocessing.py`) performs the following operations:
1. **Schema Auto-Detection**: Dynamically detects date and target columns with wide-to-long reshaping support.
2. **Datetime Standardization**: Parses timestamps into standard `pandas.Timestamp` with `dayfirst=True` handling.
3. **Missing & Invalid Value Handling**: Validates positive consumption values; non-positive entries are imputed using the respective state's historical median.
4. **Duplicate Removal**: Identifies and eliminates redundant records.
5. **Strict Chronological Sorting**: Records are sorted chronologically by `['States', 'ParsedDate']` to prevent temporal shuffle distortion.

---

## 6. Feature Engineering & Data Leakage Prevention
The feature engineering module (`src/feature_engineering.py`) extracts 17 predictive features:

### 1. Calendar & Temporal Features
- `Year`, `Month`, `Day`, `DayOfWeek` ($0 = \text{Monday}, 6 = \text{Sunday}$)
- `IsWeekend`: Binary indicator ($1$ for Saturday/Sunday, $0$ for Weekday)
- `Quarter`, `DayOfYear`, `WeekOfYear`

### 2. Backward-Looking Lag Features (Strictly Shifted $\ge 1$)
- `Usage_Lag_1`: Electricity consumption 1 day prior (immediate load inertia)
- `Usage_Lag_2`: Electricity consumption 2 days prior
- `Usage_Lag_7`: Electricity consumption 7 days prior (same day last week - weekly seasonality)

### 3. Rolling Window Aggregations
- `Usage_Rolling_Mean_7`: 7-day moving average of previous consumption
- `Usage_Rolling_Mean_14`: 14-day moving average of previous consumption
- `Usage_Rolling_Std_7`: 7-day moving standard deviation (consumption volatility)

### 4. Categorical & Spatial Features
- `States_Encoded`, `Regions_Encoded`: Categorical label encodings fitted strictly on the training set
- `latitude`, `longitude`: Geographic spatial positions

> **Data Leakage Safeguard**: All lag and rolling calculations strictly use past values (`.shift(1)`) so the model never peeks at the target date during training or inference.

---

## 7. Machine Learning Algorithms
We train and compare 5 regression algorithms:

| Algorithm | Method & Architecture | Key Advantage |
| :--- | :--- | :--- |
| **Linear Regression** | Ordinary Least Squares linear hyperplane fitting. | Fast, highly interpretable baseline. |
| **Decision Tree Regressor** | Hierarchical recursive splitting on feature thresholds (`max_depth=10`). | Captures nonlinear state thresholds. |
| **Random Forest Regressor** | Bagging ensemble of 100 decorrelated decision trees (`max_depth=12`). | Robust to variance and reduces overfitting. |
| **Gradient Boosting** | Sequential boosting optimizing pseudo-residuals (`max_depth=5`, $\eta=0.1$). | Strong predictive power on tabular trends. |
| **XGBoost Regressor** | Extreme Gradient Boosting with regularized objective and second-order Taylor expansion. | Optimal accuracy and lowest forecasting error. |

---

## 8. Evaluation Metrics
Models are evaluated on an unseen future test set (80/20 chronological split) using 7 standard metrics:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
3. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
4. **Coefficient of Determination ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
5. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
6. **Median Absolute Error (MedAE)**:
   $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, \dots, |y_n - \hat{y}_n|)$$
7. **Explained Variance Score**:
   $$\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$

---

## 9. Actual Experimental Results
Evaluated on **3,300 unseen future test records** (Chronological test period: Feb 28, 2020 – Dec 05, 2020):

| Model Name | MAE (MU) | MSE ($\text{MU}^2$) | RMSE (MU) | $R^2$ Score | MAPE (%) | MedAE (MU) | Explained Variance | Performance Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **6.040** | **197.193** | **14.043** | **0.9854** | **10.28%** | **1.755** | **0.9854** | **1st 🥇 (Best)** |
| **Gradient Boosting** | 6.111 | 207.653 | 14.410 | 0.9846 | 9.60% | 1.572 | 0.9846 | 2nd 🥈 |
| **Random Forest** | 6.128 | 220.816 | 14.860 | 0.9836 | 7.42% | 1.589 | 0.9836 | 3rd 🥉 |
| **Linear Regression** | 6.109 | 232.896 | 15.261 | 0.9827 | 11.13% | 1.695 | 0.9827 | 4th |
| **Decision Tree** | 7.044 | 344.318 | 18.556 | 0.9745 | 8.22% | 1.561 | 0.9745 | 5th |

### 🏆 Best Model: XGBoost Regressor
- Achieved **$R^2 = 0.9854$** (explaining over $98.5\%$ of all consumption variance on unseen future data).
- Achieved the lowest RMSE (**$14.043\text{ MU}$**) and lowest MSE (**$197.193\text{ MU}^2$**).

---

## 10. Streamlit Dashboard Architecture
The interactive web application (`app.py`) uses a clean **light theme** (crisp cards, blue/teal accents, high contrast typography) organized into 4 functional tabs:

1. **📊 Dashboard & Overview**: Total records, mean/min/max usage, top-10 consuming states chart, and regional grid pie chart.
2. **📈 Time Series & Seasonality**: Interactive historical trends (state-specific & all-India) with monthly and weekly demand cycle plots.
3. **🏆 Model Comparison & Evaluation**: 7-metric evaluation table, dynamic best model badge, Actual vs Predicted interactive chart, residual error distribution, and CSV export buttons.
4. **🔮 Predict Energy Consumption**: Real-time ML inference form.

---

## 11. Real Prediction Feature
The prediction module (`src/prediction.py`) allows interactive user forecasting:
1. **Inputs**: Target State, Target Date, ML Model selection, and recent consumption context.
2. **Auto-Derived Features**: Date automatically decomposes into Month, Day, DayOfWeek, IsWeekend, Quarter, and DayOfYear; State automatically resolves spatial coordinates and regional grid.
3. **Inference**: Passes aligned feature vector to the trained model artifact to generate true ML predictions in Mega Units (MU).
4. **Contextual Comparison**: Shows predicted consumption alongside historical state medians, percentage change, and weekday/weekend indicators.

---

## 12. Project Directory Structure
```
AI-Based-Energy-consumption-forecasting-usin-ML/
│
├── data/
│   └── energy_consumption.csv          # Cleaned historical dataset
│
├── models/
│   ├── best_model.joblib               # Best trained model (XGBoost)
│   ├── xgboost.joblib                  # XGBoost model artifact
│   ├── gradient_boosting.joblib        # Gradient Boosting model artifact
│   ├── random_forest.joblib            # Random Forest model artifact
│   ├── decision_tree.joblib            # Decision Tree model artifact
│   ├── linear_regression.joblib        # Linear Regression model artifact
│   ├── label_encoders.joblib           # Preprocessing categorical encoders
│   ├── feature_columns.joblib          # Exact predictor feature column list
│   ├── evaluation_data.joblib          # Test predictions & residual dataframe
│   ├── model_metrics.json              # Dynamic evaluation metrics JSON
│   ├── model_comparison.csv            # Exported metrics comparison table
│   ├── actual_vs_predicted.csv         # Exported test results CSV
│   ├── residual_analysis.csv           # Exported residual error CSV
│   ├── model_performance_comparison.png# Multi-panel model comparison plot
│   ├── actual_vs_predicted.png         # Time-series actual vs predicted plot
│   └── residuals_distribution.png      # Residual error histogram & scatter
│
├── notebooks/
│   └── exploratory_analysis.ipynb      # Step-by-step Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Ingestion, cleaning, datetime parsing
│   ├── feature_engineering.py          # Calendar, lag, rolling, and spatial features
│   ├── train_models.py                 # Chronological split, 5-model training & saving
│   ├── evaluate_models.py              # 7-metric evaluation & plot generation
│   └── prediction.py                   # Standalone inference engine
│
├── app.py                              # Streamlit Web Application (Main Dashboard)
├── generate_notebook.py                # Jupyter notebook generator script
├── requirements.txt                    # Pinned Python package dependencies
├── viva_questions.md                   # 21 Viva Questions & Answers
└── README.md                           # Documentation & Project Report
```

---

## 13. Installation Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/AI-Based-Energy-consumption-forecasting-usin-ML.git
cd AI-Based-Energy-consumption-forecasting-usin-ML
```

### Step 2: Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 14. How to Run and Deploy

### Run Data & Model Pipeline from CLI
```bash
# 1. Clean data
python -m src.data_preprocessing

# 2. Extract engineered features
python -m src.feature_engineering

# 3. Train all 5 ML models & compute metrics
python -m src.train_models

# 4. Generate evaluation plots
python -m src.evaluate_models

# 5. Test real-time prediction
python -m src.prediction
```

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```
*Open your browser and navigate to `http://localhost:8501`.*

### Launch Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"** and select your repository and `main` branch.
4. Set Main file path to `app.py`.
5. Click **"Deploy"** — the dashboard will build and deploy online in under 2 minutes.

---

## 15. License & Academic Attribution
Developed for academic study and practical demonstration in Machine Learning and Time-Series Forecasting.
