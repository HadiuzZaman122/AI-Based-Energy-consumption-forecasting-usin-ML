# ⚡ AI-Based Energy Consumption Forecasting Using Machine Learning

A complete, beginner-friendly, college-level Machine Learning project designed to forecast daily electricity consumption across Indian states and regional grids using historical time-series data.

---

## 1. Introduction
Electricity is one of the most critical resources in modern society. Power utilities and grid operators must generate and distribute the exact amount of electricity needed in real-time, because large-scale electricity cannot be easily or cheaply stored.

This project implements an end-to-end Machine Learning pipeline that analyzes historical power consumption records, discovers seasonal and geographic trends, trains multiple regression algorithms, compares their accuracy, and deploys an interactive **Streamlit Web Application** for forecasting future electricity demand.

---

## 2. Problem Statement
Electricity demand fluctuates continuously depending on:
- **Time factors**: Month, day of the week, holidays, and seasonal climate variations (e.g., summer cooling vs. winter heating).
- **Geographic factors**: State population, industrialization level, and geographic coordinates.
- **Recent demand inertia**: Yesterday's consumption and weekly moving averages.

Without accurate forecasting, power grids suffer from either:
1. **Under-generation**: Leading to power outages, brownouts, and grid failures.
2. **Over-generation**: Leading to massive wasted financial cost and excess carbon emissions.

---

## 3. Project Objectives
1. **Data Ingestion & Cleaning**: Load real historical electricity records, handle missing values, eliminate duplicates, and order records chronologically.
2. **Exploratory Data Analysis (EDA)**: Identify top-consuming states, regional grid loads, and seasonal peaks.
3. **Feature Engineering**: Extract calendar features, geographic encodings, and leakage-free lag/rolling features.
4. **Chronological Splitting**: Split data into 80% past training data and 20% future test data.
5. **Model Training & Comparison**: Train 4 Machine Learning models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. **Objective Evaluation**: Evaluate models using MAE, RMSE, and $R^2$ Score on unseen future test data.
7. **Interactive Web Application**: Provide a simple web dashboard for grid exploration and real-time prediction.

---

## 4. Dataset Overview
- **Dataset Name**: Daily Power Consumption in India (2019–2020)
- **Dataset File**: `data/energy_consumption.csv` (16,599 records)
- **Target Variable ($y$)**: `Usage` — Daily Electricity Consumption in **Mega Units (MU)**  
  *(Note: $1\text{ MU} = 1\text{ Million kWh} = 1\text{ GWh}$)*
- **Key Columns**:
  - `Dates`: Daily timestamp (`DD/MM/YYYY 00:00:00`)
  - `States`: 33 Indian States and Union Territories (e.g., Maharashtra, Gujarat, Delhi, Punjab, Tamil Nadu)
  - `Regions`: 5 Regional Grids (`NR`, `WR`, `SR`, `ER`, `NER`)
  - `latitude`, `longitude`: Spatial coordinates of the state
  - `Usage`: Daily power consumption value (in MU)

---

## 5. Technologies Used
- **Python (3.10+)**: Core programming language.
- **Pandas**: Tabular data manipulation, datetime parsing, and aggregations.
- **NumPy**: Fast vectorized numerical computations.
- **Matplotlib & Seaborn**: Statistical charting, time-series plotting, and residual analysis.
- **Scikit-learn**: Machine learning regression models, feature encoders, and metrics.
- **Streamlit**: Interactive web dashboard and user interface.
- **Joblib**: Efficient serialization and saving of trained ML models.

---

## 6. Machine Learning Algorithms Explained

| Algorithm | How it Works (Simple Terms) | Key Advantage |
| :--- | :--- | :--- |
| **Linear Regression** | Draws a straight line through the data points by finding the best mathematical weights. | Very fast baseline, easy to interpret. |
| **Decision Tree** | Splits the data using if-else questions (e.g., *Is Month $\ge 5$?*, *Is State == Maharashtra?*). | Captures non-linear patterns. |
| **Random Forest** | Builds an ensemble of 100 independent decision trees and averages their predictions. | Reduces variance and prevents overfitting. |
| **Gradient Boosting** | Builds decision trees sequentially, where each new tree fixes the mistakes made by previous trees. | Highest accuracy and lowest forecasting error. |

---

## 7. Machine Learning Methodology

```
┌─────────────────────────┐
│   1. Data Collection    │  --> Load historical CSV data
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│    2. Data Cleaning     │  --> Datetime parsing, remove duplicates, chronological sort
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 3. Feature Engineering  │  --> Calendar features, State/Region encodings, Lag-1 & Lag-7
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 4. Chronological Split  │  --> First 80% (Past) = Train Set | Last 20% (Future) = Test Set
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│    5. Model Training    │  --> Train Linear Reg, Decision Tree, Random Forest, Gradient Boosting
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 6. Model Evaluation     │  --> Calculate real MAE, RMSE, and R² Score on Test Set
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 7. Best Model Selection │  --> Select best model & save to models/best_model.joblib
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ 8. Streamlit Dashboard  │  --> Deploy interactive web UI for live forecasting
└─────────────────────────┘
```

---

## 8. Evaluation Metrics

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n} \sum |y - \hat{y}|$$
   *The average absolute difference between predicted and actual consumption in Mega Units.*

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$$
   *Penalizes larger forecast errors more heavily.*

3. **$R^2$ Score (Coefficient of Determination)**:
   $$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$$
   *Measures the proportion of variance explained by the model ($1.0$ is perfect).*

---

## 9. Actual Experimental Results

Evaluated on the **unseen 20% future test set** (3,300 records):

| Model Name | MAE (Mega Units) | RMSE (Mega Units) | $R^2$ Score | Performance Rank |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **6.116 MU** | **14.840 MU** | **0.9837** | **1st 🥇 (Best)** |
| **Random Forest** | 6.299 MU | 15.419 MU | 0.9824 | 2nd 🥈 |
| **Linear Regression** | 6.022 MU | 16.617 MU | 0.9795 | 3rd 🥉 |
| **Decision Tree** | 7.319 MU | 18.990 MU | 0.9733 | 4th |

### 🏆 Best Model: Gradient Boosting Regressor
- Achieved **$R^2 = 0.9837$** (explaining over $98.3\%$ of energy consumption variance).
- Lowest RMSE (**$14.84\text{ MU}$**), minimizing severe forecasting errors.

---

## 10. Project Directory Structure

```
energy-forecasting-ml/
│
├── data/
│   └── energy_consumption.csv          # Standardized cleaned historical dataset
│
├── models/
│   ├── best_model.joblib               # Best trained model (Gradient Boosting)
│   ├── linear_regression.joblib        # Linear Regression model
│   ├── decision_tree.joblib            # Decision Tree model
│   ├── random_forest.joblib            # Random Forest model
│   ├── gradient_boosting.joblib        # Gradient Boosting model
│   ├── label_encoders.joblib           # Preprocessing categorical encoders
│   ├── feature_columns.joblib          # Saved feature list
│   ├── evaluation_data.joblib          # Test set predictions for analysis
│   ├── model_metrics.json              # Dynamic evaluation results
│   ├── actual_vs_predicted.png         # Saved time-series evaluation plot
│   ├── model_performance_comparison.png# Saved model comparison bar chart
│   └── residuals_distribution.png      # Saved residual error histogram
│
├── notebooks/
│   └── exploratory_analysis.ipynb      # Complete educational Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Ingestion, cleaning, date formatting
│   ├── feature_engineering.py          # Temporal, lag, and categorical features
│   ├── train_models.py                 # Chronological split & model training
│   ├── evaluate_models.py              # Metrics reporting & plot generation
│   └── prediction.py                   # Standalone inference helper
│
├── app.py                              # Interactive Streamlit Web Application (Main Entry Point)
├── requirements.txt                    # Project dependencies for Streamlit Cloud & local
├── viva_questions.md                   # 21 Viva Questions & Answers
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore rules
```

---

## 11. How to Run Locally

### Step 1: Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Run Python Pipeline from Terminal
```bash
# Clean data
python -m src.data_preprocessing

# Engineer features
python -m src.feature_engineering

# Train all 4 ML models
python -m src.train_models

# Generate evaluation charts
python -m src.evaluate_models

# Test prediction from CLI
python -m src.prediction
```

### Step 3: Run the Streamlit Web Application
Launch the interactive web app:
```bash
streamlit run app.py
```

### Step 4: Run the Jupyter Notebook
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 12. Deployment to Streamlit Community Cloud

Deploying this application to Streamlit Community Cloud takes less than 2 minutes:

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Deploy AI Energy Forecasting ML project"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"**.
3. Fill in the exact deployment settings:
   - **Repository**: `<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**.
5. No extra secrets or environment variables are required.


---

## 13. Future Scope & Improvements
1. **Weather Integration**: Incorporate temperature, humidity, and heat index data from a weather API (e.g., OpenWeatherMap).
2. **Higher Temporal Resolution**: Train on hourly or 15-minute smart meter readings for real-time peak load prediction.
3. **Deep Learning**: Experiment with LSTM (Long Short-Term Memory) or Transformer-based time series models.
4. **Cloud Deployment**: Deploy the Streamlit app on Streamlit Community Cloud or AWS/GCP for public access.

---

## 14. License & Academic Attribution
This project was developed for educational purposes as an undergraduate Machine Learning course project.

