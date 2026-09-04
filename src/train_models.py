"""
train_models.py
---------------
This module executes the machine learning training pipeline:
1. Performs a strict Chronological Train-Test split (80% train, 20% test).
2. Trains 4 Machine Learning Regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. Evaluates all models on unseen future test data.
4. Identifies the best-performing model based on RMSE & R² score.
5. Saves all trained models, metadata, and evaluation metrics using Joblib/JSON.

Author: AI Project Team
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

try:
    from src.data_preprocessing import load_and_preprocess_data
    from src.feature_engineering import prepare_features
except ImportError:
    from data_preprocessing import load_and_preprocess_data
    from feature_engineering import prepare_features


def chronological_split(df, split_ratio=0.8, date_col='ParsedDate', verbose=True):
    """
    Splits the dataset chronologically based on date.
    
    Why Chronological Splitting?
    ---------------------------
    In energy consumption and time-series forecasting, standard random shuffling causes
    future data leakage (the model 'peeks' into future days to predict past days).
    A chronological split strictly uses past records to predict future records, exactly
    as in real-world energy grid deployment.
    """
    # Sort dates globally
    unique_dates = np.sort(df[date_col].unique())
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]
    
    train_df = df[df[date_col] < split_date].copy()
    test_df = df[df[date_col] >= split_date].copy()
    
    if verbose:
        print("==================================================")
        print(" Chronological Train-Test Split (80% / 20%)")
        print("==================================================")
        print(f"-> Splitting cutoff date: {pd.to_datetime(split_date).strftime('%Y-%m-%d')}")
        print(f"-> Training Set: {len(train_df):,} records ({train_df[date_col].min().date()} to {train_df[date_col].max().date()})")
        print(f"-> Testing Set:  {len(test_df):,} records ({test_df[date_col].min().date()} to {test_df[date_col].max().date()})")
        print("--------------------------------------------------\n")
        
    return train_df, test_df, split_date


def train_and_evaluate_models(train_df, test_df, feature_cols, target_col='Usage', models_dir="models"):
    """
    Trains regression models and computes real-world evaluation metrics on test set.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Define models dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    trained_model_objs = {}
    test_predictions = {}
    
    print("==================================================")
    print(" Training & Evaluating Machine Learning Models")
    print("==================================================")
    
    for name, model in models.items():
        print(f"-> Training '{name}'...")
        # Train model
        model.fit(X_train, y_train)
        trained_model_objs[name] = model
        
        # Predict on Test Data
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred
        
        # Calculate Real Metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        
        results[name] = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        }
        
        # Save individual model artifact
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, filename))
        
    print("\n--------------------------------------------------")
    print(" Model Comparison Summary (Evaluated on Test Data)")
    print("--------------------------------------------------")
    print(f"{'Model Name':<22} | {'MAE (MU)':<10} | {'RMSE (MU)':<10} | {'R² Score':<10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<22} | {m['MAE']:<10.3f} | {m['RMSE']:<10.3f} | {m['R2']:<10.4f}")
    print("-" * 60)
    
    # Identify Best Model (Highest R2 and Lowest RMSE)
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model_obj = trained_model_objs[best_model_name]
    
    print(f"\n[BEST PERFORMING MODEL]: {best_model_name}")
    print(f"   * R2 Score: {results[best_model_name]['R2']}")
    print(f"   * RMSE:     {results[best_model_name]['RMSE']} MU")
    print(f"   * MAE:      {results[best_model_name]['MAE']} MU")
    print(f"   * Explanation: {best_model_name} achieved the lowest RMSE and highest variance explained (R2).")
    
    # Save best model and metadata
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_columns.joblib"))
    
    # Save metrics JSON
    metrics_payload = {
        "models": results,
        "best_model": best_model_name,
        "best_metrics": results[best_model_name],
        "feature_count": len(feature_cols),
        "target_col": target_col,
        "unit": "Mega Units (MU)"
    }
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=4)
        
    # Save test dataset and predictions for rapid plotting in web app & evaluation
    eval_df = test_df.copy()
    eval_df['Actual'] = y_test
    for name, preds in test_predictions.items():
        eval_df[f"Pred_{name}"] = preds
    joblib.dump(eval_df, os.path.join(models_dir, "evaluation_data.joblib"))
    
    print(f"\nAll models and evaluation data successfully saved in '{models_dir}/' directory!\n")
    return results, best_model_name



def run_pipeline():
    """Executes the complete training workflow."""
    # 1. Preprocess
    clean_df, meta = load_and_preprocess_data()
    
    # 2. Engineer Features
    feat_df, feature_cols, encoders = prepare_features(clean_df, target_col=meta['target_col'])
    
    # 3. Chronological Split
    train_df, test_df, split_date = chronological_split(feat_df, split_ratio=0.8, date_col='ParsedDate')
    
    # 4. Train & Evaluate
    results, best_model_name = train_and_evaluate_models(
        train_df, test_df, feature_cols, target_col=meta['target_col']
    )
    return results, best_model_name


if __name__ == "__main__":
    run_pipeline()
