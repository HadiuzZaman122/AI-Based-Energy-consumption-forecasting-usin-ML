"""
train_models.py
---------------
This module executes the end-to-end Machine Learning training and evaluation pipeline:
1. Chronological Train-Test Split (80% past train data, 20% future test data).
2. Trains 5 Machine Learning Regression Models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
3. Evaluates all models on unseen future test data using 7 standard metrics:
   - MAE (Mean Absolute Error) ↓
   - MSE (Mean Squared Error) ↓
   - RMSE (Root Mean Squared Error) ↓
   - R² (Coefficient of Determination) ↑
   - MAPE (Mean Absolute Percentage Error) ↓
   - MedAE (Median Absolute Error) ↓
   - Explained Variance Score ↑
4. Dynamically identifies the best-performing model based on actual test results.
5. Saves all trained models, pipelines, feature columns, and evaluation datasets.

Author: AI Project Team
"""

import os
import json
from typing import Dict, Tuple, List, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score
)

# Optional XGBoost import with graceful fallback
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from src.data_preprocessing import load_and_preprocess_data
    from src.feature_engineering import prepare_features
except ImportError:
    from data_preprocessing import load_and_preprocess_data
    from feature_engineering import prepare_features


def chronological_split(
    df: pd.DataFrame,
    split_ratio: float = 0.8,
    date_col: str = 'ParsedDate',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Splits the dataset chronologically based on date.

    Why Chronological Splitting?
    ---------------------------
    In energy consumption and time-series forecasting, standard random shuffling causes
    data leakage (the model 'peeks' into future days to predict past days).
    A chronological split strictly uses past records to predict future records,
    mirroring real-world electric grid deployment.
    """
    unique_dates = np.sort(df[date_col].unique())
    split_idx = int(len(unique_dates) * split_ratio)
    split_date = unique_dates[split_idx]

    train_df = df[df[date_col] < split_date].copy().reset_index(drop=True)
    test_df = df[df[date_col] >= split_date].copy().reset_index(drop=True)

    if verbose:
        print("=" * 60)
        print(" Chronological Train-Test Split (80% Past / 20% Future)")
        print("=" * 60)
        print(f"-> Splitting Cutoff Date: {pd.to_datetime(split_date).strftime('%Y-%m-%d')}")
        print(f"-> Training Set: {len(train_df):,} records ({train_df[date_col].min().date()} to {train_df[date_col].max().date()})")
        print(f"-> Testing Set:  {len(test_df):,} records ({test_df[date_col].min().date()} to {test_df[date_col].max().date()})")
        print("=" * 60 + "\n")

    return train_df, test_df, split_date


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Usage',
    models_dir: str = "models",
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, float]], str, pd.DataFrame]:
    """
    Trains regression models and computes real-world evaluation metrics on test set.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data split.
    test_df : pd.DataFrame
        Testing data split.
    feature_cols : list of str
        Predictor feature column names.
    target_col : str, default 'Usage'
        Target variable name.
    models_dir : str, default 'models'
        Directory to save models and metrics.
    verbose : bool, default True
        Whether to print detailed comparison tables.

    Returns:
    --------
    results : dict
        Dictionary of computed metrics for each model.
    best_model_name : str
        Name of the dynamically identified best model.
    eval_df : pd.DataFrame
        Evaluation DataFrame with predictions and error metrics.
    """
    os.makedirs(models_dir, exist_ok=True)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Initialize Regression Models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

    results = {}
    trained_model_objs = {}
    test_predictions = {}

    if verbose:
        print("=" * 60)
        print(" Training & Evaluating Machine Learning Regression Models")
        print("=" * 60)

    for name, model in models.items():
        if verbose:
            print(f"-> Training '{name}'...")

        # Fit model on training split
        model.fit(X_train, y_train)
        trained_model_objs[name] = model

        # Predict on unseen future test split
        y_pred = model.predict(X_test)
        # Ensure non-negative predictions for physical energy validity
        y_pred = np.clip(y_pred, 0, None)
        test_predictions[name] = y_pred

        # Calculate all 7 evaluation metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
        medae = float(median_absolute_error(y_test, y_pred))
        evs = float(explained_variance_score(y_test, y_pred))

        results[name] = {
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2),
            "MedAE": round(medae, 3),
            "Explained_Variance": round(evs, 4)
        }

        # Save individual model artifact
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, filename))

    # Identify Best Model Dynamically (Highest R2 score, lowest RMSE)
    best_model_name = max(results, key=lambda k: (results[k]['R2'], -results[k]['RMSE']))
    best_model_obj = trained_model_objs[best_model_name]

    if verbose:
        print("\n" + "=" * 88)
        print(" Model Comparison Summary (Evaluated on Unseen Future Test Data)")
        print("=" * 88)
        header = f"{'Model Name':<20} | {'MAE (MU)':<10} | {'MSE (MU^2)':<10} | {'RMSE (MU)':<10} | {'R^2 Score':<10} | {'MAPE (%)':<10} | {'MedAE (MU)':<10}"
        print(header)
        print("-" * 88)
        for name, m in results.items():
            best_tag = " [BEST]" if name == best_model_name else ""
            row = f"{name + best_tag:<20} | {m['MAE']:<10.3f} | {m['MSE']:<10.3f} | {m['RMSE']:<10.3f} | {m['R2']:<10.4f} | {m['MAPE']:<10.2f} | {m['MedAE']:<10.3f}"
            print(row)
        print("=" * 88)

        print(f"\n[DYNAMIC BEST PERFORMING MODEL]: {best_model_name}")
        print(f"   * R^2 Score:          {results[best_model_name]['R2']}")
        print(f"   * RMSE:               {results[best_model_name]['RMSE']} MU")
        print(f"   * MAE:                {results[best_model_name]['MAE']} MU")
        print(f"   * MAPE:               {results[best_model_name]['MAPE']}%")
        print(f"   * Explained Variance: {results[best_model_name]['Explained_Variance']}")

    # Save best model and feature metadata
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_columns.joblib"))

    # Save metrics JSON
    metrics_payload = {
        "models": results,
        "best_model": best_model_name,
        "best_metrics": results[best_model_name],
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "target_col": target_col,
        "unit": "Mega Units (MU)",
        "train_records": len(train_df),
        "test_records": len(test_df)
    }
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=4)

    # Build evaluation DataFrame for test set
    eval_df = test_df.copy()
    eval_df['Actual'] = y_test

    for name, preds in test_predictions.items():
        eval_df[f"Pred_{name}"] = preds

    # Add error columns for the best model
    best_pred = eval_df[f"Pred_{best_model_name}"]
    eval_df['Predicted'] = best_pred
    eval_df['Residual'] = eval_df['Actual'] - eval_df['Predicted']
    eval_df['Absolute_Error'] = (eval_df['Actual'] - eval_df['Predicted']).abs()
    eval_df['Percentage_Error'] = (eval_df['Absolute_Error'] / eval_df['Actual'].replace(0, np.nan)) * 100

    joblib.dump(eval_df, os.path.join(models_dir, "evaluation_data.joblib"))

    # Also export CSV files for download
    comparison_table_rows = []
    for name, m in results.items():
        comparison_table_rows.append({
            "Model Name": name,
            "MAE (MU)": m["MAE"],
            "MSE (MU^2)": m["MSE"],
            "RMSE (MU)": m["RMSE"],
            "R2 Score": m["R2"],
            "MAPE (%)": m["MAPE"],
            "Median Absolute Error (MU)": m["MedAE"],
            "Explained Variance": m["Explained_Variance"],
            "Is Best Model": name == best_model_name
        })
    pd.DataFrame(comparison_table_rows).to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)

    act_vs_pred_cols = ['ParsedDate', 'States', 'Regions', 'Actual', 'Predicted', 'Absolute_Error', 'Percentage_Error']
    existing_act_cols = [c for c in act_vs_pred_cols if c in eval_df.columns]
    eval_df[existing_act_cols].to_csv(os.path.join(models_dir, "actual_vs_predicted.csv"), index=False)

    residual_cols = ['ParsedDate', 'States', 'Actual', 'Predicted', 'Residual', 'Absolute_Error']
    existing_res_cols = [c for c in residual_cols if c in eval_df.columns]
    eval_df[existing_res_cols].to_csv(os.path.join(models_dir, "residual_analysis.csv"), index=False)

    if verbose:
        print(f"\nAll models and evaluation CSV artifacts successfully saved in '{models_dir}/' directory!\n")

    return results, best_model_name, eval_df


def run_pipeline() -> Tuple[Dict[str, Dict[str, float]], str]:
    """Executes the complete machine learning workflow."""
    # 1. Preprocess raw data
    clean_df, meta = load_and_preprocess_data()

    # 2. Engineer Features
    feat_df, feature_cols, encoders = prepare_features(clean_df, target_col=meta['target_col'])

    # 3. Chronological Split (80% train, 20% test)
    train_df, test_df, split_date = chronological_split(feat_df, split_ratio=0.8, date_col='ParsedDate')

    # 4. Train & Evaluate Models
    results, best_model_name, eval_df = train_and_evaluate_models(
        train_df, test_df, feature_cols, target_col=meta['target_col']
    )

    return results, best_model_name


if __name__ == "__main__":
    run_pipeline()
