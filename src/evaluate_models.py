"""
evaluate_models.py
------------------
This module generates comprehensive evaluation reports, comparison tables,
and statistical visualization plots for the trained machine learning models.

Key Evaluation Features:
1. Multi-metric model comparison across:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Squared Error (MSE)
   - Coefficient of Determination (R²)
   - Mean Absolute Percentage Error (MAPE)
   - Median Absolute Error (MedAE)
   - Explained Variance Score
2. Actual vs. Predicted Time-Series Visualizations.
3. Residual and Error Distribution Analysis (Histogram, KDE, Scatter vs. Predicted).
4. Automated export of evaluation summary figures.

Author: AI Project Team
"""

import os
import json
from typing import Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_evaluation_data(models_dir: str = "models") -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Loads saved evaluation dataset and metrics from disk.

    Parameters:
    -----------
    models_dir : str, default 'models'
        Directory containing saved models and evaluation metadata.

    Returns:
    --------
    metrics : dict
        Parsed JSON model metrics dictionary.
    eval_df : pd.DataFrame
        Test set evaluation DataFrame with predictions.
    """
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    eval_path = os.path.join(models_dir, "evaluation_data.joblib")

    if not os.path.exists(metrics_path) or not os.path.exists(eval_path):
        # Trigger training pipeline if artifacts are not found
        try:
            from src.train_models import run_pipeline
        except ImportError:
            from train_models import run_pipeline
        run_pipeline()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    eval_df = joblib.load(eval_path)
    return metrics, eval_df


def plot_model_comparison(metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plots a multi-panel comparison chart of MAE, RMSE, R², and MAPE across all models.
    """
    save_path = save_path or os.path.join("models", "model_performance_comparison.png")
    models_data = metrics["models"]
    model_names = list(models_data.keys())

    maes = [models_data[m]["MAE"] for m in model_names]
    rmses = [models_data[m]["RMSE"] for m in model_names]
    r2s = [models_data[m]["R2"] for m in model_names]
    mapes = [models_data[m]["MAPE"] for m in model_names]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. MAE Comparison
    bars1 = axes[0, 0].bar(model_names, maes, color='#2563EB', edgecolor='black', alpha=0.85)
    axes[0, 0].set_title("Mean Absolute Error (MAE - Lower is Better)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("MAE (Mega Units)")
    axes[0, 0].tick_params(axis='x', rotation=20)
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

    # 2. RMSE Comparison
    bars2 = axes[0, 1].bar(model_names, rmses, color='#DC2626', edgecolor='black', alpha=0.85)
    axes[0, 1].set_title("Root Mean Squared Error (RMSE - Lower is Better)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("RMSE (Mega Units)")
    axes[0, 1].tick_params(axis='x', rotation=20)
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

    # 3. R² Score Comparison
    bars3 = axes[1, 0].bar(model_names, r2s, color='#059669', edgecolor='black', alpha=0.85)
    axes[1, 0].set_title("R² Score (Variance Explained - Higher is Better)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("R² Score")
    axes[1, 0].set_ylim(0.9, 1.0)
    axes[1, 0].tick_params(axis='x', rotation=20)
    for bar in bars3:
        yval = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)

    # 4. MAPE Comparison
    bars4 = axes[1, 1].bar(model_names, mapes, color='#D97706', edgecolor='black', alpha=0.85)
    axes[1, 1].set_title("Mean Absolute Percentage Error (MAPE % - Lower is Better)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("MAPE (%)")
    axes[1, 1].tick_params(axis='x', rotation=20)
    for bar in bars4:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.suptitle("Machine Learning Regression Models Performance Comparison", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved model comparison plot to: {save_path}")


def plot_actual_vs_predicted(eval_df: pd.DataFrame, metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plots Actual vs Predicted energy consumption over time on the unseen test set.
    """
    save_path = save_path or os.path.join("models", "actual_vs_predicted.png")
    best_model = metrics["best_model"]
    pred_col = f"Pred_{best_model}"

    sample_state = "Maharashtra" if "Maharashtra" in eval_df['States'].values else eval_df['States'].iloc[0]
    state_df = eval_df[eval_df['States'] == sample_state].sort_values('ParsedDate')

    plt.figure(figsize=(14, 6))
    plt.plot(state_df['ParsedDate'], state_df['Actual'], label='Actual Consumption', color='#1E40AF', linewidth=2.2)
    plt.plot(state_df['ParsedDate'], state_df[pred_col], label=f'Predicted ({best_model})', color='#EA580C', linestyle='--', linewidth=2.0)

    plt.title(f"Actual vs Predicted Energy Consumption in {sample_state} (Test Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Energy Consumption (Mega Units - MU)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved actual vs predicted plot to: {save_path}")


def plot_residuals(eval_df: pd.DataFrame, metrics: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Plots residual errors distribution and residuals vs predicted scatter plot.
    """
    save_path = save_path or os.path.join("models", "residuals_distribution.png")
    best_model = metrics["best_model"]
    residuals = eval_df['Actual'] - eval_df['Predicted']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Histogram & KDE
    sns.histplot(residuals, kde=True, color='#0D9488', bins=40, edgecolor='black', alpha=0.7, ax=axes[0])
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    axes[0].set_title(f"Residual Error Distribution ({best_model})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Residual (Actual - Predicted) in MU", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend()

    # 2. Residuals vs Predicted
    axes[1].scatter(eval_df['Predicted'], residuals, color='#2563EB', alpha=0.4, edgecolors='none', s=25)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    axes[1].set_title(f"Residuals vs Predicted Values ({best_model})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Energy Consumption (MU)", fontsize=11)
    axes[1].set_ylabel("Residual (Actual - Predicted) in MU", fontsize=11)
    axes[1].legend()

    plt.suptitle("Residual Analysis & Error Distribution", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved residual distribution plot to: {save_path}")


def evaluate_and_generate_reports() -> None:
    """Runs full evaluation and generates all comparison plots and tables."""
    print("=" * 60)
    print(" Generating Model Evaluation Graphs and Reports")
    print("=" * 60)
    metrics, eval_df = load_evaluation_data()

    plot_model_comparison(metrics)
    plot_actual_vs_predicted(eval_df, metrics)
    plot_residuals(eval_df, metrics)

    print("\nAll evaluation reports and graphs generated successfully!\n")


if __name__ == "__main__":
    evaluate_and_generate_reports()
