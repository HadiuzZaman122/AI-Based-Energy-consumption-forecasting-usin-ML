"""
evaluate_models.py
------------------
This module generates detailed evaluation reports, metrics comparison tables,
and visualization plots (Actual vs Predicted, Model Comparison, Residuals).

Key Metrics:
- MAE (Mean Absolute Error): Average absolute difference between predicted and actual consumption.
- RMSE (Root Mean Squared Error): Penalizes larger forecasting errors more severely.
- R² (Coefficient of Determination): Proportion of variance in consumption explained by the model.

Author: AI Project Team
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_evaluation_data(models_dir="models"):
    """Loads saved evaluation dataset and metrics."""
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    eval_path = os.path.join(models_dir, "evaluation_data.joblib")
    
    if not os.path.exists(metrics_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(
            "Evaluation data not found. Please run 'python -m src.train_models' first!"
        )
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    eval_df = joblib.load(eval_path)
    return metrics, eval_df


def plot_model_comparison(metrics, save_path=None):
    """
    Plots a side-by-side bar chart comparing MAE, RMSE, and R² across all models.
    """
    save_path = save_path or os.path.join("models", "model_performance_comparison.png")
    models_data = metrics["models"]
    model_names = list(models_data.keys())
    maes = [models_data[m]["MAE"] for m in model_names]
    rmses = [models_data[m]["RMSE"] for m in model_names]
    r2s = [models_data[m]["R2"] for m in model_names]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. MAE Comparison
    bars1 = axes[0].bar(model_names, maes, color='#4C72B0', edgecolor='black', alpha=0.85)
    axes[0].set_title("Mean Absolute Error (MAE - Lower is Better)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("MAE (Mega Units)")
    axes[0].tick_params(axis='x', rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        
    # 2. RMSE Comparison
    bars2 = axes[1].bar(model_names, rmses, color='#DD8452', edgecolor='black', alpha=0.85)
    axes[1].set_title("Root Mean Squared Error (RMSE - Lower is Better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("RMSE (Mega Units)")
    axes[1].tick_params(axis='x', rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        
    # 3. R² Comparison
    bars3 = axes[2].bar(model_names, r2s, color='#55A868', edgecolor='black', alpha=0.85)
    axes[2].set_title("R² Score (Higher is Better)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("R² Score")
    axes[2].set_ylim(0.9, 1.0)
    axes[2].tick_params(axis='x', rotation=25)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
        
    plt.suptitle("Machine Learning Model Performance Comparison", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved model comparison plot to: {save_path}")


def plot_actual_vs_predicted(eval_df, metrics, save_path=None):
    """
    Plots Actual vs Predicted energy consumption over time for a representative state or aggregated total.
    """
    save_path = save_path or os.path.join("models", "actual_vs_predicted.png")
    best_model = metrics["best_model"]
    pred_col = f"Pred_{best_model}"
    
    # Pick a major state (or total aggregated) for clear time-series visualization
    sample_state = "Maharashtra" if "Maharashtra" in eval_df['States'].values else eval_df['States'].iloc[0]
    state_df = eval_df[eval_df['States'] == sample_state].sort_values('ParsedDate')
    
    plt.figure(figsize=(14, 6))
    plt.plot(state_df['ParsedDate'], state_df['Actual'], label='Actual Consumption', color='#1f77b4', linewidth=2.2)
    plt.plot(state_df['ParsedDate'], state_df[pred_col], label=f'Predicted ({best_model})', color='#ff7f0e', linestyle='--', linewidth=2.0)
    
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


def plot_residuals(eval_df, metrics, save_path=None):
    """
    Plots residual errors distribution (Actual - Predicted).
    """
    save_path = save_path or os.path.join("models", "residuals_distribution.png")

    best_model = metrics["best_model"]
    pred_col = f"Pred_{best_model}"
    residuals = eval_df['Actual'] - eval_df[pred_col]
    
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True, color='#2ca02c', bins=40, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    plt.title(f"Residual Error Distribution ({best_model})", fontsize=13, fontweight='bold')
    plt.xlabel("Prediction Error (Actual - Predicted) in MU", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved residual distribution plot to: {save_path}")


def evaluate_and_generate_reports():
    """Runs full evaluation and generates all comparison plots."""
    print("==================================================")
    print(" Generating Model Evaluation Graphs and Reports")
    print("==================================================")
    metrics, eval_df = load_evaluation_data()
    
    plot_model_comparison(metrics)
    plot_actual_vs_predicted(eval_df, metrics)
    plot_residuals(eval_df, metrics)
    
    print("\nAll evaluation reports and graphs generated successfully!\n")


if __name__ == "__main__":
    evaluate_and_generate_reports()
