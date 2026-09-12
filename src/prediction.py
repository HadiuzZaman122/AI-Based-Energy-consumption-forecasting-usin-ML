"""
prediction.py
-------------
This module provides a robust, user-friendly inference pipeline for real-time
energy consumption forecasting using the trained Machine Learning regression models.

Capabilities:
1. Loads trained models, feature schemas, and categorical encoders from disk.
2. Derives all temporal features (month, day, weekday, weekend, quarter, etc.) automatically from target date.
3. Automatically maps state spatial coordinates (latitude, longitude) and regional grid codes.
4. Supports user-supplied recent consumption context (lags and rolling averages) or defaults to historical state medians.
5. Performs real inference using the designated model (or optimal Best Model).
6. Returns comprehensive prediction payload with units and comparative baseline analytics.

Author: AI Project Team
"""

import os
from typing import Dict, List, Any, Optional, Union
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date


class EnergyPredictor:
    """
    Inference engine to load trained ML models and forecast energy consumption.
    """

    def __init__(self, models_dir: str = "models", data_path: Optional[str] = None):
        self.models_dir = models_dir
        self.data_path = data_path or os.path.join("data", "energy_consumption.csv")
        self._load_artifacts()
        self._build_state_metadata()

    def _load_artifacts(self) -> None:
        """Loads trained models, categorical encoders, and feature column definitions."""
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        feat_path = os.path.join(self.models_dir, "feature_columns.joblib")
        enc_path = os.path.join(self.models_dir, "label_encoders.joblib")

        if not (os.path.exists(best_model_path) and os.path.exists(feat_path) and os.path.exists(enc_path)):
            try:
                from src.train_models import run_pipeline
            except ImportError:
                from train_models import run_pipeline
            run_pipeline()

        self.best_model = joblib.load(best_model_path)
        self.feature_cols = joblib.load(feat_path)
        self.encoders = joblib.load(enc_path)

        # Load all individual trained models
        self.all_models = {}
        if os.path.exists(self.models_dir):
            for fname in os.listdir(self.models_dir):
                if fname.endswith(".joblib") and fname not in [
                    "label_encoders.joblib",
                    "feature_columns.joblib",
                    "evaluation_data.joblib"
                ]:
                    m_key = fname.replace(".joblib", "").replace("_", " ").title()
                    self.all_models[m_key] = joblib.load(os.path.join(self.models_dir, fname))

    def _build_state_metadata(self) -> None:
        """Constructs lookup metadata for state coordinates, regions, and baseline consumption."""
        if not os.path.exists(self.data_path):
            candidates = ["long_data_.csv", os.path.join("..", "data", "energy_consumption.csv"), "dataset_tk.csv"]
            for cand in candidates:
                if os.path.exists(cand):
                    self.data_path = cand
                    break

        if os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                if 'States' in df.columns and 'Usage' in df.columns:
                    # Clean usage
                    df['Usage'] = pd.to_numeric(df['Usage'], errors='coerce')
                    grouped = df.groupby('States')
                    
                    self.state_meta = {}
                    for state, group in grouped:
                        self.state_meta[state] = {
                            "region": str(group['Regions'].iloc[0]) if 'Regions' in group.columns else "WR",
                            "latitude": float(group['latitude'].iloc[0]) if 'latitude' in group.columns else 20.0,
                            "longitude": float(group['longitude'].iloc[0]) if 'longitude' in group.columns else 77.0,
                            "mean_usage": float(group['Usage'].mean()),
                            "median_usage": float(group['Usage'].median()),
                            "min_usage": float(group['Usage'].min()),
                            "max_usage": float(group['Usage'].max()),
                            "std_usage": float(group['Usage'].std()) if group['Usage'].std() > 0 else 1.0
                        }
                    return
            except Exception:
                pass

        # Fallback default metadata dictionary
        self.state_meta = {}

    def get_available_states(self) -> List[str]:
        """Returns sorted list of available states."""
        if 'States' in self.encoders:
            return sorted(list(self.encoders['States'].classes_))
        elif self.state_meta:
            return sorted(list(self.state_meta.keys()))
        return ["Maharashtra", "Gujarat", "Tamil Nadu", "UP", "Delhi", "Punjab", "Karnataka"]

    def get_state_info(self, state_name: str) -> Dict[str, Any]:
        """Retrieves typical usage, region, and coordinates for a selected state."""
        states = self.get_available_states()
        if state_name not in states and len(states) > 0:
            state_name = states[0]

        if self.state_meta and state_name in self.state_meta:
            return self.state_meta[state_name]

        # Standard fallback for safety
        return {
            "region": "WR",
            "latitude": 20.0,
            "longitude": 77.0,
            "mean_usage": 120.0,
            "median_usage": 115.0,
            "min_usage": 10.0,
            "max_usage": 450.0,
            "std_usage": 25.0
        }

    def predict(
        self,
        state_name: str,
        date_input: Union[str, date, datetime, pd.Timestamp],
        usage_lag_1: Optional[float] = None,
        usage_lag_2: Optional[float] = None,
        usage_lag_7: Optional[float] = None,
        usage_rolling_7: Optional[float] = None,
        usage_rolling_14: Optional[float] = None,
        model_choice: str = "Best Model"
    ) -> Dict[str, Any]:
        """
        Predicts energy consumption for a given state, date, and consumption context.

        Parameters:
        -----------
        state_name : str
            Target Indian State/UT name.
        date_input : str, date, or datetime
            Target forecast date.
        usage_lag_1 : float, optional
            Yesterday's consumption. Defaults to state median.
        usage_lag_2 : float, optional
            2 days prior consumption. Defaults to state median.
        usage_lag_7 : float, optional
            Same day last week consumption. Defaults to state median.
        usage_rolling_7 : float, optional
            7-day average consumption. Defaults to state median.
        usage_rolling_14 : float, optional
            14-day average consumption. Defaults to state median.
        model_choice : str, default 'Best Model'
            Name of model to use for inference.

        Returns:
        --------
        result : dict
            Comprehensive dictionary containing predicted value, comparison, and feature payload.
        """
        # 1. Parse Datetime
        if isinstance(date_input, str):
            try:
                dt = pd.to_datetime(date_input, format='%Y-%m-%d')
            except Exception:
                dt = pd.to_datetime(date_input, dayfirst=True)
        elif isinstance(date_input, date) and not isinstance(date_input, datetime):
            dt = pd.to_datetime(date_input)
        else:
            dt = pd.to_datetime(date_input)

        # 2. Retrieve State Profile
        state_info = self.get_state_info(state_name)
        region = state_info["region"]
        lat = state_info["latitude"]
        lon = state_info["longitude"]
        baseline_median = state_info["median_usage"]
        baseline_mean = state_info["mean_usage"]

        # 3. Handle Contextual Lags and Rolling Averages
        lag1 = float(usage_lag_1) if (usage_lag_1 is not None and usage_lag_1 > 0) else baseline_median
        lag2 = float(usage_lag_2) if (usage_lag_2 is not None and usage_lag_2 > 0) else baseline_median
        lag7 = float(usage_lag_7) if (usage_lag_7 is not None and usage_lag_7 > 0) else baseline_median
        roll7 = float(usage_rolling_7) if (usage_rolling_7 is not None and usage_rolling_7 > 0) else (lag1 + lag7) / 2
        roll14 = float(usage_rolling_14) if (usage_rolling_14 is not None and usage_rolling_14 > 0) else roll7
        roll_std7 = float(state_info.get("std_usage", 5.0))

        # 4. Extract Calendar Features
        year = dt.year
        month = dt.month
        day = dt.day
        dayofweek = dt.dayofweek
        is_weekend = 1 if dayofweek >= 5 else 0
        quarter = dt.quarter
        dayofyear = dt.dayofyear
        weekofyear = int(dt.isocalendar().week)

        # 5. Encode Categorical Variables
        state_enc = (
            self.encoders['States'].transform([state_name])[0]
            if ('States' in self.encoders and state_name in set(self.encoders['States'].classes_))
            else 0
        )
        region_enc = (
            self.encoders['Regions'].transform([region])[0]
            if ('Regions' in self.encoders and region in set(self.encoders['Regions'].classes_))
            else 0
        )

        # 6. Build Candidate Feature Map
        feature_dict = {
            'Year': year,
            'Month': month,
            'Day': day,
            'DayOfWeek': dayofweek,
            'IsWeekend': is_weekend,
            'Quarter': quarter,
            'DayOfYear': dayofyear,
            'WeekOfYear': weekofyear,
            'latitude': lat,
            'longitude': lon,
            'States_Encoded': state_enc,
            'Regions_Encoded': region_enc,
            'Usage_Lag_1': lag1,
            'Usage_Lag_2': lag2,
            'Usage_Lag_7': lag7,
            'Usage_Rolling_Mean_7': roll7,
            'Usage_Rolling_Mean_14': roll14,
            'Usage_Rolling_Std_7': roll_std7
        }

        # Align with exact trained feature column order
        input_row = pd.DataFrame([feature_dict])
        aligned_row = input_row[[c for c in self.feature_cols if c in input_row.columns]]

        # Fill any missing feature column with 0
        for col in self.feature_cols:
            if col not in aligned_row.columns:
                aligned_row[col] = 0
        aligned_row = aligned_row[self.feature_cols]

        # 7. Select Model
        model_key = model_choice.title()
        if model_key in self.all_models:
            model = self.all_models[model_key]
        elif model_choice == "Best Model" or "Best" in model_choice:
            model = self.best_model
        else:
            # Match partial name
            matched = False
            for k, m in self.all_models.items():
                if model_choice.lower().replace(" ", "") in k.lower().replace(" ", ""):
                    model = m
                    matched = True
                    break
            if not matched:
                model = self.best_model

        # 8. Generate Real Model Prediction
        pred_raw = float(model.predict(aligned_row)[0])
        predicted_value = max(0.0, round(pred_raw, 2))

        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        diff_pct = round(((predicted_value - baseline_median) / baseline_median) * 100, 1) if baseline_median > 0 else 0.0

        return {
            "state": state_name,
            "region": region,
            "date": dt.strftime('%Y-%m-%d'),
            "day_name": weekday_names[dayofweek],
            "is_weekend": bool(is_weekend),
            "predicted_usage": predicted_value,
            "unit": "Mega Units (MU)",
            "model_used": model_choice,
            "state_median_historical": round(baseline_median, 2),
            "state_mean_historical": round(baseline_mean, 2),
            "state_min_historical": round(state_info["min_usage"], 2),
            "state_max_historical": round(state_info["max_usage"], 2),
            "diff_vs_baseline_pct": diff_pct,
            "input_features": {k: round(v, 3) if isinstance(v, float) else v for k, v in feature_dict.items() if k in self.feature_cols}
        }


if __name__ == "__main__":
    predictor = EnergyPredictor()
    test_states = ["Maharashtra", "Delhi", "Punjab", "Tamil Nadu"]
    test_date = "2024-06-15"

    print("=" * 60)
    print(" Energy Predictor Demonstration")
    print("=" * 60)
    for st in test_states:
        res = predictor.predict(st, test_date)
        print(f"-> State: {res['state']:<15} | Date: {res['date']} ({res['day_name']})")
        print(f"   Predicted Consumption: {res['predicted_usage']} {res['unit']}")
        print(f"   Historical Median:     {res['state_median_historical']} {res['unit']}")
        print(f"   Deviation vs Baseline: {res['diff_vs_baseline_pct']:+.1f}%\n")
