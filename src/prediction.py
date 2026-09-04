"""
prediction.py
-------------
This module provides a clean, user-friendly inference pipeline for making future
energy consumption predictions using the trained Machine Learning models.

Features:
- Handles feature transformation for single or batch inputs.
- Automatically handles coordinates and region mapping for each state.
- Supports specifying recent consumption lags or falling back to state baseline averages.
- Returns formatted prediction with units and contextual explanations.

Author: AI Project Team
"""

import os
import joblib
import pandas as pd
import numpy as np


class EnergyPredictor:
    """
    Inference class to load trained ML models and predict energy consumption.
    """
    def __init__(self, models_dir="models", data_path=None):
        self.models_dir = models_dir
        self.data_path = data_path or os.path.join("data", "energy_consumption.csv")
        self._load_artifacts()
        self._build_state_metadata()

    def _load_artifacts(self):
        """Loads saved model, encoders, and feature column list."""
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        if not os.path.exists(best_model_path):
            try:
                from src.train_models import run_pipeline
            except ImportError:
                from train_models import run_pipeline
            run_pipeline()
            
        self.best_model = joblib.load(os.path.join(self.models_dir, "best_model.joblib"))
        self.feature_cols = joblib.load(os.path.join(self.models_dir, "feature_columns.joblib"))
        self.encoders = joblib.load(os.path.join(self.models_dir, "label_encoders.joblib"))
        
        # Load other models if available
        self.all_models = {}
        if os.path.exists(self.models_dir):
            for fname in os.listdir(self.models_dir):
                if fname.endswith(".joblib") and fname not in ["label_encoders.joblib", "feature_columns.joblib", "evaluation_data.joblib"]:
                    m_name = fname.replace(".joblib", "").replace("_", " ").title()
                    self.all_models[m_name] = joblib.load(os.path.join(self.models_dir, fname))

    def _build_state_metadata(self):
        """Builds lookup table for state coordinates, regions, and baseline consumption."""
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)

            if 'States' in df.columns and 'Regions' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                self.state_meta = df.groupby('States').agg({
                    'Regions': 'first',
                    'latitude': 'first',
                    'longitude': 'first',
                    'Usage': ['mean', 'median', 'min', 'max']
                })
                return
                
        # Default fallback metadata for Indian States
        self.state_meta = None

    def get_available_states(self):
        """Returns sorted list of available states."""
        if 'States' in self.encoders:
            return sorted(list(self.encoders['States'].classes_))
        return ["Maharashtra", "Gujarat", "Tamil Nadu", "UP", "Delhi", "Punjab", "Karnataka"]

    def get_state_info(self, state_name):
        """Retrieves typical usage, region, and coordinates for a selected state."""
        states = self.get_available_states()
        if state_name not in states:
            state_name = states[0]
            
        if self.state_meta is not None and state_name in self.state_meta.index:
            row = self.state_meta.loc[state_name]
            return {
                "region": str(row[('Regions', 'first')]),
                "latitude": float(row[('latitude', 'first')]),
                "longitude": float(row[('longitude', 'first')]),
                "mean_usage": float(row[('Usage', 'mean')]),
                "median_usage": float(row[('Usage', 'median')]),
                "min_usage": float(row[('Usage', 'min')]),
                "max_usage": float(row[('Usage', 'max')])
            }
        # Fallback
        return {
            "region": "WR",
            "latitude": 20.0,
            "longitude": 77.0,
            "mean_usage": 100.0,
            "median_usage": 95.0,
            "min_usage": 10.0,
            "max_usage": 350.0
        }

    def predict(self, state_name, date_input, usage_lag_1=None, usage_lag_7=None, usage_rolling_7=None, model_choice="Best Model"):
        """
        Predicts energy consumption for a given state, date, and recent consumption context.
        
        Parameters:
        -----------
        state_name : str
            Name of the Indian State/UT.
        date_input : str or datetime or pd.Timestamp
            Target date for prediction.
        usage_lag_1 : float, optional
            Yesterday's energy consumption. If None, uses state median.
        usage_lag_7 : float, optional
            Last week's energy consumption on same day. If None, uses state median.
        usage_rolling_7 : float, optional
            7-day average consumption. If None, uses state median.
        model_choice : str
            Which trained model to use ("Best Model", "Linear Regression", "Random Forest", etc.)
            
        Returns:
        --------
        result : dict
            Dictionary containing predicted value, input features, and metadata.
        """
        # 1. Parse Date
        if isinstance(date_input, str):
            try:
                dt = pd.to_datetime(date_input, format='%Y-%m-%d')
            except Exception:
                dt = pd.to_datetime(date_input, dayfirst=True)
        else:
            dt = pd.to_datetime(date_input)

        
        # 2. Get State Info
        state_info = self.get_state_info(state_name)
        region = state_info["region"]
        lat = state_info["latitude"]
        lon = state_info["longitude"]
        baseline = state_info["median_usage"]
        
        # 3. Handle default lag inputs
        lag1 = float(usage_lag_1) if usage_lag_1 is not None and usage_lag_1 > 0 else baseline
        lag7 = float(usage_lag_7) if usage_lag_7 is not None and usage_lag_7 > 0 else baseline
        roll7 = float(usage_rolling_7) if usage_rolling_7 is not None and usage_rolling_7 > 0 else (lag1 + lag7) / 2
        
        # 4. Extract Time Features
        month = dt.month
        day = dt.day
        dayofweek = dt.dayofweek
        is_weekend = 1 if dayofweek >= 5 else 0
        quarter = dt.quarter
        dayofyear = dt.dayofyear
        weekofyear = int(dt.isocalendar().week)
        
        # 5. Encode Categorical
        state_enc = self.encoders['States'].transform([state_name])[0] if 'States' in self.encoders else 0
        region_enc = self.encoders['Regions'].transform([region])[0] if 'Regions' in self.encoders else 0
        
        # 6. Build Feature Vector dictionary
        feature_dict = {
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
            'Usage_Lag_7': lag7,
            'Usage_Rolling_Mean_7': roll7
        }
        
        # Align with exact feature columns used in training
        input_row = pd.DataFrame([feature_dict])[self.feature_cols]
        
        # 7. Select Model
        model_key = model_choice.title()
        if model_key in self.all_models:
            model = self.all_models[model_key]
        else:
            model = self.best_model
            
        # 8. Execute Prediction
        predicted_value = float(model.predict(input_row)[0])
        # Ensure non-negative output
        predicted_value = max(0.0, round(predicted_value, 2))
        
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        return {
            "state": state_name,
            "region": region,
            "date": dt.strftime('%Y-%m-%d'),
            "day_name": weekday_names[dayofweek],
            "is_weekend": bool(is_weekend),
            "predicted_usage": predicted_value,
            "unit": "Mega Units (MU)",
            "model_used": model_choice,
            "state_median_historical": round(baseline, 2),
            "state_min_historical": round(state_info["min_usage"], 2),
            "state_max_historical": round(state_info["max_usage"], 2),
            "input_features": feature_dict
        }


if __name__ == "__main__":
    predictor = EnergyPredictor()
    test_states = ["Maharashtra", "Delhi", "Punjab", "Gujarat"]
    test_date = "2024-06-15"
    
    print("==================================================")
    print(" Standalone Energy Prediction Demonstration")
    print("==================================================")
    for state in test_states:
        res = predictor.predict(state, test_date)
        print(f"-> State: {res['state']:<15} | Date: {res['date']} ({res['day_name']})")
        print(f"   Predicted Consumption: {res['predicted_usage']} {res['unit']}")
        print(f"   Historical Median:     {res['state_median_historical']} {res['unit']}\n")
