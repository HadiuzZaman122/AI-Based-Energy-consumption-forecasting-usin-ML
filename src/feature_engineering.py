"""
feature_engineering.py
----------------------
This module transforms raw cleaned datetime and categorical columns into rich,
predictive numerical features for Machine Learning models.

Key Features Created:
1. Time-based features: Year, Month, Day, DayOfWeek, IsWeekend, Quarter, DayOfYear, WeekOfYear.
2. Categorical Encodings: State and Region encodings.
3. Lag & Rolling Window features:
   - Usage_Lag_1 (Yesterday's consumption for the state)
   - Usage_Lag_7 (Consumption 7 days ago - weekly seasonality)
   - Usage_Rolling_Mean_7 (Average consumption of the past 7 days)

Data Leakage Prevention:
- All lag and rolling features only look backwards in time (using shift(1)).
- Label encoders are saved to disk so the exact same mapping is applied at prediction time.

Author: AI Project Team
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_time_features(df, date_col='ParsedDate'):
    """
    Extracts informative calendar and seasonal features from a datetime column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column.
    date_col : str
        Name of the datetime column.
        
    Returns:
    --------
    df_feat : pd.DataFrame
        DataFrame augmented with temporal features.
    """
    df_feat = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col]):
        df_feat[date_col] = pd.to_datetime(df_feat[date_col], dayfirst=True)
        
    # Extract Calendar Features
    df_feat['Year'] = df_feat[date_col].dt.year
    df_feat['Month'] = df_feat[date_col].dt.month
    df_feat['Day'] = df_feat[date_col].dt.day
    df_feat['DayOfWeek'] = df_feat[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df_feat['IsWeekend'] = df_feat['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df_feat['Quarter'] = df_feat[date_col].dt.quarter
    df_feat['DayOfYear'] = df_feat[date_col].dt.dayofyear
    df_feat['WeekOfYear'] = df_feat[date_col].dt.isocalendar().week.astype(int)
    
    return df_feat


def create_lag_features(df, target_col='Usage', group_col='States'):
    """
    Creates backward-looking lag and rolling statistics features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with chronological ordering.
    target_col : str
        The energy consumption column.
    group_col : str, optional
        Column to group by (e.g., 'States'). If present, lags are calculated per state.
        
    Returns:
    --------
    df_lagged : pd.DataFrame
        DataFrame with lag features, with initial NaN rows cleanly handled.
    """
    df_lagged = df.copy()
    
    if group_col and group_col in df_lagged.columns:
        # Group by state to compute state-specific past consumption
        # Lag 1: Consumption 1 day prior
        df_lagged['Usage_Lag_1'] = df_lagged.groupby(group_col)[target_col].shift(1)
        # Lag 7: Consumption 7 days prior (same weekday last week)
        df_lagged['Usage_Lag_7'] = df_lagged.groupby(group_col)[target_col].shift(7)
        # 7-day rolling average (past 7 days, excluding today)
        df_lagged['Usage_Rolling_Mean_7'] = (
            df_lagged.groupby(group_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
        )
    else:
        # Global lags if no grouping column exists
        df_lagged['Usage_Lag_1'] = df_lagged[target_col].shift(1)
        df_lagged['Usage_Lag_7'] = df_lagged[target_col].shift(7)
        df_lagged['Usage_Rolling_Mean_7'] = (
            df_lagged[target_col].shift(1).rolling(window=7, min_periods=1).mean()
        )
        
    # Forward fill / Back fill any remaining early NaNs so we retain maximum training data
    df_lagged['Usage_Lag_1'] = df_lagged['Usage_Lag_1'].bfill()
    df_lagged['Usage_Lag_7'] = df_lagged['Usage_Lag_7'].bfill()
    df_lagged['Usage_Rolling_Mean_7'] = df_lagged['Usage_Rolling_Mean_7'].bfill()
    
    return df_lagged


def encode_categorical_features(df, encoders=None, save_path="models/label_encoders.joblib"):
    """
    Encodes categorical features (like States and Regions) into integers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing categorical columns.
    encoders : dict, optional
        Pre-fitted LabelEncoder dictionary. If None, fits new encoders.
    save_path : str, optional
        Path to save fitted encoders for later prediction inference.
        
    Returns:
    --------
    df_encoded : pd.DataFrame
        DataFrame with integer encoded categorical columns.
    encoders : dict
        Fitted encoders dictionary.
    """
    df_encoded = df.copy()
    is_training = encoders is None
    if is_training:
        encoders = {}
        
    cat_cols = ['States', 'Regions']
    for col in cat_cols:
        if col in df_encoded.columns:
            if is_training:
                le = LabelEncoder()
                df_encoded[col + '_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le:
                    # Handle unseen categories gracefully
                    known_classes = set(le.classes_)
                    df_encoded[col + '_Encoded'] = df_encoded[col].astype(str).apply(
                        lambda s: le.transform([s])[0] if s in known_classes else -1
                    )
                    
    if is_training and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(encoders, save_path)
        
    return df_encoded, encoders


def prepare_features(df, target_col='Usage', is_training=True, encoders=None, verbose=True):
    """
    Full feature engineering pipeline: Time features -> Lag features -> Encodings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    target_col : str
        Target variable name.
    is_training : bool
        Whether this is during training phase or inference.
    encoders : dict, optional
        Encoders for inference.
    verbose : bool
        Print progress details.
        
    Returns:
    --------
    featured_df : pd.DataFrame
        DataFrame with all engineered features.
    feature_columns : list
        List of predictor column names (X).
    encoders : dict
        Fitted encoders.
    """
    if verbose:
        print("--- Feature Engineering Pipeline ---")
        
    # 1. Time Features
    df_feat = create_time_features(df, date_col='ParsedDate')
    
    # 2. Lag & Rolling Features
    df_feat = create_lag_features(df_feat, target_col=target_col, group_col='States' if 'States' in df.columns else None)
    
    # 3. Categorical Encodings
    df_feat, encoders = encode_categorical_features(df_feat, encoders=encoders)
    
    # 4. Define Feature Set (X variables)
    candidate_features = [
        'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear', 'WeekOfYear',
        'latitude', 'longitude', 'States_Encoded', 'Regions_Encoded',
        'Usage_Lag_1', 'Usage_Lag_7', 'Usage_Rolling_Mean_7'
    ]
    
    # Only keep features that exist in the dataframe
    feature_columns = [col for col in candidate_features if col in df_feat.columns]
    
    if verbose:
        print(f"-> Created {len(feature_columns)} predictor features (X):")
        for i, col in enumerate(feature_columns, 1):
            print(f"   {i}. {col}")
        print(f"-> Target variable (y): '{target_col}'")
        print("Feature engineering completed successfully!\n")
        
    return df_feat, feature_columns, encoders


if __name__ == "__main__":
    try:
        from src.data_preprocessing import load_and_preprocess_data
    except ImportError:
        from data_preprocessing import load_and_preprocess_data
    clean_df, meta = load_and_preprocess_data()
    feat_df, feat_cols, encs = prepare_features(clean_df, target_col=meta['target_col'])
    print("Sample Engineered Features:")
    print(feat_df[['ParsedDate', 'States', 'Usage'] + feat_cols[:5]].head())

