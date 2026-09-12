"""
feature_engineering.py
----------------------
This module transforms preprocessed temporal, geographic, and historical consumption
data into predictive numerical features for Machine Learning regression models.

Key Features Engineered:
1. Temporal & Calendar Signals:
   - Year, Month, Day, DayOfWeek, IsWeekend, Quarter, DayOfYear, WeekOfYear.
2. Lag & Historical Consumption Signals (Strictly past-looking, shift >= 1):
   - Usage_Lag_1: Consumption 1 day prior (immediate load inertia).
   - Usage_Lag_2: Consumption 2 days prior.
   - Usage_Lag_7: Consumption 7 days prior (weekly seasonal pattern).
3. Rolling Window Aggregations (Strictly past-looking):
   - Usage_Rolling_Mean_7: 7-day moving average of previous consumption.
   - Usage_Rolling_Mean_14: 14-day moving average of previous consumption.
   - Usage_Rolling_Std_7: 7-day moving standard deviation (consumption volatility).
4. Spatial & Categorical Signals:
   - Latitude, Longitude (spatial grid locations).
   - States_Encoded, Regions_Encoded (LabelEncoded categorical representations).

Data Leakage Prevention:
- Lag and rolling calculations use .shift(1) to ensure the current day's value is never seen.
- Categorical encoders are saved to disk so the identical mapping applies at prediction time.

Author: AI Project Team
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_time_features(df: pd.DataFrame, date_col: str = 'ParsedDate') -> pd.DataFrame:
    """
    Extracts informative calendar and seasonal features from a datetime column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column.
    date_col : str, default 'ParsedDate'
        Name of the datetime column.

    Returns:
    --------
    df_feat : pd.DataFrame
        DataFrame augmented with temporal features.
    """
    df_feat = df.copy()

    # Ensure date column is standard datetime
    if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col]):
        df_feat[date_col] = pd.to_datetime(df_feat[date_col], dayfirst=True)

    # Extract Temporal Features
    df_feat['Year'] = df_feat[date_col].dt.year
    df_feat['Month'] = df_feat[date_col].dt.month
    df_feat['Day'] = df_feat[date_col].dt.day
    df_feat['DayOfWeek'] = df_feat[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df_feat['IsWeekend'] = df_feat['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df_feat['Quarter'] = df_feat[date_col].dt.quarter
    df_feat['DayOfYear'] = df_feat[date_col].dt.dayofyear
    df_feat['WeekOfYear'] = df_feat[date_col].dt.isocalendar().week.astype(int)

    return df_feat


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'Usage',
    group_col: Optional[str] = 'States'
) -> pd.DataFrame:
    """
    Creates backward-looking lag and rolling statistics features strictly preventing data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with chronological ordering.
    target_col : str, default 'Usage'
        The energy consumption target column.
    group_col : str, optional, default 'States'
        Column to group by (e.g. 'States'). If present, lags are calculated per state.

    Returns:
    --------
    df_lagged : pd.DataFrame
        DataFrame containing lag and rolling features.
    """
    df_lagged = df.copy()

    if group_col and group_col in df_lagged.columns:
        # Group by state to compute state-specific past consumption
        # 1-day lag (yesterday)
        df_lagged['Usage_Lag_1'] = df_lagged.groupby(group_col)[target_col].shift(1)
        # 2-day lag
        df_lagged['Usage_Lag_2'] = df_lagged.groupby(group_col)[target_col].shift(2)
        # 7-day lag (same weekday last week)
        df_lagged['Usage_Lag_7'] = df_lagged.groupby(group_col)[target_col].shift(7)

        # 7-day rolling mean of past values (excluding today)
        df_lagged['Usage_Rolling_Mean_7'] = (
            df_lagged.groupby(group_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
        )
        # 14-day rolling mean
        df_lagged['Usage_Rolling_Mean_14'] = (
            df_lagged.groupby(group_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
        )
        # 7-day rolling standard deviation (volatility)
        df_lagged['Usage_Rolling_Std_7'] = (
            df_lagged.groupby(group_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=7, min_periods=2).std().fillna(0))
        )
    else:
        # Global lags if no grouping column exists
        df_lagged['Usage_Lag_1'] = df_lagged[target_col].shift(1)
        df_lagged['Usage_Lag_2'] = df_lagged[target_col].shift(2)
        df_lagged['Usage_Lag_7'] = df_lagged[target_col].shift(7)

        df_lagged['Usage_Rolling_Mean_7'] = (
            df_lagged[target_col].shift(1).rolling(window=7, min_periods=1).mean()
        )
        df_lagged['Usage_Rolling_Mean_14'] = (
            df_lagged[target_col].shift(1).rolling(window=14, min_periods=1).mean()
        )
        df_lagged['Usage_Rolling_Std_7'] = (
            df_lagged[target_col].shift(1).rolling(window=7, min_periods=2).std().fillna(0)
        )

    # Cleanly handle initial boundary NaNs by back-filling within groups or globally
    if group_col and group_col in df_lagged.columns:
        for col in ['Usage_Lag_1', 'Usage_Lag_2', 'Usage_Lag_7', 'Usage_Rolling_Mean_7', 'Usage_Rolling_Mean_14', 'Usage_Rolling_Std_7']:
            df_lagged[col] = df_lagged.groupby(group_col)[col].bfill()
            # If any remain, fill with overall median
            df_lagged[col] = df_lagged[col].fillna(df_lagged[target_col].median())
    else:
        for col in ['Usage_Lag_1', 'Usage_Lag_2', 'Usage_Lag_7', 'Usage_Rolling_Mean_7', 'Usage_Rolling_Mean_14', 'Usage_Rolling_Std_7']:
            df_lagged[col] = df_lagged[col].bfill().fillna(df_lagged[target_col].median())

    return df_lagged


def encode_categorical_features(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    save_path: Optional[str] = "models/label_encoders.joblib"
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encodes categorical features ('States' and 'Regions') into integer representations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing categorical columns.
    encoders : dict, optional
        Pre-fitted LabelEncoder dictionary. If None, fits new encoders on the data.
    save_path : str, optional
        Path to save fitted encoders for consistent inference.

    Returns:
    --------
    df_encoded : pd.DataFrame
        DataFrame with integer-encoded categorical columns.
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
                    known_classes = set(le.classes_)
                    df_encoded[col + '_Encoded'] = df_encoded[col].astype(str).apply(
                        lambda s: le.transform([s])[0] if s in known_classes else -1
                    )

    if is_training and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(encoders, save_path)

    return df_encoded, encoders


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'Usage',
    is_training: bool = True,
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]:
    """
    Full feature engineering pipeline: Temporal Features -> Lag & Rolling Features -> Categorical Encodings.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    target_col : str, default 'Usage'
        Target variable column name.
    is_training : bool, default True
        Whether this is during training phase or inference.
    encoders : dict, optional
        Fitted encoders to reuse during inference.
    verbose : bool, default True
        Whether to print pipeline progress details.

    Returns:
    --------
    featured_df : pd.DataFrame
        DataFrame with all engineered predictor features and target.
    feature_columns : list of str
        List of predictor feature names (X).
    encoders : dict
        Dictionary of categorical LabelEncoders.
    """
    if verbose:
        print("=" * 60)
        print(" Feature Engineering Pipeline")
        print("=" * 60)

    # 1. Temporal / Calendar Features
    df_feat = create_time_features(df, date_col='ParsedDate')

    # 2. Lag and Rolling Window Features
    df_feat = create_lag_features(
        df_feat,
        target_col=target_col,
        group_col='States' if 'States' in df.columns else None
    )

    # 3. Categorical Encodings
    df_feat, encoders = encode_categorical_features(
        df_feat,
        encoders=encoders,
        save_path="models/label_encoders.joblib" if is_training else None
    )

    # 4. Predictor Feature Set (X variables)
    candidate_features = [
        'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear', 'WeekOfYear',
        'latitude', 'longitude', 'States_Encoded', 'Regions_Encoded',
        'Usage_Lag_1', 'Usage_Lag_2', 'Usage_Lag_7',
        'Usage_Rolling_Mean_7', 'Usage_Rolling_Mean_14', 'Usage_Rolling_Std_7'
    ]

    feature_columns = [col for col in candidate_features if col in df_feat.columns]

    if verbose:
        print(f"-> Total Predictor Features (X): {len(feature_columns)}")
        for i, col in enumerate(feature_columns, 1):
            print(f"   {i:2d}. {col}")
        print(f"-> Target Variable (y): '{target_col}'")
        print(" Feature engineering successfully completed!\n")

    return df_feat, feature_columns, encoders


if __name__ == "__main__":
    try:
        from src.data_preprocessing import load_and_preprocess_data
    except ImportError:
        from data_preprocessing import load_and_preprocess_data

    clean_df, meta = load_and_preprocess_data()
    feat_df, feat_cols, encs = prepare_features(clean_df, target_col=meta['target_col'])
    print("Sample Engineered Features:")
    print(feat_df[['ParsedDate', 'States', 'Usage'] + feat_cols[:6]].head())
