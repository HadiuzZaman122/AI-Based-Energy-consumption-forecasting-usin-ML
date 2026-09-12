"""
data_preprocessing.py
---------------------
This module handles loading, inspecting, validating, and cleaning the energy
consumption dataset for time-series machine learning forecasting.

Key Operations:
1. Locate and load the raw energy dataset from CSV.
2. Automatically identify the Date, Target (Usage), and auxiliary columns.
3. Parse dates into standard pandas datetime format (DD/MM/YYYY support).
4. Sort records strictly chronologically to preserve temporal ordering.
5. Clean missing values, duplicates, and non-positive consumption records.
6. Provide comprehensive summary statistics and dataset metadata.

Author: AI Project Team
"""

import os
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd


def find_dataset(file_path: Optional[str] = None) -> str:
    """
    Locates the dataset file from standard paths if not explicitly provided.

    Parameters:
    -----------
    file_path : str, optional
        Custom path to the CSV dataset.

    Returns:
    --------
    str
        Verified absolute or relative path to the dataset CSV.

    Raises:
    -------
    FileNotFoundError
        If no valid dataset CSV is found in standard locations.
    """
    if file_path and os.path.exists(file_path):
        return file_path

    # Check standard project locations in order of priority
    candidates = [
        os.path.join("data", "energy_consumption.csv"),
        "long_data_.csv",
        "dataset_tk.csv",
        os.path.join("data", "long_data_.csv"),
        os.path.join("..", "data", "energy_consumption.csv"),
        os.path.join("..", "long_data_.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find any energy dataset! Please place 'energy_consumption.csv' in the 'data/' folder."
    )


def identify_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Automatically detects the Date and Target (Energy Usage) columns in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset.

    Returns:
    --------
    date_col : str
        Name of the detected date/timestamp column.
    target_col : str
        Name of the detected energy consumption target column.
    """
    date_candidates = [
        'Dates', 'Date', 'date', 'dates', 'Timestamp', 'timestamp',
        'datetime', 'DateTime', 'Unnamed: 0'
    ]
    target_candidates = [
        'Usage', 'usage', 'consumption', 'Consumption',
        'Energy_Consumption', 'Power_Consumption', 'energy', 'power',
        'load', 'Load', 'Value', 'value'
    ]

    date_col = None
    target_col = None

    # 1. Detect Date column
    for col in df.columns:
        if col in date_candidates or 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break

    # 2. Detect Target column
    for col in df.columns:
        if col in target_candidates or 'usage' in col.lower() or 'consum' in col.lower() or 'power' in col.lower():
            target_col = col
            break

    # Fallback checks
    if not date_col:
        try:
            pd.to_datetime(df.iloc[:, 0].dropna().head(10), dayfirst=True)
            date_col = df.columns[0]
        except Exception:
            raise ValueError("Unable to automatically detect the Date column. Please verify your dataset schema.")

    if not target_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
        else:
            raise ValueError("Unable to automatically detect the Energy Consumption target column.")

    return date_col, target_col


def load_and_preprocess_data(file_path: Optional[str] = None, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw energy consumption data, performs cleaning, and returns a clean, chronologically sorted DataFrame.

    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV dataset. If None, searches standard locations.
    verbose : bool, default True
        Whether to print preprocessing summary details.

    Returns:
    --------
    clean_df : pd.DataFrame
        Preprocessed and chronologically sorted DataFrame.
    metadata : dict
        Dictionary containing metadata (row counts, date range, columns, units, stats).
    """
    actual_path = find_dataset(file_path)
    if verbose:
        print("=" * 60)
        print(f" Loading dataset from: {actual_path}")
        print("=" * 60)

    df = pd.read_csv(actual_path)
    initial_rows = len(df)

    if verbose:
        print(f"-> Initial Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"-> Columns Found: {list(df.columns)}")

    # Step 1: Detect Date and Target columns
    date_col, target_col = identify_columns(df)
    if verbose:
        print(f"-> Identified Date Column: '{date_col}'")
        print(f"-> Identified Target Column: '{target_col}'")

    # Step 2: Handle Wide Format if dataset is pivoted state-by-state (e.g. dataset_tk.csv)
    if 'States' not in df.columns and len(df.columns) > 10 and target_col != 'Usage':
        if verbose:
            print("-> Detected wide-format dataset. Reshaping into standardized long format...")
        df = df.melt(id_vars=[date_col], var_name='States', value_name='Usage')
        target_col = 'Usage'

    # Step 3: Parse Datetime
    # Using dayfirst=True to handle DD/MM/YYYY formatting correctly
    df['ParsedDate'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')

    invalid_dates = int(df['ParsedDate'].isnull().sum())
    if invalid_dates > 0:
        if verbose:
            print(f"-> Warning: Found {invalid_dates} rows with invalid dates. Removing them...")
        df = df.dropna(subset=['ParsedDate'])

    # Step 4: Validate and Clean Target Variable (Usage)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    missing_targets = int(df[target_col].isnull().sum())
    if missing_targets > 0:
        if verbose:
            print(f"-> Warning: Found {missing_targets} missing target values. Dropping missing targets...")
        df = df.dropna(subset=[target_col])

    # Handle non-positive consumption values (physically impossible for power grids)
    invalid_usage = int((df[target_col] <= 0).sum())
    if invalid_usage > 0:
        if verbose:
            print(f"-> Notice: Found {invalid_usage} non-positive values. Imputing with state median...")
        if 'States' in df.columns:
            df[target_col] = df.groupby('States')[target_col].transform(
                lambda x: x.replace(0, np.nan).fillna(x.median())
            )
        else:
            df[target_col] = df[target_col].replace(0, np.nan).fillna(df[target_col].median())

    # Step 5: Remove Duplicates
    duplicates_count = int(df.duplicated().sum())
    if duplicates_count > 0:
        if verbose:
            print(f"-> Found {duplicates_count} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates()

    # Step 6: Strict Chronological Sorting
    # Crucial for time-series forecasting to avoid future-data leakage
    if 'States' in df.columns:
        df = df.sort_values(by=['States', 'ParsedDate']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['ParsedDate']).reset_index(drop=True)

    clean_df = df.copy()

    # Compile dataset metadata
    metadata = {
        'initial_rows': initial_rows,
        'final_rows': len(clean_df),
        'date_col': date_col,
        'target_col': target_col,
        'unit': 'Mega Units (MU)',
        'start_date': str(clean_df['ParsedDate'].min().strftime('%Y-%m-%d')),
        'end_date': str(clean_df['ParsedDate'].max().strftime('%Y-%m-%d')),
        'total_days': int((clean_df['ParsedDate'].max() - clean_df['ParsedDate'].min()).days) + 1,
        'states_count': int(clean_df['States'].nunique()) if 'States' in clean_df.columns else 1,
        'states': sorted(clean_df['States'].unique().tolist()) if 'States' in clean_df.columns else ['All'],
        'regions_count': int(clean_df['Regions'].nunique()) if 'Regions' in clean_df.columns else 1,
        'regions': sorted(clean_df['Regions'].unique().tolist()) if 'Regions' in clean_df.columns else ['Default'],
        'mean_usage': float(clean_df[target_col].mean()),
        'median_usage': float(clean_df[target_col].median()),
        'min_usage': float(clean_df[target_col].min()),
        'max_usage': float(clean_df[target_col].max()),
        'std_usage': float(clean_df[target_col].std()),
    }

    if verbose:
        print(f"-> Clean Dataset Shape: {clean_df.shape[0]:,} rows, {clean_df.shape[1]} columns")
        print(f"-> Date Range: {metadata['start_date']} to {metadata['end_date']} ({metadata['total_days']} days)")
        print(f"-> Target Variable '{target_col}' Summary:")
        print(f"   * Mean:   {metadata['mean_usage']:.2f} {metadata['unit']}")
        print(f"   * Median: {metadata['median_usage']:.2f} {metadata['unit']}")
        print(f"   * Min:    {metadata['min_usage']:.2f} {metadata['unit']}")
        print(f"   * Max:    {metadata['max_usage']:.2f} {metadata['unit']}")
        print(f"   * Std:    {metadata['std_usage']:.2f} {metadata['unit']}")
        print(" Preprocessing successfully completed!\n")

    return clean_df, metadata


if __name__ == "__main__":
    clean_data, meta_info = load_and_preprocess_data()
    print("Sample Clean Records:")
    print(clean_data.head())
