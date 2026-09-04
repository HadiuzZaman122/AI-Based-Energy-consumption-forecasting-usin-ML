"""
data_preprocessing.py
---------------------
This module handles loading, inspecting, and cleaning the energy consumption dataset.

Key Tasks:
1. Load dataset from CSV file.
2. Automatically identify the Date and Energy Consumption columns.
3. Parse dates into standard pandas datetime format.
4. Sort records chronologically (essential for time-series forecasting).
5. Detect and handle missing values, duplicates, and invalid records.
6. Provide clear summary statistics.

Author: AI Project Team
"""

import os
import pandas as pd
import numpy as np


def find_dataset(file_path=None):
    """
    Finds the dataset file from standard paths if not explicitly provided.
    """
    if file_path and os.path.exists(file_path):
        return file_path
    
    # Check default project locations
    candidates = [
        os.path.join("data", "energy_consumption.csv"),
        "long_data_.csv",
        "dataset_tk.csv",
        os.path.join("data", "long_data_.csv"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(
        "Could not find any energy dataset! Please place your CSV file in the 'data/' folder."
    )


def identify_columns(df):
    """
    Automatically detects the Date and Target (Energy Usage) columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset.
        
    Returns:
    --------
    date_col : str
        Name of the date column.
    target_col : str
        Name of the energy consumption column.
    """
    # Potential column names for date
    date_candidates = ['Dates', 'Date', 'date', 'dates', 'Timestamp', 'timestamp', 'datetime', 'DateTime', 'Unnamed: 0']
    # Potential column names for target consumption
    target_candidates = ['Usage', 'usage', 'consumption', 'Consumption', 'Energy_Consumption', 'Power_Consumption', 'energy', 'power', 'load', 'Load']
    
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

    # Fallbacks
    if not date_col:
        # Check if first column looks like dates
        try:
            pd.to_datetime(df.iloc[:, 0].dropna().head(10), dayfirst=True)
            date_col = df.columns[0]
        except Exception:
            raise ValueError("Unable to automatically detect the Date column. Please check your dataset.")

    if not target_col:
        # Check if wide format (e.g. multiple state columns with numbers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1] # Pick the numeric column
        else:
            raise ValueError("Unable to automatically detect the Energy Consumption target column.")

    return date_col, target_col


def load_and_preprocess_data(file_path=None, verbose=True):
    """
    Loads raw energy consumption data, performs cleaning, and returns a clean DataFrame.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the CSV dataset. If None, searches standard locations.
    verbose : bool, default True
        Whether to print preprocessing summary information.
        
    Returns:
    --------
    df : pd.DataFrame
        Preprocessed and chronologically sorted DataFrame.
    metadata : dict
        Dictionary containing metadata (date_col, target_col, units, row counts, etc.)
    """
    # Step 1: Find and load the file
    actual_path = find_dataset(file_path)
    if verbose:
        print(f"==================================================")
        print(f" Loading dataset from: {actual_path}")
        print(f"==================================================")
        
    df = pd.read_csv(actual_path)
    initial_rows = len(df)
    
    if verbose:
        print(f"-> Initial Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"-> Columns Found: {list(df.columns)}")

    # Step 2: Identify Date and Target columns
    date_col, target_col = identify_columns(df)
    if verbose:
        print(f"-> Identified Date Column: '{date_col}'")
        print(f"-> Identified Target Column (Energy Consumption): '{target_col}'")

    # Step 3: Handle Wide Format (if user uploaded wide state-by-state data like dataset_tk.csv)
    # If the dataset has many state columns and no 'States' column, reshape into long format
    if 'States' not in df.columns and len(df.columns) > 10 and target_col != 'Usage':
        if verbose:
            print("-> Detected wide format dataset. Reshaping into standardized long format...")
        df = df.melt(id_vars=[date_col], var_name='States', value_name='Usage')
        target_col = 'Usage'

    # Step 4: Clean and Parse Datetime
    # We parse the date column using dayfirst=True since Indian datasets use DD/MM/YYYY
    df['ParsedDate'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    
    # Check for unparseable dates
    invalid_dates = df['ParsedDate'].isnull().sum()
    if invalid_dates > 0:
        if verbose:
            print(f"-> Warning: Found {invalid_dates} rows with invalid dates. Removing them...")
        df = df.dropna(subset=['ParsedDate'])
    
    # Step 5: Check and Clean Target Variable (Usage)
    # Ensure usage is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Check for missing target values
    missing_targets = df[target_col].isnull().sum()
    if missing_targets > 0:
        if verbose:
            print(f"-> Warning: Found {missing_targets} missing target values. Dropping missing targets...")
        df = df.dropna(subset=[target_col])
        
    # Check for negative or zero consumption values (physically invalid for regional grids)
    invalid_usage = (df[target_col] <= 0).sum()
    if invalid_usage > 0:
        if verbose:
            print(f"-> Notice: Found {invalid_usage} non-positive values. Imputing with state median...")
        if 'States' in df.columns:
            df[target_col] = df.groupby('States')[target_col].transform(lambda x: x.replace(0, np.nan).fillna(x.median()))
        else:
            df[target_col] = df[target_col].replace(0, np.nan).fillna(df[target_col].median())

    # Step 6: Remove Duplicate Records
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        if verbose:
            print(f"-> Found {duplicates_count} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates()

    # Step 7: Sort Chronologically
    # In time-series forecasting, chronological order is strictly mandatory
    if 'States' in df.columns:
        df = df.sort_values(by=['States', 'ParsedDate']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['ParsedDate']).reset_index(drop=True)

    # Standardize column naming
    clean_df = df.copy()
    
    metadata = {
        'initial_rows': initial_rows,
        'final_rows': len(clean_df),
        'date_col': date_col,
        'target_col': target_col,
        'unit': 'Mega Units (MU)',
        'start_date': str(clean_df['ParsedDate'].min().date()),
        'end_date': str(clean_df['ParsedDate'].max().date()),
        'states_count': clean_df['States'].nunique() if 'States' in clean_df.columns else 1,
        'states': sorted(clean_df['States'].unique().tolist()) if 'States' in clean_df.columns else ['All'],
        'regions': sorted(clean_df['Regions'].unique().tolist()) if 'Regions' in clean_df.columns else ['Default']
    }

    if verbose:
        print(f"-> Clean Dataset Shape: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
        print(f"-> Date Range: {metadata['start_date']} to {metadata['end_date']}")
        print(f"-> Target Variable '{target_col}' Summary:")
        print(f"   * Mean: {clean_df[target_col].mean():.2f} {metadata['unit']}")
        print(f"   * Min:  {clean_df[target_col].min():.2f} {metadata['unit']}")
        print(f"   * Max:  {clean_df[target_col].max():.2f} {metadata['unit']}")
        print(" Preprocessing successfully completed!\n")
        
    return clean_df, metadata


if __name__ == "__main__":
    df, meta = load_and_preprocess_data()
    print("Sample Preprocessed Records:")
    print(df.head())
