# File: services/ngsa_processor.py
# Description: Self-contained and flexible script for running NGSA calculations.

from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def _interpolate_coeffs(depth, coeff_df):
    """(Internal) Performs linear interpolation on regression coefficients."""
    if coeff_df.empty:
        return np.array([np.nan] * 4)

    target_cols = ['b0', 'b1', 'b2', 'b3']

    if depth <= coeff_df['DEPTH'].min():
        return coeff_df.iloc[0][target_cols].values
    if depth >= coeff_df['DEPTH'].max():
        return coeff_df.iloc[-1][target_cols].values

    lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
    upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]

    if upper['DEPTH'] == lower['DEPTH']:
        return lower[target_cols].values

    weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
    interpolated = lower[target_cols] + weight * \
        (upper[target_cols] - lower[target_cols])

    return interpolated.values


def process_ngsa_for_well(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Processes NGSA for a single well DataFrame using dynamic parameters.
    Returns the processed DataFrame or the original if the process fails.
    """
    # --- STEP 1: VALIDATE AND PREPARE DATA ---
    # Use dynamic column names from params, with fallback to defaults
    gr_col = params.get('GR', 'GR')
    nphi_col = params.get('NEUT', 'NPHI')
    required_cols = ['DEPTH', gr_col, nphi_col]

    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Warning: Skipping well because required columns are missing: {required_cols}")
        return df_well  # Return original df to prevent crashes

    df_ngsa = df_well[required_cols].dropna().copy()

    if len(df_ngsa) < 100:
        print(
            f"Warning: Not enough valid data for NGSA regression (only {len(df_ngsa)} rows).")
        return df_well

    # --- STEP 2: SLIDING WINDOW REGRESSION ---
    # Corrected default parameters for a more effective sliding window
    window_size = int(params.get('SLIDING_WINDOW', 100))
    step = 20
    min_points_in_window = 30

    # Define filters (can be customized via params later if needed)
    gr_filter = (5, 180)
    nphi_filter = (0.05, 0.6)

    coeffs = []
    for start in range(0, len(df_ngsa) - window_size, step):
        window = df_ngsa.iloc[start:start+window_size]
        gr = window[gr_col].values
        nphi = window[nphi_col].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            nphi > nphi_filter[0]) & (nphi < nphi_filter[1])
        gr_filtered = gr[mask]
        nphi_filtered = nphi[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered
        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = nphi_filtered  # For NGSA, y is the direct NPHI value

        try:
            model = LinearRegression().fit(X, y)
            if hasattr(model, 'coef_') and len(model.coef_) == 3:
                coeffs.append({
                    'DEPTH': window['DEPTH'].mean(),
                    'b0': model.intercept_, 'b1': model.coef_[0],
                    'b2': model.coef_[1], 'b3': model.coef_[2]
                })
        except Exception as e:
            print(
                f"Warning: Regression failed at depth ~{window['DEPTH'].mean()}: {e}")
            continue

    if not coeffs:
        print("Warning: No regression coefficients were successfully calculated. Returning original data.")
        return df_well

    coeff_df = pd.DataFrame(coeffs)

    # --- STEP 3: INTERPOLATE & CALCULATE NGSA ---
    ngsa_list = []
    for _, row in df_ngsa.iterrows():
        depth, gr = row['DEPTH'], row[gr_col]
        if not (gr_filter[0] < gr < gr_filter[1]):
            ngsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        ngsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        ngsa_list.append(ngsa)

    df_ngsa['NGSA'] = ngsa_list

    # --- STEP 4: MERGE RESULTS ---
    # Drop old NGSA column to ensure the new one is used
    if 'NGSA' in df_well.columns:
        df_well = df_well.drop(columns=['NGSA'])

    df_merged = pd.merge(
        df_well, df_ngsa[['DEPTH', 'NGSA']], on='DEPTH', how='left')

    # Calculate gas effect columns
    if nphi_col in df_merged and 'NGSA' in df_merged:
        df_merged['GAS_EFFECT_NPHI'] = (
            df_merged[nphi_col] < df_merged['NGSA'])
        df_merged['NPHI_DIFF'] = df_merged['NGSA'] - df_merged[nphi_col]

    return df_merged


def process_all_wells_ngsa(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Orchestrator function: processes NGSA for a well, handling interval filtering.
    """
    # Make a copy to avoid modifying the original DataFrame
    df_processed = df_well.copy()

    # Call the core processing function
    result_df = process_ngsa_for_well(df_processed, params)

    # Robust Return: If processing fails, return the original DataFrame
    if result_df is None:
        print("❌ NGSA calculation failed. Returning original DataFrame.")
        return df_well

    print("✅ NGSA process completed.")
    return result_df
