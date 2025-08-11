# File: services/dgsa_processor.py
# Description: Self-contained and flexible script for running DGSA calculations.

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


def process_dgsa_for_well(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Processes DGSA for a single well DataFrame using dynamic parameters.
    Returns the processed DataFrame or the original if the process fails.
    """
    # --- STEP 1: VALIDATE AND PREPARE DATA ---
    # Use dynamic column names from params, with fallback to defaults
    gr_col = params.get('GR', 'GR')
    rhob_col = params.get('DENS', 'RHOB')
    required_cols = ['DEPTH', gr_col, rhob_col]

    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Warning: Skipping well because required columns are missing: {required_cols}")
        return df_well  # Return original df to prevent crashes

    df_dgsa = df_well[required_cols].dropna().copy()

    if len(df_dgsa) < 100:
        print(
            f"Warning: Not enough valid data for DGSA regression (only {len(df_dgsa)} rows).")
        return df_well

    # --- STEP 2: SLIDING WINDOW REGRESSION ---
    # Corrected default parameters for a more effective sliding window
    window_size = int(params.get('SLIDING_WINDOW', 100))
    step = 20
    min_points_in_window = 30

    # Define filters (can be customized via params later if needed)
    gr_filter = (5, 180)
    rhob_filter = (1.5, 3.0)

    coeffs = []
    for start in range(0, len(df_dgsa) - window_size, step):
        window = df_dgsa.iloc[start:start+window_size]
        gr = window[gr_col].values
        dens = window[rhob_col].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            dens > rhob_filter[0]) & (dens < rhob_filter[1])
        gr_filtered = gr[mask]
        dens_filtered = dens[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered
        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = dens_filtered  # For DGSA, y is the direct RHOB value

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

    # --- STEP 3: INTERPOLATE & CALCULATE DGSA ---
    dgsa_list = []
    for _, row in df_dgsa.iterrows():
        depth, gr = row['DEPTH'], row[gr_col]
        if not (gr_filter[0] < gr < gr_filter[1]):
            dgsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        dgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        dgsa_list.append(dgsa)

    df_dgsa['DGSA'] = dgsa_list

    # --- STEP 4: MERGE RESULTS ---
    # Drop old DGSA column to ensure the new one is used
    if 'DGSA' in df_well.columns:
        df_well = df_well.drop(columns=['DGSA'])

    df_merged = pd.merge(
        df_well, df_dgsa[['DEPTH', 'DGSA']], on='DEPTH', how='left')

    # Calculate gas effect columns
    if rhob_col in df_merged and 'DGSA' in df_merged:
        df_merged['GAS_EFFECT_RHOB'] = (
            df_merged[rhob_col] < df_merged['DGSA'])
        df_merged['DENS_DIFF'] = df_merged['DGSA'] - df_merged[rhob_col]

    return df_merged


def process_all_wells_dgsa(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Orchestrator function: processes DGSA for a well, handling interval filtering.
    """
    # Make a copy to avoid modifying the original DataFrame
    df_processed = df_well.copy()

    # Call the core processing function
    result_df = process_dgsa_for_well(df_processed, params)

    # Robust Return: If processing fails, return the original DataFrame
    if result_df is None:
        print("❌ DGSA calculation failed. Returning original DataFrame.")
        return df_well

    print("✅ DGSA process completed.")
    return result_df
