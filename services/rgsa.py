# File: services/rgsa_processor.py
# Description: Self-contained and flexible script for running RGSA calculations.

from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def process_rgsa_for_well(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Processes RGSA for a single well DataFrame using dynamic parameters.
    Returns the processed DataFrame or the original if the process fails.
    """
    # --- STEP 1: VALIDATE AND PREPARE DATA ---
    required_cols = ['DEPTH', params.get('GR', 'GR'), params.get('RES', 'RT')]
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Warning: Skipping well because required columns are missing: {required_cols}")
        return df_well  # Return original df if columns are missing

    # Use dynamic column names from params
    gr_col = params.get('GR', 'GR')
    rt_col = params.get('RES', 'RT')

    df_rgsa = df_well[['DEPTH', gr_col, rt_col]].dropna().copy()

    if len(df_rgsa) < 100:
        print(
            f"Warning: Not enough valid data for RGSA regression (only {len(df_rgsa)} rows).")
        return df_well

    # --- STEP 2: SLIDING WINDOW REGRESSION ---
    # Corrected default parameters
    window_size = int(params.get('SLIDING_WINDOW', 100))
    step = 20
    min_points_in_window = 30

    # Define filters (can be customized via params later if needed)
    gr_filter = (5, 180)
    rt_filter = (0.1, 1000)

    coeffs = []
    for start in range(0, len(df_rgsa) - window_size, step):
        window = df_rgsa.iloc[start:start+window_size]
        gr = window[gr_col].values
        rt = window[rt_col].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            rt > rt_filter[0]) & (rt < rt_filter[1])
        gr_filtered = gr[mask]
        rt_filtered = rt[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered
        log_rt = np.log10(rt_filtered)

        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = log_rt

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

    # --- STEP 3: INTERPOLATE & CALCULATE RGSA ---
    def interpolate_coeffs(depth):
        if depth <= coeff_df['DEPTH'].min():
            return coeff_df.iloc[0]
        if depth >= coeff_df['DEPTH'].max():
            return coeff_df.iloc[-1]
        lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
        upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]
        if upper['DEPTH'] == lower['DEPTH']:
            return lower
        weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
        return lower + weight * (upper - lower)

    rgsa_list = []
    for _, row in df_rgsa.iterrows():
        depth, gr = row['DEPTH'], row[gr_col]
        if not (gr_filter[0] < gr < gr_filter[1]):
            rgsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = interpolate_coeffs(
            depth)[['b0', 'b1', 'b2', 'b3']].values
        grfix = 0.01 * gr
        log_rgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        rgsa_list.append(10**log_rgsa)

    df_rgsa['RGSA'] = rgsa_list

    # --- STEP 4: MERGE RESULTS ---
    # Drop old RGSA column to ensure the new one is used
    if 'RGSA' in df_well.columns:
        df_well = df_well.drop(columns=['RGSA'])

    df_merged = pd.merge(
        df_well, df_rgsa[['DEPTH', 'RGSA']], on='DEPTH', how='left')

    # Calculate gas effect columns
    if rt_col in df_merged and 'RGSA' in df_merged:
        df_merged['GAS_EFFECT_RT'] = (df_merged[rt_col] > df_merged['RGSA'])
        df_merged['RT_RATIO'] = df_merged[rt_col] / df_merged['RGSA']
        df_merged['RT_DIFF'] = df_merged[rt_col] - df_merged['RGSA']

    return df_merged


def process_all_wells_rgsa(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Orchestrator function: processes RGSA for a well, handling interval filtering.
    """
    # Make a copy to avoid modifying the original DataFrame passed to the function
    df_processed = df_well.copy()

    # Call the core processing function
    result_df = process_rgsa_for_well(df_processed, params)

    # Robust Return: If processing fails, return the original (unprocessed) DataFrame
    if result_df is None:
        print("❌ RGSA calculation failed. Returning original DataFrame.")
        return df_well

    print("✅ RGSA process completed.")
    return result_df
