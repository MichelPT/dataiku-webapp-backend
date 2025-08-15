# # File: services/rgsa_processor.py
# # Description: Self-contained and flexible script for running RGSA calculations.

# from typing import Optional, List
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression


# def process_rgsa_for_well(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
#     """
#     Processes RGSA for a single well DataFrame using dynamic parameters.
#     Returns the processed DataFrame or the original if the process fails.
#     """
#     # --- STEP 1: VALIDATE AND PREPARE DATA ---
#     required_cols = ['DEPTH', params.get('GR', 'GR'), params.get('RES', 'RT')]
#     if not all(col in df_well.columns for col in required_cols):
#         print(
#             f"Warning: Skipping well because required columns are missing: {required_cols}")
#         return df_well  # Return original df if columns are missing

#     # Use dynamic column names from params
#     gr_col = params.get('GR', 'GR')
#     rt_col = params.get('RES', 'RT')

#     df_rgsa = df_well[['DEPTH', gr_col, rt_col]].dropna().copy()

#     if len(df_rgsa) < 100:
#         print(
#             f"Warning: Not enough valid data for RGSA regression (only {len(df_rgsa)} rows).")
#         return df_well

#     # --- STEP 2: SLIDING WINDOW REGRESSION ---
#     # Corrected default parameters
#     window_size = int(params.get('SLIDING_WINDOW', 100))
#     step = 20
#     min_points_in_window = 30

#     # Define filters (can be customized via params later if needed)
#     gr_filter = (5, 180)
#     rt_filter = (0.1, 1000)

#     coeffs = []
#     for start in range(0, len(df_rgsa) - window_size, step):
#         window = df_rgsa.iloc[start:start+window_size]
#         gr = window[gr_col].values
#         rt = window[rt_col].values

#         mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
#             rt > rt_filter[0]) & (rt < rt_filter[1])
#         gr_filtered = gr[mask]
#         rt_filtered = rt[mask]

#         if len(gr_filtered) < min_points_in_window:
#             continue

#         gr_scaled = 0.01 * gr_filtered
#         log_rt = np.log10(rt_filtered)

#         X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
#         y = log_rt

#         try:
#             model = LinearRegression().fit(X, y)
#             if hasattr(model, 'coef_') and len(model.coef_) == 3:
#                 coeffs.append({
#                     'DEPTH': window['DEPTH'].mean(),
#                     'b0': model.intercept_, 'b1': model.coef_[0],
#                     'b2': model.coef_[1], 'b3': model.coef_[2]
#                 })
#         except Exception as e:
#             print(
#                 f"Warning: Regression failed at depth ~{window['DEPTH'].mean()}: {e}")
#             continue

#     if not coeffs:
#         print("Warning: No regression coefficients were successfully calculated. Returning original data.")
#         return df_well

#     coeff_df = pd.DataFrame(coeffs)

#     # --- STEP 3: INTERPOLATE & CALCULATE RGSA ---
#     def interpolate_coeffs(depth):
#         if depth <= coeff_df['DEPTH'].min():
#             return coeff_df.iloc[0]
#         if depth >= coeff_df['DEPTH'].max():
#             return coeff_df.iloc[-1]
#         lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
#         upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]
#         if upper['DEPTH'] == lower['DEPTH']:
#             return lower
#         weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
#         return lower + weight * (upper - lower)

#     rgsa_list = []
#     for _, row in df_rgsa.iterrows():
#         depth, gr = row['DEPTH'], row[gr_col]
#         if not (gr_filter[0] < gr < gr_filter[1]):
#             rgsa_list.append(np.nan)
#             continue

#         b0, b1, b2, b3 = interpolate_coeffs(
#             depth)[['b0', 'b1', 'b2', 'b3']].values
#         grfix = 0.01 * gr
#         log_rgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
#         rgsa_list.append(10**log_rgsa)

#     df_rgsa['RGSA'] = rgsa_list

#     # --- STEP 4: MERGE RESULTS ---
#     # Drop old RGSA column to ensure the new one is used
#     if 'RGSA' in df_well.columns:
#         df_well = df_well.drop(columns=['RGSA'])

#     df_merged = pd.merge(
#         df_well, df_rgsa[['DEPTH', 'RGSA']], on='DEPTH', how='left')

#     # Calculate gas effect columns
#     if rt_col in df_merged and 'RGSA' in df_merged:
#         df_merged['GAS_EFFECT_RT'] = (df_merged[rt_col] > df_merged['RGSA'])
#         df_merged['RT_RATIO'] = df_merged[rt_col] / df_merged['RGSA']
#         df_merged['RT_DIFF'] = df_merged[rt_col] - df_merged['RGSA']

#     return df_merged


# def process_all_wells_rgsa(df_well: pd.DataFrame, params: dict) -> pd.DataFrame:
#     """
#     Orchestrator function: processes RGSA for a well, handling interval filtering.
#     """
#     # Make a copy to avoid modifying the original DataFrame passed to the function
#     df_processed = df_well.copy()

#     # Call the core processing function
#     result_df = process_rgsa_for_well(df_processed, params)

#     # Robust Return: If processing fails, return the original (unprocessed) DataFrame
#     if result_df is None:
#         print("❌ RGSA calculation failed. Returning original DataFrame.")
#         return df_well

#     print("✅ RGSA process completed.")
#     return result_df

# File: services/rgsa_processor_legacy_replica.py
# Description: Self-contained script for running RGSA calculations,
# faithfully replicating the multi-pass logic of the legacy .lls script.

from typing import List, Dict
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- PASS 1: DYNAMIC GR_MAX CALCULATION ---


def calculate_dynamic_gr_cap(df_well: pd.DataFrame, params: Dict) -> pd.DataFrame:
    # ... (kode Anda untuk fungsi ini tidak perlu diubah)
    print("INFO: Starting Pass 1 - Calculating dynamic GR cap...")
    gr_col = params.get('GR', 'GR')
    required_cols = ['DEPTH', gr_col]
    if not all(col in df_well.columns for col in required_cols):
        return None
    df_filtered = df_well[required_cols].dropna().copy()
    df_filtered = df_filtered[(df_filtered[gr_col] > 5)
                              & (df_filtered[gr_col] < 180)]
    npoints = int(params.get('SLIDING_WINDOW', 100)) * 2
    gr_caps = []
    for i in range(0, len(df_filtered), npoints):
        chunk = df_filtered.iloc[i:i + npoints]
        if len(chunk) < npoints / 2:
            continue
        gr_cap_value = chunk[gr_col].quantile(0.98)
        median_depth = chunk['DEPTH'].median()
        gr_caps.append({'DEPTH': median_depth, 'GR_MAX': gr_cap_value})
    if not gr_caps:
        return None
    print("INFO: Pass 1 Complete.")
    return pd.DataFrame(gr_caps)


def calculate_regression_coefficients(df_well: pd.DataFrame, df_gr_cap: pd.DataFrame, params: Dict) -> pd.DataFrame:
    # ... (kode Anda untuk fungsi ini tidak perlu diubah)
    print("\nINFO: Starting Pass 2 - Calculating regression coefficients...")
    gr_col, rt_col, lith_col = params.get('GR', 'GR'), params.get(
        'RES', 'RT'), params.get('LITH', 'LITHOLOGY')
    required_cols = ['DEPTH', gr_col, rt_col]
    if not all(col in df_well.columns for col in required_cols):
        return None
    df_well_sorted = df_well.sort_values('DEPTH').copy()
    df_well_sorted['GR_MAX'] = np.interp(
        df_well_sorted['DEPTH'], df_gr_cap['DEPTH'], df_gr_cap['GR_MAX'])
    excluded_lithologies = ['COAL', 'CA', 'CAOO', 'ORG']
    res_min, res_max = params.get(
        'RESWAT_MIN', 0.1), params.get('RESWAT_MAX', 1000)
    mask = ((df_well_sorted[gr_col] < df_well_sorted['GR_MAX']) & (df_well_sorted[rt_col] > res_min) & (
        df_well_sorted[rt_col] < res_max) & (df_well_sorted[gr_col].notna()) & (df_well_sorted[rt_col].notna()))
    if lith_col in df_well_sorted.columns:
        mask &= ~df_well_sorted[lith_col].str.upper().isin(
            excluded_lithologies)
    df_reg_data = df_well_sorted[mask].copy()
    npoints, min_points_in_window = int(
        params.get('SLIDING_WINDOW', 100)) * 2, 30
    coeffs = []
    for i in range(0, len(df_reg_data), npoints):
        window = df_reg_data.iloc[i:i + npoints]
        if len(window) < min_points_in_window:
            continue
        gr_scaled = 0.01 * window[gr_col]
        log_rt = np.log10(window[rt_col])
        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = log_rt
        try:
            model = LinearRegression().fit(X, y)
            coeffs.append({'DEPTH': window['DEPTH'].iloc[0], 'b0': model.intercept_, 'b1': model.coef_[
                          0], 'b2': model.coef_[1], 'b3': model.coef_[2], 'CORR_COEF': model.score(X, y), 'N_POINTS': len(window)})
        except Exception as e:
            print(
                f"Warning: Regression failed at depth ~{window['DEPTH'].mean()}: {e}")
    if not coeffs:
        return None
    print("INFO: Pass 2 Complete.")
    return pd.DataFrame(coeffs)


def calculate_and_merge_rgsa(df_well: pd.DataFrame, df_coeffs: pd.DataFrame, params: Dict) -> pd.DataFrame:
    # ... (kode Anda untuk fungsi ini tidak perlu diubah)
    print("\nINFO: Starting Pass 3 - Computing final RGSA log...")
    gr_col, rt_col = params.get('GR', 'GR'), params.get('RES', 'RT')
    df_merged = df_well.copy()
    if 'RGSA' in df_merged.columns:
        df_merged = df_merged.drop(columns=['RGSA'])
    interp_depths = df_coeffs['DEPTH']
    b0 = np.interp(df_merged['DEPTH'], interp_depths, df_coeffs['b0'])
    b1 = np.interp(df_merged['DEPTH'], interp_depths, df_coeffs['b1'])
    b2 = np.interp(df_merged['DEPTH'], interp_depths, df_coeffs['b2'])
    b3 = np.interp(df_merged['DEPTH'], interp_depths, df_coeffs['b3'])
    grfix = 0.01 * df_merged[gr_col]
    log_rgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
    df_merged['RGSA'] = 10**log_rgsa
    df_merged.loc[df_merged[gr_col].isna(), 'RGSA'] = np.nan
    if rt_col in df_merged and 'RGSA' in df_merged:
        df_merged['GAS_EFFECT_RT'] = (df_merged[rt_col] > df_merged['RGSA'])
        df_merged['RT_RATIO'] = df_merged[rt_col] / df_merged['RGSA']
        df_merged['RT_DIFF'] = df_merged[rt_col] - df_merged['RGSA']
    print("INFO: Pass 3 Complete.")
    return df_merged

# --- FUNGSI ORKESTRATOR YANG DIPERBAIKI ---


def process_all_wells_rgsa(df_well: pd.DataFrame, params: Dict,
                           target_intervals: list = None,
                           target_zones: list = None) -> pd.DataFrame:
    """
    Orkestrator yang sudah diperbaiki: menjalankan proses RGSA multi-pass
    hanya pada data yang sudah difilter.
    """
    # Buat salinan untuk menghindari modifikasi DataFrame asli
    df_original = df_well.copy()

    # --- PERBAIKAN UTAMA: Buat mask dan saring data DI SINI ---
    mask = pd.Series(True, index=df_original.index)
    if target_intervals and 'MARKER' in df_original.columns:
        mask &= df_original['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_original.columns:
        mask &= df_original['ZONE'].isin(target_zones)

    if not mask.any() or not mask.all():
        df_to_process = df_original[mask].copy()
        print(
            f"Memproses RGSA untuk {len(df_to_process)} dari {len(df_original)} baris data (difilter).")
    else:  # Jika tidak ada filter, proses semua
        df_to_process = df_original
        print(
            f"Memproses RGSA untuk seluruh {len(df_original)} baris data (tanpa filter).")

    # Buat DataFrame kerja yang HANYA berisi data yang dipilih
    df_to_process = df_original[mask].copy()
    print(
        f"Memproses RGSA untuk {len(df_to_process)} dari {len(df_original)} baris data.")
    # --- AKHIR PERBAIKAN ---

    # --- Jalankan semua pass HANYA pada data yang sudah difilter ---
    df_gr_cap = calculate_dynamic_gr_cap(df_to_process, params)
    if df_gr_cap is None or df_gr_cap.empty:
        print("❌ Gagal pada Pass 1. Mengembalikan data asli.")
        return df_original

    df_coeffs = calculate_regression_coefficients(
        df_to_process, df_gr_cap, params)
    if df_coeffs is None or df_coeffs.empty:
        print("❌ Gagal pada Pass 2. Mengembalikan data asli.")
        return df_original

    # Pass 3 menghitung RGSA dan menggabungkannya kembali ke DataFrame LENGKAP
    result_df = calculate_and_merge_rgsa(df_original, df_coeffs, params)
    if result_df is None:
        print("❌ Gagal pada Pass 3. Mengembalikan data asli.")
        return df_original

    print("\n✅ Proses RGSA selesai dengan sukses.")
    return result_df
