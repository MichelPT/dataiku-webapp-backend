from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def _interpolate_coeffs(depth, coeff_df):
    # (Fungsi internal ini tidak berubah)
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


def process_dgsa_for_well(df_well: pd.DataFrame, params: dict, target_intervals: list, target_zones: list) -> pd.DataFrame:
    gr_col, rhob_col = params.get('GR', 'GR'), params.get('DENS', 'RHOB')
    required_cols = ['DEPTH', gr_col, rhob_col]
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom hilang: {required_cols}")
        return df_well

    mask = pd.Series(True, index=df_well.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_well.columns:
        mask = df_well['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_well.columns:
        if has_filters:
            mask |= df_well['ZONE'].isin(target_zones)
        else:
            mask = df_well['ZONE'].isin(target_zones)

    df_dgsa = df_well.loc[mask, required_cols].dropna().copy()
    if len(df_dgsa) < 100:
        print(
            f"Peringatan: Data tidak cukup untuk regresi DGSA (hanya {len(df_dgsa)} baris).")
        return df_well

    window_size, step, min_points = int(
        params.get('SLIDING_WINDOW', 100)), 20, 30
    gr_filter, rhob_filter = (5, 180), (1.5, 3.0)
    coeffs = []
    for start in range(0, len(df_dgsa) - window_size, step):
        window = df_dgsa.iloc[start:start+window_size]
        gr, dens = window[gr_col].values, window[rhob_col].values
        mask_filter = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            dens > rhob_filter[0]) & (dens < rhob_filter[1])
        gr_filtered, dens_filtered = gr[mask_filter], dens[mask_filter]
        if len(gr_filtered) < min_points:
            continue
        gr_scaled = 0.01 * gr_filtered
        X, y = np.vstack(
            [gr_scaled, gr_scaled**2, gr_scaled**3]).T, dens_filtered
        try:
            model = LinearRegression().fit(X, y)
            coeffs.append({'DEPTH': window['DEPTH'].mean(), 'b0': model.intercept_, 'b1': model.coef_[
                          0], 'b2': model.coef_[1], 'b3': model.coef_[2]})
        except Exception as e:
            print(
                f"Peringatan: Regresi gagal pada ~{window['DEPTH'].mean()}: {e}")

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return df_well

    coeff_df = pd.DataFrame(coeffs)
    dgsa_list = []
    for _, row in df_dgsa.iterrows():
        depth, gr = row['DEPTH'], row[gr_col]
        if not (gr_filter[0] < gr < gr_filter[1]):
            dgsa_list.append(np.nan)
            continue
        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        dgsa_list.append(b0 + b1*grfix + b2*grfix**2 + b3*grfix**3)
    df_dgsa['DGSA'] = dgsa_list

    if 'DGSA' in df_well.columns:
        df_well = df_well.drop(columns=['DGSA'])
    df_merged = pd.merge(
        df_well, df_dgsa[['DEPTH', 'DGSA']], on='DEPTH', how='left')
    if rhob_col in df_merged and 'DGSA' in df_merged:
        df_merged['GAS_EFFECT_RHOB'] = (
            df_merged[rhob_col] < df_merged['DGSA'])
        df_merged['DENS_DIFF'] = df_merged['DGSA'] - df_merged[rhob_col]
    return df_merged


def process_all_wells_dgsa(df_well: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    print("Memulai proses DGSA...")
    result_df = process_dgsa_for_well(
        df_well, params, target_intervals, target_zones)
    print("✅ Proses DGSA selesai.")
    return result_df
