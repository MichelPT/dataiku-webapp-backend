from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def process_rgsa_for_well(df_well: pd.DataFrame, params: dict, target_intervals: list, target_zones: list) -> pd.DataFrame:
    """
    Memproses RGSA untuk satu sumur, dengan perhitungan terbatas pada interval/zona target.
    """
    # --- LANGKAH 1: VALIDASI DAN PERSIAPAN DATA ---
    gr_col = params.get('GR', 'GR')
    rt_col = params.get('RES', 'RT')
    required_cols = ['DEPTH', gr_col, rt_col]

    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom yang dibutuhkan hilang: {required_cols}")
        return df_well

    # --- PERUBAHAN KUNCI: Membuat mask untuk memilih baris target ---
    # Default ke semua baris jika tidak ada filter
    mask = pd.Series(True, index=df_well.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_well.columns:
        mask = df_well['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_well.columns:
        # Jika sudah ada filter dari marker, gabungkan dengan logika OR
        if has_filters:
            mask |= df_well['ZONE'].isin(target_zones)
        else:
            mask = df_well['ZONE'].isin(target_zones)

    # Buat DataFrame kerja HANYA dari baris yang cocok dengan mask
    df_rgsa = df_well.loc[mask, required_cols].dropna().copy()

    if len(df_rgsa) < 100:
        print(
            f"Peringatan: Data tidak cukup untuk regresi RGSA (hanya {len(df_rgsa)} baris dalam interval/zona yang dipilih).")
        return df_well
    # --- AKHIR PERUBAHAN KUNCI ---

    # --- LANGKAH 2: REGRESI SLIDING WINDOW ---
    window_size = int(params.get('SLIDING_WINDOW', 100))
    step = 20
    min_points_in_window = 30
    gr_filter = (5, 180)
    rt_filter = (0.1, 1000)

    coeffs = []
    for start in range(0, len(df_rgsa) - window_size, step):
        window = df_rgsa.iloc[start:start+window_size]
        # ... (Sisa logika regresi tetap sama) ...
        gr = window[gr_col].values
        rt = window[rt_col].values
        mask_filter = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            rt > rt_filter[0]) & (rt < rt_filter[1])
        gr_filtered, rt_filtered = gr[mask_filter], rt[mask_filter]
        if len(gr_filtered) < min_points_in_window:
            continue
        gr_scaled = 0.01 * gr_filtered
        log_rt = np.log10(rt_filtered)
        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = log_rt
        try:
            model = LinearRegression().fit(X, y)
            coeffs.append({'DEPTH': window['DEPTH'].mean(), 'b0': model.intercept_, 'b1': model.coef_[
                          0], 'b2': model.coef_[1], 'b3': model.coef_[2]})
        except Exception as e:
            print(
                f"Peringatan: Regresi gagal pada kedalaman ~{window['DEPTH'].mean()}: {e}")

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return df_well

    coeff_df = pd.DataFrame(coeffs)

    # --- LANGKAH 3: INTERPOLASI & HITUNG RGSA ---
    def interpolate_coeffs(depth):
        # ... (Fungsi interpolasi tetap sama) ...
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
        # ... (Sisa logika kalkulasi RGSA tetap sama) ...
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

    # --- LANGKAH 4: GABUNGKAN HASIL ---
    if 'RGSA' in df_well.columns:
        df_well = df_well.drop(columns=['RGSA'])

    # Merge kembali ke DataFrame LENGKAP asli
    df_merged = pd.merge(
        df_well, df_rgsa[['DEPTH', 'RGSA']], on='DEPTH', how='left')

    # Hitung kolom efek gas
    if rt_col in df_merged and 'RGSA' in df_merged:
        df_merged['GAS_EFFECT_RT'] = (df_merged[rt_col] > df_merged['RGSA'])
        df_merged['RT_RATIO'] = df_merged[rt_col] / df_merged['RGSA']
        df_merged['RT_DIFF'] = df_merged[rt_col] - df_merged['RGSA']

    return df_merged


def process_all_wells_rgsa(df_well: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Fungsi orkestrator: memproses RGSA untuk sebuah sumur.
    """
    print("Memulai proses RGSA...")
    # Teruskan semua parameter ke fungsi pemroses inti
    result_df = process_rgsa_for_well(
        df_well, params, target_intervals, target_zones)
    print("✅ Proses RGSA selesai.")
    return result_df
