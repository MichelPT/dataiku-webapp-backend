from typing import Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from app.services.rgsa import process_rgsa_for_well
from app.services.ngsa import process_ngsa_for_well
from app.services.dgsa import process_dgsa_for_well


def _interpolate_coeffs(depth, coeff_df):
    """(Internal) Melakukan interpolasi linear pada koefisien regresi."""
    if coeff_df.empty:
        return np.array([np.nan] * 4)
    if depth <= coeff_df['DEPTH'].min():
        return coeff_df.iloc[0][['b0', 'b1', 'b2', 'b3']].values
    if depth >= coeff_df['DEPTH'].max():
        return coeff_df.iloc[-1][['b0', 'b1', 'b2', 'b3']].values
    lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
    upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]
    weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
    row = lower + weight * (upper - lower)
    return row[['b0', 'b1', 'b2', 'b3']].values


def _classify_zone(score):
    """(Internal) Memberikan nama zona berdasarkan skor anomali."""
    if score == 3:
        return 'Zona Prospek Kuat'
    elif score == 2:
        return 'Zona Menarik'
    elif score == 1:
        return 'Zona Lemah'
    else:
        return 'Non Prospek'


def calculate_gsa_log(df_input: pd.DataFrame, params: dict, ref_log: str, target_log: str, output_log_name: str) -> Optional[pd.DataFrame]:
    """
    Fungsi generik untuk menghitung baseline log (GSA) menggunakan sliding window regression.
    """
    df_input = df_input.copy()
    df_valid = df_input[['DEPTH', ref_log, target_log]].dropna().copy()
    coeffs = []

    window_size = int(params.get('window_size', 106))
    step = int(params.get('step', 20))
    min_points = int(params.get('min_points_in_window', 30))

    print(f"Memulai kalkulasi {output_log_name}...")

    for start in range(0, len(df_valid) - window_size, step):
        window = df_valid.iloc[start:start + window_size]
        gr = window[ref_log].values
        target = window[target_log].values

        mask = (gr > 5) & (gr < 180) & (target > 0.1) & (target < 1000)
        gr, target = gr[mask], target[mask]

        if len(gr) < min_points:
            continue

        gr_scaled = 0.01 * gr
        y = np.log10(target) if output_log_name == 'RGSA' else target
        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T

        model = LinearRegression().fit(X, y)
        coeffs.append({
            'DEPTH': window['DEPTH'].mean(),
            'b0': model.intercept_,
            'b1': model.coef_[0],
            'b2': model.coef_[1],
            'b3': model.coef_[2]
        })

    if not coeffs:
        print(
            f"⚠️ Tidak ada koefisien dihitung untuk {output_log_name}, seluruh output akan NaN.")
        df_input[output_log_name] = np.nan
        return df_input

    coeff_df = pd.DataFrame(coeffs)

    # Siapkan array hasil
    gsa_array = np.full(len(df_input), np.nan)

    for i, row in df_input.iterrows():
        if pd.isna(row[ref_log]) or pd.isna(row['DEPTH']):
            continue

        gr_val = row[ref_log]
        if gr_val < 5 or gr_val > 180:
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(row['DEPTH'], coeff_df)
        if np.isnan(b0):
            continue

        grfix = 0.01 * gr_val
        log_gsa = b0 + b1 * grfix + b2 * grfix**2 + b3 * grfix**3
        gsa_val = 10**log_gsa if output_log_name == 'RGSA' else log_gsa
        gsa_array[i] = gsa_val

    df_input[output_log_name] = gsa_array
    return df_input
    return pd.merge(df_input, df_gsa[['DEPTH', output_log_name]], on='DEPTH', how='left')


def run_full_gsa_analysis(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Orkestrator untuk menjalankan seluruh alur kerja GSA (RGSA, NGSA, DGSA).
    """
    df_processed = df.copy()

    # --- Proses RGSA ---
    print("Memproses RGSA...")
    df_processed = process_rgsa_for_well(
        df_well=df_processed, params=params,
        ref_log='GR', target_log='RT', output_log_name='RGSA'
    )
    if 'RT' in df_processed and 'RGSA' in df_processed:
        df_processed['RGSA_GAS_EFFECT'] = df_processed['RT'] > df_processed['RGSA']
        print("Kolom RGSA_GAS_EFFECT ditambahkan.")

    # --- Proses NGSA ---
    print("Memproses NGSA...")
    df_processed = process_ngsa_for_well(
        df_well=df_processed, params=params,
        ref_log='GR', target_log='NPHI', output_log_name='NGSA'
    )
    if 'NPHI' in df_processed and 'NGSA' in df_processed:
        df_processed['NGSA_GAS_EFFECT'] = df_processed['NPHI'] < df_processed['NGSA']
        print("Kolom NGSA_GAS_EFFECT ditambahkan.")

    # --- Proses DGSA ---
    print("Memproses DGSA...")
    df_processed = process_dgsa_for_well(
        df_well=df_processed, params=params,
        ref_log='GR', target_log='RHOB', output_log_name='DGSA'
    )
    if 'RHOB' in df_processed and 'DGSA' in df_processed:
        df_processed['DGSA_GAS_EFFECT'] = df_processed['RHOB'] < df_processed['DGSA']
        df_processed['DENS_DIFF'] = df_processed['DGSA'] - df_processed['RHOB']
        print("Kolom DGSA_GAS_EFFECT dan DENS_DIFF ditambahkan.")

    print("Proses Kalkulasi GSA selesai.")
    print("Menghitung anomali, skor, dan klasifikasi zona...")

    # Hitung anomali dan skor
    df_processed['RGSA_ANOM'] = df_processed['RT'] > df_processed['RGSA']
    df_processed['NGSA_ANOM'] = df_processed['NPHI'] < df_processed['NGSA']
    df_processed['DGSA_ANOM'] = df_processed['RHOB'] < df_processed['DGSA']

    df_processed['SCORE'] = df_processed[[
        'RGSA_ANOM', 'NGSA_ANOM', 'DGSA_ANOM']].sum(axis=1)
    df_processed['ZONA'] = df_processed['SCORE'].apply(_classify_zone)
    print("Kolom ZONA berhasil dibuat.")

    return df_processed
