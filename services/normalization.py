import pandas as pd
import numpy as np


def calculate_gr_normalized(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Menormalisasi log berdasarkan rumus yang diberikan dari LOGLAN.

    Formula:
        grz = (pct_high_new - pct_low_new) / (pct_high_old - pct_low_old)
        grn = grz * (gr - pct_low_old) + pct_low_new

    Args:
        df (pd.DataFrame): DataFrame input yang berisi data log.
        params (dict): Dictionary berisi parameter dari frontend.
        target_intervals (list, optional): Daftar marker interval yang akan diproses.
        target_zones (list, optional): Daftar zona yang akan diproses.

    Returns:
        pd.DataFrame: DataFrame dengan kolom log yang sudah dinormalisasi.
    """
    df_processed = df.copy()

    # --- 1. Ekstrak semua parameter dari dictionary 'params' ---
    # Parameter 'NEW' dan 'OLD' sekarang semuanya adalah nilai numerik (float)
    pct_low_new = float(params.get('PCT_LOW_NEW', 0.0))
    pct_high_new = float(params.get('PCT_HIGH_NEW', 0.0))
    pct_low_old = float(params.get('PCT_LOW_OLD', 0.0))
    pct_high_old = float(params.get('PCT_HIGH_OLD', 0.0))

    # Nama kolom untuk log input dan output
    gr_col = params.get('LOG_IN', 'GR')
    grn_col = params.get('LOG_OUT', f"{gr_col}_NORM")

    # --- 2. Validasi kolom input ---
    if gr_col not in df_processed.columns:
        raise ValueError(
            f"Kolom input log '{gr_col}' tidak ditemukan di dalam file.")

    # Inisialisasi kolom output jika belum ada
    if grn_col not in df_processed.columns:
        # Mengisi kolom output dengan nilai original dari log input sebagai dasar
        df_processed[grn_col] = pd.to_numeric(
            df_processed[gr_col], errors='coerce')

    # --- 3. Buat mask untuk memfilter baris yang akan diproses (jika ada) ---
    mask = pd.Series(True, index=df_processed.index)
    has_filters = (target_intervals and 'MARKER' in df_processed.columns) or \
                  (target_zones and 'ZONE' in df_processed.columns)

    if has_filters:
        combined_mask = pd.Series(False, index=df_processed.index)
        if target_intervals and 'MARKER' in df_processed.columns:
            combined_mask |= df_processed['MARKER'].isin(target_intervals)
        if target_zones and 'ZONE' in df_processed.columns:
            combined_mask |= df_processed['ZONE'].isin(target_zones)
        mask = combined_mask

    if not mask.any():
        print("Peringatan: Tidak ada baris yang cocok dengan filter. Normalisasi tidak dilakukan.")
        return df_processed

    # --- 4. Terapkan rumus pada baris yang sesuai dengan mask ---

    # Ambil data log input (gr) dari kolom yang dipilih
    gr = df_processed.loc[mask, gr_col]

    # Hitung 'grz' menggunakan nilai konstanta
    denominator = pct_high_old - pct_low_old
    grz = (pct_high_new - pct_low_new) / denominator

    # Hitung 'grn' (hasil normalisasi)
    grn = grz * (gr - pct_low_old) + pct_low_new

    # Masukkan hasil perhitungan ke kolom output
    df_processed.loc[mask, grn_col] = grn

    print(
        f"Normalisasi berhasil diterapkan pada {mask.sum()} baris untuk kolom '{grn_col}'.")
    return df_processed
