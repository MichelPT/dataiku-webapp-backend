import numpy as np
import pandas as pd


def calculate_iqual(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Menghitung IQUAL berdasarkan threshold dinamis dan hanya pada interval/zona yang dipilih.
    """
    df_processed = df.copy()

    # 1. Ekstrak parameter threshold dari frontend, dengan nilai default
    phie_threshold = float(params.get('PHIE_THRESHOLD', 0.1))
    vsh_threshold = float(params.get('VSH_THRESHOLD', 0.5))

    # 2. Validasi kolom input
    required_cols = ['PHIE', 'VSH']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(
            "Kolom 'PHIE' dan 'VSH' dibutuhkan untuk kalkulasi IQUAL.")

    # 3. Buat mask untuk filter zona/interval dari frontend
    mask = pd.Series(True, index=df_processed.index)
    if target_intervals and 'MARKER' in df_processed.columns:
        mask &= df_processed['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_processed.columns:
        mask &= df_processed['ZONE'].isin(target_zones)

    if not mask.any():
        print("Peringatan: Tidak ada data yang cocok dengan filter. Tidak ada kalkulasi yang dilakukan.")
        return df_processed  # Kembalikan DataFrame asli

    print(
        f"Menghitung IQUAL untuk {mask.sum()} dari {len(df_processed)} baris data (difilter).")

    # 4. Inisialisasi atau bersihkan kolom IQUAL
    if 'IQUAL' in df_processed.columns:
        # Set nilai menjadi 0 (non-reservoir) untuk semua baris yang difilter terlebih dahulu
        df_processed.loc[mask, 'IQUAL'] = 0
    else:
        df_processed['IQUAL'] = 0

    # 5. Lakukan perhitungan HANYA pada baris yang cocok dengan mask
    # Buat sub-mask untuk kondisi reservoir di dalam area yang sudah difilter
    reservoir_condition = (df_processed.loc[mask, 'PHIE'] >= phie_threshold) & \
                          (df_processed.loc[mask, 'VSH'] <= vsh_threshold)

    # Dapatkan indeks absolut dari baris yang memenuhi kondisi reservoir
    reservoir_indices = df_processed[mask][reservoir_condition].index

    # Update kolom IQUAL menjadi 1 hanya pada indeks tersebut
    df_processed.loc[reservoir_indices, 'IQUAL'] = 1

    return df_processed
