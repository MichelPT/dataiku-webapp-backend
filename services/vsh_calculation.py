import pandas as pd
import numpy as np


def calculate_vsh_from_gr(df: pd.DataFrame, gr_log: str, gr_ma: float, gr_sh: float, output_col: str) -> pd.DataFrame:
    """
    Menghitung VSH dari Gamma Ray menggunakan metode linear.

    Args:
        df (pd.DataFrame): DataFrame input yang berisi data log.
        gr_log (str): Nama kolom Gamma Ray yang akan digunakan.
        gr_ma (float): Nilai GR matriks (zona bersih).
        gr_sh (float): Nilai GR shale.
        output_col (str): Nama kolom baru untuk menyimpan hasil VSH.

    Returns:
        pd.DataFrame: DataFrame asli dengan tambahan kolom VSH.
    """
    # Pastikan kolom yang dibutuhkan ada
    if gr_log not in df.columns:
        print(
            f"Peringatan: Kolom '{gr_log}' tidak ditemukan. Melewatkan kalkulasi VSH.")
        df[output_col] = np.nan
        return df

    # Salin untuk menghindari SettingWithCopyWarning
    df_processed = df.copy()

    # Hitung VSH dengan rumus linear
    v_gr = (df_processed[gr_log] - gr_ma) / (gr_sh - gr_ma)

    # Batasi nilai antara 0 dan 1
    df_processed[output_col] = v_gr.clip(0, 1)

    return df_processed
