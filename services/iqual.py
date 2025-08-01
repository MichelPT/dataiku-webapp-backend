import numpy as np
import pandas as pd


def calculate_iqual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung kolom IQUAL berdasarkan kondisi PHIE dan VSH.
    IQUAL = 1 jika (PHIE >= 0.1) dan (VSH <= 0.5), selain itu 0.
    """
    df = df.copy()
    # Pastikan kolom yang dibutuhkan ada
    if 'PHIE' in df.columns and 'VSH' in df.columns:
        df['IQUAL'] = np.where((df['PHIE'] >= 0.1) & (df['VSH'] <= 0.5), 1, 0)
    else:
        # Jika kolom tidak ada, buat IQUAL dengan nilai default 0
        df['IQUAL'] = 0
        print("Peringatan: Kolom 'PHIE' atau 'VSH' tidak ditemukan. Kolom 'IQUAL' diisi dengan 0.")
    return df
