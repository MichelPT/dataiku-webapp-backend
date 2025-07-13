import pandas as pd
import numpy as np


def calculate_sw(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Fungsi utama untuk menghitung Saturasi Air (SW Indonesia) dan klasifikasi reservoir.
    """
    df_processed = df.copy()

    # Ekstrak parameter dari frontend, dengan nilai default yang aman
    RWS = float(params.get('RWS', 0.529))
    RWT = float(params.get('RWT', 227))
    FTEMP = float(params.get('FTEMP', 80))
    RT_SH = float(params.get('RT_SH', 2.2))
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))
    SW = 'SW'
    VSH = 'VSH'
    PHIE = 'PHIE'
    RT = 'RT'

    # Asumsi kolom PHIE dan VSH sudah ada dari proses sebelumnya
    required_cols = ['GR', 'RT', 'PHIE', 'VSH']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(
            "Kolom input (GR, RT, PHIE, VSH) belum lengkap. Jalankan modul sebelumnya.")

    print("Menghitung RW pada suhu formasi...")
    df_processed["RW_TEMP"] = RWS * (RWT + 21.5) / (FTEMP + 21.5)

    print("Menghitung Saturasi Air (SW Indonesia)...")
    v = df_processed[VSH] ** 2
    ff = A / df_processed[PHIE] ** M
    # Hindari pembagian dengan nol
    ff_times_rw_temp = ff * df_processed["RW_TEMP"]
    # Ganti 0 dengan NaN untuk sementara
    ff_times_rw_temp[ff_times_rw_temp == 0] = np.nan

    f1 = 1 / ff_times_rw_temp
    f2 = 2 * np.sqrt(v / (ff_times_rw_temp * RT_SH))
    f3 = v / RT_SH

    denom = f1 + f2 + f3
    denom[denom == 0] = np.nan

    df_processed[SW] = (1 / (df_processed[RT] * denom)) ** (1 / N)
    df_processed.loc[df_processed[PHIE] < 0.005, SW] = 1.0
    df_processed[SW] = df_processed[SW].clip(lower=0, upper=1)

    return df_processed
