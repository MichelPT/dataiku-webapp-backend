import pandas as pd
import numpy as np


def calculate_rwa(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Fungsi utama untuk menghitung RWA (Full, Simple, Tar) berdasarkan parameter.
    Fungsi ini mandiri dan akan menghitung VSH & PHIE jika belum ada.
    """
    df_processed = df.copy()

    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    RT_SH = float(params.get('RT_SH', 5.0))

    phie = df_processed["PHIE"]
    rt = df_processed["RT"]
    vsh = df_processed["VSH"]

    rt[rt == 0] = np.nan
    phie[phie == 0] = np.nan

    f1 = (phie ** M) / A
    f2 = 1 / rt

    # Full Indonesia
    v_full = vsh ** (2 - vsh)
    f3_full = v_full / RT_SH
    f4_full = np.sqrt(v_full / (rt * RT_SH))
    rwaf = f1 / (f2 + f3_full - f4_full)
    df_processed["RWA_FULL"] = rwaf.clip(lower=0)

    # Simple Indonesia
    v_simple = vsh ** 2
    f3_simple = v_simple / RT_SH
    f4_simple = np.sqrt(v_simple / (rt * RT_SH))
    rwas = f1 / (f2 + f3_simple - f4_simple)
    df_processed["RWA_SIMPLE"] = rwas.clip(lower=0)

    # Tar Sand
    v_tar = vsh ** (2 - 2 * vsh)
    f3_tar = v_tar / RT_SH
    f4_tar = np.sqrt(v_tar / (rt * RT_SH))
    rwat = f1 / (f2 + f3_tar - f4_tar)
    df_processed["RWA_TAR"] = rwat.clip(lower=0)

    print("Kolom RWA baru telah ditambahkan: RWA_FULL, RWA_SIMPLE, RWA_TAR")
    return df_processed
