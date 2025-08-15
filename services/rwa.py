import pandas as pd
import numpy as np


def calculate_rwa(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Menghitung RWA (Full, Simple, Tar) beserta koreksi temperaturnya (RWT),
    dengan filter internal untuk interval/zona.
    """
    df_processed = df.copy()

    # --- 1. Ekstrak Parameter & Validasi Kolom ---
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    RT_SH = float(params.get('RT_SH', 5.0))

    # Parameter untuk koreksi temperatur RWA_RWT
    FTEMP = float(params.get('FTEMP', 80))
    RWT = float(params.get('RWT', 227))

    required_cols = ['PHIE', 'RT', 'VSH']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Kolom input {required_cols} tidak lengkap.")

    # Buat proses idempoten dengan menghapus hasil lama dan inisialisasi ulang
    rwa_cols = [
        "RWA_FULL", "RWA_SIMPLE", "RWA_TAR",
        "RWA_FULL_RWT", "RWA_SIMPLE_RWT", "RWA_TAR_RWT"
    ]
    df_processed.drop(
        columns=df_processed.columns.intersection(rwa_cols), inplace=True)
    for col in rwa_cols:
        df_processed[col] = np.nan

    # --- 2. Buat Mask Filter ---
    mask = pd.Series(True, index=df_processed.index)
    if target_intervals and 'MARKER' in df_processed.columns:
        mask &= df_processed['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_processed.columns:
        mask &= df_processed['ZONE'].isin(target_zones)

    valid_data_mask = df_processed[required_cols].notna().all(axis=1)
    final_mask = mask & valid_data_mask

    if not final_mask.any():
        print("Peringatan: Tidak ada data yang cocok dengan filter. Tidak ada kalkulasi RWA yang dilakukan.")
        return df_processed

    print(
        f"Menghitung RWA untuk {final_mask.sum()} dari {len(df_processed)} baris.")

    # --- 3. Lakukan Perhitungan HANYA pada Baris yang Dipilih ---
    phie = df_processed.loc[final_mask, "PHIE"].copy()
    rt = df_processed.loc[final_mask, "RT"].copy()
    vsh = df_processed.loc[final_mask, "VSH"].copy()

    rt[rt == 0] = np.nan
    phie[phie == 0] = np.nan

    # Perhitungan umum
    f1 = (phie ** M) / A
    f2 = 1 / rt

    # Kalkulasi untuk setiap metode
    # a. Full Indonesia
    v_full = vsh ** (2 - vsh)
    f3_full = v_full / RT_SH
    f4_full = np.sqrt(v_full / (rt * RT_SH))
    denominator_full = f2 + f3_full - f4_full
    rwaf = np.where(denominator_full != 0, f1 / denominator_full, np.nan)
    df_processed.loc[final_mask, "RWA_FULL"] = np.clip(rwaf, 0, None)

    # b. Simple Indonesia
    v_simple = vsh ** 2
    f3_simple = v_simple / RT_SH
    f4_simple = np.sqrt(v_simple / (rt * RT_SH))
    denominator_simple = f2 + f3_simple - f4_simple
    rwas = np.where(denominator_simple != 0, f1 / denominator_simple, np.nan)
    df_processed.loc[final_mask, "RWA_SIMPLE"] = np.clip(rwas, 0, None)

    # c. Tar Sand
    v_tar = vsh ** (2 - 2 * vsh)
    f3_tar = v_tar / RT_SH
    f4_tar = np.sqrt(v_tar / (rt * RT_SH))
    denominator_tar = f2 + f3_tar - f4_tar
    rwat = np.where(denominator_tar != 0, f1 / denominator_tar, np.nan)
    df_processed.loc[final_mask, "RWA_TAR"] = np.clip(rwat, 0, None)

    # --- 4. Hitung RWA_RWT untuk setiap metode ---
    temp_denominator = RWT + 21.5
    if temp_denominator != 0:
        temp_correction = (FTEMP + 21.5) / temp_denominator
        df_processed.loc[final_mask, "RWA_FULL_RWT"] = df_processed.loc[final_mask,
                                                                        "RWA_FULL"] * temp_correction
        df_processed.loc[final_mask, "RWA_SIMPLE_RWT"] = df_processed.loc[final_mask,
                                                                          "RWA_SIMPLE"] * temp_correction
        df_processed.loc[final_mask, "RWA_TAR_RWT"] = df_processed.loc[final_mask,
                                                                       "RWA_TAR"] * temp_correction

    print("Kolom RWA dan RWA_RWT telah ditambahkan/diperbarui.")
    return df_processed
