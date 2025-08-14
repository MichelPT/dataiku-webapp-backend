import pandas as pd
import numpy as np


def calculate_vsh_from_gr(
    df: pd.DataFrame,
    gr_log: str,
    gr_ma: float,
    gr_sh: float,
    output_col: str,
    target_intervals: list = None,
    target_zones: list = None
) -> pd.DataFrame:
    """
    Menghitung VSH dari Gamma Ray menggunakan metode linear, dengan opsi untuk
    membatasi perhitungan hanya pada interval atau zona tertentu.

    Args:
        df (pd.DataFrame): DataFrame input yang berisi data log.
        gr_log (str): Nama kolom Gamma Ray yang akan digunakan.
        gr_ma (float): Nilai GR matriks (zona bersih).
        gr_sh (float): Nilai GR shale.
        output_col (str): Nama kolom baru untuk menyimpan hasil VSH.
        target_intervals (list, optional): Daftar interval (dari kolom 'MARKER') 
                                            yang akan dihitung. Defaults to None.
        target_zones (list, optional): Daftar zona (dari kolom 'ZONE') yang 
                                       akan dihitung. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame asli dengan tambahan atau pembaruan kolom VSH.
    """
    # Pastikan kolom input yang dibutuhkan ada
    if gr_log not in df.columns:
        print(
            f"Peringatan: Kolom '{gr_log}' tidak ditemukan. Melewatkan kalkulasi VSH.")
        # Buat kolom output kosong jika belum ada, agar tidak error saat menyimpan
        if output_col not in df.columns:
            df[output_col] = np.nan
        return df

    # Salin untuk menghindari SettingWithCopyWarning
    df_processed = df.copy()

    # Jika kolom output belum ada, inisialisasi dengan nilai kosong (NaN)
    if output_col not in df_processed.columns:
        df_processed[output_col] = np.nan

    # --- BAGIAN BARU: Membuat mask untuk memilih baris target ---
    # Inisialisasi mask untuk tidak memilih baris apa pun pada awalnya
    mask = pd.Series(False, index=df_processed.index)

    has_filters = False
    # Tambahkan baris yang cocok dengan target_intervals ke mask
    if target_intervals and 'MARKER' in df_processed.columns:
        mask |= df_processed['MARKER'].isin(target_intervals)
        has_filters = True

    # Tambahkan baris yang cocok dengan target_zones ke mask
    if target_zones and 'ZONE' in df_processed.columns:
        mask |= df_processed['ZONE'].isin(target_zones)
        has_filters = True

    # Jika tidak ada filter yang diberikan (target_intervals dan target_zones kosong),
    # maka pilih semua baris untuk dihitung.
    if not has_filters:
        mask = pd.Series(True, index=df_processed.index)

    # Jika tidak ada baris yang cocok dengan filter, kembalikan DataFrame tanpa perubahan
    if not mask.any():
        print(
            "Peringatan: Tidak ada baris yang cocok dengan interval atau zona yang dipilih.")
        return df_processed

    # --- AKHIR BAGIAN BARU ---

    # Hitung VSH dengan rumus linear HANYA pada baris yang dipilih oleh mask
    v_gr = (df_processed.loc[mask, gr_log] - gr_ma) / (gr_sh - gr_ma)

    # Batasi nilai antara 0 dan 1 dan tetapkan ke kolom output HANYA untuk baris yang dipilih
    df_processed.loc[mask, output_col] = v_gr.clip(0, 1)

    return df_processed
