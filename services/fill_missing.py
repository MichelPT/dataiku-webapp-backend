import pandas as pd
import numpy as np

from services.data_processing import fill_flagged_missing_values

import pandas as pd
import numpy as np


def flag_missing_values(df, logs_to_check):
    """
    Membuat atau memperbarui kolom MISSING_FLAG berdasarkan log yang dipilih.
    Nilai 1 = ada NaN di salah satu log yang diperiksa.
    Nilai 0 = semua log yang diperiksa punya data (tidak NaN).
    """
    df_flagged = df.copy()

    # Jika kolom MISSING_FLAG belum ada, inisialisasi dengan 0
    if 'MISSING_FLAG' not in df_flagged.columns:
        df_flagged['MISSING_FLAG'] = 0

    # Buat mask untuk baris yang mengandung NaN di salah satu log yang dipilih
    mask_missing = df_flagged[logs_to_check].isna().any(axis=1)

    # Set flag berdasarkan mask
    df_flagged.loc[mask_missing, 'MISSING_FLAG'] = 1
    df_flagged.loc[~mask_missing, 'MISSING_FLAG'] = 0

    return df_flagged


def fill_flagged_values(df, logs_to_fill, max_consecutive_nan):
    """
    Mengisi nilai NaN pada log yang dipilih jika flag-nya 1 dan dalam batas max_consecutive.
    """
    df_filled = df.copy()

    for log in logs_to_fill:
        if log in df_filled.columns and 'MISSING_FLAG' in df_filled.columns:
            # Buat grup dari data yang hilang secara berurutan
            df_filled['block'] = (
                df_filled['MISSING_FLAG'].diff() != 0).cumsum()
            missing_blocks = df_filled[df_filled['MISSING_FLAG'] == 1]

            # Hitung ukuran setiap blok data yang hilang
            block_sizes = missing_blocks.groupby('block').size()

            # Dapatkan ID blok yang ukurannya kurang dari atau sama dengan batas maksimum
            blocks_to_fill = block_sizes[block_sizes <=
                                         max_consecutive_nan].index

            # Buat mask boolean untuk semua baris yang akan diisi
            fill_mask = (df_filled['MISSING_FLAG'] == 1) & (
                df_filled['block'].isin(blocks_to_fill))

            # Lakukan interpolasi pada seluruh kolom untuk mendapatkan nilai pengisi
            interpolated_series = df_filled[log].interpolate(
                method='linear', limit_direction='both')

            # Gunakan mask untuk mengisi nilai hanya pada baris yang memenuhi syarat
            df_filled.loc[fill_mask, log] = interpolated_series[fill_mask]

            # Hapus flag untuk data yang sudah berhasil diisi
            df_filled.loc[fill_mask, 'MISSING_FLAG'] = 0

    # Hapus kolom 'block' temporary yang digunakan untuk grouping
    if 'block' in df_filled.columns:
        df_filled = df_filled.drop(columns=['block'])

    return df_filled
