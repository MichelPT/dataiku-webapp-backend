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


def fill_flagged_values(df, logs_to_fill, max_consecutive, isDataPrep):
    """
    Mengisi nilai yang hilang (NaN) pada log yang dipilih berdasarkan MISSING_FLAG == 1,
    namun HANYA jika jumlah data hilang yang berurutan kurang dari atau sama dengan
    'max_consecutive'.

    Proses ini menggunakan metode backward fill diikuti forward fill dan membuat
    kolom baru dengan akhiran '_FM' untuk menyimpan hasilnya.

    Args:
        df (pd.DataFrame): DataFrame input.
        logs_to_fill (list): Daftar nama kolom (log) yang akan diproses.
        max_consecutive (int): Jumlah maksimum baris NaN berurutan yang diizinkan untuk diisi.

    Returns:
        pd.DataFrame: DataFrame dengan kolom baru '_FM' yang sudah diisi secara kondisional.
    """
    df_filled = df.copy()

    if 'MISSING_FLAG' not in df_filled.columns:
        print("Peringatan: Kolom 'MISSING_FLAG' tidak ditemukan.")
        return df_filled

    # --- Bagian Inti untuk Identifikasi Blok ---
    # 1. Buat kolom 'block' untuk mengidentifikasi setiap grup/blok data yang
    #    berurutan di mana MISSING_FLAG == 1.
    #    Ini adalah trik pandas yang efisien: .diff() mendeteksi perubahan,
    #    dan .cumsum() memberi nomor unik untuk setiap blok.
    df_filled['block'] = (df_filled['MISSING_FLAG'].diff() != 0).cumsum()

    # 2. Dapatkan DataFrame yang hanya berisi baris-baris yang hilang
    missing_rows = df_filled[df_filled['MISSING_FLAG'] == 1]

    # 3. Hitung ukuran (jumlah baris) untuk setiap blok data yang hilang
    if not missing_rows.empty:
        block_sizes = missing_rows.groupby('block').size()

        # 4. Dapatkan ID dari blok-blok yang ukurannya <= batas maksimum dari input user
        blocks_to_fill_ids = block_sizes[block_sizes <= max_consecutive].index
    else:
        # Jika tidak ada baris yang hilang sama sekali, buat daftar kosong
        blocks_to_fill_ids = []

    # 5. Buat 'mask' final yang lebih cerdas. Sebuah baris akan diisi HANYA JIKA:
    #    a. MISSING_FLAG-nya adalah 1
    #    b. ID 'block' dari baris tersebut ada di dalam daftar 'blocks_to_fill_ids'
    fill_mask = (df_filled['MISSING_FLAG'] == 1) & \
                (df_filled['block'].isin(blocks_to_fill_ids))
    # --- Akhir Bagian Identifikasi Blok ---

    for log in logs_to_fill:
        if log not in df_filled.columns:
            print(
                f"Peringatan: Kolom log '{log}' tidak ditemukan. Melewati...")
            continue

        if isDataPrep:
            new_col_name = f"{log}_FM"
        else:
            new_col_name = log
        df_filled[new_col_name] = df_filled[log]

        # Siapkan nilai untuk mengisi menggunakan metode bfill -> ffill
        fill_values = df_filled[log].bfill().ffill()

        # Lakukan pengisian hanya pada baris yang telah lolos seleksi mask
        df_filled.loc[fill_mask, new_col_name] = fill_values[fill_mask]

    # Hapus kolom 'block' sementara setelah semua log selesai diproses
    if 'block' in df_filled.columns:
        df_filled = df_filled.drop(columns=['block'])

    return df_filled
