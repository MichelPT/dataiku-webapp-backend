# /services/data_processing.py
import os
from flask import jsonify
from narwhals import col
import pandas as pd
import numpy as np
import io


def handle_null_values(csv_content: str) -> str:
    """
    Reads CSV content, fills nulls using linear interpolation,
    and returns the cleaned CSV content.
    Raises an exception if an error occurs.
    """
    # Use io.StringIO to read the string content as if it were a file
    csv_file_like_object = io.StringIO(csv_content)

    df = pd.read_csv(csv_file_like_object)

    # Identify numeric columns to interpolate
    numeric_cols = df.select_dtypes(include='number').columns

    if not numeric_cols.empty:
        # Fill nulls using linear interpolation
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='linear', limit_direction='both', axis=0)

    # Fill any remaining nulls (e.g., in non-numeric columns) with a placeholder
    df.fillna('NA', inplace=True)

    # Convert the cleaned DataFrame back to a CSV string
    cleaned_csv_content = df.to_csv(index=False)

    return cleaned_csv_content


# fill_null_handler.py


def fill_null_values_in_marker_range(df, selected_logs):
    """
    Isi nilai null pada GR, RT, NPHI, RHOB berdasarkan range marker.
    Hanya data dalam range marker yang akan diproses menggunakan backward-fill dan forward-fill.
    """
    df_filled = df.copy().sort_values('DEPTH').reset_index(drop=True)

    marker_range_mask = pd.Series(
        [True] * len(df_filled))  # default: isi semua

    if 'MARKER' in df_filled.columns and 'DEPTH' in df_filled.columns:
        marker_rows = df_filled['MARKER'].notna() & (df_filled['MARKER'] != '')
        if marker_rows.any():
            first_marker_idx = df_filled[marker_rows].index[0]
            first_marker_depth = df_filled.loc[first_marker_idx, 'DEPTH']
            last_marker_idx = df_filled.index[-1]
            last_marker_depth = df_filled.loc[last_marker_idx, 'DEPTH']

            for idx in range(first_marker_idx + 1, len(df_filled)):
                if pd.isna(df_filled.loc[idx, 'MARKER']) or df_filled.loc[idx, 'MARKER'] == '':
                    last_marker_idx = idx - 1
                    last_marker_depth = df_filled.loc[last_marker_idx, 'DEPTH']
                    break

            marker_range_mask = (
                (df_filled['DEPTH'] >= first_marker_depth) &
                (df_filled['DEPTH'] <= last_marker_depth)
            )

    for col in selected_logs:
        if col not in df_filled.columns:
            continue
        marker_indices = df_filled[marker_range_mask].index
        series = df_filled.loc[marker_indices, col]
        filled_series = series.bfill().ffill()
        df_filled.loc[marker_indices, col] = filled_series

    return df_filled


def min_max_normalize(log_in,
                      low_ref=40, high_ref=140,
                      low_in=5, high_in=95,
                      cutoff_min=0, cutoff_max=250):
    """
    Geolog-style MIN-MAX normalization.
    """
    log = np.array(log_in, dtype=float)

    if cutoff_min is not None:
        log[log < cutoff_min] = np.nan
    if cutoff_max is not None:
        log[log > cutoff_max] = np.nan

    if high_in == low_in:
        return np.full_like(log, low_ref)

    m = (high_ref - low_ref) / (high_in - low_in)
    log_out = low_ref + m * (log - low_in)

    return log_out


def selective_normalize_handler(df, log_column, marker_column,
                                target_markers=None,
                                low_ref=40, high_ref=140,
                                low_in=5, high_in=95,
                                cutoff_min=0, cutoff_max=250,
                                log_out_col=None):
    """
    Handles normalization. If target_markers is empty or None, it normalizes the entire log.
    Otherwise, it normalizes only within the specified markers.
    """
    result_df = df.copy()
    log_data = result_df[log_column].values

    # Jika tidak ada interval yang dipilih (kasus Data Prep),
    # buat 'mask' yang mencakup semua data valid di log tersebut.
    if not target_markers:
        print("No intervals selected. Normalizing the entire log.")
        # Mask akan memilih semua baris di mana log input tidak NaN
        target_mask = ~np.isnan(log_data)

        # Data yang tidak dipilih (log_raw) akan kosong
        log_raw = np.full_like(log_data, np.nan, dtype=float)

    # Jika ada interval yang dipilih (kasus Dashboard)
    else:
        print(f"Normalizing for selected intervals: {target_markers}")
        # Mask akan memilih baris yang cocok dengan marker
        target_mask = result_df[marker_column].isin(target_markers)

        # Data yang tidak dipilih (log_raw) adalah data di luar interval
        log_raw = log_data.copy()
        log_raw[target_mask] = np.nan

    log_norm = np.full_like(log_data, np.nan, dtype=float)
    log_raw_norm = log_raw.copy()

    if np.any(target_mask):
        target_data = log_data[target_mask]

        if len(target_data) > 0 and not np.all(np.isnan(target_data)):
            normalized_target = min_max_normalize(
                target_data,
                low_ref=low_ref, high_ref=high_ref,
                low_in=low_in, high_in=high_in,
                cutoff_min=cutoff_min, cutoff_max=cutoff_max
            )
            log_norm[target_mask] = normalized_target
            log_raw_norm[target_mask] = normalized_target

    # Gunakan nama kolom output yang diberikan dari frontend
    if not log_out_col:
        log_out_col = f'{log_column}_NO'

    result_df[log_out_col] = log_raw_norm

    return result_df


# Trimming data


def trim_data_auto(df, required_columns):
    valid_index = {
        col: df[(df[col] != -999.0) & (~df[col].isna())].index
        for col in required_columns if col in df.columns
    }

    start_idx = min(idx.min() for idx in valid_index.values())
    end_idx = max(idx.max() for idx in valid_index.values())
    trimmed_df = df.loc[start_idx:end_idx].copy()

    return trimmed_df


def trim_data_depth(df, depth_above=0.0, depth_below=0.0, above=0, below=0, mode=None):
    depth_above = float(depth_above)
    depth_below = float(depth_below)

    if above == 1 and below == 0:
        df = df[df.index >= depth_above]
    elif above == 0 and below == 1:
        df = df[df.index <= depth_below]
    elif above == 1 and below == 1 and mode == 'CUSTOM_TRIM':
        df = df[(df.index >= depth_above) & (df.index <= depth_below)]

    return df


def smoothing(df, window, col_in, col_out):
    df_smooth = df.copy()
    df_smooth[col_out] = df[col_in].rolling(
        window=window, center=True).mean()
    return df_smooth


def trim_log_by_masking(df: pd.DataFrame, columns_to_trim: list, trim_mode: str, depth_above: float = None, depth_below: float = None) -> pd.DataFrame:
    """
    Melakukan trimming log dengan cara masking dan menyimpan hasilnya di kolom baru
    yang namanya dibuat secara dinamis (_TR, _TR_TR, dst.) untuk menghindari penimpaan.
    """
    if 'DEPTH' not in df.columns:
        raise ValueError("DataFrame input harus memiliki kolom 'DEPTH'.")

    df_out = df.copy()

    # Tentukan mask terlebih dahulu, karena akan digunakan untuk semua kolom
    mask = None
    if trim_mode == 'CUSTOM_TRIM' and depth_above is not None and depth_below is not None:
        mask = (df_out['DEPTH'] < depth_above) | (
            df_out['DEPTH'] > depth_below)
    elif trim_mode == 'DEPTH_ABOVE' and depth_above is not None:
        mask = df_out['DEPTH'] < depth_above
    elif trim_mode == 'DEPTH_BELOW' and depth_below is not None:
        mask = df_out['DEPTH'] > depth_below

    if mask is None:
        print("Peringatan: Kondisi trim tidak valid. Tidak ada kolom baru yang dibuat.")
        return df

    # --- PERBAIKAN UTAMA: Logika Penamaan Kolom Dinamis ---
    for col_in in columns_to_trim:
        if col_in not in df_out.columns:
            print(f"Peringatan: Kolom '{col_in}' tidak ditemukan, melewati.")
            continue

        # Tentukan nama kolom output awal
        col_out = f"{col_in}_TR"

        # Loop untuk menemukan nama kolom yang unik dengan menambahkan '_TR'
        while col_out in df_out.columns:
            col_out = f"{col_out}_TR"

        print(
            f"Membuat kolom output baru: '{col_out}' dari kolom input '{col_in}'")

        # Buat kolom baru dengan menyalin data ASLI dari 'col_in'
        df_out[col_out] = df_out[col_in]

        # Terapkan mask ke kolom BARU yang unik ini
        df_out.loc[mask, col_out] = np.nan
    # --- AKHIR PERBAIKAN ---

    return df_out


def flag_missing_values_in_range(df: pd.DataFrame, logs_to_check: list, flag_col_name: str = 'MISSING_FLAG') -> pd.DataFrame:
    """
    Membuat atau memperbarui kolom penanda ('MISSING_FLAG') untuk nilai null.
    Menghasilkan flag 0 (lengkap) dan 1 (hilang).

    Args:
        df (pd.DataFrame): DataFrame input.
        logs_to_check (list): Daftar nama kolom log yang akan diperiksa.
        flag_col_name (str, optional): Nama untuk kolom penanda.

    Returns:
        pd.DataFrame: DataFrame dengan kolom penanda yang sudah dibuat/diperbarui.
    """
    df_out = df.copy()

    # Inisialisasi kolom flag dengan 0 jika belum ada
    if flag_col_name not in df_out.columns:
        df_out[flag_col_name] = 0

    existing_logs = [col for col in logs_to_check if col in df_out.columns]
    if not existing_logs:
        print("Peringatan: Tidak ada log yang dipilih ditemukan di DataFrame.")
        return df_out

    # Buat masker untuk baris yang memiliki nilai NaN di salah satu log yang diperiksa
    missing_mask = df_out[existing_logs].isnull().any(axis=1)

    # Terapkan flag 1 untuk baris dengan data hilang
    # Hanya ubah yang saat ini bernilai 0 untuk menghindari menimpa flag 2 dari splicing
    df_out.loc[missing_mask & (df_out[flag_col_name] == 0), flag_col_name] = 1

    print(f"--> Penandaan nilai hilang selesai.")
    return df_out


def fill_flagged_missing_values(df: pd.DataFrame, logs_to_fill: list, max_consecutive_nan: int = 3) -> pd.DataFrame:
    """
    Mengisi nilai null pada kolom yang dipilih, TAPI HANYA jika:
    1. Baris tersebut memiliki MISSING_FLAG == 1.
    2. Jumlah nilai null yang berurutan <= max_consecutive_nan.
    """
    df_out = df.copy()

    if 'MISSING_FLAG' not in df_out.columns:
        print("Peringatan: Kolom 'MISSING_FLAG' tidak ditemukan. Tidak ada nilai yang akan diisi.")
        return df_out

    existing_logs = [col for col in logs_to_fill if col in df_out.columns]
    if not existing_logs:
        return df_out

    print(
        f"--> Mengisi nilai hilang untuk baris dengan flag 1 (maks {max_consecutive_nan} NaN berurutan).")

    # Masker dasar: hanya pertimbangkan baris dengan flag 1
    base_fill_mask = (df_out['MISSING_FLAG'] == 1)

    for col in existing_logs:
        is_nan = df_out[col].isnull()
        group_id = (is_nan != is_nan.shift()).cumsum()
        group_size = group_id.map(group_id.value_counts())

        # Masker untuk NaN yang berada dalam grup kecil (boleh diisi)
        nan_in_small_gap_mask = (is_nan) & (group_size <= max_consecutive_nan)

        # Kondisi final: baris harus punya flag 1 DAN merupakan bagian dari celah kecil
        final_col_mask = base_fill_mask & nan_in_small_gap_mask

        if final_col_mask.any():
            temp_col = df_out[col].copy()
            filled_temp_col = temp_col.bfill().ffill()
            df_out.loc[final_col_mask, col] = filled_temp_col[final_col_mask]

    return df_out
