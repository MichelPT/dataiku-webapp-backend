# /api/modules/data_utils.py
import os
from flask import jsonify
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
    Geolog-style MIN-MAX normalization using percentiles
    """
    log = np.array(log_in, dtype=float)

    if cutoff_min is not None:
        log[log < cutoff_min] = np.nan
    if cutoff_max is not None:
        log[log > cutoff_max] = np.nan

    # low_in = np.nanpercentile(log, pct_min)
    # high_in = np.nanpercentile(log, pct_max)

    # Hindari pembagian dengan nol jika semua data sama
    if high_in == low_in:
        return np.full_like(log, low_ref)

    m = (high_ref - low_ref) / (high_in - low_in)
    log_out = low_ref + m * (log - low_in)

    return log_out


def selective_normalize_handler(df, log_column, marker_column,
                                target_markers=None,
                                low_ref=40, high_ref=140,
                                low_in=5, high_in=95,
                                cutoff_min=0, cutoff_max=250):

    # Copy DataFrame untuk menghindari modifikasi original
    result_df = df.copy()

    # Ambil data log asli
    log_data = result_df[log_column].values

    # Jika target_markers tidak didefinisikan, gunakan semua marker
    if target_markers is None:
        target_markers = result_df[marker_column].dropna().unique()

    # Inisialisasi log_raw: data asli untuk marker yang TIDAK dipilih
    target_mask = result_df[marker_column].isin(target_markers)
    log_raw = log_data.copy()
    log_raw[target_mask] = np.nan  # Set NaN untuk marker yang dipilih

    # Inisialisasi log_norm dengan NaN
    log_norm = np.full_like(log_data, np.nan, dtype=float)

    # Inisialisasi log_raw_norm dengan log_raw
    log_raw_norm = log_raw.copy()

    # Normalisasi hanya untuk target markers menggunakan function asli
    if target_markers:
        target_data = log_data[target_mask]

        # Hanya lakukan normalisasi jika ada data yang valid
        if len(target_data) > 0 and not np.all(np.isnan(target_data)):

            # PANGGIL FUNCTION ASLI min_max_normalize
            normalized_target = min_max_normalize(target_data,
                                                  low_ref=low_ref,
                                                  high_ref=high_ref,
                                                  low_in=low_in,
                                                  high_in=high_in,
                                                  cutoff_min=cutoff_min,
                                                  cutoff_max=cutoff_max)

            # Masukkan hasil normalisasi ke log_norm
            log_norm[target_mask] = normalized_target

            # Overwrite log_raw_norm dengan data normalisasi untuk target markers
            log_raw_norm[target_mask] = normalized_target

    # Tambahkan kolom hasil ke DataFrame
    result_df[f'{log_column}_RAW'] = log_raw
    result_df[f'{log_column}_NORM'] = log_norm
    result_df[f'{log_column}_RAW_NORM'] = log_raw_norm

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


def smoothing(df):
    df_smooth = df.copy()
    df_smooth["GR_MovingAvg_5"] = df["GR"].rolling(
        window=5, center=True).mean()
    df_smooth["GR_MovingAvg_10"] = df["GR"].rolling(
        window=10, center=True).mean()
    return df_smooth
