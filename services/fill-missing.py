import pandas as pd
import numpy as np

from services.data_processing import fill_flagged_missing_values


def splice_and_flag_logs(df_run1: pd.DataFrame, df_run2: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Fungsi lengkap untuk splicing, merging, dan flagging.
    Fungsi ini sekarang juga memanggil fill_missing_values secara internal
    jika opsi yang sesuai diaktifkan di 'params'.
    """
    try:
        splice_depth = float(params.get('SPLICEDEPTH', 1520.0))
        # Ambil opsi baru dari parameter
        fill_option = params.get('fill_missing', False)
        max_consecutive = int(params.get('max_consecutive_nan', 3))
    except (ValueError, TypeError):
        raise ValueError(
            "Parameter SPLICEDEPTH atau max_consecutive_nan tidak valid.")

    # --- 1. Splicing & Penentuan Batas Gap ---
    if 'DEPTH' not in df_run1.columns or 'DEPTH' not in df_run2.columns:
        raise KeyError("Kolom 'DEPTH' tidak ditemukan.")

    df_run1_indexed = df_run1.set_index('DEPTH')
    df_run2_indexed = df_run2.set_index('DEPTH')
    upper_part = df_run1_indexed[df_run1_indexed.index < splice_depth]
    lower_part = df_run2_indexed[df_run2_indexed.index >= splice_depth]
    max_depth_upper = upper_part.index.max()
    min_depth_bottom = lower_part.index.min()
    spliced_df = pd.concat([upper_part, lower_part], sort=True).reset_index()

    # --- 2. Merging Kurva ---
    final_df = pd.DataFrame({'DEPTH': spliced_df['DEPTH']})
    output_cols = []
    curve_mapping = {
        'GR':  (params.get('GR_RUN1'), params.get('GR_RUN2'), params.get('GR_SPL')),
        'NPHI': (params.get('NPHI_RUN1'), params.get('NPHI_RUN2'), params.get('NPHI_SPL')),
        'RHOB': (params.get('RHOB_RUN1'), params.get('RHOB_RUN2'), params.get('RHOB_SPL')),
        'RT':  (params.get('RT_RUN1'), params.get('RT_RUN2'), params.get('RT_SPL')),
    }
    for _, (col_run1, col_run2, col_out) in curve_mapping.items():
        if not all([col_run1, col_run2, col_out]):
            continue
        output_cols.append(col_out)
        series_run1 = spliced_df[col_run1] if col_run1 in spliced_df else pd.Series(
            dtype=float)
        series_run2 = spliced_df[col_run2] if col_run2 in spliced_df else pd.Series(
            dtype=float)
        final_df[col_out] = series_run1.combine_first(series_run2)

    # --- 3. Proses Flagging Tiga Tingkat ---
    final_df['MISSING_FLAG'] = 0
    if pd.notna(max_depth_upper) and pd.notna(min_depth_bottom) and min_depth_bottom > max_depth_upper:
        gap_mask = (final_df['DEPTH'] > max_depth_upper) & (
            final_df['DEPTH'] < min_depth_bottom)
        final_df.loc[gap_mask, 'MISSING_FLAG'] = 2

    not_flagged_as_gap = final_df['MISSING_FLAG'] == 0
    missing_in_merged_logs = final_df[output_cols].isnull().any(axis=1)
    final_flag_1_mask = not_flagged_as_gap & missing_in_merged_logs
    final_df.loc[final_flag_1_mask, 'MISSING_FLAG'] = 1

    # --- 4. Proses Fill Missing Opsional ---
    if fill_option:
        print("--> Opsi 'fill_missing' aktif. Memanggil fungsi fill_missing_values...")
        final_df = fill_flagged_missing_values(
            df=final_df,
            logs_to_fill=output_cols,
            max_consecutive_nan=max_consecutive
        )

    return final_df
