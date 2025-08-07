import pandas as pd
import numpy as np


def splice_and_merge_logs(df_run1: pd.DataFrame, df_run2: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Fungsi utama untuk melakukan splicing dan merging dua set data log.
    LOGIKA BARU: Run 1 adalah bagian ATAS, Run 2 adalah bagian BAWAH.

    Args:
        df_run1 (pd.DataFrame): DataFrame untuk data di ATAS splice depth (RUN 1).
        df_run2 (pd.DataFrame): DataFrame untuk data di BAWAH/PADA splice depth (RUN 2).
        params (dict): Dictionary berisi parameter dari frontend, termasuk
                       'SPLICEDEPTH' dan nama-nama kolom log.

    Returns:
        pd.DataFrame: DataFrame hasil splicing yang kolomnya sudah digabungkan.
    """
    try:
        splice_depth = float(params.get('SPLICEDEPTH', 1520.0))
    except (ValueError, TypeError):
        raise ValueError("SPLICEDEPTH harus berupa angka yang valid.")

    print(f"--> Memulai proses splicing pada kedalaman: {splice_depth}")
    print(f"--> LOGIKA BARU: Run 1 (atas), Run 2 (bawah)")

    # --- 1. Splicing ---
    # Pastikan 'DEPTH' ada dan atur sebagai index
    if 'DEPTH' not in df_run1.columns or 'DEPTH' not in df_run2.columns:
        raise KeyError(
            "Kolom 'DEPTH' tidak ditemukan di salah satu DataFrame input.")

    df_run1_indexed = df_run1.set_index('DEPTH')
    df_run2_indexed = df_run2.set_index('DEPTH')

    # Ambil bagian ATAS dari data RUN 1
    upper_part = df_run1_indexed[df_run1_indexed.index < splice_depth]

    # Ambil bagian BAWAH dari data RUN 2
    lower_part = df_run2_indexed[df_run2_indexed.index >= splice_depth]

    # Gabungkan keduanya. Pandas akan secara otomatis menyatukan semua kolom.
    spliced_df = pd.concat([upper_part, lower_part], sort=True)
    spliced_df.sort_index(inplace=True)
    spliced_df.reset_index(inplace=True)
    print("--> Splicing data mentah selesai.")

    # --- 2. Merging (Penggabungan Kurva) ---
    final_df = pd.DataFrame()
    final_df['DEPTH'] = spliced_df['DEPTH']

    # Mapping dari parameter frontend ke pasangan kolom input dan nama kolom output
    curve_mapping = {
        'GR':  (params.get('GR_RUN1'),  params.get('GR_RUN2'),  params.get('GR')),
        'NPHI': (params.get('NPHI_RUN1'), params.get('NPHI_RUN2'), params.get('NPHI')),
        'RHOB': (params.get('RHOB_RUN1'), params.get('RHOB_RUN2'), params.get('RHOB')),
        'RT':  (params.get('RT_RUN1'),  params.get('RT_RUN2'),  params.get('RT')),
    }

    for log_type, (col_run1, col_run2, col_out) in curve_mapping.items():
        if not all([col_run1, col_run2, col_out]):
            print(
                f"Peringatan: Parameter untuk log {log_type} tidak lengkap, melewati...")
            continue

        print(
            f"--> Menggabungkan {log_type}: '{col_run1}' (Run 1 - Atas) dan '{col_run2}' (Run 2 - Bawah) -> '{col_out}'")

        # Ambil data dari kolom Run 1 (atas) dan Run 2 (bawah) jika ada
        series_run1 = spliced_df[col_run1] if col_run1 in spliced_df else pd.Series(
            index=spliced_df.index, dtype=float)
        series_run2 = spliced_df[col_run2] if col_run2 in spliced_df else pd.Series(
            index=spliced_df.index, dtype=float)

        # Logika penggabungan: prioritaskan data dari Run 1 (atas), lalu isi kekosongan dengan Run 2 (bawah)
        final_df[col_out] = series_run1.combine_first(series_run2)

    print("--> Penggabungan semua kurva selesai.")
    return final_df


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
        'GR':  (params.get('GR_RUN1'), params.get('GR_RUN2'), params.get('GR')),
        'NPHI': (params.get('NPHI_RUN1'), params.get('NPHI_RUN2'), params.get('NPHI')),
        'RHOB': (params.get('RHOB_RUN1'), params.get('RHOB_RUN2'), params.get('RHOB')),
        'RT':  (params.get('RT_RUN1'), params.get('RT_RUN2'), params.get('RT')),
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

    # # --- 4. Proses Fill Missing Opsional ---
    # if fill_option:
    #     print("--> Opsi 'fill_missing' aktif. Memanggil fungsi fill_missing_values...")
    #     final_df = fill_flagged_missing_values(
    #         df=final_df,
    #         logs_to_fill=output_cols,
    #         max_consecutive_nan=max_consecutive
    #     )

    return final_df
