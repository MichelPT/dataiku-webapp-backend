import pandas as pd
import numpy as np


def calculate_vsh_from_gr(df: pd.DataFrame, gr_log: str, gr_ma: float, gr_sh: float, output_col: str,
                          marker_column: str = 'MARKER', target_markers: list = None) -> pd.DataFrame:
    """
    Menghitung VSH dari Gamma Ray, terbatas pada marker tertentu jika diberikan.

    Args:
        df (pd.DataFrame): DataFrame input log.
        gr_log (str): Nama kolom GR.
        gr_ma (float): Nilai GR matrix.
        gr_sh (float): Nilai GR shale.
        output_col (str): Nama kolom hasil.
        marker_column (str): Nama kolom marker.
        target_markers (list, optional): Daftar marker yang ingin dihitung.

    Returns:
        pd.DataFrame: DataFrame hasil dengan kolom VSH.
    """

    if gr_log not in df.columns:
        print(f"⚠️ Kolom {gr_log} tidak ditemukan.")
        df_processed[output_col] = np.nan
        return df_processed

    df_processed = df[df[gr_log] != -999.0].copy()

    # Ambil data GR
    gr_values = df_processed[gr_log].values

    # Jika output_col sudah ada, mulai dari sana; jika tidak, inisialisasi NaN
    if output_col in df_processed.columns:
        vsh_values = df_processed[output_col].values.copy()
    else:
        vsh_values = np.full_like(gr_values, np.nan, dtype=np.float64)

    # Mask untuk baris yang akan dihitung (berdasarkan marker)
    if target_markers is not None and marker_column in df.columns:
        marker_mask = df_processed[marker_column].isin(target_markers)
    else:
        marker_mask = np.ones(len(df_processed), dtype=bool)  # Semua True

    v_gr = (gr_values - gr_ma) / (gr_sh - gr_ma)
    v_gr_clipped = np.clip(v_gr, 0, 1)

    vsh_values[marker_mask] = v_gr_clipped[marker_mask]
    df_processed[output_col] = vsh_values

    return df_processed
