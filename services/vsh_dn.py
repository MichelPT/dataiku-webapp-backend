import pandas as pd
import numpy as np


def calculate_vsh_dn(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Menghitung VSH dari crossplot Density-Neutron.
    """
    df_processed = df.copy()

    # Ekstrak parameter dari frontend, dengan nilai default yang sesuai
    RHO_MA = float(params.get('RHO_MA', 2.645))
    RHO_SH = float(params.get('RHO_SH', 2.61))
    RHO_FL = float(params.get('RHO_FL', 0.85))
    NPHI_MA = float(params.get('NPHI_MA', -0.02))
    NPHI_SH = float(params.get('NPHI_SH', 0.398))
    NPHI_FL = float(params.get('NPHI_FL', 0.85))

    # Ambil nama kolom log dari parameter
    RHO_LOG = params.get('RHOB', 'RHOB')
    NPHI_LOG = params.get('NPHI', 'NPHI')
    VSH_OUTPUT_LOG = params.get('VSH', 'VSH_DN')

    # Pastikan kolom yang dibutuhkan ada
    required_cols = [RHO_LOG, NPHI_LOG]
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Kolom input {required_cols} tidak ditemukan.")

    print("Menghitung VSH dari Density-Neutron...")
    # --- HITUNG VSH DARI DENSITY-NEUTRON ---
    a = (RHO_MA - RHO_FL) * (NPHI_FL - df_processed[NPHI_LOG])
    b = (df_processed[RHO_LOG] - RHO_FL) * (NPHI_FL - NPHI_MA)
    c = (RHO_MA - RHO_FL) * (NPHI_FL - NPHI_SH)
    d = (RHO_SH - RHO_FL) * (NPHI_FL - NPHI_MA)

    denominator = c - d
    if denominator != 0:
        df_processed[VSH_OUTPUT_LOG] = (a - b) / denominator
    else:
        df_processed[VSH_OUTPUT_LOG] = np.nan  # Hindari pembagian dengan nol

    df_processed[VSH_OUTPUT_LOG] = df_processed[VSH_OUTPUT_LOG].clip(0, 1)

    # Opsional: Hitung perbedaan jika VSH_GR sudah ada
    if 'VSH_GR' in df_processed.columns:
        df_processed['VSH_DIFF'] = df_processed['VSH_GR'] - \
            df_processed[VSH_OUTPUT_LOG]

    return df_processed
