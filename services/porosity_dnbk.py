import pandas as pd
import numpy as np

# Fungsi ini tidak perlu diubah
def dn_xplot(rho0, nphi0, rho_ma, rho_max, rho_fl):
    """(Internal) Menghitung porositas dan densitas matriks dari crossplot D-N."""
    try:
        # Konversi rho0 dari g/cc ke kg/m3 jika perlu (asumsi input rho0 adalah g/cc)
        rho0_kgm3 = rho0 * 1000
        
        phid = (rho_ma - rho0_kgm3) / (rho_ma - rho_fl)
        
        if nphi0 >= phid:
            pda = (rho_ma - rho_max) / (rho_ma - rho_fl)
            pna = 0.7 - 10 ** (-5 * nphi0 - 0.16)
        else:
            pda = 1.0
            pna = -2.06 * nphi0 - 1.17 + 10 ** (-16 * nphi0 - 0.4)

        denom = pda - pna
        if np.isclose(denom, 0) or np.isnan(denom):
            return np.nan, np.nan

        phix = (pda * nphi0 - phid * pna) / denom
        if np.isclose(1 - phix, 0) or np.isnan(phix):
            return np.nan, np.nan

        # rma akan dalam kg/m3
        rma = (rho0_kgm3 - phix * rho_fl) / (1 - phix)
        return phix, rma
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
        return np.nan, np.nan

# Fungsi ini tidak perlu diubah
def _klasifikasi_reservoir_numeric(phie):
    """(Internal) Memberikan KODE kelas reservoir berdasarkan nilai PHIE."""
    if pd.isna(phie):
        return 0
    elif phie >= 0.20:
        return 4
    elif phie >= 0.15:
        return 3
    elif phie >= 0.10:
        return 2
    else:
        return 1

# ==============================================================================
# FUNGSI UTAMA YANG DIPERBAIKI
# ==============================================================================
def calculate_porosity_loglan(
    df: pd.DataFrame,
    params: dict,
    target_intervals: list = None,
    target_zones: list = None
) -> pd.DataFrame:
    """
    Menghitung porositas menggunakan alur logika yang diadaptasi dari pseudocode Loglan.
    Fungsi ini melakukan dua kali perhitungan crossplot:
    1. Pada log mentah untuk mendapatkan Apparent Matrix Density (RHO_MAA).
    2. Pada log yang dikoreksi serpih untuk mendapatkan True Matrix Density (RHO_MAT) dan PHIE.
    """
    df_processed = df.copy()

    # --- 1. Persiapan dan Ekstraksi Parameter ---
    
    # Pastikan tipe data numerik
    for col in ["VSH", "RHOB", "NPHI"]:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    # Ekstraksi parameter dari dictionary
    RHO_FL = params.get('rhob_fl', 1.0) * 1000 # Konversi ke kg/m3
    RHO_W = params.get('rhob_w', 1.0)   # g/cc
    RHO_SH = params.get('rhob_sh', 2.45) # g/cc
    RHO_DSH = params.get('rhob_dsh', 2.60) # g/cc
    NPHI_SH = params.get('nphi_sh', 0.35)
    PHIE_MAX = params.get('phie_max', 0.3)
    RHO_MA_BASE = params.get('rhob_ma_base', 2.71) * 1000 # kg/m3 (e.g., Sandstone)
    RHO_MAX = params.get('rhob_max', 4.0) * 1000 # kg/m3

    # Validasi kolom input
    required_cols = ['RHOB', 'NPHI', 'VSH']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Kolom input yang dibutuhkan ({required_cols}) tidak ditemukan.")
    
    # Inisialisasi semua kolom output sesuai Loglan
    output_cols = [
        "MTH_PHI", "PHIT_DN", "PHIT", "PHIE_DN", "PHIE", 
        "RHO_MAT", "RHO_MAA", "RHOB_SR", "NPHI_SR",
        "VOL_DRYSHL", "VOL_SHLWAT"
    ]
    for col in output_cols:
        df_processed[col] = np.nan
        if col == "MTH_PHI":
             df_processed[col] = "" # Inisialisasi sebagai string kosong

    # --- 2. Perhitungan Awal (di luar loop utama) ---
    
    # Hitung shale porosity (PHIT_SH)
    phit_sh_denom = RHO_DSH - RHO_W
    PHIT_SH = (RHO_DSH - RHO_SH) / phit_sh_denom if phit_sh_denom != 0 else np.nan
    
    # Masking untuk filter interval/zona
    mask = pd.Series(True, index=df_processed.index)
    if target_intervals and 'MARKER' in df_processed.columns:
        mask &= df_processed['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_processed.columns:
        mask &= df_processed['ZONE'].isin(target_zones)

    # --- 3. Alur Logika Utama (Shale vs. Normal) ---
    
    # Kondisi 1: Zona Serpih Murni (VSH >= 0.95)
    high_shale_mask = (df_processed['VSH'] >= 0.95) & mask
    if high_shale_mask.any():
        df_processed.loc[high_shale_mask, 'MTH_PHI'] = 'SHALE'
        df_processed.loc[high_shale_mask, 'PHIT_DN'] = PHIT_SH
        df_processed.loc[high_shale_mask, 'PHIT'] = PHIT_SH
        df_processed.loc[high_shale_mask, 'PHIE_DN'] = 0
        df_processed.loc[high_shale_mask, 'PHIE'] = 0
        # RHO_MAT dan RHO_MAA tetap NaN (Missing)

    # Kondisi 2: Zona Perhitungan Normal (VSH < 0.95)
    normal_calc_mask = (df_processed['VSH'] < 0.95) & mask
    if normal_calc_mask.any():
        df_processed.loc[normal_calc_mask, 'MTH_PHI'] = 'DEN/NEUT'
        
        # --- Langkah-langkah perhitungan untuk zona normal ---
        subset_df = df_processed.loc[normal_calc_mask].copy()

        # a. Hitung Apparent Matrix Density (RHO_MAA) dari log mentah
        results_maa = subset_df.apply(
            lambda row: dn_xplot(row["RHOB"], row["NPHI"], RHO_MA_BASE, RHO_MAX, RHO_FL),
            axis=1, result_type='expand'
        )
        # Ambil hanya rma (indeks 1) dan konversi ke g/cc
        df_processed.loc[normal_calc_mask, 'RHO_MAA'] = results_maa[1] / 1000

        # b. Koreksi log untuk efek serpih (Shale Reduced Logs)
        vsh = subset_df["VSH"]
        denom = 1 - vsh
        # Hindari pembagian dengan nol
        rhob_sr = np.where(np.isclose(denom, 0), np.nan, (subset_df["RHOB"] - vsh * RHO_SH) / denom)
        nphi_sr = np.where(np.isclose(denom, 0), np.nan, (subset_df["NPHI"] - vsh * NPHI_SH) / denom)
        nphi_sr = np.clip(nphi_sr, -0.015, 1.0)
        
        df_processed.loc[normal_calc_mask, 'RHOB_SR'] = rhob_sr
        df_processed.loc[normal_calc_mask, 'NPHI_SR'] = nphi_sr
        
        # c. Hitung True Matrix Density (RHO_MAT) dan Porositas (phix) dari log terkoreksi
        subset_df['RHOB_SR_TEMP'] = rhob_sr
        subset_df['NPHI_SR_TEMP'] = nphi_sr
        
        results_mat = subset_df.apply(
            lambda row: dn_xplot(row["RHOB_SR_TEMP"], row["NPHI_SR_TEMP"], RHO_MA_BASE, RHO_MAX, RHO_FL),
            axis=1, result_type='expand'
        )
        phix = results_mat[0]
        rma = results_mat[1] # dalam kg/m3

        # d. Hitung PHIE_DN dan PHIT_DN
        phie_dn = phix * (1 - vsh)
        phit_dn = phie_dn + (vsh * PHIT_SH)

        # e. Limit porositas untuk mendapatkan PHIE dan PHIT final
        # Opsi 'SHALE_REDUCED' dari Loglan digunakan sebagai default
        phie_lim = PHIE_MAX * (1 - vsh)
        phie = np.clip(phie_dn, 0, phie_lim)
        phit = phie + (vsh * PHIT_SH)

        # f. Update kolom-kolom di DataFrame utama
        df_processed.loc[normal_calc_mask, 'RHO_MAT'] = rma / 1000 # konversi ke g/cc
        df_processed.loc[normal_calc_mask, 'PHIE_DN'] = phie_dn
        df_processed.loc[normal_calc_mask, 'PHIT_DN'] = phit_dn
        df_processed.loc[normal_calc_mask, 'PHIE'] = phie
        df_processed.loc[normal_calc_mask, 'PHIT'] = phit
        
    # --- 4. Perhitungan Akhir (setelah semua kondisi diproses) ---
    df_processed['VOL_DRYSHL'] = df_processed['VSH'] * (1 - PHIT_SH)
    df_processed['VOL_SHLWAT'] = df_processed['VSH'] * PHIT_SH

    print("Perhitungan porositas sesuai alur Loglan selesai.")
    return df_processed