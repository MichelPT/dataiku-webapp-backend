import pandas as pd
import numpy as np


def dn_xplot(rho0, nphi0, rho_ma, rho_max, rho_fl):
    """(Internal) Menghitung porositas dan densitas matriks dari crossplot D-N."""
    # Fungsi ini tetap sama, karena logikanya per baris.
    try:
        phid = (rho_ma - rho0 * 1000) / (rho_ma - rho_fl)
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

        rma = (rho0 * 1000 - phix * rho_fl) / (1 - phix)
        return phix, rma
    except (ValueError, TypeError, ZeroDivisionError):
        return np.nan, np.nan


def _klasifikasi_reservoir_numeric(phie):
    """(Internal) Memberikan KODE kelas reservoir berdasarkan nilai PHIE."""
    # Fungsi ini tetap sama.
    if pd.isna(phie):
        return 0  # NoData
    elif phie >= 0.20:
        return 4  # Prospek Kuat
    elif phie >= 0.15:
        return 3  # Zona Menarik
    elif phie >= 0.10:
        return 2  # Zona Lemah
    else:
        return 1  # Non Prospek

def calculate_porosity(
    df: pd.DataFrame,
    params: dict,
    target_intervals: list = None,
    target_zones: list = None
) -> pd.DataFrame:
    """
    Menghitung porositas dengan penanganan kondisi khusus (VSH >= 0.95) sesuai Loglan.
    """
    df_processed = df.copy()

    # --- Ensure numeric types ---
    for col in ["VSH_LINEAR", "RHOB", "NPHI"]:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")


    # --- Masking untuk filter interval/zona (tidak berubah) ---
    mask = pd.Series(True, index=df_processed.index)
    if target_intervals and 'MARKER' in df_processed.columns:
        mask &= df_processed['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_processed.columns:
        mask &= df_processed['ZONE'].isin(target_zones)

    if not mask.any():
        print("Peringatan: Tidak ada baris yang cocok dengan filter.")
        return df_processed

    # --- Ekstraksi parameter dan validasi (tidak berubah) ---
    RHO_FL = params.get('rhob_fl', 1.00)
    RHO_W = params.get('rhob_w', 1.00)
    RHO_SH = params.get('rhob_sh', 2.45)
    RHO_DSH = params.get('rhob_dsh', 2.60)
    NPHI_SH = params.get('nphi_sh', 0.35)
    PHIE_MAX = params.get('phie_max', 0.3)
    RHO_MA_BASE = params.get('rhob_ma_base', 2.71) * 1000
    RHO_MAX = params.get('rhob_max', 4.00) * 1000

    required_cols = ['RHOB', 'NPHI']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Kolom input {required_cols} tidak ditemukan.")

    if 'VSH_LINEAR' not in df_processed.columns:
        df_processed['_LINEAR'] = df_processed.get('VSH_LINEAR', 'VSH_DN')
    if 'VSH_LINEAR' not in df_processed.columns:
        raise ValueError("Kolom VSH tidak ditemukan.")

    output_cols = ["RHOB_SR", "NPHI_SR", "PHIE_DEN",
                   "PHIT_DEN", "PHIE", "PHIT", "RHO_MAT"]
    for col in output_cols:
        df_processed[col] = np.nan

    # --- PERBAIKAN: PENANGANAN KONDISI KHUSUS SESUAI LOGLAN ---

    # Hitung PHIT_SH terlebih dahulu, karena dibutuhkan di beberapa kondisi
    PHIT_SH = (RHO_DSH - RHO_SH) / (RHO_DSH -
                                    RHO_W) if (RHO_DSH - RHO_W) != 0 else np.nan

    # 1. Buat mask untuk kondisi serpih murni (VSH >= 0.95)
    high_shale_mask = (df_processed['VSH_LINEAR'] >= 0.95) & mask

    # Terapkan nilai default untuk zona serpih murni
    if high_shale_mask.any():
        print(
            f"Menerapkan nilai default untuk {high_shale_mask.sum()} baris serpih murni (VSH >= 0.95).")
        df_processed.loc[high_shale_mask, 'PHIE_DEN'] = 0
        df_processed.loc[high_shale_mask, 'PHIE'] = 0
        df_processed.loc[high_shale_mask, 'PHIT_DEN'] = PHIT_SH
        df_processed.loc[high_shale_mask, 'PHIT'] = PHIT_SH
        df_processed.loc[high_shale_mask,
                         'RHO_MAT'] = np.nan  # Tidak ada matriks

    # 2. Buat mask untuk perhitungan normal (bukan serpih murni)
    normal_calc_mask = (df_processed['VSH_LINEAR'] < 0.95) & mask

    if not normal_calc_mask.any():
        print("Tidak ada baris untuk perhitungan normal, semua adalah serpih murni.")
        return df_processed

    print(
        f"Menjalankan perhitungan crossplot untuk {normal_calc_mask.sum()} baris normal.")

    # --- Perhitungan normal hanya dijalankan pada mask yang sesuai ---
    vsh_masked = df_processed.loc[normal_calc_mask, "VSH_LINEAR"]
    denominator = 1 - vsh_masked

    # Hitung RHOB_SR & NPHI_SR yang aman
    rhob_sr_num = df_processed.loc[normal_calc_mask,
                                   "RHOB"] - vsh_masked * RHO_SH
    df_processed.loc[normal_calc_mask, "RHOB_SR"] = np.where(
        np.isclose(denominator, 0), np.nan, rhob_sr_num / denominator)

    nphi_sr_num = df_processed.loc[normal_calc_mask,
                                   "NPHI"] - vsh_masked * NPHI_SH
    df_processed.loc[normal_calc_mask, "NPHI_SR"] = np.where(
        np.isclose(denominator, 0), np.nan, nphi_sr_num / denominator)

    df_processed.loc[normal_calc_mask, "NPHI_SR"] = df_processed.loc[normal_calc_mask, "NPHI_SR"].clip(
        lower=-0.015, upper=1)

    # Lanjutkan dengan dn_xplot hanya pada baris normal
    target_rows = df_processed.loc[normal_calc_mask].copy()
    valid_rows_mask = target_rows["RHOB_SR"].notna(
    ) & target_rows["NPHI_SR"].notna()

    if valid_rows_mask.any():
        results = target_rows[valid_rows_mask].apply(
            lambda row: dn_xplot(
                row["RHOB_SR"], row["NPHI_SR"], RHO_MA_BASE, RHO_MAX, RHO_FL * 1000),
            axis=1
        )
        phix_vals, rma_vals = zip(*results)
        target_rows.loc[valid_rows_mask, 'phix_temp'] = phix_vals
        target_rows.loc[valid_rows_mask, 'rma_temp'] = rma_vals

        # Hitung porositas untuk baris normal
        target_rows["PHIE_DEN"] = np.array(
            target_rows['phix_temp']) * (1 - target_rows["VSH_LINEAR"])
        target_rows["PHIT_DEN"] = target_rows["PHIE_DEN"] + \
            target_rows["VSH_LINEAR"] * PHIT_SH
        target_rows["PHIE"] = target_rows["PHIE_DEN"].clip(
            lower=0, upper=PHIE_MAX * (1 - target_rows["VSH_LINEAR"]))
        target_rows["PHIT"] = target_rows["PHIE"] + \
            target_rows["VSH_LINEAR"] * PHIT_SH
        target_rows["RHO_MAT"] = np.array(target_rows['rma_temp']) / 1000

        # Update DataFrame utama HANYA untuk baris normal
        df_processed.update(target_rows[output_cols])

    print("Kolom Porositas baru telah ditambahkan/diperbarui.")
    return df_processed