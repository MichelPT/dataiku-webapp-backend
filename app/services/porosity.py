# File: porosity.py
import pandas as pd
import numpy as np


def dn_xplot(rho0, nphi0, rho_ma, rho_max, rho_fl):
    """(Internal) Menghitung porositas dan densitas matriks dari crossplot D-N."""
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
    except Exception:
        return np.nan, np.nan


def _klasifikasi_reservoir_numeric(phie):
    """(Internal) Memberikan KODE kelas reservoir berdasarkan nilai PHIE."""
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


def calculate_porosity(df: pd.DataFrame, params: dict,
                       marker_column: str = 'MARKER',
                       target_markers: list = None) -> pd.DataFrame:
    """
    Menghitung porositas berdasarkan parameter, hanya pada marker tertentu jika diberikan.
    Nilai sebelumnya akan dipertahankan di marker lain.
    """
    df_processed = df.copy()

    # Ekstrak parameter
    RHO_FL = params.get('RHO_FL', 1.00)
    RHO_W = params.get('RHO_W', 1.00)
    RHO_SH = params.get('RHO_SH', 2.45)
    RHO_DSH = params.get('RHO_DSH', 2.60)
    NPHI_SH = params.get('NPHI_SH', 0.35)
    PHIE_MAX = params.get('PHIE_MAX', 0.3)
    RHO_MA_BASE = params.get('RHO_MA_BASE', 2.71) * 1000
    RHO_MAX = params.get('RHO_MAX', 4.00) * 1000

    if 'VSH' not in df_processed.columns:
        if 'VSH_GR' in df_processed.columns:
            df_processed['VSH'] = df_processed['VSH_GR']
        else:
            raise ValueError("Kolom VSH atau VSH_GR tidak ditemukan.")

    # Buat masker berdasarkan marker
    if target_markers and marker_column in df_processed.columns:
        mask = df_processed[marker_column].isin(target_markers)
    else:
        mask = np.ones(len(df_processed), dtype=bool)

    # Inisialisasi/pertahankan nilai lama untuk kolom hasil
    def keep_or_init(col, default=np.nan):
        return df_processed[col].copy() if col in df_processed else np.full(len(df_processed), default)

    PHIE_DEN = keep_or_init('PHIE_DEN')
    PHIT_DEN = keep_or_init('PHIT_DEN')
    PHIE = keep_or_init('PHIE')
    PHIT = keep_or_init('PHIT')
    RHO_MAT = keep_or_init('RHO_MAT')
    RES_CLASS = df_processed['RESERVOIR_CLASS'].copy(
    ) if 'RESERVOIR_CLASS' in df_processed else pd.Series([None]*len(df_processed))

    VSH = df_processed['VSH'].clip(0, 1).values
    RHOB = df_processed['RHOB'].values
    NPHI = df_processed['NPHI'].values

    RHOB_SR = (RHOB - VSH * RHO_SH) / (1 - VSH)
    NPHI_SR = (NPHI - VSH * NPHI_SH) / (1 - VSH)
    NPHI_SR = np.clip(NPHI_SR, -0.015, 1)

    PHIT_SH = (RHO_DSH - RHO_SH) / (RHO_DSH - RHO_W)

    # Perhitungan hanya untuk baris dengan mask = True
    for i in np.where(mask)[0]:
        if not np.isnan(RHOB_SR[i]) and not np.isnan(NPHI_SR[i]):
            phix, rma = dn_xplot(
                RHOB_SR[i], NPHI_SR[i], RHO_MA_BASE, RHO_MAX, RHO_FL * 1000)
        else:
            phix, rma = np.nan, np.nan

        PHIE_DEN[i] = phix * (1 - VSH[i]) if not np.isnan(phix) else np.nan
        PHIT_DEN[i] = PHIE_DEN[i] + VSH[i] * \
            PHIT_SH if not np.isnan(PHIE_DEN[i]) else np.nan
        PHIE[i] = np.clip(PHIE_DEN[i], 0, PHIE_MAX * (1 - VSH[i])
                          ) if not np.isnan(PHIE_DEN[i]) else np.nan
        PHIT[i] = PHIE[i] + VSH[i] * \
            PHIT_SH if not np.isnan(PHIE[i]) else np.nan
        RHO_MAT[i] = rma / 1000 if not np.isnan(rma) else np.nan
        RES_CLASS[i] = _klasifikasi_reservoir_numeric(
            PHIE[i]) if not np.isnan(PHIE[i]) else None

    # Masukkan kembali ke DataFrame
    df_processed["PHIE_DEN"] = PHIE_DEN
    df_processed["PHIT_DEN"] = PHIT_DEN
    df_processed["PHIE"] = PHIE
    df_processed["PHIT"] = PHIT
    df_processed["RHO_MAT"] = RHO_MAT
    df_processed["RESERVOIR_CLASS"] = RES_CLASS

    print(f"Porosity dihitung pada marker: {target_markers}")
    return df_processed
