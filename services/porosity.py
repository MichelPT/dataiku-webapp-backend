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


def calculate_porosity(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Menghitung berbagai jenis porositas berdasarkan parameter yang diberikan.
    """
    df_processed = df.copy()

    # Ekstrak parameter dengan nilai default
    RHO_FL = params.get('rho_fl', 1.00)
    RHO_W = params.get('rho_w', 1.00)
    RHO_SH = params.get('rho_sh', 2.45)
    RHO_DSH = params.get('rho_dsh', 2.60)
    NPHI_SH = params.get('nphi_sh', 0.35)
    PHIE_MAX = params.get('phie_max', 0.3)
    RHO_MA_BASE = params.get('rho_ma_base', 2.71) * 1000
    RHO_MAX = params.get('rho_max', 4.00) * 1000

    # Pastikan kolom VSH ada, jika tidak, hitung terlebih dahulu
    if 'VSH' not in df_processed.columns:
        # Asumsi VSH_GR adalah representasi VSH utama
        if 'VSH_GR' in df_processed.columns:
            df_processed['VSH'] = df_processed['VSH_GR']
        else:
            raise ValueError("Kolom VSH atau VSH_GR tidak ditemukan.")

    # Perhitungan
    PHIT_SH = (RHO_DSH - RHO_SH) / (RHO_DSH - RHO_W)
    df_processed["RHOB_SR"] = (
        df_processed["RHOB"] - df_processed["VSH"] * RHO_SH) / (1 - df_processed["VSH"])
    df_processed["NPHI_SR"] = (
        df_processed["NPHI"] - df_processed["VSH"] * NPHI_SH) / (1 - df_processed["VSH"])
    df_processed["NPHI_SR"] = df_processed["NPHI_SR"].clip(
        lower=-0.015, upper=1)

    phix_vals, rma_vals = [], []
    for i, row in df_processed.iterrows():
        if pd.notna(row["RHOB_SR"]) and pd.notna(row["NPHI_SR"]):
            phix, rma = dn_xplot(
                row["RHOB_SR"], row["NPHI_SR"], RHO_MA_BASE, RHO_MAX, RHO_FL * 1000)
        else:
            phix, rma = np.nan, np.nan
        phix_vals.append(phix)
        rma_vals.append(rma)

    df_processed["PHIE_DN"] = np.array(phix_vals) * (1 - df_processed["VSH"])
    df_processed["PHIT_DN"] = df_processed["PHIE_DN"] + \
        df_processed["VSH"] * PHIT_SH
    df_processed["PHIE"] = df_processed["PHIE_DN"].clip(
        lower=0, upper=PHIE_MAX * (1 - df_processed["VSH"]))
    df_processed["PHIT"] = df_processed["PHIE"] + df_processed["VSH"] * PHIT_SH
    df_processed["RHO_MAT"] = np.array(rma_vals) / 1000

    df_processed.rename(
        columns={"PHIE_DN": "PHIE_DEN", "PHIT_DN": "PHIT_DEN"}, inplace=True)

    # df_processed["RESERVOIR_CLASS"] = df_processed["PHIE"].apply(
    #     _klasifikasi_reservoir_numeric)

    print("Kolom Porosity baru telah ditambahkan: PHIE, PHIT, PHIE_DN, RHO_MAT")
    return df_processed
