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
    Menghitung berbagai jenis porositas berdasarkan parameter yang diberikan,
    dengan opsi untuk membatasi perhitungan pada interval atau zona tertentu.
    """
    df_processed = df.copy()

    # --- BAGIAN BARU: Membuat mask untuk memilih baris target ---
    mask = pd.Series(False, index=df_processed.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_processed.columns:
        mask |= df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        mask |= df_processed['ZONE'].isin(target_zones)
        has_filters = True
    if not has_filters:
        mask = pd.Series(True, index=df_processed.index)

    if not mask.any():
        print("Peringatan: Tidak ada baris yang cocok dengan filter interval/zona yang dipilih.")
        return df_processed
    # --- AKHIR BAGIAN BARU ---

    # Ekstrak parameter dengan nilai default
    RHO_FL = params.get('rho_fl', 1.00)
    RHO_W = params.get('rho_w', 1.00)
    RHO_SH = params.get('rho_sh', 2.45)
    RHO_DSH = params.get('rho_dsh', 2.60)
    NPHI_SH = params.get('nphi_sh', 0.35)
    PHIE_MAX = params.get('phie_max', 0.3)
    RHO_MA_BASE = params.get('rho_ma_base', 2.71) * 1000
    RHO_MAX = params.get('rho_max', 4.00) * 1000

    # Pastikan kolom input utama ada
    required_cols = ['RHOB', 'NPHI']
    for col in required_cols:
        if col not in df_processed.columns:
            raise ValueError(
                f"Kolom input yang dibutuhkan '{col}' tidak ditemukan.")

    # Pastikan kolom VSH ada, jika tidak, coba gunakan VSH_GR
    if 'VSH' not in df_processed.columns:
        if 'VSH_GR' in df_processed.columns:
            df_processed['VSH'] = df_processed['VSH_GR']
        else:
            raise ValueError(
                "Kolom VSH atau VSH_GR tidak ditemukan untuk perhitungan porositas.")

    # Inisialisasi kolom output jika belum ada
    output_cols = ["RHOB_SR", "NPHI_SR", "PHIE_DEN",
                   "PHIT_DEN", "PHIE", "PHIT", "RHO_MAT"]
    for col in output_cols:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # --- PERHITUNGAN VEKTORISASI PADA BARIS TERPILIH (MASK) ---
    vsh_masked = df_processed.loc[mask, "VSH"]

    # Perhitungan awal
    PHIT_SH = (RHO_DSH - RHO_SH) / (RHO_DSH - RHO_W)
    df_processed.loc[mask, "RHOB_SR"] = (
        df_processed.loc[mask, "RHOB"] - vsh_masked * RHO_SH) / (1 - vsh_masked)
    df_processed.loc[mask, "NPHI_SR"] = (
        df_processed.loc[mask, "NPHI"] - vsh_masked * NPHI_SH) / (1 - vsh_masked)
    df_processed.loc[mask, "NPHI_SR"] = df_processed.loc[mask,
                                                         "NPHI_SR"].clip(lower=-0.015, upper=1)

    # Jalankan fungsi dn_xplot yang kompleks hanya pada baris yang relevan
    target_rows = df_processed.loc[mask].copy()

    # Pastikan kolom yang dibutuhkan tidak NaN sebelum apply
    valid_rows_mask = target_rows["RHOB_SR"].notna(
    ) & target_rows["NPHI_SR"].notna()

    if valid_rows_mask.any():
        results = target_rows[valid_rows_mask].apply(
            lambda row: dn_xplot(
                row["RHOB_SR"], row["NPHI_SR"], RHO_MA_BASE, RHO_MAX, RHO_FL * 1000
            ),
            axis=1
        )
        # Pisahkan hasil tuple (phix, rma) ke dalam dua kolom
        phix_vals, rma_vals = zip(*results)
        target_rows.loc[valid_rows_mask, 'phix_temp'] = phix_vals
        target_rows.loc[valid_rows_mask, 'rma_temp'] = rma_vals

        # Perhitungan akhir berdasarkan hasil dn_xplot
        target_rows["PHIE_DEN"] = np.array(
            target_rows['phix_temp']) * (1 - target_rows["VSH"])
        target_rows["PHIT_DEN"] = target_rows["PHIE_DEN"] + \
            target_rows["VSH"] * PHIT_SH
        target_rows["PHIE"] = target_rows["PHIE_DEN"].clip(
            lower=0, upper=PHIE_MAX * (1 - target_rows["VSH"]))
        target_rows["PHIT"] = target_rows["PHIE"] + \
            target_rows["VSH"] * PHIT_SH
        target_rows["RHO_MAT"] = np.array(target_rows['rma_temp']) / 1000

        # Update DataFrame utama dengan hasil perhitungan
        update_cols = ["RHOB_SR", "NPHI_SR", "PHIE_DEN",
                       "PHIT_DEN", "PHIE", "PHIT", "RHO_MAT"]
        df_processed.update(target_rows[update_cols])

    print("Kolom Porositas baru telah ditambahkan/diperbarui: PHIE, PHIT, dll.")
    return df_processed
