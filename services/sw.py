import pandas as pd
import numpy as np


def newton_simandoux(rt, ff, rwtemp, rtsh, vsh, n, opt='MODIFIED', c=1, max_iter=20, tol=1e-5):
    """
    Newton-Raphson method for solving Simandoux equation for water saturation.

    Args:
        rt: True resistivity
        ff: Formation factor
        rwtemp: Formation water resistivity at formation temperature
        rtsh: Shale resistivity
        vsh: Shale volume
        n: Saturation exponent
        opt: 'MODIFIED' or 'SCHLUMBERGER'
        c: Shale exponent (for Schlumberger)
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Water saturation value
    """
    if opt == 'MODIFIED':
        g1 = 1 / (ff * rwtemp)
        g2 = vsh / rtsh
    else:  # Schlumberger
        g1 = 1 / (ff * rwtemp * (1 - vsh))
        g2 = (vsh ** c) / rtsh

    g3 = -1 / rt
    sw = 0.5  # Initial guess

    for _ in range(max_iter):
        fx = g1 * sw ** n + g2 * sw + g3
        fxp = n * g1 * sw ** (n - 1) + g2

        if fxp == 0:
            return np.nan

        delta = fx / fxp
        sw -= delta
        sw = max(0, sw)  # Ensure sw stays positive

        if abs(delta) < tol:
            return sw

    return np.nan


def calculate_sw_simandoux(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Calculates Water Saturation (Simandoux) with internal filtering.
    """
    df_processed = df.copy()
    df_processed.columns = [col.upper() for col in df_processed.columns]
    df_processed.replace(-999.0, np.nan, inplace=True)

    # 1. Get parameters and check for required columns
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))
    C = float(params.get('C', 1.0))
    RWS = float(params.get('RWS', 0.1))
    RWT = float(params.get('RWT', 75))
    FTEMP = float(params.get('FTEMP', 90))
    SWE_IRR = float(params.get('SWE_IRR', 0.0))
    RT_SH = float(params.get('RT_SH', 5.0))
    OPT_SIM = params.get('OPT_SIM', 'MODIFIED')

    if 'ILD' not in df_processed.columns and 'RT' in df_processed.columns:
        df_processed['ILD'] = df_processed['RT']

    required_cols = ['GR', 'RHOB', 'ILD']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")

    # 2. Create a mask for filtering
    mask = pd.Series(True, index=df_processed.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_processed.columns:
        mask = df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        mask = (mask | zone_mask) if has_filters else zone_mask

    # 3. Prepare data (VSH, PHIE) for the masked rows if they don't exist
    df_processed["RW_TEMP"] = RWS * (RWT + 21.5) / (FTEMP + 21.5)

    if 'VSH' not in df_processed.columns:
        print("Calculating VSH from GR for selected intervals...")
        GR_clean = df_processed.loc[mask, "GR"].quantile(0.05)
        GR_shale = df_processed.loc[mask, "GR"].quantile(0.95)
        if GR_shale > GR_clean:
            vsh_calc = (df_processed.loc[mask, "GR"] -
                        GR_clean) / (GR_shale - GR_clean)
            df_processed.loc[mask, 'VSH'] = vsh_calc.clip(0, 1)
        else:
            df_processed.loc[mask, 'VSH'] = 0

    if 'PHIE' not in df_processed.columns:
        print("Calculating PHIE from density for selected intervals...")
        RHO_MA = float(params.get('RHO_MA', 2.65))
        RHO_SH = float(params.get('RHO_SH', 2.45))
        RHO_FL = float(params.get('RHO_FL', 1.0))
        if (RHO_MA - RHO_FL) != 0:
            phie_calc = ((RHO_MA - df_processed.loc[mask, "RHOB"]) / (RHO_MA - RHO_FL)) - \
                df_processed.loc[mask, "VSH"] * \
                ((RHO_MA - RHO_SH) / (RHO_MA - RHO_FL))
            df_processed.loc[mask, 'PHIE'] = phie_calc.clip(lower=0)
        else:
            df_processed.loc[mask, 'PHIE'] = 0

    # 4. Perform calculations ONLY on the masked rows
    print(
        f"Calculating Simandoux SW for {mask.sum()} of {len(df_processed)} rows.")

    SW_COL = 'SW'
    df_processed[SW_COL] = np.nan
    df_processed['VOL_UWAT'] = np.nan

    # Apply the calculation row-by-row only on the filtered data
    def apply_newton(row):
        if pd.isna(row["ILD"]) or pd.isna(row["PHIE"]) or pd.isna(row["VSH"]):
            return np.nan
        if row["PHIE"] < 0.005:
            return 1.0
        ff = A / (row["PHIE"] ** M)
        return newton_simandoux(
            rt=row["ILD"], ff=ff, rwtemp=row["RW_TEMP"], rtsh=RT_SH,
            vsh=row["VSH"], n=N, opt=OPT_SIM, c=C
        )

    # Use .apply() for a cleaner, though not necessarily faster, implementation on the masked data
    sw_values = df_processed[mask].apply(apply_newton, axis=1)

    # Assign calculated values back to the main DataFrame using the mask
    df_processed.loc[mask, SW_COL] = sw_values

    # Clip the results and calculate final parameters on the masked data
    df_processed.loc[mask, SW_COL] = df_processed.loc[mask,
                                                      SW_COL].clip(lower=SWE_IRR, upper=1.0)
    df_processed.loc[mask, "VOL_UWAT"] = df_processed.loc[mask,
                                                          "PHIE"] * df_processed.loc[mask, SW_COL]

    print("Water Saturation calculation (Simandoux) completed.")
    return df_processed


def calculate_sw(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Fungsi utama untuk menghitung Saturasi Air (SW Indonesia) dengan filter internal.
    """
    df_processed = df.copy()

    # 1. Persiapan: Ekstrak parameter dan verifikasi kolom
    RWS = float(params.get('RWS', 0.529))
    RWT = float(params.get('RWT', 227))
    FTEMP = float(params.get('FTEMP', 80))
    RT_SH = float(params.get('RT_SH', 2.2))
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))

    # Nama kolom
    SW = 'SW'
    VSH = 'VSH'
    PHIE = 'PHIE'
    RT = 'RT'
    RW_TEMP = "RW_TEMP"

    required_cols = ['GR', RT, PHIE, VSH]
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(
            "Kolom input (GR, RT, PHIE, VSH) tidak lengkap. Jalankan modul sebelumnya.")

    # 2. Buat mask untuk memilih baris yang akan diproses
    mask = pd.Series(True, index=df_processed.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_processed.columns:
        mask = df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        mask = (mask | zone_mask) if has_filters else zone_mask

    if not mask.any():
        print("Peringatan: Tidak ada data yang cocok dengan filter. Tidak ada kalkulasi yang dilakukan.")
        return df  # Kembalikan DataFrame asli jika tidak ada yang cocok

    print(f"Menghitung SW untuk {mask.sum()} dari {len(df_processed)} baris.")

    # 3. Lakukan perhitungan HANYA pada baris yang cocok dengan mask

    # Hitung RW_TEMP untuk semua baris karena mungkin dibutuhkan di modul lain
    df_processed[RW_TEMP] = RWS * (RWT + 21.5) / (FTEMP + 21.5)

    # Ambil data yang sudah difilter untuk kalkulasi
    v = df_processed.loc[mask, VSH] ** 2
    phie_masked = df_processed.loc[mask, PHIE]
    ff = A / phie_masked ** M

    ff_times_rw_temp = ff * df_processed.loc[mask, RW_TEMP]
    # Hindari pembagian dengan nol
    ff_times_rw_temp[ff_times_rw_temp == 0] = np.nan

    f1 = 1 / ff_times_rw_temp
    f2 = 2 * np.sqrt(v / (ff_times_rw_temp * RT_SH))
    f3 = v / RT_SH

    denom = f1 + f2 + f3
    denom[denom == 0] = np.nan

    # Hitung SW dan terapkan pada baris yang difilter
    sw_calculated = (1 / (df_processed.loc[mask, RT] * denom)) ** (1 / N)

    # Inisialisasi kolom SW jika belum ada
    if SW not in df_processed.columns:
        df_processed[SW] = np.nan

    df_processed.loc[mask, SW] = sw_calculated

    # Terapkan kondisi dan batasan pada kolom SW yang sudah diupdate
    df_processed.loc[df_processed[PHIE] < 0.005, SW] = 1.0
    df_processed[SW] = df_processed[SW].clip(lower=0, upper=1)

    return df_processed
