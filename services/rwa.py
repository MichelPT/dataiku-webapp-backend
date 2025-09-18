import pandas as pd
import numpy as np


def calculate_ftemp(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    if 'TVDSS' not in df_processed.columns:
        raise ValueError("Input DataFrame must contain a 'TVDSS' column.")
    df_processed['FTEMP'] = 75 + (0.05 * df_processed['TVDSS'])
    print("Successfully created 'FTEMP' log.")
    return df_processed


def calculate_rwa(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Menghitung RWA berdasarkan metode Indonesia yang dipilih, beserta koreksi temperatur (RWT).
    Logika disesuaikan dengan Loglan: menghitung satu metode RWA saja.
    """
    df_processed = df.copy()

    # Pastikan FTEMP dihitung jika tidak ada (opsional, bisa dihapus jika FTEMP selalu ada)
    if 'FTEMP' not in df_processed.columns and 'TVDSS' in df_processed.columns:
        df_processed = calculate_ftemp(df_processed)

    # --- 1. Ekstrak Parameter & Validasi Kolom ---
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    RT_SH = float(params.get('RT_SH', 2.2))
    RWT = float(params.get('RWT', 227))

    # Ambil pilihan metode dari frontend
    opt_indo = params.get('OPT_INDO', 'RWA_FULL')

    # Ambil nama log dinamis
    rt_log = params.get('RT', 'RT')
    phie_log = params.get('PHIE', 'PHIE')
    vsh_log = params.get('VSH', 'VSH_LINEAR')
    ftemp_log = params.get('FTEMP', 'FTEMP')

    # Ambil nama log output dinamis
    rwa_indo_out = params.get('RWA_INDO', 'RWA_INDO')
    rwa_out = params.get('RWA', 'RWA')
    rwa_rwt_out = params.get('RWA_RWT', 'RWA_RWT')

    # Validasi kolom input yang wajib ada
    required_cols = [rt_log, phie_log, vsh_log, ftemp_log]
    missing_cols = [
        col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Kolom input tidak lengkap: {missing_cols}")

    # Buat proses idempoten
    output_cols = [rwa_indo_out, rwa_out, rwa_rwt_out]
    df_processed.drop(columns=df_processed.columns.intersection(
        output_cols), inplace=True, errors='ignore')
    for col in output_cols:
        df_processed[col] = np.nan

    # --- 2. Buat Mask Filter ---
    # Inisialisasi mask untuk seluruh DataFrame
    final_mask = pd.Series(True, index=df_processed.index)

    # Gabungkan filter interval dan zona jika ada
    interval_mask = pd.Series(False, index=df_processed.index)
    zone_mask = pd.Series(False, index=df_processed.index)
    has_filters = False

    if target_intervals and 'MARKER' in df_processed.columns:
        interval_mask = df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        has_filters = True

    if has_filters:
        final_mask = interval_mask | zone_mask

    # Filter tambahan untuk memastikan data input valid (tidak NaN)
    valid_data_mask = df_processed[required_cols].notna().all(axis=1)
    final_mask &= valid_data_mask

    if not final_mask.any():
        print("Peringatan: Tidak ada data valid. Kalkulasi RWA dilewati.")
        return df_processed

    print(
        f"Menghitung RWA untuk {final_mask.sum()} baris menggunakan metode: {opt_indo}")

    # --- 3. Perhitungan Sesuai Loglan ---
    phie = df_processed.loc[final_mask, phie_log]
    rt = df_processed.loc[final_mask, rt_log]
    vsh = df_processed.loc[final_mask, vsh_log]
    ftemp = df_processed.loc[final_mask, ftemp_log]

    if opt_indo == 'RWA_FULL':
        v = vsh ** (2 - vsh)
    elif opt_indo == 'RWA_SIMPLE':
        v = vsh ** 2
    elif opt_indo == 'RWA_TAR_SAND':
        v = vsh ** (2 - 2 * vsh)
    else:
        print(
            f"Peringatan: Opsi '{opt_indo}' tidak dikenali. Menggunakan metode 'RWA_FULL'.")
        v = vsh ** (2 - vsh)  # Default ke FULL

    # Hitung RWA_INDO
    f1 = (phie ** M) / A
    f2 = 1 / rt
    f3 = v / RT_SH
    f4 = np.sqrt(v / (rt * RT_SH))

    denominator = f2 + f3 - f4
    rwa_indo = np.where(denominator > 0, f1 / denominator, np.nan)

    # Limit RWA (tidak boleh negatif)
    rwa = np.maximum(0, rwa_indo)

    # Koreksi temperatur ke RWT
    temp_correction = (ftemp + 21.5) / (RWT + 21.5)
    rwa_rwt = rwa * temp_correction

    # --- 4. Simpan Hasil ke DataFrame ---
    df_processed.loc[final_mask, rwa_indo_out] = rwa_indo
    df_processed.loc[final_mask, rwa_out] = rwa
    df_processed.loc[final_mask, rwa_rwt_out] = rwa_rwt

    print(
        f"Kolom '{rwa_indo_out}', '{rwa_out}', dan '{rwa_rwt_out}' telah dihitung.")
    return df_processed
