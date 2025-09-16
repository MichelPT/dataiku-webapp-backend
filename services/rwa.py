import pandas as pd
import numpy as np

def calculate_ftemp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Formation Temperature (FTEMP) log based on the TVDSS log.

    Formula: FTEMP = 75 + (0.05 * TVDSS)

    Args:
        df: DataFrame containing a 'TVDSS' column.

    Returns:
        DataFrame with a new 'FTEMP' column added.
    """
    df_processed = df.copy()

    # Check if the required TVDSS column exists
    if 'TVDSS' not in df_processed.columns:
        raise ValueError("Input DataFrame must contain a 'TVDSS' column.")

    # Apply the formula vectorized for efficiency
    df_processed['FTEMP'] = 75 + (0.05 * df_processed['TVDSS'])
    
    print("Successfully created 'FTEMP' log.")
    return df_processed

def calculate_rwa(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Menghitung RWA (Full, Simple, Tar) beserta koreksi temperaturnya (RWT),
    dengan filter internal untuk interval/zona.
    
    FIXED:
    1. Menggunakan log FTEMP dari DataFrame untuk koreksi temperatur per baris.
    2. Memperbaiki formula RWA dengan menambahkan faktor 2 yang hilang.
    """
    df_processed = df.copy()
    df_processed = calculate_ftemp(df_processed)

    # --- 1. Ekstrak Parameter & Validasi Kolom ---
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    RT_SH = float(params.get('RT_SH', 5.0))
    RWT = float(params.get('RWT', 227)) # Suhu referensi Rwa (misal: suhu lab)

    # --- FIX: 'FTEMP' kini menjadi kolom wajib, bukan parameter konstan ---
    required_cols = ['PHIE', 'RT', 'VSH', 'FTEMP']
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Kolom input tidak lengkap. Pastikan kolom berikut ada: {missing_cols}")

    # Buat proses idempoten dengan menghapus hasil lama dan inisialisasi ulang
    rwa_cols = [
        "RWA_FULL", "RWA_SIMPLE", "RWA_TAR",
        "RWA_FULL_RWT", "RWA_SIMPLE_RWT", "RWA_TAR_RWT"
    ]
    df_processed.drop(
        columns=df_processed.columns.intersection(rwa_cols), inplace=True, errors='ignore')
    for col in rwa_cols:
        df_processed[col] = np.nan

    # --- 2. Buat Mask Filter ---
    # Inisialisasi mask untuk seluruh DataFrame
    mask = pd.Series(True, index=df_processed.index)
    
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
        mask = interval_mask | zone_mask

    # Filter tambahan untuk memastikan data input valid (tidak NaN)
    valid_data_mask = df_processed[required_cols].notna().all(axis=1)
    final_mask = mask & valid_data_mask

    if not final_mask.any():
        print("Peringatan: Tidak ada data valid yang cocok dengan filter. Tidak ada kalkulasi RWA yang dilakukan.")
        return df_processed

    print(f"Menghitung RWA untuk {final_mask.sum()} dari {len(df_processed)} baris.")

    # --- 3. Lakukan Perhitungan HANYA pada Baris yang Dipilih ---
    phie = df_processed.loc[final_mask, "PHIE"]
    rt = df_processed.loc[final_mask, "RT"]
    vsh = df_processed.loc[final_mask, "VSH"]

    # Hindari pembagian dengan nol atau log dari nol
    rt = rt.replace(0, np.nan)
    phie = phie.replace(0, np.nan)

    # Perhitungan umum berdasarkan rearrangement formula Indonesia
    # Rwa = (PHIE^M / A) / ( (1/RT) + (Vsh_term/RT_SH) - 2 * sqrt(Vsh_term / (RT * RT_SH)) )
    f1 = (phie ** M) / A
    f2 = 1 / rt

    # Kalkulasi untuk setiap metode
    # a. Full Indonesia
    v_full = vsh ** (2 - vsh)
    f3_full = v_full / RT_SH
    f4_full = np.sqrt(v_full / (rt * RT_SH))
    # --- FIX: Menambahkan faktor 2 pada suku akar kuadrat ---
    denominator_full = f2 + f3_full - (2 * f4_full)
    rwaf = np.where(denominator_full > 0, f1 / denominator_full, np.nan) # Hanya ambil hasil positif
    df_processed.loc[final_mask, "RWA_FULL"] = rwaf

    # b. Simple Indonesia
    v_simple = vsh ** 2
    f3_simple = v_simple / RT_SH
    f4_simple = np.sqrt(v_simple / (rt * RT_SH))
    # --- FIX: Menambahkan faktor 2 pada suku akar kuadrat ---
    denominator_simple = f2 + f3_simple - (2 * f4_simple)
    rwas = np.where(denominator_simple > 0, f1 / denominator_simple, np.nan)
    df_processed.loc[final_mask, "RWA_SIMPLE"] = rwas

    # c. Tar Sand
    v_tar = vsh ** (2 - 2 * vsh)
    f3_tar = v_tar / RT_SH
    f4_tar = np.sqrt(v_tar / (rt * RT_SH))
    # --- FIX: Menambahkan faktor 2 pada suku akar kuadrat ---
    denominator_tar = f2 + f3_tar - (2 * f4_tar)
    rwat = np.where(denominator_tar > 0, f1 / denominator_tar, np.nan)
    df_processed.loc[final_mask, "RWA_TAR"] = rwat

    # --- 4. Hitung RWA_RWT untuk setiap metode menggunakan FTEMP log ---
    temp_denominator = RWT + 21.5
    if temp_denominator != 0:
        # Ambil log temperatur untuk baris yang relevan
        ftemp_masked = df_processed.loc[final_mask, "FTEMP"]
        # Hitung koreksi temperatur untuk setiap baris (vectorized)
        temp_correction = (RWT + 21.5) / (ftemp_masked + 21.5) # Arp's Law: R2 = R1 * (T1+k)/(T2+k)
        
        # Terapkan koreksi
        df_processed.loc[final_mask, "RWA_FULL_RWT"] = df_processed.loc[final_mask, "RWA_FULL"] * temp_correction
        df_processed.loc[final_mask, "RWA_SIMPLE_RWT"] = df_processed.loc[final_mask, "RWA_SIMPLE"] * temp_correction
        df_processed.loc[final_mask, "RWA_TAR_RWT"] = df_processed.loc[final_mask, "RWA_TAR"] * temp_correction

    print("Kolom RWA dan RWA_RWT telah ditambahkan/diperbarui.")
    return df_processed