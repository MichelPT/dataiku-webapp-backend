import numpy as np
import pandas as pd

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
    
def indonesia_computation(rw_in, phie, ct, a, m, n, rtsh, vsh):
    """
    Fungsi untuk menghitung water saturation menggunakan metode Indonesia.
    (Fungsi ini tidak perlu diubah)
    """
    # Hindari nilai invalid pada input
    if pd.isna(vsh) or pd.isna(phie) or pd.isna(ct) or pd.isna(rtsh):
        return np.nan

    ddd = 2 - vsh
    aaa = vsh**ddd / rtsh
    bbb = phie**m / (a * rw_in)

    # Pastikan argumen untuk akar pangkat tidak negatif
    sqrt_arg = (vsh**ddd * phie**m) / (a * rw_in * rtsh)
    if sqrt_arg < 0:
        ccc = 0  # Atau tangani sebagai error
    else:
        ccc = 2 * np.sqrt(sqrt_arg)

    denominator = aaa + bbb + ccc

    if denominator == 0 or np.isclose(denominator, 0):
        return 1.0

    base = ct / denominator
    if base < 0:
        return 1.0

    swe = base ** (1 / n)
    return max(0.0, min(1.0, swe))


def process_swgrad(df: pd.DataFrame, params: dict = None, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Memproses perhitungan SWGRAD, dengan filter internal untuk interval/zona.
    MODIFIED: Kini menggunakan log 'FTEMP' dari DataFrame, bukan parameter konstan.
    """
    if params is None:
        params = {}

    try:
        df_processed = df.copy()
        df_processed = calculate_ftemp(df_processed)

        # 1. Persiapan: Hapus kolom lama dan siapkan parameter
        cols_to_drop = ['SWGRAD'] + [f'SWARRAY_{i}' for i in range(1, 26)]
        df_processed.drop(columns=df_processed.columns.intersection(
            cols_to_drop), inplace=True, errors='ignore')

        # Inisialisasi kolom output dengan NaN
        for col in cols_to_drop:
            df_processed[col] = np.nan

        # --- MODIFICATION START ---
        # Pastikan kolom input ada, termasuk 'FTEMP'
        required_cols = ['RT', 'VSH', 'PHIE', 'FTEMP']
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            print(f"Peringatan: Kolom input {missing_cols} tidak lengkap. Melewatkan SWGRAD.")
            return df
        # --- MODIFICATION END ---

        df_processed['CT'] = 1 / df_processed['RT']

        a = float(params.get('A', 1.0))
        m = float(params.get('M', 2.0))
        n = float(params.get('N', 2.0))
        rtsh = float(params.get('RTSH', 2.2))
        
        # --- MODIFICATION START ---
        # Parameter ftemp_const tidak lagi digunakan karena kita memakai log FTEMP
        # ftemp_const = float(params.get('FTEMP', 75.0))
        # --- MODIFICATION END ---

        # 2. Buat mask untuk memilih baris yang akan diproses (kode tidak berubah)
        mask = pd.Series(True, index=df_processed.index)
        has_filters = False
        if target_intervals and 'MARKER' in df_processed.columns:
            mask = df_processed['MARKER'].isin(target_intervals)
            has_filters = True
        if target_zones and 'ZONE' in df_processed.columns:
            zone_mask = df_processed['ZONE'].isin(target_zones)
            mask = (mask | zone_mask) if has_filters else zone_mask

        # 3. Lakukan perhitungan HANYA pada baris yang cocok dengan mask
        print(
            f"Memproses SWGRAD untuk {mask.sum()} dari {len(df_processed)} baris.")

        # Loop hanya pada indeks yang dipilih oleh mask
        for i in df_processed[mask].index:
            sw = np.zeros(26)

            # Ambil nilai sekali per baris untuk efisiensi
            vsh_val = df_processed.at[i, 'VSH']
            phie_val = df_processed.at[i, 'PHIE']
            ct_val = df_processed.at[i, 'CT']
            
            # --- MODIFICATION START ---
            # Ambil nilai FTEMP langsung dari kolom DataFrame
            ftemp_val = df_processed.at[i, 'FTEMP']
            # --- MODIFICATION END ---

            # Loop untuk 25 tingkat salinitas
            for j in range(1, 26):
                sal_j = j * 1000.0
                x_j = 0.0123 + 3647.5 / (sal_j**0.955)
                # Gunakan Arp's Law untuk Fahrenheit (sesuai konstanta 6.77)
                rw_in = x_j * 81.77 / (ftemp_val + 6.77)

                sw[j] = indonesia_computation(
                    rw_in, phie_val, ct_val, a, m, n, rtsh, vsh_val)
                df_processed.at[i, f'SWARRAY_{j}'] = sw[j]

            # Hitung gradien (kode tidak berubah)
            sx, sx2, sy, sxy, n_grad = 0, 0, 0, 0, 0
            for j in range(1, 26):
                x_val, y_val = j, sw[j]
                if not pd.isna(y_val):
                    sx += x_val
                    sx2 += x_val**2
                    sy += y_val
                    sxy += x_val * y_val
                    n_grad += 1

            if n_grad > 1:
                denominator = (sx * sx - n_grad * sx2)
                if denominator != 0:
                    swgrad = (sx * sy - n_grad * sxy) / denominator
                    df_processed.at[i, 'SWGRAD'] = swgrad

        # Hapus kolom CT yang hanya sementara
        df_processed.drop(columns=['CT'], inplace=True, errors='ignore')

        return df_processed

    except Exception as e:
        print(f"Error dalam process_swgrad: {str(e)}")
        raise e