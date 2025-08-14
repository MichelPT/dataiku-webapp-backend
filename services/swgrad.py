# Di dalam file services/swgrad.py atau file pemrosesan Anda

import numpy as np
import pandas as pd
# Hapus impor linregress karena tidak lagi digunakan
# from scipy.stats import linregress


import numpy as np
import pandas as pd


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
    """
    if params is None:
        params = {}

    try:
        df_processed = df.copy()

        # 1. Persiapan: Hapus kolom lama dan siapkan parameter
        cols_to_drop = ['SWGRAD'] + [f'SWARRAY_{i}' for i in range(1, 26)]
        df_processed.drop(columns=df_processed.columns.intersection(
            cols_to_drop), inplace=True)

        # Inisialisasi kolom output dengan NaN
        for col in cols_to_drop:
            df_processed[col] = np.nan

        # Pastikan kolom input ada
        required_cols = ['RT', 'VSH', 'PHIE', 'DEPTH']
        if not all(col in df_processed.columns for col in required_cols):
            print(
                "Peringatan: Kolom input (RT, VSH, PHIE, DEPTH) tidak lengkap. Melewatkan SWGRAD.")
            return df

        df_processed['CT'] = 1 / df_processed['RT']

        a = params.get('A', 1.0)
        m = params.get('M', 2.0)
        n = params.get('N', 2.0)
        rtsh = params.get('RTSH', 2.2)
        ftemp_const = params.get('FTEMP', 75.0)

        # 2. Buat mask untuk memilih baris yang akan diproses
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
            depth_val = df_processed.at[i, 'DEPTH']
            ct_val = df_processed.at[i, 'CT']
            ftemp_val = ftemp_const + 0.05 * depth_val

            # Loop untuk 25 tingkat salinitas
            for j in range(1, 26):
                sal_j = j * 1000.0
                x_j = 0.0123 + 3647.5 / (sal_j**0.955)
                rw_in = x_j * 81.77 / (ftemp_val + 6.77)

                sw[j] = indonesia_computation(
                    rw_in, phie_val, ct_val, a, m, n, rtsh, vsh_val)
                df_processed.at[i, f'SWARRAY_{j}'] = sw[j]

            # Hitung gradien
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

# import numpy as np
# import pandas as pd
# from scipy.stats import linregress


# def indonesia_computation(rw_in, phie, ct, a, m, n, rtsh, vsh):
#     """
#     Fungsi untuk menghitung water saturation menggunakan metode Indonesia
#     """
#     dd = 2 - vsh
#     aa = vsh**dd / rtsh
#     bb = phie**m / (a * rw_in)
#     cc = 2 * np.sqrt((vsh**dd * phie**m) / (a * rw_in * rtsh))
#     denominator = aa + bb + cc

#     if denominator == 0:
#         return 1.0

#     swe = (ct / denominator) ** (1 / n)
#     return max(0.0, min(1.0, swe))


# def process_swgrad(df, params=None):
#     """
#     Proses perhitungan untuk seluruh dataset
#     """
#     if params is None:
#         params = {}

#     try:
#         # Initialize SWARRAY columns
#         for i in range(1, 26):
#             df[f'SWARRAY_{i}'] = np.nan
#         df['SWGRAD'] = np.nan

#         # Data non dummy
#         df['CT'] = 1 / df['RT']

#         # Konstanta dummy per zona
#         a = params.get('A_PARAM', 1)
#         m = params.get('M_PARAM', 1.8)
#         n = params.get('N_PARAM', 1.8)
#         rtsh = params.get('RTSH', 1)

#         # Data dari kolom dataframe
#         vsh = df['VSH'].values  # VSH dari kolom dataframe
#         phie = df['PHIE'].values  # PHIE dari kolom dataframe
#         # formation temperature (fahrenheit)
#         ftemp = 75 + 0.05 * df['DEPTH'].values
#         ct = df['CT'].values
#         df['FTEMP'] = ftemp
#         df['A'] = a
#         df['M'] = m
#         df['N'] = n
#         df['RTSH'] = rtsh

#         # Proses perhitungan untuk setiap baris dalam well ini
#         for i in range(len(df)):
#             sal = np.zeros(16)
#             x = np.zeros(16)
#             sw = np.zeros(16)

#             # Loop untuk setiap salinitas (1-25)
#             for j in range(1, 16):
#                 sal[j] = j * 1000
#                 x[j] = 0.0123 + 3647.5 / sal[j]**0.955
#                 rw_in = x[j] * 81.77 / (ftemp[i] + 6.77)

#                 # Hitung water saturation
#                 sw[j] = indonesia_computation(
#                     rw_in, phie[i], ct[i], a, m, n, rtsh, vsh[i])

#                 # Simpan ke SWARRAY
#                 df.iloc[i, df.columns.get_loc(f'SWARRAY_{j}')] = sw[j]

#             # HITUNG SWGRAD SETELAH SEMUA SW DIHITUNG
#             # Gunakan data SW pada salinitas 10k, 15k, 20k, 25k ppm (indeks 10, 15, 20, 25)
#             try:
#                 # sw[10], sw[15], sw[20], sw[25]
#                 data_SW = np.array([sw[5*k] for k in range(2, 6)])
#                 # [10, 15, 20, 25]
#                 data_SAL = np.array([5*k for k in range(2, 6)])

#                 # Hitung gradient menggunakan linear regression
#                 SWGRAD, _, _, _, _ = linregress(data_SAL, data_SW)
#                 df.iloc[i, df.columns.get_loc('SWGRAD')] = SWGRAD

#             except Exception as e:
#                 print(f"Error calculating SWGRAD for row {i}: {str(e)}")
#                 df.iloc[i, df.columns.get_loc('SWGRAD')] = np.nan

#         return df

#     except Exception as e:
#         print(f"Error in process_swgrad: {str(e)}")
#         raise e
