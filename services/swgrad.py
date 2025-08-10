# Di dalam file services/swgrad.py atau file pemrosesan Anda

import numpy as np
import pandas as pd
# Hapus impor linregress karena tidak lagi digunakan
# from scipy.stats import linregress


def indonesia_computation(rw_in, phie, ct, a, m, n, rtsh, vsh):
    """
    Fungsi untuk menghitung water saturation menggunakan metode Indonesia.
    (Fungsi ini sudah benar dan tidak perlu diubah)
    """
    ddd = 2 - vsh
    aaa = vsh**ddd / rtsh
    bbb = phie**m / (a * rw_in)
    ccc = 2 * np.sqrt((vsh**ddd * phie**m) / (a * rw_in * rtsh))
    denominator = aaa + bbb + ccc

    # Hindari pembagian dengan nol
    if denominator == 0 or np.isclose(denominator, 0):
        return 1.0

    # Hindari nilai negatif di dalam akar pangkat
    base = ct / denominator
    if base < 0:
        return 1.0

    swe = base ** (1 / n)
    return max(0.0, min(1.0, swe))


def process_swgrad(df, params=None):
    """
    Proses perhitungan SWGRAD untuk seluruh dataset sesuai dengan Loglan.
    """
    if params is None:
        params = {}

    try:
        # Inisialisasi kolom-kolom output
        for i in range(1, 26):
            df[f'SWARRAY_{i}'] = np.nan
        df['SWGRAD'] = np.nan

        df['CT'] = 1 / df['RT']

        # Ambil parameter atau gunakan nilai default
        a = params.get('A', 1.0)
        m = params.get('M', 2)
        n = params.get('N', 2)
        rtsh = params.get('RTSH', 2.2)
        ftemp_const = params.get('FTEMP', 75.0)

        # Siapkan data dari kolom DataFrame untuk pemrosesan berbasis vektor
        vsh_vals = df['VSH'].values
        phie_vals = df['PHIE'].values
        ftemp_vals = ftemp_const + 0.05 * df['DEPTH'].values
        ct_vals = df['CT'].values

        # Proses perhitungan untuk setiap baris (setiap titik kedalaman)
        for i in range(len(df)):
            sw = np.zeros(26)  # Ukuran 26 untuk mengakomodasi indeks 1-25

            # Loop untuk menghitung SW pada 25 tingkat salinitas
            for j in range(1, 26):
                sal_j = j * 1000.0
                x_j = 0.0123 + 3647.5 / (sal_j**0.955)
                rw_in = x_j * 81.77 / (ftemp_vals[i] + 6.77)

                sw[j] = indonesia_computation(
                    rw_in, phie_vals[i], ct_vals[i], a, m, n, rtsh, vsh_vals[i]
                )
                df.at[i, f'SWARRAY_{j}'] = sw[j]

            # Inisialisasi variabel sumasi seperti di Loglan
            sx = 0
            sx2 = 0
            sy = 0
            sxy = 0
            n_grad = 0  # Jumlah titik untuk regresi

            # Loop dari 1 hingga 25 untuk menghitung gradien
            for j in range(1, 26):
                # Loglan hanya menggunakan 3 titik, tapi kita bisa membuatnya dinamis
                # Di sini kita akan mengikuti Loglan dan menggunakan semua 25 titik
                # Anda bisa mengubah `range(1, 26)` jika ingin titik yang lebih spesifik

                # Asumsikan sumbu x adalah salinitas (direpresentasikan oleh j)
                x_val = j
                # Sumbu y adalah nilai SW yang sesuai
                y_val = sw[j]

                # Lakukan sumasi hanya jika sw valid
                if not np.isnan(y_val):
                    sx += x_val
                    sx2 += x_val**2
                    sy += y_val
                    sxy += x_val * y_val
                    n_grad += 1

            # Hitung gradien hanya jika ada cukup data
            if n_grad > 1:
                denominator = (sx * sx - n_grad * sx2)
                if denominator != 0:
                    # Implementasi rumus gradien dari Loglan
                    swgrad = (sx * sy - n_grad * sxy) / denominator
                    df.at[i, 'SWGRAD'] = swgrad
            # --- AKHIR PERBAIKAN ---

        return df

    except Exception as e:
        print(f"Error in process_swgrad: {str(e)}")
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
