import numpy as np
import pandas as pd
from scipy.stats import linregress

def indonesia_computation(rw_in, phie, ct, a, m, n, rtsh, vsh):
    """
    Fungsi untuk menghitung water saturation menggunakan metode Indonesia
    """
    dd = 2 - vsh
    aa = vsh**dd / rtsh
    bb = phie**m / (a * rw_in)
    cc = 2 * np.sqrt((vsh**dd * phie**m) / (a * rw_in * rtsh))
    denominator = aa + bb + cc

    if denominator == 0:
        return 1.0

    swe = (ct / denominator) ** (1 / n)
    return max(0.0, min(1.0, swe))

def process_swgrad(df, params=None):
    """
    Proses perhitungan untuk seluruh dataset
    """
    if params is None:
        params = {}

    try:
        # Initialize SWARRAY columns
        for i in range(1, 26):
            df[f'SWARRAY_{i}'] = np.nan
        df['SWGRAD'] = np.nan

        # Data non dummy
        df['CT'] = 1 / df['RT']

        # Konstanta dummy per zona
        a = params.get('A_PARAM', 1)
        m = params.get('M_PARAM', 1.8)
        n = params.get('N_PARAM', 1.8)
        rtsh = params.get('RTSH', 1)

        # Data dari kolom dataframe
        vsh = df['VSH'].values  # VSH dari kolom dataframe
        phie = df['PHIE'].values  # PHIE dari kolom dataframe
        ftemp = 75 + 0.05 * df['DEPTH'].values  # formation temperature (fahrenheit)
        ct = df['CT'].values
        df['FTEMP'] = ftemp
        df['A_PARAM'] = a
        df['M_PARAM'] = m
        df['N_PARAM'] = n
        df['RTSH'] = rtsh

        # Proses perhitungan untuk setiap baris dalam well ini
        for i in range(len(df)):
            sal = np.zeros(26)
            x = np.zeros(26)
            sw = np.zeros(26)

            # Loop untuk setiap salinitas (1-25)
            for j in range(1, 26):
                sal[j] = j * 1000
                x[j] = 0.0123 + 3647.5 / sal[j]**0.955
                rw_in = x[j] * 81.77 / (ftemp[i] + 6.77)

                # Hitung water saturation
                sw[j] = indonesia_computation(rw_in, phie[i], ct[i], a, m, n, rtsh, vsh[i])

                # Simpan ke SWARRAY
                df.iloc[i, df.columns.get_loc(f'SWARRAY_{j}')] = sw[j]

            # HITUNG SWGRAD SETELAH SEMUA SW DIHITUNG
            # Gunakan data SW pada salinitas 10k, 15k, 20k, 25k ppm (indeks 10, 15, 20, 25)
            try:
                data_SW = np.array([sw[5*k] for k in range(2, 6)])  # sw[10], sw[15], sw[20], sw[25]
                data_SAL = np.array([5*k for k in range(2, 6)])     # [10, 15, 20, 25]

                # Hitung gradient menggunakan linear regression
                SWGRAD, _, _, _, _ = linregress(data_SAL, data_SW)
                df.iloc[i, df.columns.get_loc('SWGRAD')] = SWGRAD

            except Exception as e:
                print(f"Error calculating SWGRAD for row {i}: {str(e)}")
                df.iloc[i, df.columns.get_loc('SWGRAD')] = np.nan

        return df

    except Exception as e:
        print(f"Error in process_swgrad: {str(e)}")
        raise e
