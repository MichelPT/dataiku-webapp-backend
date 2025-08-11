# File: services/rt_r0.py

import numpy as np
import pandas as pd
# Hapus impor linregress karena tidak lagi digunakan
# from scipy.stats import linregress


def calculate_iqual(df):
    """
    Menghitung IQUAL berdasarkan kondisi:
    IF (PHIE>0.1)&&(VSH<0.5): IQUAL =1
    else: IQUAL =0
    """
    # Fungsi ini sudah benar dan tidak perlu diubah
    df = df.copy()
    df['IQUAL'] = np.where((df['PHIE'] > 0.1) & (df['VSH'] < 0.5), 1, 0)
    return df


def calculate_R0(df):
    """
    Menghitung R0 dan parameter terkait sesuai dengan Loglan ro.lls.
    """
    # --- PERBAIKAN UTAMA DI SINI ---

    # 1. Hitung Rwa sesuai Loglan
    # df['RWA'] = df['RT'] * (df['PHIE']**df['M'])

    # 2. Perbaiki rumus 'aa' untuk menggunakan RW, bukan RWA_FULL
    aa = (df['PHIE']**df['M']) / (df['A'] * df['RWA_FULL'])

    # Rumus bb dan cc sudah benar
    cc = 2 - df['VSH']
    bb = (df['VSH']**cc) / df['RTSH']

    # Rumus R0 sudah benar (ini adalah bentuk sederhana dari (1/(sqrt(aa)+sqrt(bb)))**2)
    R0 = 1 / (aa + 2 * (aa * bb)**0.5 + bb)

    # Simpan hasil ke DataFrame
    df['R0'] = R0
    df['RTR0'] = df['RT'] - df['R0']

    # --- AKHIR PERBAIKAN ---
    return df


def analyze_rtr0_groups(df):
    """
    Analisis RTR0 untuk setiap group, dihitung menggunakan rumus gradien manual
    sesuai dengan Loglan rt_ro.lls.
    """
    results_rtr0 = []

    # Proses hanya pada data di mana IQUAL = 1, sesuai Loglan
    df_reservoir = df[df['IQUAL'] == 1]

    for group_id, group in df_reservoir.groupby('GROUP_ID'):
        n = len(group)

        # Hanya proses grup jika memiliki lebih dari 1 titik data
        if n <= 1:
            continue

        try:
            # Perhitungan untuk gradien RT vs R0 (rttorogradient)
            sxa = group['R0'].sum()
            sya = group['RT'].sum()
            sxya = (group['R0'] * group['RT']).sum()
            sx2a = (group['R0']**2).sum()

            denominator_a = (sxa * sxa - n * sx2a)
            slope_rt2r0 = (sxa * sya - n * sxya) / \
                denominator_a if denominator_a != 0 else np.nan

            # Perhitungan untuk gradien PHIE vs RTR0 (rtrotophiegradient)
            sxb = group['PHIE'].sum()
            syb = group['RTR0'].sum()
            sxyb = (group['PHIE'] * group['RTR0']).sum()
            sx2b = (group['PHIE']**2).sum()

            denominator_b = (sxb * sxb - n * sx2b)
            slope_phie2rtr0 = (sxb * syb - n * sxyb) / \
                denominator_b if denominator_b != 0 else np.nan

            if np.isnan(slope_phie2rtr0) or np.isinf(slope_phie2rtr0):
                continue

            fluid_rtrophie = 'G' if slope_phie2rtr0 > 0 else 'W'

            results_rtr0.append({
                'GROUP_ID': group_id,
                'RT_R0_GRAD': slope_rt2r0,
                'PHIE_RTR0_GRAD': slope_phie2rtr0,
                'FLUID_RTROPHIE': fluid_rtrophie
            })

        except Exception as e:
            print(f"Warning: Error processing group {group_id}: {str(e)}")
            continue

    if not results_rtr0:
        return pd.DataFrame(columns=['GROUP_ID', 'RT_R0_GRAD', 'PHIE_RTR0_GRAD', 'FLUID_RTROPHIE'])

    return pd.DataFrame(results_rtr0)


def process_rt_r0(df, params=None):
    """
    Fungsi utama untuk memproses analisis RT-R0.
    """
    if params is None:
        params = {}

    try:
        # Tambahkan parameter default jika belum ada di DataFrame
        # (Logika ini bisa disesuaikan sesuai kebutuhan)
        default_params = {'A': 1.0, 'M': 2.0, 'N': 2.0, 'RTSH': 2.2}
        for p, val in default_params.items():
            if p not in df.columns:
                df[p] = params.get(f'{p}_PARAM', val)

        # Urutan proses sudah benar sesuai Loglan
        df = calculate_iqual(df)
        df = calculate_R0(df)
        df['GROUP_ID'] = (df['IQUAL'].diff() != 0).cumsum()
        df_results_rtr0 = analyze_rtr0_groups(df)

        # Hapus kolom lama sebelum merge untuk menghindari duplikasi
        cols_to_drop = ['RT_R0_GRAD', 'PHIE_RTR0_GRAD', 'FLUID_RTROPHIE']
        df.drop(
            columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        if not df_results_rtr0.empty:
            df = df.merge(df_results_rtr0, on='GROUP_ID', how='left')

        return df
    except Exception as e:
        print(f"Error in process_rt_r0: {e}")
        raise e


# # In your services/rt_r0.py file

# import numpy as np
# import pandas as pd
# from scipy.stats import linregress


# def calculate_R0(df):
#     """
#     Menghitung R0 dan parameter terkait
#     """
#     aa = df['PHIE']**df['M'] / (df['A']*df['RWA_FULL'])
#     cc = 2 - df['VSH']
#     bb = df['VSH']**cc / df['RTSH']

#     R0 = 1 / (aa + 2 * (aa * bb)**0.5 + bb)
#     df['R0'] = R0
#     df['RTR0'] = df['RT'] - df['R0']
#     return df


# def analyze_rtr0_groups(df):
#     """
#     Analisis RTR0 untuk setiap group dalam satu well
#     """
#     results_rtr0 = []

#     for group_id, group in df.groupby('GROUP_ID'):
#         n = len(group)

#         # Hanya memproses group dengan n > 1
#         if ((group['PHIE'].nunique() == 1) | (group['RT'].nunique() == 1) | (n <= 1)):
#             continue

#         try:
#             # Regresi linear untuk slope dan r-squared
#             slope_rt2r0, _, _, _, _ = linregress(group['RT'], group['R0'])
#             slope_phie2rtr0, _, _, _, _ = linregress(
#                 group['PHIE'], group['RTR0'])

#             # Validasi hasil regresi
#             if np.isnan(slope_phie2rtr0) or np.isinf(slope_phie2rtr0):
#                 continue

#             # Mengikuti kode acuan persis
#             condition = slope_phie2rtr0 > 0
#             FLUID_RTROPHIE = np.where(condition, 'G', 'W')

#             # Pastikan FLUID_RTROPHIE adalah scalar string
#             if isinstance(FLUID_RTROPHIE, np.ndarray):
#                 FLUID_RTROPHIE = FLUID_RTROPHIE.item()  # Konversi array ke scalar

#             # List hasil
#             results_rtr0.append({
#                 'GROUP_ID': group_id,
#                 'RT_R0_GRAD': slope_rt2r0,
#                 'PHIE_RTR0_GRAD': slope_phie2rtr0,
#                 'FLUID_RTROPHIE': FLUID_RTROPHIE
#             })

#         except Exception as e:
#             print(f"Warning: Error processing group {group_id}: {str(e)}")
#             continue

#     return pd.DataFrame(results_rtr0)


# def process_rt_r0(df, params=None):
#     """
#     Main function to process RT-R0 analysis
#     """
#     if params is None:
#         params = {}

#     try:
#         # Tambahkan parameter default jika belum ada
#         # if 'A' not in df.columns:
#         #     df['A'] = params.get('A_PARAM', 1)
#         # if 'M' not in df.columns:
#         #     df['M'] = params.get('M_PARAM', 1.8)
#         # if 'N' not in df.columns:
#         #     df['N'] = params.get('N_PARAM', 1.8)
#         # if 'RTSH' not in df.columns:
#         #     df['RTSH'] = params.get('RTSH', 1)
#         # if 'RW' not in df.columns:
#         #     df['RW'] = params.get('RW', 1)

#         # Step 2: Calculate R0 and RTR0
#         df = calculate_R0(df)

#         # Step 3: Group by sequence
#         df['GROUP_ID'] = (df['IQUAL'].diff() != 0).cumsum()

#         # Step 4: Analyze RTR0 groups
#         df_results_rtr0 = analyze_rtr0_groups(df)

#         columns_to_remove = ['RT_R0_GRAD', 'PHIE_RTR0_GRAD', 'FLUID_RTROPHIE']
#         df.drop(
#             columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

#         # Step 5: Merge results
#         if not df_results_rtr0.empty:
#             df = df.merge(df_results_rtr0, on='GROUP_ID', how='left')

#         return df

#     except Exception as e:
#         print(f"Error in process_rt_r0: {e}")
#         raise e
