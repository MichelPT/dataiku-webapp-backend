# File: services/rt_r0.py

import numpy as np
import pandas as pd
# Hapus impor linregress karena tidak lagi digunakan
# from scipy.stats import linregress


import numpy as np
import pandas as pd


def calculate_iqual(df):
    """
    Menghitung IQUAL berdasarkan kondisi: PHIE > 0.1 AND VSH < 0.5.
    """
    df_copy = df.copy()
    # Pastikan kolom yang dibutuhkan ada
    if 'PHIE' in df_copy.columns and 'VSH' in df_copy.columns:
        df_copy['IQUAL'] = np.where(
            (df_copy['PHIE'] > 0.1) & (df_copy['VSH'] < 0.5), 1, 0)
    else:
        # Jika kolom tidak ada, IQUAL diisi 0 agar tidak error
        df_copy['IQUAL'] = 0
        print("Peringatan: Kolom 'PHIE' atau 'VSH' tidak ditemukan, IQUAL diatur ke 0.")
    return df_copy


def calculate_R0(df):
    """
    Menghitung R0 dan parameter terkait.
    """
    df_copy = df.copy()
    # Pastikan semua kolom yang dibutuhkan untuk kalkulasi ada
    required_cols = ['PHIE', 'M', 'A', 'RWA_FULL', 'VSH', 'RTSH', 'RT']
    if not all(col in df_copy.columns for col in required_cols):
        print("Peringatan: Kolom yang dibutuhkan untuk kalkulasi R0 tidak lengkap. Melewatkan.")
        # Kembalikan kolom kosong agar tidak error saat merge
        df_copy['R0'] = np.nan
        df_copy['RTR0'] = np.nan
        return df_copy

    aa = (df_copy['PHIE']**df_copy['M']) / (df_copy['A'] * df_copy['RWA_FULL'])
    cc = 2 - df_copy['VSH']
    bb = (df_copy['VSH']**cc) / df_copy['RTSH']

    # Hindari pembagian dengan nol atau nilai invalid
    denominator = aa + 2 * (aa * bb)**0.5 + bb
    R0 = np.where(denominator != 0, 1 / denominator, np.nan)

    df_copy['R0'] = R0
    df_copy['RTR0'] = df_copy['RT'] - df_copy['R0']
    return df_copy


def analyze_rtr0_groups(df):
    """
    Menganalisis gradien untuk setiap grup data reservoir (IQUAL=1).
    """
    results_rtr0 = []
    # Proses hanya pada data di mana IQUAL = 1
    df_reservoir = df[df['IQUAL'] == 1].copy()

    if df_reservoir.empty or 'GROUP_ID' not in df_reservoir.columns:
        return pd.DataFrame()

    for group_id, group in df_reservoir.groupby('GROUP_ID'):
        n = len(group)
        if n <= 1:
            continue

        # Perhitungan gradien RT vs R0
        sxa = group['R0'].sum()
        sya = group['RT'].sum()
        sxya = (group['R0'] * group['RT']).sum()
        sx2a = (group['R0']**2).sum()
        denominator_a = (sxa * sxa - n * sx2a)
        slope_rt2r0 = (sxa * sya - n * sxya) / \
            denominator_a if denominator_a != 0 else np.nan

        # Perhitungan gradien PHIE vs RTR0
        sxb = group['PHIE'].sum()
        syb = group['RTR0'].sum()
        sxyb = (group['PHIE'] * group['RTR0']).sum()
        sx2b = (group['PHIE']**2).sum()
        denominator_b = (sxb * sxb - n * sx2b)
        slope_phie2rtr0 = (sxb * syb - n * sxyb) / \
            denominator_b if denominator_b != 0 else np.nan

        fluid_rtrophie = 'G' if slope_phie2rtr0 > 0 else 'W' if not pd.isna(
            slope_phie2rtr0) else np.nan

        results_rtr0.append({
            'GROUP_ID': group_id,
            'RT_R0_GRAD': slope_rt2r0,
            'PHIE_RTR0_GRAD': slope_phie2rtr0,
            'FLUID_RTROPHIE': fluid_rtrophie
        })

    return pd.DataFrame(results_rtr0) if results_rtr0 else pd.DataFrame()


def process_rt_r0(df: pd.DataFrame, params: dict = None, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Fungsi utama untuk memproses analisis RT-R0, dengan penanganan filter internal.
    """
    if params is None:
        params = {}

    try:
        df_final = df.copy()

        # 1. Tentukan baris mana yang akan diproses berdasarkan filter
        mask = pd.Series(True, index=df_final.index)
        has_filters = False
        if target_intervals and 'MARKER' in df_final.columns:
            mask = df_final['MARKER'].isin(target_intervals)
            has_filters = True
        if target_zones and 'ZONE' in df_final.columns:
            mask = (mask | df_final['ZONE'].isin(
                target_zones)) if has_filters else df_final['ZONE'].isin(target_zones)

        # Buat DataFrame kerja dari baris yang dipilih
        df_to_process = df_final[mask].copy()
        if df_to_process.empty:
            print("Peringatan: Tidak ada data yang cocok dengan filter yang dipilih.")
            return df_final

        # 2. Lakukan semua perhitungan pada DataFrame kerja
        default_params = {'A': 1.0, 'M': 2.0,
                          'N': 2.0, 'RTSH': 2.2, 'RWA_FULL': 0.1}
        for p, val in default_params.items():
            if p not in df_to_process.columns:
                df_to_process[p] = params.get(f'{p}_PARAM', val)

        df_to_process = calculate_iqual(df_to_process)
        df_to_process = calculate_R0(df_to_process)
        df_to_process['GROUP_ID'] = (
            df_to_process['IQUAL'].diff() != 0).cumsum()

        df_results_rtr0 = analyze_rtr0_groups(df_to_process)

        if not df_results_rtr0.empty:
            df_to_process = df_to_process.merge(
                df_results_rtr0, on='GROUP_ID', how='left')

        # 3. Gabungkan hasilnya kembali ke DataFrame lengkap
        # Kolom yang akan di-update atau ditambahkan
        result_cols = [
            'IQUAL', 'R0', 'RTR0', 'GROUP_ID',
            'RT_R0_GRAD', 'PHIE_RTR0_GRAD', 'FLUID_RTROPHIE'
        ]
        # Pastikan hanya kolom yang ada di df_to_process yang akan di-merge
        cols_to_merge = [
            col for col in result_cols if col in df_to_process.columns]

        # Hapus kolom lama dari df_final untuk menghindari konflik
        df_final = df_final.drop(columns=cols_to_merge, errors='ignore')

        # Gunakan 'DEPTH' sebagai kunci unik untuk merge
        if 'DEPTH' in df_to_process.columns:
            df_final = pd.merge(
                df_final,
                df_to_process[['DEPTH'] + cols_to_merge],
                on='DEPTH',
                how='left'
            )
        else:
            print(
                "Error: Kolom 'DEPTH' tidak ditemukan, tidak dapat menggabungkan hasil.")
            return df

        return df_final

    except Exception as e:
        print(f"Error dalam process_rt_r0: {e}")
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
