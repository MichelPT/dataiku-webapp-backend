# In your services/rt_r0.py file

import numpy as np
import pandas as pd
from scipy.stats import linregress

def calculate_iqual(df):
    """
    Menghitung IQUAL berdasarkan kondisi:
    IF (PHIE>0.1)&&(VSH<0.5): IQUAL =1
    else: IQUAL =0
    """
    df = df.copy()
    df['IQUAL'] = np.where((df['PHIE'] > 0.1) & (df['VSH'] < 0.5), 1, 0)
    return df

def calculate_R0(df):
    """
    Menghitung R0 dan parameter terkait
    """
    rwa = df['RT'] * df['PHIE']**df['M']
    aa = df['PHIE']**df['M'] / (df['A']*df['RW'])
    cc = 2 - df['VSH']
    bb = df['VSH']**cc / df['RTSH']

    R0 = 1 / (aa + 2 * (aa * bb)**0.5 + bb)
    df['RWA'] = rwa
    df['R0'] = R0
    df['RTR0'] = df['RT'] - df['R0']
    return df

def analyze_rtr0_groups(df):
    """
    Analisis RTR0 untuk setiap group dalam satu well
    """
    results_rtr0 = []

    for group_id, group in df.groupby('GROUP_ID'):
        n = len(group)

        # Hanya memproses group dengan n > 1
        if ((group['PHIE'].nunique() == 1) | (group['RT'].nunique() == 1) | (n <= 1)):
            continue

        try:
            # Regresi linear untuk slope dan r-squared
            slope_rt2r0, _, _, _, _ = linregress(group['RT'], group['R0'])
            slope_phie2rtr0, _, _, _, _ = linregress(group['PHIE'], group['RTR0'])

            # Validasi hasil regresi
            if np.isnan(slope_phie2rtr0) or np.isinf(slope_phie2rtr0):
                continue

            # Mengikuti kode acuan persis
            condition = slope_phie2rtr0 > 0
            FLUID_RTROPHIE = np.where(condition, 'G', 'W')

            # Pastikan FLUID_RTROPHIE adalah scalar string
            if isinstance(FLUID_RTROPHIE, np.ndarray):
                FLUID_RTROPHIE = FLUID_RTROPHIE.item()  # Konversi array ke scalar

            # List hasil
            results_rtr0.append({
                'GROUP_ID': group_id,
                'RT_R0_GRAD': slope_rt2r0,
                'PHIE_RTR0_GRAD': slope_phie2rtr0,
                'FLUID_RTROPHIE': FLUID_RTROPHIE
            })

        except Exception as e:
            print(f"Warning: Error processing group {group_id}: {str(e)}")
            continue

    return pd.DataFrame(results_rtr0)

def process_rt_r0(df, params=None):
    """
    Main function to process RT-R0 analysis
    """
    if params is None:
        params = {}
    
    try:
        # Tambahkan parameter default jika belum ada
        if 'A' not in df.columns:
            df['A'] = params.get('A_PARAM', 1)
        if 'M' not in df.columns:
            df['M'] = params.get('M_PARAM', 1.8)
        if 'N' not in df.columns:
            df['N'] = params.get('N_PARAM', 1.8)
        if 'RTSH' not in df.columns:
            df['RTSH'] = params.get('RTSH', 1)
        if 'RW' not in df.columns:
            df['RW'] = params.get('RW', 1)

        # Step 1: Calculate IQUAL
        df = calculate_iqual(df)

        # Step 2: Calculate R0 and RTR0
        df = calculate_R0(df)

        # Step 3: Group by sequence
        df['GROUP_ID'] = (df['IQUAL'].diff() != 0).cumsum()

        # Step 4: Analyze RTR0 groups
        df_results_rtr0 = analyze_rtr0_groups(df)

        # Step 5: Merge results
        if not df_results_rtr0.empty:
            df = df.merge(df_results_rtr0, on='GROUP_ID', how='left')

        return df

    except Exception as e:
        print(f"Error in process_rt_r0: {e}")
        raise e