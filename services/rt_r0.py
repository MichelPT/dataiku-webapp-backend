import numpy as np
import pandas as pd
from scipy.stats import linregress

def calculate_iqual(df):
    """Calculate IQUAL based on conditions"""
    df = df.copy()
    df['IQUAL'] = np.where((df['PHIE'] > 0.1) & (df['VSH'] < 0.5), 1, 0)
    return df

def calculate_R0(df, params):
    """Calculate R0 and related parameters"""
    rwa = df['RT'] * df['PHIE']**params.get('M_PARAM', 1.8)
    aa = df['PHIE']**params.get('M_PARAM', 1.8) / (params.get('A_PARAM', 1) * params.get('RW', 1))
    cc = 2 - df['VSH']
    bb = df['VSH']**cc / params.get('RTSH', 1)

    R0 = 1 / (aa + 2 * (aa * bb)**0.5 + bb)
    df['RWA'] = rwa
    df['R0'] = R0
    df['RTR0'] = df['RT'] - df['R0']
    return df

def analyze_rtr0_groups(df):
    """Analyze RTR0 for each group in the well"""
    results_rtr0 = []
    df = df.copy()
    df['GROUP_ID'] = (df['IQUAL'].diff() != 0).cumsum()

    for group_id, group in df.groupby('GROUP_ID'):
        n = len(group)
        if ((group['PHIE'].nunique() == 1) | (group['RT'].nunique() == 1) | (n <= 1)):
            continue

        try:
            slope_rt2r0, _, _, _, _ = linregress(group['RT'], group['R0'])
            slope_phie2rtr0, _, _, _, _ = linregress(group['PHIE'], group['RTR0'])

            if np.isnan(slope_phie2rtr0) or np.isinf(slope_phie2rtr0):
                continue

            FLUID_RTROPHIE = 'G' if slope_phie2rtr0 > 0 else 'W'

            results_rtr0.append({
                'GROUP_ID': group_id,
                'RT_R0_GRAD': slope_rt2r0,
                'PHIE_RTR0_GRAD': slope_phie2rtr0,
                'FLUID_RTROPHIE': FLUID_RTROPHIE
            })
        except Exception:
            continue

    return pd.DataFrame(results_rtr0) if results_rtr0 else pd.DataFrame()

def process_rt_r0(df, params=None):
    """Main function to process RT-R0 analysis"""
    if params is None:
        params = {}
    
    try:
        # Calculate IQUAL
        df = calculate_iqual(df)

        # Calculate R0 and RTR0
        df = calculate_R0(df, params)

        # Analyze RTR0 groups
        df_results = analyze_rtr0_groups(df)

        if not df_results.empty:
            df = df.merge(df_results, on='GROUP_ID', how='left')

        return df
    except Exception as e:
        print(f"Error in process_rt_r0: {str(e)}")
        raise e
