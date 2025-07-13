# In your services/rt_r0.py file

import numpy as np
import pandas as pd
from scipy.stats import linregress

def calculate_iqual(df):
    """Calculate IQUAL based on conditions"""
    df['IQUAL'] = np.where((df['PHIE'] > 0.1) & (df['VSH'] < 0.5), 1, 0)
    return df

def calculate_R0(df, params):
    """Calculate R0 and related parameters"""
    # Ensure all required columns are numeric
    for col in ['RT', 'PHIE', 'VSH']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['RT', 'PHIE', 'VSH'], inplace=True)
    
    # Safely get parameters
    m_param = params.get('M_PARAM', 1.8)
    a_param = params.get('A_PARAM', 1)
    rw_param = params.get('RW', 1)
    rtsh_param = params.get('RTSH', 1)

    rwa = df['RT'] * df['PHIE']**m_param
    
    aa = df['PHIE']**m_param / (a_param * rw_param) if (a_param * rw_param) != 0 else 0
    cc = 2 - df['VSH']
    bb = df['VSH']**cc / rtsh_param if rtsh_param != 0 else 0
    
    # Calculate R0 with safety for square root
    sqrt_term_val = aa * bb
    sqrt_term = np.sqrt(sqrt_term_val, where=(sqrt_term_val >= 0), out=np.full_like(sqrt_term_val, np.nan))
    
    denominator = aa + 2 * sqrt_term + bb
    R0 = 1 / denominator
    
    df['RWA'] = rwa
    df['R0'] = R0
    df['RTR0'] = df['RT'] - df['R0']
    return df

def analyze_rtr0_groups(df):
    """Analyze RTR0 for each group in the well. The input df must already have GROUP_ID."""
    results_rtr0 = []
    for group_id, group in df.groupby('GROUP_ID'):
        n = len(group)
        if n <= 1 or group['PHIE'].nunique() == 1 or group['RT'].nunique() == 1:
            continue

        try:
            # Drop NaN/inf values for regression
            clean_group = group.dropna(subset=['RT', 'R0', 'PHIE', 'RTR0']).replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_group) <= 1:
                continue

            slope_rt2r0, _, _, _, _ = linregress(clean_group['RT'], clean_group['R0'])
            slope_phie2rtr0, _, _, _, _ = linregress(clean_group['PHIE'], clean_group['RTR0'])
            
            if np.isnan(slope_phie2rtr0) or np.isinf(slope_phie2rtr0):
                continue

            FLUID_RTROPHIE = 'G' if slope_phie2rtr0 > 0 else 'W'
            results_rtr0.append({
                'GROUP_ID': group_id,
                'RT_R0_GRAD': slope_rt2r0,
                'PHIE_RTR0_GRAD': slope_phie2rtr0,
                'FLUID_RTROPHIE': FLUID_RTROPHIE
            })
        except (ValueError, TypeError):
            continue

    return pd.DataFrame(results_rtr0) if results_rtr0 else pd.DataFrame(columns=['GROUP_ID'])

def process_rt_r0(df, params=None):
    """Main function to process RT-R0 analysis."""
    if params is None:
        params = {}
    
    try:
        # Calculate IQUAL first
        df = calculate_iqual(df)
        
        # ✨ FIX IS HERE: Create GROUP_ID on the main DataFrame
        # This column is now available for both analyze_rtr0_groups and the final merge.
        df['GROUP_ID'] = (df['IQUAL'].diff() != 0).cumsum()

        # Calculate R0 and RTR0
        df = calculate_R0(df, params)

        # Analyze RTR0 groups (now that df has GROUP_ID)
        df_results = analyze_rtr0_groups(df)

        # Merge results back if any were generated
        if not df_results.empty:
            df = df.merge(df_results, on='GROUP_ID', how='left')

        return df
    except Exception as e:
        print(f"Error in process_rt_r0: {e}")
        raise e