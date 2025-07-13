import numpy as np
import pandas as pd
from scipy.stats import linregress

def indonesia_computation(rw_in, phie, ct, params):
    """Calculate water saturation using Indonesia method"""
    a = params.get('A_PARAM', 1)
    m = params.get('M_PARAM', 1.8)
    n = params.get('N_PARAM', 1.8)
    rtsh = params.get('RTSH', 1)
    vsh = params.get('VSH', 0)

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
    """Main function to process SWGRAD analysis"""
    if params is None:
        params = {}

    try:
        # Initialize SWARRAY columns
        for i in range(1, 26):
            df[f'SWARRAY_{i}'] = np.nan
        df['SWGRAD'] = np.nan

        # Calculate CT
        df['CT'] = 1 / df['RT']

        # Calculate formation temperature
        df['FTEMP'] = 75 + 0.05 * df['DEPTH']

        # Process each row
        for i in range(len(df)):
            sal = np.zeros(26)
            x = np.zeros(26)
            sw = np.zeros(26)

            # Calculate for each salinity
            for j in range(1, 26):
                sal[j] = j * 1000
                x[j] = 0.0123 + 3647.5 / sal[j]**0.955
                rw_in = x[j] * 81.77 / (df['FTEMP'].iloc[i] + 6.77)

                # Calculate water saturation
                sw[j] = indonesia_computation(
                    rw_in, 
                    df['PHIE'].iloc[i], 
                    df['CT'].iloc[i], 
                    params
                )

                # Store in SWARRAY
                df.iloc[i, df.columns.get_loc(f'SWARRAY_{j}')] = sw[j]

            # Calculate SWGRAD
            try:
                data_SW = np.array([sw[5*k] for k in range(2, 6)])
                data_SAL = np.array([5*k for k in range(2, 6)])
                SWGRAD, _, _, _, _ = linregress(data_SAL, data_SW)
                df.iloc[i, df.columns.get_loc('SWGRAD')] = SWGRAD
            except:
                df.iloc[i, df.columns.get_loc('SWGRAD')] = np.nan

        return df

    except Exception as e:
        print(f"Error in process_swgrad: {str(e)}")
        raise e
