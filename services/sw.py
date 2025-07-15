import pandas as pd
import numpy as np


def calculate_sw(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Main function to calculate Water Saturation (SW Indonesia) and reservoir classification.
    """
    df_processed = df.copy()

    # Extract parameters from frontend with safe defaults
    RWS = float(params.get('RWS', 0.529))
    RWT = float(params.get('RWT', 227))
    FTEMP = float(params.get('FTEMP', 80))
    RT_SH = float(params.get('RT_SH', 2.2))
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))
    SW = 'SW'
    VSH = 'VSH'
    PHIE = 'PHIE'
    RT = 'RT'

    # Verify required columns exist
    required_cols = ['GR', 'RT', 'PHIE', 'VSH']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(
            "Required input columns (GR, RT, PHIE, VSH) not complete. Run previous modules first.")

    print("Calculating RW at formation temperature...")
    df_processed["RW_TEMP"] = RWS * (RWT + 21.5) / (FTEMP + 21.5)

    print("Calculating Water Saturation (SW Indonesia)...")
    v = df_processed[VSH] ** 2
    ff = A / df_processed[PHIE] ** M

    # Avoid division by zero
    ff_times_rw_temp = ff * df_processed["RW_TEMP"]
    ff_times_rw_temp[ff_times_rw_temp == 0] = np.nan

    f1 = 1 / ff_times_rw_temp
    f2 = 2 * np.sqrt(v / (ff_times_rw_temp * RT_SH))
    f3 = v / RT_SH

    denom = f1 + f2 + f3
    denom[denom == 0] = np.nan

    df_processed[SW] = (1 / (df_processed[RT] * denom)) ** (1 / N)
    df_processed.loc[df_processed[PHIE] < 0.005, SW] = 1.0
    df_processed[SW] = df_processed[SW].clip(lower=0, upper=1)

    return df_processed
