import pandas as pd
import numpy as np


def calculate_rwa(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Calculates RWA (Full, Simple, Tar) with internal filtering for intervals/zones.
    """
    df_processed = df.copy()

    # 1. Get parameters and verify required columns
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    RT_SH = float(params.get('RT_SH', 5.0))

    required_cols = ['PHIE', 'RT', 'VSH']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(
            "Required columns (PHIE, RT, VSH) are not present in the data.")

    # Make process idempotent by dropping old results and re-initializing
    rwa_cols = ["RWA_FULL", "RWA_SIMPLE", "RWA_TAR"]
    df_processed.drop(
        columns=df_processed.columns.intersection(rwa_cols), inplace=True)
    for col in rwa_cols:
        df_processed[col] = np.nan

    # 2. Create a mask to select rows for calculation
    mask = pd.Series(True, index=df_processed.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_processed.columns:
        mask = df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        mask = (mask | zone_mask) if has_filters else zone_mask

    # Also ensure we only calculate on valid data points within the mask
    valid_data_mask = df_processed[required_cols].notna().all(axis=1)
    final_mask = mask & valid_data_mask

    if not final_mask.any():
        print(
            "Warning: No data matched the filter criteria. No RWA calculations performed.")
        return df_processed

    print(
        f"Calculating RWA for {final_mask.sum()} of {len(df_processed)} rows.")

    # 3. Perform calculations ONLY on the masked (selected) rows
    phie = df_processed.loc[final_mask, "PHIE"].copy()
    rt = df_processed.loc[final_mask, "RT"].copy()
    vsh = df_processed.loc[final_mask, "VSH"].copy()

    # Handle zero values in the selection
    rt[rt == 0] = np.nan
    phie[phie == 0] = np.nan

    # Common calculations for the selection
    f1 = (phie ** M) / A
    f2 = 1 / rt

    # Full Indonesia calculation
    v_full = vsh ** (2 - vsh)
    f3_full = v_full / RT_SH
    f4_full = np.sqrt(v_full / (rt * RT_SH))
    rwaf = f1 / (f2 + f3_full - f4_full)
    df_processed.loc[final_mask, "RWA_FULL"] = rwaf.clip(lower=0)

    # Simple Indonesia calculation
    v_simple = vsh ** 2
    f3_simple = v_simple / RT_SH
    f4_simple = np.sqrt(v_simple / (rt * RT_SH))
    rwas = f1 / (f2 + f3_simple - f4_simple)
    df_processed.loc[final_mask, "RWA_SIMPLE"] = rwas.clip(lower=0)

    # Tar Sand calculation
    v_tar = vsh ** (2 - 2 * vsh)
    f3_tar = v_tar / RT_SH
    f4_tar = np.sqrt(v_tar / (rt * RT_SH))
    rwat = f1 / (f2 + f3_tar - f4_tar)
    df_processed.loc[final_mask, "RWA_TAR"] = rwat.clip(lower=0)

    print("Added/updated RWA columns: RWA_FULL, RWA_SIMPLE, RWA_TAR")
    return df_processed
