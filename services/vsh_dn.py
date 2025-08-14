import pandas as pd
import numpy as np


def calculate_vsh_dn(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Calculates VSH from Density-Neutron crossplot with internal filtering.
    """
    df_processed = df.copy()

    # 1. Get parameters and verify required columns
    RHO_MA = float(params.get('RHO_MA', 2.645))
    RHO_SH = float(params.get('RHO_SH', 2.61))
    RHO_FL = float(params.get('RHO_FL', 0.85))
    NPHI_MA = float(params.get('NPHI_MA', -0.02))
    NPHI_SH = float(params.get('NPHI_SH', 0.398))
    NPHI_FL = float(params.get('NPHI_FL', 0.85))

    RHO_LOG = params.get('RHOB', 'RHOB')
    NPHI_LOG = params.get('NPHI', 'NPHI')
    VSH_OUTPUT_LOG = params.get('VSH', 'VSH_DN')

    required_cols = [RHO_LOG, NPHI_LOG]
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Required input columns {required_cols} not found.")

    # Make process idempotent by dropping old results and re-initializing
    vsh_cols = [VSH_OUTPUT_LOG, 'VSH_DIFF']
    df_processed.drop(
        columns=df_processed.columns.intersection(vsh_cols), inplace=True)
    for col in vsh_cols:
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
        print("Warning: No data matched the filter criteria. No VSH-DN calculations performed.")
        return df_processed

    print(
        f"Calculating VSH-DN for {final_mask.sum()} of {len(df_processed)} rows.")

    # 3. Perform calculations ONLY on the masked (selected) rows
    rho_log_masked = df_processed.loc[final_mask, RHO_LOG]
    nphi_log_masked = df_processed.loc[final_mask, NPHI_LOG]

    a = (RHO_MA - RHO_FL) * (NPHI_FL - nphi_log_masked)
    b = (rho_log_masked - RHO_FL) * (NPHI_FL - NPHI_MA)
    c = (RHO_MA - RHO_FL) * (NPHI_FL - NPHI_SH)
    d = (RHO_SH - RHO_FL) * (NPHI_FL - NPHI_MA)

    denominator = c - d
    if denominator != 0:
        vsh_dn_calculated = (a - b) / denominator
        df_processed.loc[final_mask,
                         VSH_OUTPUT_LOG] = vsh_dn_calculated.clip(0, 1)
    else:
        df_processed.loc[final_mask, VSH_OUTPUT_LOG] = np.nan

    # Optional: Calculate difference if VSH_GR exists
    if 'VSH_GR' in df_processed.columns:
        vsh_gr_masked = df_processed.loc[final_mask, 'VSH_GR']
        vsh_dn_masked = df_processed.loc[final_mask, VSH_OUTPUT_LOG]
        df_processed.loc[final_mask,
                         'VSH_DIFF'] = vsh_gr_masked - vsh_dn_masked

    print(f"Added/updated VSH-DN columns: {VSH_OUTPUT_LOG}, VSH_DIFF")
    return df_processed
