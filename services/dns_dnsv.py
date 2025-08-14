# app.py

import os
import pandas as pd
from services.autoplot import calculate_nphi_rhob_intersection


def dns(rhob_in, nphi_in):
    """Calculate DNS (Density-Neutron Separation)"""
    return ((2.71 - rhob_in) / 1.71) - nphi_in


def dnsv(rhob_in, nphi_in, rhob_sh, nphi_sh, vsh):
    """Calculate DNSV (Density-Neutron Separation corrected for shale Volume)"""
    rhob_corv = rhob_in + vsh * (2.65 - rhob_sh)
    nphi_corv = nphi_in + vsh * (0 - nphi_sh)
    return ((2.71 - rhob_corv) / 1.71) - nphi_corv


def process_dns_dnsv(df: pd.DataFrame, params: dict = None, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Main function to process DNS-DNSV analysis with internal filtering.
    """
    if params is None:
        params = {}

    try:
        df_processed = df.copy()

        # 1. Prepare data and parameters
        # Make process idempotent by dropping old results
        df_processed.drop(columns=['DNS', 'DNSV'],
                          inplace=True, errors='ignore')

        # Rename VSH_LINEAR if it exists and VSH does not
        if 'VSH_LINEAR' in df_processed.columns and 'VSH' not in df_processed.columns:
            df_processed['VSH'] = df_processed['VSH_LINEAR']

        # Ensure required columns exist before proceeding
        required_cols = ['RHOB', 'NPHI', 'VSH']
        if not all(col in df_processed.columns for col in required_cols):
            print(
                "Warning: Required columns (RHOB, NPHI, VSH) not found. Skipping calculation.")
            return df

        # Coerce to numeric, turning errors into NaN
        for col in required_cols:
            df_processed[col] = pd.to_numeric(
                df_processed[col], errors='coerce')

        # Calculate shale point from the full dataset for consistency
        shale_point = calculate_nphi_rhob_intersection(
            df_processed, params.get(
                'prcntz_qz', 5), params.get('prcntz_wtr', 5)
        )
        nphi_sh = shale_point['nphi_sh']
        rhob_sh = shale_point['rhob_sh']

        # 2. Create a mask to select rows for calculation
        mask = pd.Series(True, index=df_processed.index)
        has_filters = False
        if target_intervals and 'MARKER' in df_processed.columns:
            mask = df_processed['MARKER'].isin(target_intervals)
            has_filters = True
        if target_zones and 'ZONE' in df_processed.columns:
            zone_mask = df_processed['ZONE'].isin(target_zones)
            mask = (mask | zone_mask) if has_filters else zone_mask

        # Also, ensure we only calculate on valid data points
        valid_data_mask = df_processed[required_cols].notna().all(axis=1)
        final_mask = mask & valid_data_mask

        if not final_mask.any():
            print(
                "Warning: No data matched the filter criteria. No calculations performed.")
            return df

        # 3. Perform calculations only on the masked (selected) rows
        print(f"Calculating DNS-DNSV for {final_mask.sum()} rows.")

        rhob_masked = df_processed.loc[final_mask, 'RHOB']
        nphi_masked = df_processed.loc[final_mask, 'NPHI']
        vsh_masked = df_processed.loc[final_mask, 'VSH']

        df_processed.loc[final_mask, 'DNS'] = dns(rhob_masked, nphi_masked)
        df_processed.loc[final_mask, 'DNSV'] = dnsv(
            rhob_masked, nphi_masked, rhob_sh, nphi_sh, vsh_masked)

        return df_processed

    except Exception as e:
        print(f"Error in process_dns_dnsv: {str(e)}")
        raise e
