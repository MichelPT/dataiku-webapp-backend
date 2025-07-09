import numpy as np
import pandas as pd

def dns(rhob_in, nphi_in):
    """Calculate DNS (Density-Neutron Separation)"""
    return ((2.71 - rhob_in) / 1.71) - nphi_in

def dnsv(rhob_in, nphi_in, rhob_sh, nphi_sh, vsh):
    """Calculate DNSV (Density-Neutron Separation corrected for shale Volume)"""
    rhob_corv = rhob_in + vsh * (2.65 - rhob_sh)
    nphi_corv = nphi_in + vsh * (0 - nphi_sh)
    return ((2.71 - rhob_corv) / 1.71) - nphi_corv

def process_dns_dnsv(df, params=None):
    """Main function to process DNS-DNSV analysis"""
    if params is None:
        params = {}

    try:
        rhob_sh = params.get('RHOB_SH', 0.35)
        nphi_sh = params.get('NPHI_SH', 2.528)

        # Calculate DNS and DNSV
        df['DNS'] = dns(df['RHOB'], df['NPHI'])
        df['DNSV'] = dnsv(df['RHOB'], df['NPHI'], rhob_sh, nphi_sh, df['VSH'])

        return df

    except Exception as e:
        print(f"Error in process_dns_dnsv: {str(e)}")
        raise e
