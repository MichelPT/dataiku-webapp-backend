import pandas as pd
import numpy as np

def calculate_ftemp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Formation Temperature (FTEMP) log based on the TVDSS log.

    Formula: FTEMP = 75 + (0.05 * TVDSS)

    Args:
        df: DataFrame containing a 'TVDSS' column.

    Returns:
        DataFrame with a new 'FTEMP' column added.
    """
    df_processed = df.copy()

    # Check if the required TVDSS column exists
    if 'TVDSS' not in df_processed.columns:
        raise ValueError("Input DataFrame must contain a 'TVDSS' column.")

    # Apply the formula vectorized for efficiency
    df_processed['FTEMP'] = 75 + (0.05 * df_processed['TVDSS'])
    
    print("Successfully created 'FTEMP' log.")
    return df_processed

def newton_simandoux(rt, ff, rwtemp, rtsh, vsh, n, opt='MODIFIED', c=1, max_iter=20, tol=1e-5):
    """
    Newton-Raphson method for solving Simandoux equation for water saturation.

    Args:
        rt: True resistivity
        ff: Formation factor
        rwtemp: Formation water resistivity at formation temperature
        rtsh: Shale resistivity
        vsh: Shale volume
        n: Saturation exponent
        opt: 'MODIFIED' or 'SCHLUMBERGER'
        c: Shale exponent (for Schlumberger)
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Water saturation value
    """
    if opt == 'MODIFIED':
        g1 = 1 / (ff * rwtemp)
        g2 = vsh / rtsh
    else:  # Schlumberger
        g1 = 1 / (ff * rwtemp * (1 - vsh))
        g2 = (vsh ** c) / rtsh

    g3 = -1 / rt
    sw = 0.5  # Initial guess

    for _ in range(max_iter):
        fx = g1 * sw ** n + g2 * sw + g3
        fxp = n * g1 * sw ** (n - 1) + g2

        if fxp == 0:
            return np.nan

        delta = fx / fxp
        sw -= delta
        sw = max(0, sw)  # Ensure sw stays positive

        if abs(delta) < tol:
            return sw

    return np.nan


def calculate_sw_simandoux(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Calculates Water Saturation (Simandoux) with internal filtering.
    """
    df_processed = df.copy()
    df_processed.columns = [col.upper() for col in df_processed.columns]
    df_processed.replace(-999.0, np.nan, inplace=True)
    df_processed = calculate_ftemp(df_processed)

    # 1. Get parameters and check for required columns
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))
    C = float(params.get('C', 1.0))
    RWS = float(params.get('RWS', 0.1))
    RWT = float(params.get('RWT', 75))
    SWE_IRR = float(params.get('SWE_IRR', 0.0))
    RT_SH = float(params.get('RT_SH', 5.0))
    OPT_SIM = params.get('OPT_SIM', 'MODIFIED')

    if 'ILD' not in df_processed.columns and 'RT' in df_processed.columns:
        df_processed['ILD'] = df_processed['RT']

    required_cols = ['GR', 'RHOB', 'ILD', 'FTEMP']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")

    # 2. Create a mask for filtering
    mask = pd.Series(True, index=df_processed.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_processed.columns:
        mask = df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        mask = (mask | zone_mask) if has_filters else zone_mask

    # 3. Prepare data (VSH, PHIE) for the masked rows if they don't exist
    df_processed["RW_TEMP"] = RWS * (RWT + 21.5) / (df_processed['FTEMP'] + 21.5)

    if 'VSH' not in df_processed.columns:
        print("Calculating VSH from GR for selected intervals...")
        GR_clean = df_processed.loc[mask, "GR"].quantile(0.05)
        GR_shale = df_processed.loc[mask, "GR"].quantile(0.95)
        if GR_shale > GR_clean:
            vsh_calc = (df_processed.loc[mask, "GR"] -
                        GR_clean) / (GR_shale - GR_clean)
            df_processed.loc[mask, 'VSH'] = vsh_calc.clip(0, 1)
        else:
            df_processed.loc[mask, 'VSH'] = 0

    if 'PHIE' not in df_processed.columns:
        print("Calculating PHIE from density for selected intervals...")
        RHO_MA = float(params.get('RHO_MA', 2.65))
        RHO_SH = float(params.get('RHO_SH', 2.45))
        RHO_FL = float(params.get('RHO_FL', 1.0))
        if (RHO_MA - RHO_FL) != 0:
            phie_calc = ((RHO_MA - df_processed.loc[mask, "RHOB"]) / (RHO_MA - RHO_FL)) - \
                df_processed.loc[mask, "VSH"] * \
                ((RHO_MA - RHO_SH) / (RHO_MA - RHO_FL))
            df_processed.loc[mask, 'PHIE'] = phie_calc.clip(lower=0)
        else:
            df_processed.loc[mask, 'PHIE'] = 0

    # 4. Perform calculations ONLY on the masked rows
    print(
        f"Calculating Simandoux SW for {mask.sum()} of {len(df_processed)} rows.")

    SW_COL = 'SW'
    df_processed[SW_COL] = np.nan
    df_processed['VOL_UWAT'] = np.nan

    # Apply the calculation row-by-row only on the filtered data
    def apply_newton(row):
        if pd.isna(row["ILD"]) or pd.isna(row["PHIE"]) or pd.isna(row["VSH"]):
            return np.nan
        if row["PHIE"] < 0.005:
            return 1.0
        ff = A / (row["PHIE"] ** M)
        return newton_simandoux(
            rt=row["ILD"], ff=ff, rwtemp=row["RW_TEMP"], rtsh=RT_SH,
            vsh=row["VSH"], n=N, opt=OPT_SIM, c=C
        )

    # Use .apply() for a cleaner, though not necessarily faster, implementation on the masked data
    sw_values = df_processed[mask].apply(apply_newton, axis=1)

    # Assign calculated values back to the main DataFrame using the mask
    df_processed.loc[mask, SW_COL] = sw_values

    # Clip the results and calculate final parameters on the masked data
    df_processed.loc[mask, SW_COL] = df_processed.loc[mask,
                                                      SW_COL].clip(lower=SWE_IRR, upper=1.0)
    df_processed.loc[mask, "VOL_UWAT"] = df_processed.loc[mask,
                                                          "PHIE"] * df_processed.loc[mask, SW_COL]

    print("Water Saturation calculation (Simandoux) completed.")
    return df_processed


def calculate_sw_lama(df: pd.DataFrame, params: dict, target_intervals: list = None, target_zones: list = None) -> pd.DataFrame:
    """
    Fungsi utama untuk menghitung Saturasi Air (SW Indonesia) dengan filter internal.
    """
    df_processed = df.copy()

    # 1. Persiapan: Ekstrak parameter dan verifikasi kolom
    RWS = float(params.get('RWS', 0.529))
    RWT = float(params.get('RWT', 227))
    RT_SH = float(params.get('RT_SH', 2.2))
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))

    # Nama kolom
    SW = 'SW'
    VSH = 'VSH_LINEAR'
    PHIE = 'PHIE'
    RT = 'RT'
    RW_TEMP = "RW_TEMP"

    required_cols = ['GR', RT, PHIE, VSH, 'FTEMP']
    if not all(col in df_processed.columns for col in required_cols):
        raise ValueError(
            "Kolom input (GR, RT, PHIE, VSH) tidak lengkap. Jalankan modul sebelumnya.")

    # 2. Buat mask untuk memilih baris yang akan diproses
    mask = pd.Series(True, index=df_processed.index)
    has_filters = False
    if target_intervals and 'MARKER' in df_processed.columns:
        mask = df_processed['MARKER'].isin(target_intervals)
        has_filters = True
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        mask = (mask | zone_mask) if has_filters else zone_mask

    if not mask.any():
        print("Peringatan: Tidak ada data yang cocok dengan filter. Tidak ada kalkulasi yang dilakukan.")
        return df  # Kembalikan DataFrame asli jika tidak ada yang cocok

    print(f"Menghitung SW untuk {mask.sum()} dari {len(df_processed)} baris.")

    # 3. Lakukan perhitungan HANYA pada baris yang cocok dengan mask

    # Hitung RW_TEMP untuk semua baris karena mungkin dibutuhkan di modul lain
    df_processed[RW_TEMP] = RWS * (RWT + 21.5) / (df_processed['FTEMP'] + 21.5)

    # Ambil data yang sudah difilter untuk kalkulasi
    v = df_processed.loc[mask, VSH] ** 2
    phie_masked = df_processed.loc[mask, PHIE]
    ff = A / phie_masked ** M

    ff_times_rw_temp = ff * df_processed.loc[mask, RW_TEMP]
    # Hindari pembagian dengan nol
    ff_times_rw_temp[ff_times_rw_temp == 0] = np.nan

    f1 = 1 / ff_times_rw_temp
    f2 = 2 * np.sqrt(v / (ff_times_rw_temp * RT_SH))
    f3 = v / RT_SH

    denom = f1 + f2 + f3
    denom[denom == 0] = np.nan

    # Hitung SW dan terapkan pada baris yang difilter
    sw_calculated = (1 / (df_processed.loc[mask, RT] * denom)) ** (1 / N)

    # Inisialisasi kolom SW jika belum ada
    if SW not in df_processed.columns:
        df_processed[SW] = np.nan

    df_processed.loc[mask, SW] = sw_calculated

    # Terapkan kondisi dan batasan pada kolom SW yang sudah diupdate
    df_processed.loc[df_processed[PHIE] < 0.005, SW] = 1.0
    df_processed[SW] = df_processed[SW].clip(lower=0, upper=1)

    return df_processed

# def calculate_sw_full_code(
#     df: pd.DataFrame,
#     params: dict,
#     target_intervals: list = None,
#     target_zones: list = None
# ) -> pd.DataFrame:
#     """
#     Corrected SW calculation following the exact LLS certified code logic.
#     """
#     df_processed = df.copy()
    
#     # 1. Extract parameters
#     OPT_INDO = params.get('OPT_INDO', 'FULL').upper()  # Fixed: was hardcoded
#     OPT_RW = params.get('OPT_RW', 'MEASURED').upper()
#     OPT_M = params.get('OPT_M', 'CONSTANT').upper()
    
#     # Numeric parameters
#     RWS = float(params.get('RWS', 0.529))
#     RWT = float(params.get('RWT', 227))
#     RT_SH = float(params.get('RT_SH', 2.2))
#     RMF = params.get('RMF')  # Can be None
#     A = float(params.get('A', 1.0))
#     M = float(params.get('M', 2.0))
#     N = float(params.get('N', 2.0))
#     SWE_IRR = float(params.get('SWE_IRR', 0.0))
    
#     # 2. Verify required columns
#     required_cols = ['PHIE', 'VSH_LINEAR', 'RT']
#     if OPT_RW == 'MEASURED':
#         required_cols.append('FTEMP')
#     elif OPT_RW == 'LOG':
#         required_cols.append('RW')
#     if OPT_M == 'VARIABLE':
#         required_cols.append('M_EXP')
        
#     missing_cols = [col for col in required_cols if col not in df_processed.columns]
#     if missing_cols:
#         raise ValueError(f"Missing required columns: {missing_cols}")
    
#     # 3. Initialize output columns (following LLS naming)
#     output_cols = ['OPT_SW', 'SWE', 'SWE_INDO', 'VOL_UWAT', 'SXOE', 'VOL_XWAT']
#     for col in output_cols:
#         if col == 'OPT_SW':
#             df_processed[col] = ''
#         else:
#             df_processed[col] = np.nan
    
#     # 4. Apply filters
#     mask = pd.Series(True, index=df_processed.index)
#     if target_intervals and 'MARKER' in df_processed.columns:
#         mask = df_processed['MARKER'].isin(target_intervals)
#     if target_zones and 'ZONE' in df_processed.columns:
#         zone_mask = df_processed['ZONE'].isin(target_zones)
#         mask = mask | zone_mask if target_intervals else zone_mask
    
#     if not mask.any():
#         print("Warning: No data matches filter criteria")
#         return df_processed
    
#     # 5. Process each row following LLS logic
#     for idx in df_processed[mask].index:
#         vsh = df_processed.loc[idx, 'VSH_LINEAR']
#         phie = df_processed.loc[idx, 'PHIE']
#         rt = df_processed.loc[idx, 'RT']
        
#         # Step 1: Assign saturation method and calculate shale term
#         if OPT_INDO == 'FULL':
#             df_processed.loc[idx, 'OPT_SW'] = 'INDO_FUL'
#             v = vsh ** (2 - vsh)
#         elif OPT_INDO == 'SIMPLE':
#             df_processed.loc[idx, 'OPT_SW'] = 'INDO_SIM'
#             v = vsh ** 2
#         elif OPT_INDO == 'TAR_SAND':
#             df_processed.loc[idx, 'OPT_SW'] = 'INDO_TAR'
#             v = vsh ** (2 - 2 * vsh)
#         else:
#             # Default to SIMPLE if unknown
#             df_processed.loc[idx, 'OPT_SW'] = 'INDO_SIM'
#             v = vsh ** 2
        
#         # Step 2: Check for low effective porosity
#         if phie < 0.005:
#             df_processed.loc[idx, 'SWE'] = 1.0
#             df_processed.loc[idx, 'SWE_INDO'] = 1.0
#             df_processed.loc[idx, 'VOL_UWAT'] = phie
#             df_processed.loc[idx, 'SXOE'] = 1.0
#             df_processed.loc[idx, 'VOL_XWAT'] = phie
#             continue  # Skip to next iteration (equivalent to goto END_MODEL)
        
#         # Step 3: Calculate Rw
#         if OPT_RW == 'MEASURED':
#             ftemp = df_processed.loc[idx, 'FTEMP']
#             rwtemp = RWS * (RWT + 21.5) / (ftemp + 21.5)
#         elif OPT_RW == 'LOG':
#             rwtemp = df_processed.loc[idx, 'RW']
#         # Note: SALINITY option not implemented as it wasn't in your original code
#         else:
#             # Default to MEASURED method
#             ftemp = df_processed.loc[idx, 'FTEMP'] if 'FTEMP' in df_processed.columns else 75
#             rwtemp = RWS * (RWT + 21.5) / (ftemp + 21.5)
        
#         # Step 4: Choose constant or variable cementation exponent
#         if OPT_M == 'CONSTANT':
#             mtemp = M
#         else:  # VARIABLE
#             if pd.isna(df_processed.loc[idx, 'M_EXP']):
#                 # Set all values to missing and continue (following LLS logic)
#                 df_processed.loc[idx, 'SWE_INDO'] = np.nan
#                 df_processed.loc[idx, 'SWE'] = np.nan
#                 df_processed.loc[idx, 'SXOE'] = np.nan
#                 df_processed.loc[idx, 'VOL_UWAT'] = np.nan
#                 df_processed.loc[idx, 'VOL_XWAT'] = np.nan
#                 continue
#             mtemp = df_processed.loc[idx, 'M_EXP']
        
#         # Step 5: Calculate formation factor
#         ff = A / (phie ** mtemp)
        
#         # Step 6: Calculate SWE (following exact LLS formula)
#         f1 = 1 / (ff * rwtemp)
#         f2 = 2 * np.sqrt(v / (rwtemp * ff * RT_SH))
#         f3 = v / RT_SH
        
#         swe = (1 / (rt * (f1 + f2 + f3))) ** (1/N)
#         df_processed.loc[idx, 'SWE_INDO'] = swe  # Store original value
#         df_processed.loc[idx, 'SWE'] = np.clip(swe, SWE_IRR, 1.0)  # Apply limits
#         df_processed.loc[idx, 'VOL_UWAT'] = phie * df_processed.loc[idx, 'SWE']
        
#         # Step 7: Calculate SXOE
#         if 'RXO' in df_processed.columns and not pd.isna(df_processed.loc[idx, 'RXO']) and RMF is not None:
#             rxo = df_processed.loc[idx, 'RXO']
#             f1_xo = 1 / (ff * RMF)
#             f2_xo = 2 * np.sqrt(v / (RMF * ff * RT_SH))
#             f3_xo = v / RT_SH
            
#             sxoe = (1 / (rxo * (f1_xo + f2_xo + f3_xo))) ** (1/N)
#             df_processed.loc[idx, 'SXOE'] = np.clip(sxoe, SWE_IRR, 1.0)
#             df_processed.loc[idx, 'VOL_XWAT'] = phie * df_processed.loc[idx, 'SXOE']
#         else:
#             df_processed.loc[idx, 'SXOE'] = np.nan  # Missing, as per LLS
#             df_processed.loc[idx, 'VOL_XWAT'] = df_processed.loc[idx, 'VOL_UWAT']  # Copy VOL_UWAT
    
#     return df_processed


# # Simplified version that only returns SW (SWE column)
# def calculate_sw(
#     df: pd.DataFrame,
#     params: dict,
#     target_intervals: list = None,
#     target_zones: list = None
# ) -> pd.DataFrame:
#     """
#     Simplified version that only calculates SW following LLS standard.
#     Returns only the SWE column as 'SW'.
#     """
#     # Use the corrected function
#     result_df = calculate_sw_full_code(df, params, target_intervals, target_zones)
    
#     # Keep only the original columns plus SW
#     original_cols = df.columns.tolist()
#     result_df['SW'] = result_df['SWE']  # Rename SWE to SW for consistency
    
#     # Return only original columns plus SW
#     return result_df[original_cols + ['SW']]

def calculate_sw_with_shale_cutoff(
    df: pd.DataFrame,
    params: dict,
    target_intervals: list = None,
    target_zones: list = None
) -> pd.DataFrame:
    """
    SW calculation with shale cutoff logic (Geolog style).
    
    Additional parameter in params:
    - 'VSH_CUTOFF': VSH threshold above which SW is set to 1.0 (default 0.5)
    """
    df_processed = df.copy()
    
    # Extract shale cutoff parameter
    VSH_CUTOFF = float(params.get('VSH_CUTOFF', 0.5))
    
    # Other parameters (same as before)
    OPT_INDO = params.get('OPT_INDO', 'FULL').upper()
    OPT_RW = params.get('OPT_RW', 'MEASURED').upper()
    OPT_M = params.get('OPT_M', 'CONSTANT').upper()
    
    RWS = float(params.get('RWS', 0.529))
    RWT = float(params.get('RWT', 227))
    RT_SH = float(params.get('RT_SH', 2.2))
    RMF = params.get('RMF')
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))
    SWE_IRR = float(params.get('SWE_IRR', 0.0))
    
    # Verify required columns
    required_cols = ['PHIE', 'VSH_LINEAR', 'RT']
    if OPT_RW == 'MEASURED':
        required_cols.append('FTEMP')
    elif OPT_RW == 'LOG':
        required_cols.append('RW')
    if OPT_M == 'VARIABLE':
        required_cols.append('M_EXP')
        
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize SW column
    df_processed['SW'] = np.nan
    
    # Apply filters
    mask = pd.Series(True, index=df_processed.index)
    if target_intervals and 'MARKER' in df_processed.columns:
        mask = df_processed['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_processed.columns:
        zone_mask = df_processed['ZONE'].isin(target_zones)
        mask = mask | zone_mask if target_intervals else zone_mask
    
    if not mask.any():
        print("Warning: No data matches filter criteria")
        return df_processed
    
    print(f"Calculating SW for {mask.sum()} rows with VSH cutoff = {VSH_CUTOFF}")
    
    # Process each row
    for idx in df_processed[mask].index:
        vsh = df_processed.loc[idx, 'VSH_LINEAR']
        phie = df_processed.loc[idx, 'PHIE']
        rt = df_processed.loc[idx, 'RT']
        
        # CRITICAL: Apply shale cutoff FIRST (Geolog logic)
        if vsh >= VSH_CUTOFF:
            df_processed.loc[idx, 'SW'] = 1.0
            continue  # Skip Indonesian calculation for shaly intervals
        
        # Calculate shale term for non-shaly intervals
        if OPT_INDO == 'FULL':
            v = vsh ** (2 - vsh)
        elif OPT_INDO == 'SIMPLE':
            v = vsh ** 2
        elif OPT_INDO == 'TAR_SAND':
            v = vsh ** (2 - 2 * vsh)
        else:
            v = vsh ** 2
        
        # Check for low effective porosity
        if phie < 0.005:
            df_processed.loc[idx, 'SW'] = 1.0
            continue
        
        # Calculate Rw
        if OPT_RW == 'MEASURED':
            ftemp = df_processed.loc[idx, 'FTEMP']
            rwtemp = RWS * (RWT + 21.5) / (ftemp + 21.5)
        elif OPT_RW == 'LOG':
            rwtemp = df_processed.loc[idx, 'RW']
        else:
            ftemp = df_processed.loc[idx, 'FTEMP'] if 'FTEMP' in df_processed.columns else 75
            rwtemp = RWS * (RWT + 21.5) / (ftemp + 21.5)
        
        # Choose cementation exponent
        if OPT_M == 'CONSTANT':
            mtemp = M
        else:
            if pd.isna(df_processed.loc[idx, 'M_EXP']):
                df_processed.loc[idx, 'SW'] = np.nan
                continue
            mtemp = df_processed.loc[idx, 'M_EXP']
        
        # Calculate formation factor
        ff = A / (phie ** mtemp)
        
        # Calculate SW using Indonesian equation
        f1 = 1 / (ff * rwtemp)
        f2 = 2 * np.sqrt(v / (rwtemp * ff * RT_SH))
        f3 = v / RT_SH
        
        sw = (1 / (rt * (f1 + f2 + f3))) ** (1/N)
        df_processed.loc[idx, 'SW'] = np.clip(sw, SWE_IRR, 1.0)
    
    return df_processed


def calculate_sw(
    df: pd.DataFrame,
    params: dict,
    target_intervals: list = None,
    target_zones: list = None
) -> pd.DataFrame:
    """
    Advanced version with multiple cutoff options similar to commercial software.
    """
    df_processed = df.copy()
    
    # Cutoff parameters
    VSH_CUTOFF = float(params.get('VSH_CUTOFF', 0.5))
    PHIE_CUTOFF = float(params.get('PHIE_CUTOFF', 0.05))  # Below this, SW = 1.0
    # RT_CUTOFF = float(params.get('RT_CUTOFF', None))  # Above this, use different logic
    
    # Apply the basic calculation first
    df_result = calculate_sw_with_shale_cutoff(df, params, target_intervals, target_zones)
    
    # Apply additional cutoffs if specified
    mask = pd.Series(True, index=df_result.index)
    if target_intervals and 'MARKER' in df_result.columns:
        mask = df_result['MARKER'].isin(target_intervals)
    if target_zones and 'ZONE' in df_result.columns:
        zone_mask = df_result['ZONE'].isin(target_zones)
        mask = mask | zone_mask if target_intervals else zone_mask
    
    # Additional porosity cutoff
    low_poro_mask = (df_result['PHIE'] < PHIE_CUTOFF) & mask
    if low_poro_mask.any():
        df_result.loc[low_poro_mask, 'SW'] = 1.0
        print(f"Applied porosity cutoff: {low_poro_mask.sum()} points set to SW=1.0")
    
    # High resistivity cutoff (optional)
    # if RT_CUTOFF and 'RT' in df_result.columns:
    #     high_rt_mask = (df_result['RT'] > RT_CUTOFF) & mask
    #     if high_rt_mask.any():
    #         # For very high resistivity, you might want to cap SW at a lower value
    #         # This is formation-specific logic
    #         print(f"High resistivity detected: {high_rt_mask.sum()} points above {RT_CUTOFF} ohm-m")
    
    return df_result