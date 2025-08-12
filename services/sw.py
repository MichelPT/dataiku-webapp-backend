import pandas as pd
import numpy as np


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


def calculate_sw_simandoux(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate Water Saturation using Modified Simandoux method with Newton-Raphson iteration.
    This is an advanced method that accounts for shale effects in resistivity calculations.
    The final output column is 'SW'.
    """
    df_processed = df.copy()
    
    # Normalize column names to uppercase
    df_processed.columns = [col.upper() for col in df_processed.columns]
    
    # Replace missing values
    df_processed.replace(-999.0, np.nan, inplace=True)
    
    # Extract parameters with defaults
    A = float(params.get('A', 1.0))
    M = float(params.get('M', 2.0))
    N = float(params.get('N', 2.0))
    C = float(params.get('C', 1.0))  # Shale exponent for Schlumberger
    RWS = float(params.get('RWS', 0.1))
    RWT = float(params.get('RWT', 75))
    FTEMP = float(params.get('FTEMP', 90))
    SWE_IRR = float(params.get('SWE_IRR', 0.0))
    RT_SH = float(params.get('RT_SH', 5.0))
    OPT_SIM = params.get('OPT_SIM', 'MODIFIED')  # 'MODIFIED' or 'SCHLUMBERGER'
    
    # Check required columns
    required_cols = ['GR', 'RHOB', 'ILD']  # ILD is deep resistivity
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        # Try alternative column names
        if 'RT' in df_processed.columns and 'ILD' not in df_processed.columns:
            df_processed['ILD'] = df_processed['RT']
        else:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("Calculating RW at formation temperature...")
    df_processed["RW_TEMP"] = RWS * (RWT + 21.5) / (FTEMP + 21.5)
    
    # Calculate VSH if not present
    if 'VSH' not in df_processed.columns:
        print("Calculating VSH from GR...")
        GR_clean = df_processed["GR"].quantile(0.05)
        GR_shale = df_processed["GR"].quantile(0.95)
        df_processed["VSH"] = ((df_processed["GR"] - GR_clean) / (GR_shale - GR_clean)).clip(0, 1)
    
    # Calculate PHIE if not present
    if 'PHIE' not in df_processed.columns:
        print("Calculating PHIE from density...")
        RHO_MA = float(params.get('RHO_MA', 2.65))
        RHO_SH = float(params.get('RHO_SH', 2.45))
        RHO_FL = float(params.get('RHO_FL', 1.0))
        
        df_processed["PHIE"] = ((RHO_MA - df_processed["RHOB"]) / (RHO_MA - RHO_FL)) - \
                               df_processed["VSH"] * ((RHO_MA - RHO_SH) / (RHO_MA - RHO_FL))
        df_processed["PHIE"] = df_processed["PHIE"].clip(lower=0)
    
    print(f"Calculating Water Saturation using {OPT_SIM} Simandoux method...")
    
    # --- MODIFICATION START ---
    # Define the final output column name for consistency
    SW_COL = 'SW'

    # Calculate SW using Newton-Raphson by iterating over rows
    sw_values = []
    for _, row in df_processed.iterrows():
        if pd.notna(row["ILD"]) and pd.notna(row["PHIE"]) and pd.notna(row["VSH"]):
            phie = row["PHIE"]
            if phie < 0.005:
                # For very low porosity, saturation is considered 100%
                sw = 1.0
            else:
                ff = A / (phie ** M)
                sw = newton_simandoux(
                    rt=row["ILD"], 
                    ff=ff, 
                    rwtemp=row["RW_TEMP"], 
                    rtsh=RT_SH, 
                    vsh=row["VSH"], 
                    n=N, 
                    opt=OPT_SIM, 
                    c=C
                )
            sw_values.append(sw)
        else:
            sw_values.append(np.nan)
    
    # Assign the raw calculated values to a temporary series
    temp_sw = pd.Series(sw_values, index=df_processed.index)
    
    # Clip the result to be between SWE_IRR and 1.0 and assign to the final 'SW' column
    df_processed[SW_COL] = temp_sw.clip(lower=SWE_IRR, upper=1.0)
    
    # Calculate additional parameters based on the final SW
    df_processed["VOL_UWAT"] = df_processed["PHIE"] * df_processed[SW_COL]
    # --- MODIFICATION END ---
    
    print("Water Saturation calculation (Simandoux) completed successfully.")
    return df_processed


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
