# FILE 2: api/app/services/qc_service.py
# (Ini adalah kode run_quality_control.py dan handle_nulls_script.py Anda yang digabungkan)
import os
import lasio
import pandas as pd
import numpy as np
import io
import logging

def add_markers_to_df(df, well_name, all_markers_df, logger):
    """Menambahkan marker ke DataFrame, dengan logging."""
    df['Marker'] = None
    well_name_cleaned = well_name.strip()
    logger.info(f"[Markers] Memulai pencarian marker untuk Sumur: '{well_name_cleaned}'")

    if all_markers_df.empty:
        logger.warning("[Markers] Data marker kosong.")
        return False
        
    try:
        well_markers = all_markers_df[all_markers_df['Well identifier_cleaned'] == well_name_cleaned.upper()].copy()
        if well_markers.empty:
            logger.warning(f"[Markers] Tidak ada marker ditemukan untuk sumur '{well_name_cleaned}'.")
            return False

        logger.info(f"[Markers] Ditemukan {len(well_markers)} entri marker untuk '{well_name_cleaned}'.")
        well_markers.sort_values(by='MD', inplace=True)
        
        last_depth = 0.0
        for _, marker_row in well_markers.iterrows():
            current_depth = marker_row['MD']
            surface_name = str(marker_row['Surface'])
            mask = (df['DEPTH'] >= last_depth) & (df['DEPTH'] < current_depth)
            df.loc[mask, 'Marker'] = surface_name
            last_depth = current_depth

        if not well_markers.empty:
            last_marker = well_markers.iloc[-1]
            df.loc[df['DEPTH'] >= last_marker['MD'], 'Marker'] = str(last_marker['Surface'])

        logger.info(f"[Markers] Penandaan marker selesai.")
        return True
    except Exception as e:
        logger.error(f"[Markers] Terjadi error: {e}", exc_info=True)
        return False

def check_extreme_values(df, column):
    if pd.api.types.is_numeric_dtype(df[column]) and not df[column].isna().all():
        mean, std = df[column].mean(), df[column].std()
        if std == 0: return False
        mask = (df[column] > mean + 3 * std) | (df[column] < mean - 3 * std)
        return mask.any()
    return False

def run_full_qc_pipeline(files_data: list, logger):
    """Fungsi utama dari qc_logic.py Anda, sekarang di dalam service."""
    qc_results = []
    output_files = {}
    required_logs = ['GR', 'NPHI', 'RT', 'RHOB']
    skip_files_lower = ['abb-032.las', 'abb-033.las', 'abb-059.las']

    all_markers_df = pd.DataFrame()
    for file_info in files_data:
        if file_info['name'].lower().endswith('.csv') and 'marker' in file_info['name'].lower():
            try:
                marker_content = io.StringIO(file_info['content'])
                df_marker = pd.read_csv(marker_content, sep='[;,]', engine='python', on_bad_lines='skip')
                if all(col in df_marker.columns for col in ['Well identifier', 'MD', 'Surface']):
                    all_markers_df = pd.concat([all_markers_df, df_marker], ignore_index=True)
            except Exception as e:
                logger.warning(f"Tidak bisa membaca file marker '{file_info['name']}'. Error: {e}")
    
    if not all_markers_df.empty:
        logger.info("[Markers] Membersihkan dan menyiapkan data marker...")
        all_markers_df['Well identifier_cleaned'] = all_markers_df['Well identifier'].str.strip().str.upper()
        if all_markers_df['MD'].dtype == object:
            all_markers_df['MD'] = pd.to_numeric(all_markers_df['MD'].str.replace(',', '.', regex=False), errors='coerce')
        all_markers_df.dropna(subset=['MD', 'Well identifier_cleaned'], inplace=True)
        all_markers_df['Surface'] = all_markers_df['Surface'].astype(str)
        logger.info(f"[Markers] Data marker bersih. {len(all_markers_df)} baris valid dimuat.")

    las_files = [f for f in files_data if f['name'].lower().endswith('.las')]
    for file_info in las_files:
        filename = file_info['name']
        if filename.lower() in skip_files_lower:
            logger.info(f"--- MELEWATI: {filename} ---")
            continue
            
        well_name = os.path.splitext(filename)[0]
        status = "PASS"
        details = {}
        try:
            logger.info(f"--- [Memproses] MULAI: {filename} ---")
            las_content = io.StringIO(file_info['content'])
            las = lasio.read(las_content)
            df = las.df().reset_index()
            df.rename(columns=lambda c: c.upper(), inplace=True)
            column_mapping = { 'DEPT': 'DEPTH', 'ILD': 'RT', 'LLD': 'RT', 'RESD': 'RT', 'RHOZ': 'RHOB', 'DENS': 'RHOB', 'TNPH': 'NPHI', 'GR_CAL': 'GR' }
            df.rename(columns=column_mapping, inplace=True)
            
            if 'DEPTH' not in df.columns: raise ValueError("Kolom DEPTH tidak ditemukan.")
            df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce')
            df.dropna(subset=['DEPTH'], inplace=True)

            details['missing_columns'] = [log for log in required_logs if log not in df.columns]
            if details['missing_columns']:
                status = "MISSING_LOGS"
                qc_results.append({'well_name': well_name, 'status': status, 'details': ', '.join(details['missing_columns'])})
                output_files[f"{well_name}_{status}.csv"] = df.to_csv(index=False)
                continue

            for col in required_logs: df[col] = df[col].replace([-999.0, -999.25], np.nan)
            
            has_markers = add_markers_to_df(df, well_name, all_markers_df, logger)
            
            zone_df = df.dropna(subset=['MARKER']) if has_markers and not df['MARKER'].isna().all() else df
            if zone_df.empty: zone_df = df
            
            details['null_columns'] = [log for log in required_logs if zone_df[log].isna().any()]
            if details['null_columns']:
                status = "HAS_NULL"
                qc_results.append({'well_name': well_name, 'status': status, 'details': ', '.join(details['null_columns'])})
                output_files[f"{well_name}_{status}.csv"] = df.to_csv(index=False)
                continue

            details['extreme_columns'] = [log for log in required_logs if check_extreme_values(zone_df, log)]
            if details['extreme_columns']:
                status = "EXTREME_VALUES"
                qc_results.append({'well_name': well_name, 'status': status, 'details': ', '.join(details['extreme_columns'])})
                output_files[f"{well_name}_{status}.csv"] = df.to_csv(index=False)
                continue
            
            qc_results.append({'well_name': well_name, 'status': status, 'details': 'All checks passed'})
            output_files[f"{well_name}_{status}.csv"] = df.to_csv(index=False)

        except Exception as e:
            logger.error(f"Error memproses {filename}: {e}", exc_info=True)
            qc_results.append({'well_name': well_name, 'status': 'ERROR', 'details': str(e)})

    return {'qc_summary': qc_results, 'output_files': output_files}

def append_zones_to_dataframe(df, well_name, depth_column='DEPTH'):
    """
    Append zone information to a DataFrame based on predefined depth ranges.
    Only applies to BNG wells with specific zone classifications.
    
    Args:
        df (pd.DataFrame): Main DataFrame containing well log data with depth column
        well_name (str): Well identifier to check (must contain 'BNG')
        depth_column (str): Name of the depth column in df (default: 'DEPTH')
    
    Returns:
        pd.DataFrame: DataFrame with added 'ZONE' column
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Initialize ZONE column
    result_df['ZONE'] = None
    
    # Check if well name contains BNG (case insensitive)
    if 'BNG' not in well_name.upper():
        print(f"Zone classification only applies to BNG wells. Skipping well: {well_name}")
        return result_df
    
    # Define zone boundaries
    zones = [
        {'name': 'ABF', 'top': 554.0, 'bottom': 1138.3},
        {'name': 'GUF', 'top': 1138.3, 'bottom': 1539.5},
        {'name': 'BRF', 'top': 1539.5, 'bottom': 1579.2},
        {'name': 'TAF', 'top': 1579.2, 'bottom': 2301.0}
    ]
    
    # Apply zones based on depth ranges
    for zone in zones:
        mask = (result_df[depth_column] >= zone['top']) & (result_df[depth_column] < zone['bottom'])
        result_df.loc[mask, 'ZONE'] = zone['name']
    
    # Count how many rows were assigned a zone
    zone_count = result_df['ZONE'].notna().sum()
    print(f"Applied zones to {well_name}: {zone_count} depth points classified")
    
    return result_df

def append_markers_to_dataframe(df, marker_df, well_name, depth_column='DEPTH'):
    """
    Append marker information to a DataFrame based on depth ranges and well name.
    
    Args:
        df (pd.DataFrame): Main DataFrame containing well log data with depth column
        marker_df (pd.DataFrame): Marker DataFrame with columns ['Well identifier', 'MD', 'Surface']
        well_name (str): Well identifier to match (e.g., 'BNG-007')
        depth_column (str): Name of the depth column in df (default: 'DEPTH')
    
    Returns:
        pd.DataFrame: DataFrame with added 'MARKER' column
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Initialize MARKER column
    result_df['MARKER'] = None
    
    # Clean well name for matching
    well_name_cleaned = well_name.strip().upper()
    
    # Filter marker data for the specific well
    well_markers = marker_df[marker_df['Well identifier'].str.strip().str.upper() == well_name_cleaned].copy()
    
    if well_markers.empty:
        print(f"No markers found for well: {well_name}")
        return result_df
    
    # Clean and convert MD column (handle comma decimal separator)
    if well_markers['MD'].dtype == object:
        well_markers['MD'] = pd.to_numeric(
            well_markers['MD'].astype(str).str.replace(',', '.', regex=False), 
            errors='coerce'
        )
    
    # Remove rows with invalid MD values
    well_markers = well_markers.dropna(subset=['MD'])
    
    if well_markers.empty:
        print(f"No valid marker depths found for well: {well_name}")
        return result_df
    
    # Sort markers by depth
    well_markers = well_markers.sort_values('MD').reset_index(drop=True)
    
    # Apply markers based on depth ranges
    for i in range(len(well_markers)):
        current_depth = well_markers.loc[i, 'MD']
        surface_name = str(well_markers.loc[i, 'Surface'])
        
        if i == 0:
            # For the first marker, assign from the beginning up to this depth
            mask = result_df[depth_column] <= current_depth
        else:
            # For subsequent markers, assign from previous depth to current depth
            previous_depth = well_markers.loc[i-1, 'MD']
            mask = (result_df[depth_column] > previous_depth) & (result_df[depth_column] <= current_depth)
        
        result_df.loc[mask, 'MARKER'] = surface_name
    
    # For depths beyond the last marker, assign the last surface
    if len(well_markers) > 0:
        last_depth = well_markers.iloc[-1]['MD']
        last_surface = str(well_markers.iloc[-1]['Surface'])
        mask = result_df[depth_column] > last_depth
        result_df.loc[mask, 'MARKER'] = last_surface
    
    print(f"Successfully applied {len(well_markers)} markers to {well_name}")
    return result_df


def read_marker_file(marker_file_path):
    """
    Read marker file and return cleaned DataFrame.
    
    Args:
        marker_file_path (str): Path to the marker CSV file
    
    Returns:
        pd.DataFrame: Cleaned marker DataFrame
    """
    try:
        # Try reading with different separators
        try:
            marker_df = pd.read_csv(marker_file_path, sep=';')
        except:
            try:
                marker_df = pd.read_csv(marker_file_path, sep=',')
            except:
                marker_df = pd.read_csv(marker_file_path, sep='\t')
        
        # Verify required columns exist
        required_columns = ['Well identifier', 'MD', 'Surface']
        if not all(col in marker_df.columns for col in required_columns):
            raise ValueError(f"Marker file must contain columns: {required_columns}")
        
        return marker_df
        
    except Exception as e:
        print(f"Error reading marker file: {e}")
        return pd.DataFrame()


def handle_null_values(csv_content: str) -> str:
    """Fungsi dari data_utils.py lama Anda."""
    csv_file_like_object = io.StringIO(csv_content)
    df = pd.read_csv(csv_file_like_object)
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both', axis=0)
    df.fillna('NA', inplace=True)
    return df.to_csv(index=False)

