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


def handle_null_values(csv_content: str) -> str:
    """Fungsi dari data_utils.py lama Anda."""
    csv_file_like_object = io.StringIO(csv_content)
    df = pd.read_csv(csv_file_like_object)
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both', axis=0)
    df.fillna('NA', inplace=True)
    return df.to_csv(index=False)