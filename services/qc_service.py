# FILE 2: api/app/services/qc_service.py

import os
import lasio
import pandas as pd
import numpy as np
import io
import logging


def validate_and_convert_rhob_units(las, df, logger):
    """
    Validasi unit RHOB dari header LAS dan konversi dari K/M3 ke G/C3 jika diperlukan.

    Args:
        las: objek lasio LAS file
        df: DataFrame hasil dari las.df()
        logger: logger object

    Returns:
        tuple: (df_modified, conversion_info)
    """
    conversion_info = {
        'rhob_columns_found': [],
        'conversions_performed': [],
        'warnings': []
    }

    # Cari kolom RHOB dengan berbagai variasi nama
    rhob_variants = ['RHOB', 'RHOZ', 'DENS', 'DENSITY']
    rhob_columns_in_df = []

    for col in df.columns:
        col_upper = str(col).upper().strip()
        if any(variant in col_upper for variant in rhob_variants):
            rhob_columns_in_df.append(col)

    conversion_info['rhob_columns_found'] = rhob_columns_in_df

    if not rhob_columns_in_df:
        logger.info("Tidak ditemukan kolom RHOB untuk validasi unit.")
        return df, conversion_info

    # Periksa unit untuk setiap kolom RHOB yang ditemukan
    for rhob_col in rhob_columns_in_df:
        try:
            # Cari informasi curve di header LAS
            curve_info = None
            original_col_name = rhob_col

            # Cari di curves section
            for curve in las.curves:
                if curve.mnemonic.upper().strip() == rhob_col.upper().strip():
                    curve_info = curve
                    break

            if curve_info is None:
                # Coba cari dengan nama yang sudah di-rename
                for curve in las.curves:
                    curve_upper = curve.mnemonic.upper().strip()
                    if any(variant in curve_upper for variant in rhob_variants):
                        curve_info = curve
                        original_col_name = curve.mnemonic
                        break

            if curve_info is None:
                warning_msg = f"Informasi curve untuk {rhob_col} tidak ditemukan di header LAS"
                logger.warning(warning_msg)
                conversion_info['warnings'].append(warning_msg)
                continue

            # Ambil unit dari curve info
            unit = curve_info.unit.strip().upper() if curve_info.unit else ''

            logger.info(
                f"Kolom {rhob_col} (original: {original_col_name}) memiliki unit: '{unit}'")

            # Cek apakah unit adalah K/M3 (dengan berbagai variasi penulisan)
            km3_variants = ['K/M3', 'KG/M3', 'K/M³', 'KG/M³', 'KGM3', 'KM3']

            if any(variant == unit for variant in km3_variants):
                logger.info(
                    f"🔄 Mengkonversi {rhob_col} dari {unit} ke G/C3...")

                # Konversi dari kg/m³ ke g/cm³ (dibagi 1000)
                # 1 kg/m³ = 0.001 g/cm³
                original_values = df[rhob_col].copy()
                df[rhob_col] = df[rhob_col] / 1000.0

                # Hitung statistik konversi
                valid_original = original_values.dropna()
                valid_converted = df[rhob_col].dropna()

                conversion_summary = {
                    'column': rhob_col,
                    'original_unit': unit,
                    'new_unit': 'G/C3',
                    'conversion_factor': 0.001,
                    'rows_converted': len(valid_converted),
                    'original_range': f"{valid_original.min():.2f} - {valid_original.max():.2f}" if len(valid_original) > 0 else "No valid data",
                    'converted_range': f"{valid_converted.min():.4f} - {valid_converted.max():.4f}" if len(valid_converted) > 0 else "No valid data"
                }

                conversion_info['conversions_performed'].append(
                    conversion_summary)

                logger.info(f"✅ Konversi selesai untuk {rhob_col}:")
                logger.info(f"   - {len(valid_converted)} baris dikonversi")
                logger.info(
                    f"   - Range original ({unit}): {conversion_summary['original_range']}")
                logger.info(
                    f"   - Range hasil (G/C3): {conversion_summary['converted_range']}")

            elif unit in ['G/C3', 'G/CM3', 'G/CM³', 'GCC', 'GRCM3']:
                logger.info(
                    f"✅ {rhob_col} sudah dalam unit yang benar: {unit}")

            else:
                warning_msg = f"⚠️ Unit {rhob_col} tidak dikenali: '{unit}'. Tidak ada konversi yang dilakukan."
                logger.warning(warning_msg)
                conversion_info['warnings'].append(warning_msg)

        except Exception as e:
            error_msg = f"Error saat memproses unit untuk {rhob_col}: {str(e)}"
            logger.error(error_msg)
            conversion_info['warnings'].append(error_msg)

    return df, conversion_info


def add_formation_data_to_df(df, formation_df, well_name, column_name, logger):
    """
    Fungsi generik yang disempurnakan untuk menambahkan data formasi (Marker atau Zone).
    """
    df[column_name] = np.nan
    well_name_cleaned = well_name.strip().upper()

    logger.info(
        f"[{column_name}] Mencoba menggabungkan data untuk Sumur: '{well_name_cleaned}'")

    if formation_df is None or formation_df.empty:
        logger.warning(
            f"[{column_name}] Data {column_name.lower()} tidak tersedia untuk digabungkan.")
        return df

    available_wells = formation_df['well_identifier'].unique()
    logger.info(
        f"[{column_name}] Sumur yang tersedia di data {column_name.lower()}: {available_wells}")

    well_formation = formation_df[formation_df['well_identifier']
                                  == well_name_cleaned].copy()

    if well_formation.empty:
        logger.warning(
            f"[{column_name}] Tidak ada data cocok ditemukan untuk sumur '{well_name_cleaned}'.")
        return df

    logger.info(
        f"[{column_name}] Ditemukan {len(well_formation)} entri untuk '{well_name_cleaned}'.")

    well_formation = well_formation.sort_values('md').reset_index(drop=True)

    for i in range(len(well_formation)):
        top_depth = well_formation.loc[i, 'md']
        surface_name = str(well_formation.loc[i, 'surface'])
        bottom_depth = well_formation.loc[i + 1, 'md'] if i + \
            1 < len(well_formation) else float('inf')

        mask = (df['DEPTH'] >= top_depth) & (df['DEPTH'] < bottom_depth)
        df.loc[mask, column_name] = surface_name

    logger.info(
        f"[{column_name}] Penandaan selesai untuk '{well_name_cleaned}'. {df[column_name].notna().sum()} baris terisi.")
    return df


def process_formation_data(formation_data, name, logger):
    """
    Helper paling tangguh untuk membaca dan membersihkan data Marker/Zone dari payload.
    """
    if not formation_data or 'content' not in formation_data:
        logger.info(f"Data {name} tidak disediakan dalam payload.")
        return None
    try:
        content_str = formation_data['content']

        first_line = content_str.splitlines()[0]
        separator = ';' if ';' in first_line else ','

        logger.info(
            f"Mencoba membaca file {name} dengan pemisah: '{separator}'")
        df = pd.read_csv(io.StringIO(content_str), sep=separator,
                         engine='python', on_bad_lines='warn', skipinitialspace=True)

        logger.info(f"Kolom mentah terdeteksi di {name}: {list(df.columns)}")

        cleaned_columns = [str(col).strip().lower() for col in df.columns]
        df.columns = cleaned_columns

        # --- PERBAIKAN UTAMA DI SINI ---
        # Logika pemetaan sekarang bekerja pada nama kolom yang sudah bersih (cleaned_columns)
        column_map = {}
        for col in cleaned_columns:
            if 'well' in col and 'identifier' in col:
                column_map[col] = 'well_identifier'
            elif col == 'md':
                column_map[col] = 'md'
            elif 'surface' in col:
                column_map[col] = 'surface'

        df.rename(columns=column_map, inplace=True)

        required_cols = ['well_identifier', 'md', 'surface']
        if not all(col in df.columns for col in required_cols):
            logger.warning(
                f"GAGAL: File {name} tidak dapat diproses. Kolom standar {required_cols} tidak ditemukan setelah pembersihan. Kolom yang ada: {list(df.columns)}")
            return None

        df['well_identifier'] = df['well_identifier'].astype(
            str).str.strip().str.upper()
        if df['md'].dtype == object:
            df['md'] = pd.to_numeric(df['md'].astype(str).str.replace(
                ',', '.', regex=False), errors='coerce')

        df.dropna(subset=['md', 'well_identifier'], inplace=True)

        logger.info(
            f"Data {name} berhasil diproses. {len(df)} baris valid dimuat.")
        return df
    except Exception as e:
        logger.error(
            f"Gagal total saat memproses data {name}: {e}", exc_info=True)
        return None


def run_full_qc_pipeline(well_logs_data: list, marker_data: dict, zone_data: dict, logger: logging.Logger):
    """Fungsi utama pipeline QC."""
    qc_results = []
    conversion_log = []
    output_files = {}
    required_logs = ['GR', 'NPHI', 'RT', 'RHOB']

    all_markers_df = process_formation_data(marker_data, 'Marker', logger)
    all_zones_df = process_formation_data(zone_data, 'Zone', logger)

    for file_info in well_logs_data:
        filename = file_info['name']
        well_name = os.path.splitext(filename)[0]
        status = "PASS"
        try:
            logger.info(f"--- [Memproses] MULAI: {filename} ---")
            las = lasio.read(io.StringIO(file_info['content']))
            df = las.df().reset_index()

            # Validasi dan konversi unit RHOB SEBELUM rename kolom
            logger.info(f"🔍 Melakukan validasi unit untuk {filename}...")
            df, conversion_info = validate_and_convert_rhob_units(
                las, df, logger)

            # Simpan info konversi untuk log
            if conversion_info['conversions_performed']:
                conversion_log.append({
                    'well_name': well_name,
                    'conversions': conversion_info['conversions_performed'],
                    'warnings': conversion_info['warnings']
                })

            df.rename(columns=lambda c: c.upper(), inplace=True)

            column_mapping = {'DEPT': 'DEPTH', 'ILD': 'RT', 'LLD': 'RT', 'RESD': 'RT',
                              'RHOZ': 'RHOB', 'DENS': 'RHOB', 'TNPH': 'NPHI', 'GR_CAL': 'GR'}
            df.rename(columns=column_mapping, inplace=True)

            if 'DEPTH' not in df.columns:
                raise ValueError("Kolom DEPTH tidak ditemukan.")
            df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce')
            df.dropna(subset=['DEPTH'], inplace=True)

            df = add_formation_data_to_df(
                df, all_markers_df, well_name, 'MARKER', logger)
            df = add_formation_data_to_df(
                df, all_zones_df, well_name, 'ZONE', logger)

            missing_columns = [
                log for log in required_logs if log not in df.columns]
            if missing_columns:
                status = "MISSING_LOGS"
                details = ', '.join(missing_columns)
            else:
                for col in required_logs:
                    df[col] = df[col].replace([-999.0, -999.25], np.nan)

                null_columns = [
                    log for log in required_logs if df[log].isna().any()]
                if null_columns:
                    status = "HAS_NULL"
                    details = ', '.join(null_columns)
                else:
                    details = 'All checks passed'

            qc_results.append(
                {'well_name': well_name, 'status': status, 'details': details})
            output_files[f"{well_name}.csv"] = df.to_csv(index=False)

        except Exception as e:
            logger.error(f"Error memproses {filename}: {e}", exc_info=True)
            qc_results.append(
                {'well_name': well_name, 'status': 'ERROR', 'details': str(e)})

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
        print(
            f"Zone classification only applies to BNG wells. Skipping well: {well_name}")
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
        mask = (result_df[depth_column] >= zone['top']) & (
            result_df[depth_column] < zone['bottom'])
        result_df.loc[mask, 'ZONE'] = zone['name']

    # Count how many rows were assigned a zone
    zone_count = result_df['ZONE'].notna().sum()
    print(
        f"Applied zones to {well_name}: {zone_count} depth points classified")

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
    well_markers = marker_df[marker_df['Well identifier'].str.strip(
    ).str.upper() == well_name_cleaned].copy()

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
            mask = (result_df[depth_column] > previous_depth) & (
                result_df[depth_column] <= current_depth)

        result_df.loc[mask, 'MARKER'] = surface_name

    # For depths beyond the last marker, assign the last surface
    if len(well_markers) > 0:
        last_depth = well_markers.iloc[-1]['MD']
        last_surface = str(well_markers.iloc[-1]['Surface'])
        mask = result_df[depth_column] > last_depth
        result_df.loc[mask, 'MARKER'] = last_surface

    print(f"Successfully applied {len(well_markers)} markers to {well_name}")
    return result_df
