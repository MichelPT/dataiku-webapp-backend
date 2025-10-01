# /api/app.py
import glob
from flask import json, request, jsonify
from services.autoplot import calculate_gr_ma_sh_from_nphi_rhob, calculate_nphi_rhob_intersection
from services.iqual import calculate_iqual
from services.normalization import calculate_gr_normalized
from services.splicing import splice_and_flag_logs
from services.vsh_dn import calculate_vsh_dn
from services.rwa import calculate_rwa
from services.sw import calculate_sw, calculate_sw_simandoux
from services.crossplot import generate_crossplot
from services.histogram import plot_histogram
from services.ngsa import process_all_wells_ngsa
from services.dgsa import process_all_wells_dgsa
from services.rgsa import process_all_wells_rgsa
from services.depth_matching import run_depth_matching
from services.porosity import calculate_porosity
# from routes.qc_routes import qc_bp
from services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_custom,
    plot_depth_matching,
    plot_fill_missing,
    plot_gsa_main,
    plot_gwd,
    plot_iqual,
    plot_log_default,
    plot_matching_results,
    plot_module_2,
    plot_module1,
    plot_module_3,
    plot_norm_prep,
    plot_smoothing,
    plot_phie_den,
    plot_gsa_main,
    plot_smoothing_prep,
    plot_splicing,
    plot_trimming,
    plot_vsh_linear,
    plot_sw_indo,
    plot_rwa_indo,
)
from services.plotting_service import plot_normalization
from services.trim_data import trim_well_log
from services.rgbe_rpbe import process_rgbe_rpbe, plot_rgbe_rpbe
from services.rt_r0 import process_rt_r0
from services.swgrad import process_swgrad
from services.dns_dnsv import process_dns_dnsv
from services.rt_r0_plot import plot_rt_r0
from services.swgrad_plot import plot_swgrad
from services.dns_dnsv_plot import plot_dns_dnsv
from services.rwa import calculate_rwa
from services.sw import calculate_sw
from services.structures_service import get_fields_list, get_field_structures, get_structure_details, get_well_details
from services.folder_nav_service import get_structure_wells_folders, get_well_folder_files
from services.module1_service import get_module1_plot_data
from services.fill_missing import fill_flagged_values, flag_missing_values, flag_missing_values
from services.ml_imputer import run_ml_imputation_for_well

from typing import Optional
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import logging
import lasio
from services.vsh_calculation import calculate_vsh_from_gr
from services.data_processing import fill_flagged_missing_values, fill_null_values_in_marker_range, flag_missing_values_in_range, handle_null_values, selective_normalize_handler, smoothing, trim_data_depth, trim_log_by_masking
from services.qc_service import append_zones_to_dataframe, run_full_qc_pipeline
from services.vsh_calculation import calculate_vsh_from_gr
from services.data_processing import handle_null_values, fill_null_values_in_marker_range, min_max_normalize, selective_normalize_handler, smoothing, trim_data_auto, trim_data_depth
from services.qc_service import run_full_qc_pipeline
from flask import Flask, request, jsonify, Response
import plotly.io as pio


app = Flask(__name__)

# Izinkan permintaan dari semua sumber (penting untuk development)
CORS(app)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
# app.register_blueprint(qc_bp, url_prefix='/api/qc')


@app.route('/')
def home():
    # A simple route to confirm the API is running
    return "Flask backend is running!"


@app.route('/api/run-qc', methods=['POST'])
def qc_route():
    """
    Receives a structured payload with well logs, marker data, and zone data,
    runs the QC process, and returns the results.
    """
    app.logger.info("Received request for /api/run-qc")
    try:
        data = request.get_json()

        # Ekstrak data dari payload yang terstruktur
        well_logs_data = data.get('well_logs')
        marker_data = data.get('marker_data')
        zone_data = data.get('zone_data')

        if not well_logs_data or not isinstance(well_logs_data, list):
            return jsonify({"error": "Invalid input: 'well_logs' key with a list of file objects is required."}), 400

        # Panggil pipeline QC dengan data yang sudah dipisahkan
        results = run_full_qc_pipeline(
            well_logs_data, marker_data, zone_data, app.logger)

        return jsonify(results)

    except Exception as e:
        app.logger.error(
            f"An unexpected error occurred in /api/run-qc: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route('/api/handle-nulls', methods=['POST'])
def handle_nulls_route():
    """
    Receives CSV content as plain text, processes it, and returns the cleaned CSV.
    """
    app.logger.info("Received request for /api/handle-nulls")
    try:
        # Get the raw text content from the request body
        csv_content = request.get_data(as_text=True)

        if not csv_content:
            return jsonify({"error": "Request body cannot be empty."}), 400

        # Call the refactored utility function
        cleaned_csv = handle_null_values(csv_content)

        # Return the result as a downloadable CSV file
        return Response(
            cleaned_csv,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=cleaned_data.csv"}
        )

    except Exception as e:
        app.logger.error(
            f"An unexpected error occurred in /api/handle-nulls: {e}", exc_info=True)
        return jsonify({"error": "Failed to process CSV data."}), 500


# Tentukan path ke file data secara andal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAS_DIR = 'data/depth-matching'
GWD_DIR = 'data/gwd'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
BASE_STRUCTURE_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'structures')

# --- Endpoint 1: Mengambil Struktur Folder ---


@app.route('/api/get-folder-structure', methods=['GET'])
def get_folder_structure():
    """
    Memindai direktori 'data/structures' dan mengembalikan hierarki folder
    dalam format JSON.
    Contoh: { "Adera": ["Abab", "Benuang"], "Limau": [...] }
    """
    app.logger.info("Menerima permintaan untuk /api/get-folder-structure")

    # Pastikan path dasar ada
    if not os.path.isdir(BASE_STRUCTURE_PATH):
        app.logger.error(
            f"Direktori dasar tidak ditemukan di: {BASE_STRUCTURE_PATH}")
        return jsonify({"error": f"Direktori dasar tidak ditemukan di server: {BASE_STRUCTURE_PATH}"}), 500

    structure_dict = {}
    try:
        # Dapatkan daftar semua 'Fields' (direktori level pertama)
        fields = [f for f in os.listdir(BASE_STRUCTURE_PATH) if os.path.isdir(
            os.path.join(BASE_STRUCTURE_PATH, f))]

        for field in fields:
            field_path = os.path.join(BASE_STRUCTURE_PATH, field)
            # Dapatkan daftar semua 'Structures' di dalam setiap 'Field'
            structures = [s for s in os.listdir(
                field_path) if os.path.isdir(os.path.join(field_path, s))]
            structure_dict[field] = sorted(structures)

        app.logger.info(
            f"Struktur folder berhasil dibaca: {len(structure_dict)} fields ditemukan.")
        return jsonify(structure_dict)

    except Exception as e:
        app.logger.error(
            f"Gagal saat memindai struktur folder: {e}", exc_info=True)
        return jsonify({"error": "Gagal membaca struktur folder di server."}), 500


# --- Endpoint 2: Menyimpan Hasil QC ---
@app.route('/api/save-qc-results', methods=['POST'])
def save_qc_results():
    """
    Menerima field, structure, dan daftar sumur yang telah diproses (well) 
    lalu menyimpannya ke dalam file CSV di lokasi yang sesuai.
    """
    app.logger.info("Menerima permintaan untuk /api/save-qc-results")
    try:
        data = request.get_json()
        field = data.get('field')
        structure = data.get('structure')
        wells = data.get('wells')

        if not all([field, structure, wells]):
            return jsonify({"error": "Payload tidak lengkap. Membutuhkan 'field', 'structure', dan 'wells'."}), 400

        # Buat path target berdasarkan input dari frontend
        target_directory = os.path.join(BASE_STRUCTURE_PATH, field, structure)

        # Buat direktori jika belum ada (aman untuk dijalankan meskipun sudah ada)
        os.makedirs(target_directory, exist_ok=True)

        saved_files = []
        for well in wells:
            well_name = well.get('wellName')
            csv_content = well.get('csvContent')

            if not all([well_name, csv_content]):
                app.logger.warning(
                    f"Data sumur tidak lengkap, dilewati: {well}")
                continue

            # Tentukan nama file dan path lengkapnya
            file_name = f"{well_name}.csv"
            file_path = os.path.join(target_directory, file_name)

            # Tulis konten CSV ke dalam file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            saved_files.append(file_path)
            app.logger.info(f"Berhasil menyimpan file ke: {file_path}")

        return jsonify({
            "message": f"Berhasil menyimpan {len(saved_files)} file.",
            "location": target_directory,
            "files": saved_files
        }), 200

    except Exception as e:
        app.logger.error(f"Gagal saat menyimpan hasil QC: {e}", exc_info=True)
        return jsonify({"error": "Gagal menyimpan file di server."}), 500


@app.route('/api/list-zones', methods=['POST'])
def list_zones():
    """
    Reads the main data files, finds all unique values in the 'ZONE' column,
    and returns them as a JSON list.
    """
    request_data = request.get_json()
    full_path = request_data.get('full_path', '')

    try:
        if not os.path.exists(full_path):
            return jsonify({"error": f"Main data directory not found at {full_path}"}), 404

        files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
        if not files:
            return jsonify({"error": "No CSV files found in the 'wells' folder."}), 404

        data = []

        for filename in files:
            file_path = os.path.join(full_path, filename)
            try:
                df_temp = pd.read_csv(file_path, on_bad_lines='warn')
                data.append(df_temp)
            except Exception as read_error:
                print(
                    f"Warning: Failed to read file {file_path}. Error: {read_error}")

        df = pd.concat(data, ignore_index=True)

        if 'ZONE' not in df.columns:
            return jsonify({"error": "Column 'ZONE' not found in the CSV files."}), 404

        if df.empty:
            return jsonify({"error": "Combined data is empty after processing all files."}), 500

        unique_zones = df['ZONE'].dropna().unique().tolist()

        print(f"Sending {len(unique_zones)} unique zones to frontend.")

        return jsonify(unique_zones)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/list-intervals', methods=['POST'])
def list_intervals():
    """
    Membaca file data utama, menemukan semua nilai unik di kolom 'MARKER',
    dan mengembalikannya sebagai daftar JSON.
    """
    data = request.get_json()
    full_path = data.get('full_path', '')
    try:
        if not os.path.exists(full_path):
            return jsonify({"error": f"File data utama tidak ditemukan di {full_path}"}), 404

        files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
        if not files:
            return jsonify({"error": "Tidak ada file CSV ditemukan di folder 'wells'."}), 404

        data = []

        for filename in files:
            file_path = os.path.join(full_path, filename)
            try:
                df_temp = pd.read_csv(file_path, on_bad_lines='warn')
                data.append(df_temp)
            except Exception as read_error:
                print(
                    f"Peringatan: Gagal membaca file {file_path}. Error: {read_error}")

        df = pd.concat(data, ignore_index=True)

        if 'MARKER' not in df.columns:
            return jsonify({"error": "Kolom 'MARKER' tidak ditemukan dalam file CSV."}), 404

        if df.empty:
            return jsonify({"error": "Data gabungan kosong setelah memproses semua file."}), 500

        unique_markers = df['MARKER'].dropna().unique().tolist()
        # unique_markers.sort()

        print(f"Mengirim {len(unique_markers)} interval unik ke frontend.")

        return jsonify(unique_markers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-well-columns', methods=['POST'])
def get_well_columns():
    """
    Handles requests from both Dashboard and Data Prep.
    It prioritizes 'file_paths' if available, otherwise falls back to 'wells'.
    """
    try:
        data = request.get_json()
        file_paths = data.get('file_paths', [])
        full_path = data.get('full_path', '')
        wells = data.get('wells', [])
        result = {}

        # If file_paths are not provided, construct them from 'wells' for backward compatibility
        if not file_paths and wells:
            print("Received 'wells', constructing file paths...")
            # This assumes your "clean" data files are directly in WELLS_DIR
            file_paths = [os.path.join(
                full_path, f"{well}.csv") for well in wells]

        if not file_paths:
            return jsonify({"error": "No wells or file paths provided."}), 400

        for path in file_paths:
            # IMPORTANT: Security check to prevent accessing files outside the 'data' directory
            safe_path = os.path.abspath(path)
            print(f"Backend is trying to access file at: {safe_path}")
            if not safe_path.startswith(os.path.abspath('data')):
                print(f"Warning: Access denied for path '{path}'")
                continue

            if os.path.exists(safe_path):
                df = pd.read_csv(safe_path, nrows=1, on_bad_lines='warn')
                # Use the filename as the key for consistency
                file_name = os.path.basename(safe_path)
                result[file_name] = df.columns.tolist()
            else:
                result[os.path.basename(path)] = []

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-log-percentiles', methods=['POST'])
def get_log_percentiles():
    """
    Menghitung persentil dinamis untuk SUMUR YANG DIPILIH (selected wells).
    Endpoint ini menerima `low_percentile` dan `high_percentile` dari payload
    untuk menentukan persentil mana yang akan dihitung.
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Request body tidak valid."}), 400

        # --- Ekstrak Parameter dari Payload ---
        log_column = payload.get('log_column')
        low_percentile = float(payload.get(
            'low_percentile', 5.0))   # Default P5
        high_percentile = float(payload.get(
            'high_percentile', 95.0))  # Default P95

        full_path = payload.get('full_path')
        selected_wells = payload.get('selected_wells', [])

        if not log_column or not full_path or not selected_wells:
            return jsonify({"error": "Parameter 'log_column', 'full_path', dan 'selected_wells' dibutuhkan."}), 400

        # --- Proses File ---
        base_dir = os.path.abspath(os.path.join(
            'data', full_path.replace('data/', '')))
        if not base_dir.startswith(os.path.abspath('data')):
            return jsonify({"error": f"Akses ke direktori '{full_path}' ditolak."}), 403

        paths_to_process = [os.path.join(
            base_dir, f"{well}.csv") for well in selected_wells]

        all_data_frames = []
        for file_path in paths_to_process:
            if os.path.exists(file_path):
                df_temp = pd.read_csv(file_path)
                all_data_frames.append(df_temp)
            else:
                print(f"Peringatan: File tidak ditemukan di path: {file_path}")
                continue

        if not all_data_frames:
            return jsonify({"error": "Tidak ada data valid yang ditemukan untuk sumur yang dipilih."}), 404

        df = pd.concat(all_data_frames, ignore_index=True)

        # --- Filter Data (Opsional) ---
        selected_intervals = payload.get('selected_intervals', [])
        if selected_intervals and 'MARKER' in df.columns:
            df = df[df['MARKER'].isin(selected_intervals)]

        selected_zones = payload.get('selected_zones', [])
        if selected_zones and 'ZONE' in df.columns:
            df = df[df['ZONE'].isin(selected_zones)]

        if df.empty or log_column not in df.columns:
            return jsonify({"error": f"Tidak ada data valid setelah filter atau kolom '{log_column}' tidak ditemukan."}), 404

        # --- Hitung Persentil ---
        log_data = df[log_column].dropna()
        if log_data.empty:
            return jsonify({"error": f"Tidak ada nilai numerik yang valid di kolom '{log_column}'."}), 404

        p_low_value = np.nanpercentile(log_data, low_percentile)
        p_high_value = np.nanpercentile(log_data, high_percentile)

        return jsonify({
            "p_low": round(p_low_value, 4),
            "p_high": round(p_high_value, 4)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-folder-percentiles', methods=['POST'])
def get_folder_percentiles():
    """
    Menghitung persentil dinamis untuk SEMUA SUMUR DALAM SATU FOLDER,
    dengan menerapkan filter interval/zona jika ada.
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Request body tidak valid."}), 400

        # --- Ekstrak Parameter dari Payload ---
        full_path = payload.get('full_path')
        log_column = payload.get('log_column')
        low_percentile = float(payload.get('low_percentile', 5.0))
        high_percentile = float(payload.get('high_percentile', 95.0))

        if not all([full_path, log_column]):
            return jsonify({"error": "Parameter 'full_path' dan 'log_column' dibutuhkan."}), 400

        # --- Proses File ---
        base_dir = os.path.abspath(os.path.join(
            'data', full_path.replace('data/', '')))
        if not base_dir.startswith(os.path.abspath('data')):
            return jsonify({"error": f"Akses ke direktori '{full_path}' ditolak."}), 403

        all_files_in_folder = glob.glob(os.path.join(base_dir, '*.csv'))
        if not all_files_in_folder:
            return jsonify({"error": f"Tidak ada file .csv yang ditemukan di {full_path}"}), 404

        df_list = [pd.read_csv(f) for f in all_files_in_folder]
        df_all_wells = pd.concat(df_list, ignore_index=True)

        if log_column not in df_all_wells.columns:
            return jsonify({"error": f"Kolom '{log_column}' tidak ditemukan pada data di folder tersebut."}), 404

        # --- PENAMBAHAN: Filter Data (Opsional) ---
        selected_intervals = payload.get('selected_intervals', [])
        if selected_intervals and 'MARKER' in df_all_wells.columns:
            df_all_wells = df_all_wells[df_all_wells['MARKER'].isin(
                selected_intervals)]

        selected_zones = payload.get('selected_zones', [])
        if selected_zones and 'ZONE' in df_all_wells.columns:
            df_all_wells = df_all_wells[df_all_wells['ZONE'].isin(
                selected_zones)]

        if df_all_wells.empty:
            return jsonify({"error": f"Tidak ada data valid setelah filter pada folder '{full_path}'."}), 404

        # --- Hitung Persentil ---
        log_data = df_all_wells[log_column].dropna()
        if log_data.empty:
            return jsonify({"error": f"Tidak ada data numerik yang valid di kolom '{log_column}' setelah filter."}), 404

        p_low_value = np.nanpercentile(log_data, low_percentile)
        p_high_value = np.nanpercentile(log_data, high_percentile)

        return jsonify({
            "p_low_folder": round(p_low_value, 4),
            "p_high_folder": round(p_high_value, 4)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-intersection-point', methods=['POST'])
def get_intersection_point():
    """
    Endpoint API ringan yang MENGGUNAKAN KEMBALI logika terpusat.
    """
    try:
        payload = request.get_json()
        selected_wells = payload.get('selected_wells', [])
        full_path = payload.get('full_path', [])
        selected_intervals = payload.get('selected_intervals', [])
        prcnt_qz = float(payload.get('prcnt_qz', 5))
        prcnt_wtr = float(payload.get('prcnt_wtr', 5))

        if not selected_wells:
            return jsonify({"error": "Well harus dipilih."}), 400

        df_list = [pd.read_csv(os.path.join(
            full_path, f"{w}.csv")) for w in selected_wells]
        df = pd.concat(df_list, ignore_index=True)

        if selected_intervals and 'MARKER' in df.columns:
            df = df[df['MARKER'].isin(selected_intervals)]

        # Panggil fungsi terpusat yang sama
        intersection_coords = calculate_nphi_rhob_intersection(
            df, prcnt_qz, prcnt_wtr)

        return jsonify(intersection_coords)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500


@app.route('/api/get-gr-ma-sh', methods=['POST'])
def get_gr_ma_sh_defaults():
    """
    Endpoint untuk menghitung nilai default GR_MA dan GR_SH
    berdasarkan titik terdekat dari crossplot NPHI-RHOB.
    """
    try:
        payload = request.get_json()
        full_path = payload.get('full_path', [])
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])
        selected_zones = payload.get('selected_zones', [])
        prcnt_qz = float(payload.get('prcnt_qz', 5))
        prcnt_wtr = float(payload.get('prcnt_wtr', 5))

        if not selected_wells:
            return jsonify({"error": "Well harus dipilih."}), 400

        df_list = [pd.read_csv(os.path.join(
            full_path, f"{w}.csv")) for w in selected_wells]
        df = pd.concat(df_list, ignore_index=True)

        if selected_intervals and 'MARKER' in df.columns:
            df = df[df['MARKER'].isin(selected_intervals)]

        if selected_zones and 'ZONE' in df.columns:
            df = df[df['ZONE'].isin(selected_zones)]

        # Panggil fungsi autoplot Anda
        # Kita bisa berikan nilai default untuk percentile di sini
        gr_params = calculate_gr_ma_sh_from_nphi_rhob(
            df, prcnt_qz, prcnt_wtr)

        return jsonify(gr_params)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

# Tambahkan ini di dalam app.py


@app.route('/api/get-default-rtsh', methods=['POST'])
def get_default_rtsh():
    """
    Endpoint untuk menghitung nilai RT_SH default dari sumur yang dipilih.
    """
    try:
        payload = request.get_json()
        selected_wells = payload.get('selected_wells', [])
        full_path = payload.get('full_path', '')

        if not selected_wells:
            # Jika tidak ada sumur, kembalikan nilai default standar
            return jsonify({"default_rt_sh": 2.2})

        # --- Logika Perhitungan ---
        all_well_data = []
        for well_name in selected_wells:
            file_path = os.path.join(full_path, f"{well_name}.csv")
            if os.path.exists(file_path):
                # Baca data dari setiap sumur yang ada
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                all_well_data.append(df_well)

        if not all_well_data:
            # Jika file dari sumur yang dipilih tidak ada, kembalikan default
            return jsonify({"default_rt_sh": 2.2})

        # 1. Gabungkan semua data dari sumur yang dipilih menjadi satu DataFrame
        combined_df = pd.concat(all_well_data, ignore_index=True)

        # 2. Filter data di mana VSH > cutoff
        # CATATAN: VSH (V/V) biasanya antara 0-1. Cutoff '5' tidak mungkin.
        # Saya asumsikan maksud Anda adalah VSH > 0.5 (50%). Anda bisa ubah nilai ini.
        VSH_SHALE_CUTOFF = 0.5
        shale_zones_df = combined_df[combined_df['VSH_LINEAR']
                                     > VSH_SHALE_CUTOFF]

        # 3. Hitung nilai rata-rata (mean) dari kolom RT pada data yang sudah difilter
        if not shale_zones_df.empty and 'RT' in shale_zones_df.columns:
            # Hitung rata-rata jika ada data shale
            rt_sh_mean = shale_zones_df['RT'].mean()
        else:
            # Jika tidak ada zona shale yang ditemukan, gunakan default standar
            rt_sh_mean = 2.2

        # 4. Periksa apakah hasilnya valid, jika tidak, kembalikan default
        if pd.isna(rt_sh_mean):
            rt_sh_mean = 2.2

        # Kembalikan hasilnya dalam format JSON, dibulatkan untuk kebersihan
        return jsonify({"default_rt_sh": round(rt_sh_mean, 4)})

    except Exception as e:
        print(f"Error calculating default RT_SH: {e}")
        # Jika terjadi error, kembalikan saja nilai default standar
        return jsonify({"default_rt_sh": 2.2})

# PLOT AND CALCULATION API


@app.route('/api/get-plot', methods=['POST'])
def get_plot():
    """
    Handles requests to generate a default well log plot for multiple wells.
    """
    try:
        request_data = request.get_json()
        full_path = request_data.get('full_path')
        selected_wells = request_data.get('selected_wells')
        selected_intervals = request_data.get('selected_intervals')
        selected_zones = request_data.get('selected_zones')

        if not selected_wells:
            return jsonify({"error": "No wells were selected."}), 400

        print(f"Request received to plot wells: {selected_wells}")

        df_list = []
        for well_name in selected_wells:
            file_path = os.path.join(full_path, f"{well_name}.csv")
            if os.path.exists(file_path):
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                df_list.append(df_well)

        if not df_list:
            return jsonify({"error": "No valid data could be found for the selected wells."}), 404

        df = pd.concat(df_list, ignore_index=True)

        # df_marker = extract_markers_with_mean_depth(df)
        # df = normalize_xover(df, 'NPHI', 'RHOB')
        # df = normalize_xover(df, 'RT', 'RHOB')

        if selected_intervals:  # <-- ADD THIS BLOCK
            if 'MARKER' in df.columns:
                # Keep only the rows where 'MARKER' is in our list
                df = df[df['MARKER'].isin(selected_intervals)]
            else:
                print("Warning: 'MARKER' column not found, cannot filter by interval.")

        if selected_zones:
            if 'ZONE' in df.columns:
                df = df[df['ZONE'].isin(selected_zones)]
            else:
                print(
                    "Warning: 'ZONE' column not found, cannot filter by zone.")

        if df.empty:
            return jsonify({"error": "No data available for the selected wells and intervals."}), 404

        # 1. Generate both plots from your plotting function
        fig, fig_header = plot_log_default(df=df)

        # 2. Check if the plots were created successfully
        if fig is None or fig_header is None:
            print("Error: plot_log_default() failed to return one or both figures.")
            return jsonify({"error": "Failed to generate plot figures."}), 500

        print(fig_header)

        # 3. Create the response object with two distinct keys
        response_data = {
            "main_plot": json.loads(fig.to_json()),
            "header_plot": json.loads(fig_header.to_json())
        }

        # 4. Return the combined JSON object
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500


@app.route('/api/get-normalization-plot', methods=['POST', 'OPTIONS'])
def get_normalization_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            data = request.get_json()
            full_path = data.get('full_path', [])
            selected_wells = data.get('selected_wells', [])
            selected_intervals = data.get('selected_intervals', [])
            selected_zones = data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih"}), 400

            # Baca dan gabungkan HANYA data dari sumur yang dipilih
            df_list = []
            for well_name in selected_wells:
                file_path = os.path.join(
                    full_path, f"{well_name}.csv")
                if os.path.exists(file_path):
                    df_list.append(pd.read_csv(file_path, on_bad_lines='warn'))

            if not df_list:
                return jsonify({"error": "Data untuk sumur yang dipilih tidak ditemukan."}), 404

            df = pd.concat(df_list, ignore_index=True)

            log_in_col = 'GR'
            log_out_col = 'GR_NORM'

            # Validasi kolom hasil normalisasi
            if log_out_col not in df.columns or df[log_out_col].isnull().all():
                return jsonify({"error": f"Tidak ada data normalisasi yang valid untuk sumur yang dipilih. Jalankan proses pada interval yang benar."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            fig, fig_header = plot_normalization(
                df=df
            )

            # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/list-wells', methods=['POST'])
def list_wells():
    data = request.get_json()
    full_path = data.get('full_path', '')
    try:
        if not os.path.exists(full_path):
            # This might happen on the very first run before any files are there
            return jsonify({"error": 'no file found'}), 200

        well_files = [f.replace('.csv', '')
                      for f in os.listdir(full_path) if f.endswith('.csv')]
        well_files.sort()
        return jsonify(well_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/api/run-interval-normalization', methods=['POST', 'OPTIONS'])
# def run_interval_normalization():
#     if request.method == 'OPTIONS':
#         return jsonify({'status': 'ok'}), 200

#     try:
#         payload = request.get_json()
#         params = payload.get('params', {})
#         full_path = payload.get('full_path')
#         selected_wells = payload.get('selected_wells', [])
#         selected_intervals = payload.get('selected_intervals', [])
#         selected_zones = payload.get('selected_zones', [])
#         isDataPrep = payload.get('isDataPrep', False)

#         if not selected_wells:
#             return jsonify({"error": "No wells were provided to process."}), 400

#         file_paths = [os.path.join(
#             full_path, f"{well_name}.csv") for well_name in selected_wells]

#         print(f"Mulai normalisasi untuk {len(file_paths)} file...")
#         processed_dfs = []

#         for path in file_paths:
#             safe_path = os.path.abspath(path)
#             if not safe_path.startswith(os.path.abspath('data')):
#                 print(f"Warning: Access denied for path '{path}'")
#                 continue
#             if not os.path.exists(safe_path):
#                 print(f"Peringatan: File di {safe_path} tidak ditemukan.")
#                 continue

#             df = pd.read_csv(safe_path)

#             # Panggil fungsi normalisasi baru dari normalization.py
#             # Semua parameter yang dibutuhkan ada di dalam 'params'
#             df_norm = calculate_gr_normalized(
#                 df=df,
#                 params=params,
#                 target_intervals=selected_intervals,
#                 target_zones=selected_zones
#             )

#             # Simpan data yang telah dimodifikasi kembali ke file CSV
#             df_norm.to_csv(safe_path, index=False)
#             processed_dfs.append(df_norm)
#             print(f"Normalisasi selesai untuk {os.path.basename(safe_path)}")

#         if not processed_dfs:
#             return jsonify({"error": "Tidak ada file yang berhasil diproses."}), 400

#         final_df = pd.concat(processed_dfs, ignore_index=True)
#         result_json = final_df.to_json(orient='records')

#         return jsonify({
#             "message": f"Normalisasi selesai untuk {len(processed_dfs)} file.",
#             "data": result_json
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500


@app.route('/api/run-interval-normalization', methods=['POST', 'OPTIONS'])
def run_interval_normalization():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        payload = request.get_json()
        params = payload.get('params', {})
        full_path = payload.get('full_path')
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])
        selected_zones = payload.get('selected_zones', [])

        if not selected_wells:
            return jsonify({"error": "No wells were provided to process."}), 400

        file_paths = [os.path.join(
            full_path, f"{well_name}.csv") for well_name in selected_wells]

        print(f"Mulai normalisasi individual untuk {len(file_paths)} file...")
        processed_dfs = []

        # Ekstrak parameter yang akan digunakan untuk menghitung persentil
        log_column = params.get('LOG_IN', 'GR')
        low_percentile = float(params.get('LOW_IN', 5.0))
        high_percentile = float(params.get('HIGH_IN', 95.0))

        for path in file_paths:
            safe_path = os.path.abspath(path)
            if not safe_path.startswith(os.path.abspath('data')):
                print(f"Warning: Access denied for path '{path}'")
                continue
            if not os.path.exists(safe_path):
                print(f"Peringatan: File di {safe_path} tidak ditemukan.")
                continue

            df = pd.read_csv(safe_path)

            # --- PERUBAHAN UTAMA DIMULAI DI SINI ---

            # 1. Buat salinan DataFrame untuk difilter
            df_filtered = df.copy()

            # 2. Terapkan filter interval dan zona PADA SUMUR INI SAJA
            if selected_intervals and 'MARKER' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['MARKER'].isin(
                    selected_intervals)]

            if selected_zones and 'ZONE' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['ZONE'].isin(
                    selected_zones)]

            if df_filtered.empty or log_column not in df_filtered.columns:
                print(
                    f"Peringatan: Tidak ada data yang cocok dengan filter untuk {os.path.basename(safe_path)}. Melewati sumur ini.")
                # Tetap proses sumur lain, jangan hentikan semuanya
                # Tambahkan df original jika tidak ada yg bisa dinormalisasi
                processed_dfs.append(df)
                continue

            # 3. Hitung persentil spesifik UNTUK SUMUR INI
            log_data = df_filtered[log_column].dropna()
            if log_data.empty:
                print(
                    f"Peringatan: Tidak ada data log valid untuk {os.path.basename(safe_path)}. Melewati sumur ini.")
                processed_dfs.append(df)
                continue

            well_specific_pct_low_old = np.nanpercentile(
                log_data, low_percentile)
            well_specific_pct_high_old = np.nanpercentile(
                log_data, high_percentile)

            # 4. Buat salinan parameter dan update dengan nilai persentil sumur ini
            current_params = params.copy()
            current_params['PCT_LOW_OLD'] = well_specific_pct_low_old
            current_params['PCT_HIGH_OLD'] = well_specific_pct_high_old

            print(
                f"  - Menjalankan normalisasi untuk {os.path.basename(safe_path)} dengan PCT_OLD: {well_specific_pct_low_old:.2f} & {well_specific_pct_high_old:.2f}")

            # 5. Panggil fungsi normalisasi dengan parameter yang sudah disesuaikan
            df_norm = calculate_gr_normalized(
                df=df,  # Gunakan df original agar normalisasi diterapkan ke semua data, bukan hanya yg difilter
                params=current_params,  # Gunakan parameter yang sudah di-update
                target_intervals=selected_intervals,
                target_zones=selected_zones
            )

            # --- AKHIR DARI PERUBAHAN UTAMA ---

            # Simpan data yang telah dimodifikasi kembali ke file CSV
            df_norm.to_csv(safe_path, index=False)
            processed_dfs.append(df_norm)
            print(f"Normalisasi selesai untuk {os.path.basename(safe_path)}")

        if not processed_dfs:
            return jsonify({"error": "Tidak ada file yang berhasil diproses."}), 400

        final_df = pd.concat(processed_dfs, ignore_index=True)
        result_json = final_df.to_json(orient='records')

        return jsonify({
            "message": f"Normalisasi individual selesai untuk {len(processed_dfs)} file.",
            "data": result_json
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-smoothing', methods=['POST', 'OPTIONS'])
def run_smoothing():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        payload = request.get_json()
        params = payload.get('params', {})

        window = int(params.get('WINDOW', 5))
        col_in = params.get('LOG_IN', 'GR')
        col_out = params.get('LOG_OUT', f"{col_in}_SM")

        isDataPrep = payload.get('isDataPrep', False)

        # --- LOGIKA PENANGANAN PATH YANG DISESUAIKAN ---
        paths_to_process = []
        if isDataPrep:
            print(f"Mulai smoothing untuk Data Prep...")
            # Untuk Data Prep, 'file_paths' berisi path relatif lengkap
            file_paths = payload.get('file_paths', [])
            for path in file_paths:
                full_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))
                if full_path.startswith(DATA_ROOT):
                    paths_to_process.append(full_path)
                else:
                    print(
                        f"Peringatan Keamanan: Akses ke path '{path}' ditolak.")
        else:
            # Untuk Dashboard, bangun path dari 'selected_wells'
            selected_wells = payload.get('selected_wells', [])
            full_path_dir = payload.get('full_path', '')  # Ini adalah wellsDir
            print(
                f"Mulai smoothing untuk Dashboard pada {len(selected_wells)} sumur...")

            paths_to_process = [os.path.join(
                full_path_dir, f"{well_name}.csv") for well_name in selected_wells]

        if not paths_to_process:
            return jsonify({"error": "Tidak ada file valid yang ditemukan untuk diproses."}), 400

        # --- LOGIKA VALIDASI DAN PEMROSESAN ---
        selected_intervals = payload.get('selected_intervals', [])
        selected_zones = payload.get('selected_zones', [])

        # Validasi untuk Dashboard: harus ada interval atau zona
        if not isDataPrep and not selected_intervals and not selected_zones:
            return jsonify({"error": "Sumur dan setidaknya satu interval atau zona harus dipilih."}), 400

        processed_dfs = []
        for file_path in paths_to_process:
            if not os.path.exists(file_path):
                print(f"Peringatan: File tidak ditemukan di path: {file_path}")
                continue

            df = pd.read_csv(file_path, on_bad_lines='warn')

            # Tentukan baris mana yang akan diproses
            # Default: proses semua baris (untuk Data Prep)
            mask = pd.Series(True, index=df.index)
            if not isDataPrep:
                if selected_intervals and 'MARKER' in df.columns:
                    mask = df['MARKER'].isin(selected_intervals)
                elif selected_zones and 'ZONE' in df.columns:
                    mask = df['ZONE'].isin(selected_zones)
                else:
                    # Jika tidak ada interval/zona yang cocok, lewati file ini
                    print(
                        f"Peringatan: Tidak ada baris yang cocok dengan interval/zona di {os.path.basename(file_path)}")
                    continue

            if not mask.any():
                print(
                    f"Peringatan: Tidak ada data yang cocok dengan filter di {os.path.basename(file_path)}")
                continue

            # Terapkan smoothing hanya pada baris yang cocok dengan mask
            df_to_smooth = df[mask].copy()
            df_to_smooth[col_out] = df_to_smooth[col_in].rolling(
                window=window, center=True, min_periods=1).mean()

            # Gabungkan kembali hasil smoothing ke dataframe asli
            df[col_out] = np.nan  # Buat kolom baru
            # Update hanya baris yang telah di-smooth
            df.update(df_to_smooth[[col_out]])

            # Simpan kembali file dengan data yang sudah diperbarui
            df.to_csv(file_path, index=False)
            processed_dfs.append(df)
            print(f"Smoothing selesai untuk {os.path.basename(file_path)}")

        if not processed_dfs:
            return jsonify({"error": "Tidak ada file yang berhasil diproses."}), 400

        final_df = pd.concat(processed_dfs, ignore_index=True)
        result_json = final_df.to_json(orient='records')

        return jsonify({
            "message": f"Smoothing selesai untuk {len(processed_dfs)} file.",
            "data": result_json
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-depth-matching', methods=['POST', 'OPTIONS'])
def run_depth_matching_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        # --- (Tidak ada perubahan di bagian ini) ---
        payload = request.get_json()
        ref_path = payload.get('ref_las_path', '')
        lwd_path = payload.get('lwd_las_path', '')
        ref_curve = payload.get('ref_curve')
        lwd_curve = payload.get('lwd_curve')
        slack = int(payload.get('slack', 25))
        num_chunks = int(payload.get('num_chunks', 10))
        output_lwd_curve = payload.get('output_lwd_curve')

        # Memanggil fungsi pemrosesan data (tidak berubah)
        result_df = run_depth_matching(
            ref_path, lwd_path, ref_curve, lwd_curve, num_chunks, slack)

        # --------------------------------------------------------------------
        # --- PERUBAHAN UTAMA: Logika Penyimpanan Hasil Dimodifikasi ---
        # --------------------------------------------------------------------

        # 1. Baca file Referensi (REF/WL) asli
        original_ref_df = pd.read_csv(ref_path)
        if 'DEPT' in original_ref_df.columns:
            original_ref_df.rename(columns={'DEPT': 'DEPTH'}, inplace=True)

        # 2. Siapkan data kurva baru yang akan ditambahkan/diperbarui
        new_curve_col_name = f"{lwd_curve}_DM"
        new_curve_df = result_df[['DEPTH', new_curve_col_name]]

        # 3. LOGIKA OVERWRITE: Jika kolom dengan nama output sudah ada di file REF, hapus dulu.
        if output_lwd_curve in original_ref_df.columns:
            original_ref_df.drop(columns=[output_lwd_curve], inplace=True)

        # 4. Gabungkan (merge) dataframe referensi asli dengan kurva baru
        #    'how=left' memastikan semua baris dari file referensi asli tetap ada.
        merged_ref_df = pd.merge(
            original_ref_df, new_curve_df, on='DEPTH', how='left')

        # 5. Ganti nama kolom '_DM' menjadi nama output akhir yang diinginkan
        merged_ref_df.rename(
            columns={new_curve_col_name: output_lwd_curve}, inplace=True)

        # 6. Simpan kembali ke file Referensi (REF/WL), menimpa file yang lama
        merged_ref_df.to_csv(ref_path, index=False)

        # ---------------------------------------------------------------
        # --- Akhir dari Perubahan ---
        # ---------------------------------------------------------------

        # Bagian untuk menyimpan file 'MATCHING.csv' tetap dipertahankan karena berguna untuk perbandingan
        matching_df = pd.DataFrame({
            'DEPTH': result_df['DEPTH'],
            ref_curve: result_df[ref_curve],
            lwd_curve: result_df[lwd_curve],
            output_lwd_curve: result_df[new_curve_col_name]
        })
        matching_file_path = os.path.join(
            os.path.dirname(lwd_path), "MATCHING.csv")
        matching_df.to_csv(matching_file_path, index=False)

        # Perbarui pesan sukses untuk mencerminkan file yang diubah
        return jsonify({"message": f"Proses berhasil. Kurva '{output_lwd_curve}' ditambahkan/diperbarui di file {os.path.basename(ref_path)}."})

    except Exception as e:
        logging.error(f"Error di run-depth-matching: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-vsh-calculation', methods=['POST', 'OPTIONS'])
def run_vsh_calculation():
    """
    Endpoint untuk menjalankan kalkulasi VSH berdasarkan parameter dari frontend,
    dan menyimpan hasilnya kembali ke file CSV.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"Memulai kalkulasi VSH untuk {len(selected_wells)} sumur...")

            # Ekstrak parameter dari frontend, dengan nilai default
            gr_ma = float(params.get('GR_MA', 30))
            gr_sh = float(params.get('GR_SH', 120))
            input_log = params.get('GR', 'GR')
            output_log = params.get('output_log', 'VSH_LINEAR')

            # Loop melalui setiap sumur yang dipilih
            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {file_path}, file tidak ditemukan.")
                    continue

                # Baca data sumur secara lengkap, tanpa difilter di sini
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # Panggil fungsi logika untuk menghitung VSH dan teruskan filter sebagai argumen
                df_updated = calculate_vsh_from_gr(
                    df=df_well,
                    gr_log=input_log,
                    gr_ma=gr_ma,
                    gr_sh=gr_sh,
                    output_col=output_log,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )

                # Simpan (overwrite) file CSV dengan DataFrame LENGKAP yang sudah diperbarui
                df_updated.to_csv(file_path, index=False)
                print(f"Hasil VSH untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi VSH berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-porosity-calculation', methods=['POST', 'OPTIONS'])
def run_porosity_calculation():
    """
    Endpoint untuk menjalankan kalkulasi Porositas dan menyimpan hasilnya.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {file_path}, file tidak ditemukan.")
                    continue

                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                df_updated = calculate_porosity(
                    df=df_well,
                    params=params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )

                df_updated.to_csv(file_path, index=False)
                print(
                    f"Hasil Porositas untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi Porositas berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-porosity-plot', methods=['POST', 'OPTIONS'])
def get_porosity_plot():
    """
    Endpoint untuk membuat dan menampilkan plot hasil kalkulasi porositas.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom hasil kalkulasi sebelumnya (VSH, PHIE) sudah ada
            required_cols = ['VSH', 'PHIE', 'PHIT', 'PHIE_DEN', 'PHIT_DEN']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi VSH dan Porosity terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting yang baru
            fig, fig_header = plot_phie_den(
                df=df,
            )

            # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


def _run_gsa_process(payload, gsa_function_to_run):
    """
    Helper generik yang sudah diperbaiki untuk menjalankan proses GSA.
    """
    try:
        # Ekstrak semua data yang dibutuhkan dari payload
        params = payload.get('params', {})
        full_path = payload.get('full_path', '')
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])
        selected_zones = payload.get('selected_zones', [])

        if not selected_wells:
            return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

        for well_name in selected_wells:
            file_path = os.path.join(full_path, f"{well_name}.csv")
            if not os.path.exists(file_path):
                print(f"Peringatan: File untuk {well_name} tidak ditemukan.")
                continue

            df_well = pd.read_csv(file_path, on_bad_lines='warn')

            # --- PERBAIKAN UTAMA ---
            # Teruskan SEMUA parameter yang relevan, termasuk filter, ke fungsi pemroses
            df_processed = gsa_function_to_run(
                df_well,
                params,
                target_intervals=selected_intervals,
                target_zones=selected_zones
            )
            # --- AKHIR PERBAIKAN ---

            df_processed.to_csv(file_path, index=False)
            print(f"Hasil untuk sumur '{well_name}' telah disimpan.")

        return jsonify({"message": f"Kalkulasi berhasil untuk {len(selected_wells)} sumur."}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-rgsa', methods=['POST', 'OPTIONS'])
def run_rgsa():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        payload = request.get_json()
        # Call the helper with the RGSA processing function
        return _run_gsa_process(payload, process_all_wells_rgsa)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-ngsa', methods=['POST', 'OPTIONS'])
def run_ngsa():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        payload = request.get_json()
        # Call the helper with the NGSA processing function
        return _run_gsa_process(payload, process_all_wells_ngsa)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-dgsa', methods=['POST', 'OPTIONS'])
def run_dgsa():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        payload = request.get_json()
        # Call the helper with the DGSA processing function
        return _run_gsa_process(payload, process_all_wells_dgsa)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-gsa-plot', methods=['POST', 'OPTIONS'])
def get_gsa_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting GSA yang baru
            fig, fig_header = plot_gsa_main(df)

            # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/trim-data-dash', methods=['POST'])
def run_trim_dashboard():
    """
    Endpoint untuk melakukan trim pada data log sumur berdasarkan parameter dari frontend.
    """
    try:
        data = request.get_json()

        # 1. Ekstrak data dari payload
        full_path = data.get('full_path')
        well_names = data.get('selected_wells', [])
        params = data.get('params', {})

        if not full_path:
            return jsonify({'error': 'Parameter "full_path" wajib diisi.'}), 400
        if not well_names:
            return jsonify({'error': 'Tidak ada sumur yang dipilih (selected_wells).'}), 400

        responses = []

        # 2. Lakukan iterasi untuk setiap sumur yang dipilih
        for well_name in well_names:
            file_path = os.path.join(full_path, f"{well_name}.csv")

            if not os.path.exists(file_path):
                print(
                    f"Peringatan: File {file_path} tidak ditemukan, melewati.")
                continue

            df = pd.read_csv(file_path, on_bad_lines='warn')

            if 'DEPTH' not in df.columns:
                print(
                    f"Peringatan: Kolom 'DEPTH' tidak ditemukan di {well_name}, melewati.")
                continue

            original_rows = len(df)

            # 3. Ambil parameter trim dan konversi ke float dengan aman
            trim_mode = params.get('TRIM_MODE')

            try:
                # Konversi depth_above, anggap None jika string kosong
                depth_above_str = params.get('DEPTH_ABOVE')
                depth_above = float(depth_above_str) if depth_above_str not in [
                    None, ''] else None

                # Konversi depth_below, anggap None jika string kosong
                depth_below_str = params.get('DEPTH_BELOW')
                depth_below = float(depth_below_str) if depth_below_str not in [
                    None, ''] else None
            except (ValueError, TypeError):
                return jsonify({'error': 'Nilai DEPTH_ABOVE atau DEPTH_BELOW harus berupa angka.'}), 400

            # 4. Terapkan logika trimming berdasarkan mode
            if trim_mode == 'DEPTH_ABOVE':
                if depth_above is not None:
                    df = df[df['DEPTH'] >= depth_above].copy()
                else:
                    return jsonify({'error': 'DEPTH_ABOVE harus diisi untuk mode ini.'}), 400

            elif trim_mode == 'DEPTH_BELOW':
                if depth_below is not None:
                    df = df[df['DEPTH'] <= depth_below].copy()
                else:
                    return jsonify({'error': 'DEPTH_BELOW harus diisi untuk mode ini.'}), 400

            elif trim_mode == 'CUSTOM_TRIM':
                if depth_above is not None:
                    df = df[df['DEPTH'] >= depth_above].copy()
                if depth_below is not None:
                    df = df[df['DEPTH'] <= depth_below].copy()
                if depth_above is None and depth_below is None:
                    return jsonify({'error': 'Setidaknya satu nilai (DEPTH_ABOVE atau DEPTH_BELOW) harus diisi untuk CUSTOM_TRIM.'}), 400

            # 5. Simpan file yang sudah di-trim (menimpa file asli)
            df.to_csv(file_path, index=False)

            responses.append({
                'well': well_name,
                'original_rows': original_rows,
                'trimmed_rows': len(df),
                'file_saved': file_path
            })

        return jsonify({'message': 'Proses trimming berhasil diselesaikan.', 'results': responses}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/trim-data', methods=['POST'])
def run_trim_well_log():
    """
    Endpoint API fleksibel untuk menjalankan proses trimming.
    """
    try:
        data = request.get_json()
        params = data.get('params', {})
        full_path = data.get('full_path', '')
        file_paths = data.get('file_paths', [])
        well_names = data.get('selected_wells', [])

        # Fallback: Jika 'file_paths' tidak ada (dari Dashboard), buat dari 'selected_wells'
        if not file_paths and well_names:
            print(f"Menerima 'selected_wells' dari Dashboard: {well_names}")
            file_paths = [os.path.join(
                full_path, f"{well}.csv") for well in well_names]

        if not file_paths:
            return jsonify({'error': 'Tidak ada sumur atau file yang dipilih untuk diproses.'}), 400

        trim_mode = params.get('TRIM_MODE')
        if not trim_mode:
            return jsonify({'error': 'Parameter `TRIM_MODE` wajib diisi'}), 400

        # --- BAGIAN YANG HILANG & DIPERBAIKI ---
        depth_above = None
        depth_below = None

        try:
            # Hanya ambil dan konversi depth_above jika mode memerlukannya
            if trim_mode in ['DEPTH_ABOVE', 'CUSTOM_TRIM']:
                val = params.get('DEPTH_ABOVE')
                if val is None or str(val).strip() == '':
                    return jsonify({'error': f'DEPTH_ABOVE wajib diisi untuk mode {trim_mode}.'}), 400
                depth_above = float(val)

            # Hanya ambil dan konversi depth_below jika mode memerlukannya
            if trim_mode in ['DEPTH_BELOW', 'CUSTOM_TRIM']:
                val = params.get('DEPTH_BELOW')
                if val is None or str(val).strip() == '':
                    return jsonify({'error': f'DEPTH_BELOW wajib diisi untuk mode {trim_mode}.'}), 400
                depth_below = float(val)

        except (ValueError, TypeError):
            return jsonify({'error': 'Nilai DEPTH_ABOVE atau DEPTH_BELOW harus berupa angka yang valid.'}), 400
        # --- AKHIR PERBAIKAN ---

        responses = []
        for path in file_paths:
            full_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))

            if not full_path.startswith(DATA_ROOT):  # Validasi keamanan
                print(f"Peringatan Keamanan: Akses ke path '{path}' ditolak.")
                continue

            if not os.path.exists(full_path):
                print(
                    f"Peringatan: File {full_path} tidak ditemukan, melewati.")
                continue

            df = pd.read_csv(full_path, on_bad_lines='warn')
            if 'DEPTH' not in df.columns:
                print(
                    f"Peringatan: Kolom DEPTH tidak ditemukan di {os.path.basename(full_path)}, melewati.")
                continue

            # Logika pemilihan kolom asli (sudah benar)
            all_columns = df.columns.tolist()
            processed_suffixes = ('_TR', '_NO', '_SM', '_FM')
            excluded_columns = ['DEPTH', 'WELL_NAME',
                                'MARKER', 'ZONE', 'GROUP_ID', 'IQUAL']
            columns_to_trim = [
                col for col in all_columns
                if not col.endswith(processed_suffixes) and col not in excluded_columns
            ]

            # Panggil fungsi logika dengan daftar kolom dan parameter depth yang sudah benar
            processed_df = trim_log_by_masking(
                df=df,
                columns_to_trim=columns_to_trim,
                trim_mode=trim_mode,
                depth_above=depth_above,
                depth_below=depth_below
            )

            processed_df.to_csv(full_path, index=False)

            original_cols_set = set(df.columns)
            new_cols_set = set(processed_df.columns)
            created_cols = list(new_cols_set - original_cols_set)

            responses.append({
                'file_updated': os.path.basename(full_path),
                'columns_created': created_cols
            })

        return jsonify({'message': 'Trimming berhasil diselesaikan.', 'results': responses}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-smoothing-plot', methods=['POST', 'OPTIONS'])
def get_smoothing_plot():
    """
    Endpoint untuk membuat dan menampilkan plot hasil kalkulasi porositas.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path')
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            df_marker_info = extract_markers_with_mean_depth(df)

            # Panggil fungsi plotting yang baru
            fig_result = plot_smoothing(
                df=df,
                df_marker=df_marker_info,
                df_well_marker=df
            )

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-rgbe-rpbe', methods=['POST', 'OPTIONS'])
def run_rgbe_rpbe_calculation():
    """
    Endpoint untuk menjalankan kalkulasi RGBE-RPBE pada sumur yang dipilih.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            # Harusnya string, bukan list
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"[INFO] Memulai kalkulasi RGBE-RPBE untuk {len(selected_wells)} sumur.")

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # Baca seluruh data sumur, jangan difilter di sini
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # --- PERUBAHAN KUNCI ---
                # Panggil fungsi pemroses dengan DataFrame lengkap dan filter sebagai argumen
                df_processed = process_rgbe_rpbe(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )
                # --- AKHIR PERUBAHAN KUNCI ---

                # Simpan kembali DataFrame yang sudah diproses dan lengkap
                df_processed.to_csv(file_path, index=False)
                print(
                    f"Hasil RGBE-RPBE untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi RGBE-RPBE berhasil untuk {len(selected_wells)} sumur."}), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-rgbe-rpbe-plot', methods=['POST', 'OPTIONS'])
def get_rgbe_rpbe_plot():
    """
    Endpoint for generating RGBE-RPBE visualization plot
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            df_list = []
            for well in selected_wells:
                file_path = os.path.join(full_path, f"{well}.csv")
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                df_list.append(df_well)

            df = pd.concat(df_list, ignore_index=True)

            required_cols = ['DEPTH', 'GR', 'RT', 'NPHI',
                             'RHOB', 'VSH', 'IQUAL', 'RGBE', 'RPBE']
            # if not all(col in df.columns for col in required_cols):
            #     # This error will now be correctly shown if the calculation hasn't been run
            #     return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi RGBE-RPBE terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            fig, fig_header = plot_rgbe_rpbe(df)
            # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-rt-r0', methods=['POST', 'OPTIONS'])
def run_rt_r0_calculation():
    """
    Endpoint untuk menjalankan kalkulasi RT-R0 pada sumur yang dipilih.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"Memulai kalkulasi RT-R0 untuk {len(selected_wells)} sumur...")

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # Baca seluruh data sumur, jangan difilter di sini
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # Panggil fungsi pemroses dengan DataFrame lengkap dan filter sebagai argumen
                df_processed = process_rt_r0(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )

                # Ganti nama kolom sesuai permintaan
                df_processed = df_processed.rename(
                    columns={'R0': 'RO', 'RTR0': 'RT_RO'})

                # Simpan kembali DataFrame yang sudah diproses dan lengkap
                df_processed.to_csv(file_path, index=False)
                print(f"Hasil RT-R0 untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi RT-R0 berhasil untuk {len(selected_wells)} sumur."}), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-swgrad', methods=['POST', 'OPTIONS'])
def run_swgrad_calculation():
    """Endpoint untuk menjalankan kalkulasi SWGRAD pada sumur yang dipilih."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            # Ambil parameter dari payload
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])
            params = payload.get('params', {})  # Ambil params untuk diteruskan

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # 1. Baca seluruh data sumur, jangan difilter di sini
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # --- PERUBAHAN KUNCI ---
                # 2. Panggil fungsi pemroses dengan DataFrame lengkap dan filter sebagai argumen
                df_processed = process_swgrad(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )
                # --- AKHIR PERUBAHAN KUNCI ---

                # 3. Simpan kembali DataFrame yang sudah diproses dan lengkap
                df_processed.to_csv(file_path, index=False)
                print(
                    f"Hasil SWGRAD untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi SWGRAD berhasil untuk {len(selected_wells)} sumur."}), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-dns-dnsv', methods=['POST', 'OPTIONS'])
def run_dns_dnsv_calculation():
    """
    Endpoint for running DNS-DNSV calculations on selected wells.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "No wells selected."}), 400

            print(
                f"Starting DNS-DNSV calculation for {len(selected_wells)} wells...")

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Warning: Skipping well {well_name}, file not found.")
                    continue

                # 1. Read the full well data without filtering
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # 2. Call the processing function with the full DataFrame and filters
                df_processed = process_dns_dnsv(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )

                # 3. Save the processed, complete DataFrame back to the file
                df_processed.to_csv(file_path, index=False)
                print(
                    f"DNS-DNSV results for well '{well_name}' have been saved.")

            return jsonify({"message": f"DNS-DNSV calculation successful for {len(selected_wells)} wells."}), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-rt-r0-plot', methods=['POST', 'OPTIONS'])
def get_rt_r0_plot():
    """
    Endpoint for generating RT-R0 visualization plot
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Read and combine data from selected wells
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validate required columns
            required_cols = ['DEPTH', 'GR', 'RT', 'NPHI',
                             'RHOB', 'VSH', 'IQUAL', 'RO', 'RT_RO', 'RWA_FULL']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi RT-R0 terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")
            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Generate plot
            fig, fig_header = plot_rt_r0(df)

            # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-swgrad-plot', methods=['POST', 'OPTIONS'])
def get_swgrad_plot():
    """Endpoint for generating SWGRAD visualization plot."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            df_list = []
            for well in selected_wells:
                file_path = os.path.join(full_path, f"{well}.csv")
                # 1. Read data robustly
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                df_list.append(df_well)

            df = pd.concat(df_list, ignore_index=True)

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # 2. Add the normalization step before plotting
            df = normalize_xover(df, 'NPHI', 'RHOB')

            # 3. Validate that all required columns now exist
            required_cols = ['DEPTH', 'GR', 'RT', 'VSH', 'NPHI',
                             'RHOB', 'SWGRAD'] + [f'SWARRAY_{i}' for i in range(1, 26)]
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi SWGRAD terlebih dahulu."}), 400

            # 4. Generate plot
            fig, fig_header = plot_swgrad(df)
           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-dns-dnsv-plot', methods=['POST', 'OPTIONS'])
def get_dns_dnsv_plot():
    """
    Endpoint for generating DNS-DNSV visualization plot
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', '')
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Read and combine data from selected wells
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validate required columns
            required_cols = ['DEPTH', 'GR', 'RHOB',
                             'NPHI', 'VSH', 'DNS', 'DNSV']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi DNS-DNSV terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Generate plot
            fig, fig_header = plot_dns_dnsv(df)

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-histogram-plot', methods=['POST', 'OPTIONS'])
def get_histogram_plot():
    """
    Endpoint untuk membuat plot histogram berdasarkan sumur dan parameter
    yang dipilih oleh pengguna.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            full_path = payload.get('full_path', [])
            selected_wells = payload.get('selected_wells', [])
            log_to_plot = payload.get('log_column')
            num_bins = int(payload.get('bins', 50))
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400
            if not log_to_plot:
                return jsonify({"error": "Tidak ada log yang dipilih untuk dianalisis."}), 400

            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            fig_result = plot_histogram(df, log_to_plot, num_bins)

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-crossplot', methods=['POST', 'OPTIONS'])
def get_crossplot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        payload = request.get_json()
        full_path = payload.get('full_path', [])
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])
        selected_zones = payload.get('selected_zones', [])
        x_col = payload.get('x_col', 'NPHI')
        y_col = payload.get('y_col', 'RHOB')
        gr_ma = float(payload.get('gr_ma', 30))
        gr_sh = float(payload.get('gr_sh', 120))
        rho_ma = float(payload.get('RHO_MA', 2.645))
        rho_sh = float(payload.get('RHO_SH', 2.61))
        nphi_ma = float(payload.get('NPHI_MA', -0.02))
        nphi_sh = float(payload.get('NPHI_SH', 0.398))

        prcnt_qz = float(payload.get('prcnt_qz', 5))
        prcnt_wtr = float(payload.get('prcnt_wtr', 5))

        if prcnt_qz < 0 or prcnt_wtr < 0:
            return jsonify({'error': 'Persentase kuarsa dan air tidak boleh negatif'}), 400
        if prcnt_qz is None:
            prcnt_qz = 0
        if prcnt_wtr is None:
            prcnt_wtr = 0

        if not selected_wells:
            return jsonify({'error': 'Well belum dipilih'}), 400

        df_list = [pd.read_csv(os.path.join(
            full_path, f"{w}.csv"), on_bad_lines='warn') for w in selected_wells]
        df = pd.concat(df_list, ignore_index=True)

        fig = generate_crossplot(
            df, x_col, y_col, gr_ma, gr_sh, rho_ma, rho_sh, nphi_ma, nphi_sh, prcnt_qz=prcnt_qz, prcnt_wtr=prcnt_wtr, selected_intervals=selected_intervals, selected_zones=selected_zones)
        fig_json = pio.to_json(fig)
        return Response(response=fig_json, status=200, mimetype='application/json')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Silahkan modifikasi nilai persentase'}), 500


@app.route('/api/get-vsh-plot', methods=['POST', 'OPTIONS'])
def get_vsh_plot():
    """
    Endpoint untuk membuat dan menampilkan plot hasil kalkulasi VSH.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', '')
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom VSH sudah ada
            if 'VSH_LINEAR' not in df.columns:
                # Jika menggunakan 'VSH' sebagai nama kolom utama
                if 'VSH_GR' in df.columns:
                    df.rename(columns={'VSH_GR': 'VSH_LINEAR'}, inplace=True)
                else:
                    return jsonify({"error": "Data VSH_GR belum dihitung. Jalankan modul VSH_GR Calculation terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            fig, fig_header = plot_vsh_linear(
                df=df,
            )

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-sw-calculation', methods=['POST', 'OPTIONS'])
def run_sw_calculation():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # 1. Baca seluruh data sumur tanpa filtering
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # 2. Panggil fungsi kalkulasi dengan DataFrame lengkap dan filter
                df_updated = calculate_sw(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )

                # 3. Simpan kembali DataFrame yang sudah diproses
                df_updated.to_csv(file_path, index=False)
                print(f"Hasil SW untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi Saturasi Air berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-sw-simandoux-calculation', methods=['POST', 'OPTIONS'])
def run_sw_simandoux_calculation():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Warning: Skipping well {well_name}, file not found.")
                    continue

                # 1. Read the entire well data file without filtering
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # --- KEY CHANGE ---
                # 2. Call the calculation function with the full DataFrame and filters
                df_updated = calculate_sw_simandoux(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )
                # --- END OF CHANGE ---

                # 3. Save the processed, complete DataFrame back to the file
                df_updated.to_csv(file_path, index=False)
                print(
                    f"Hasil SW (Simandoux) untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi Saturasi Air (Simandoux) berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-sw-plot', methods=['POST', 'OPTIONS'])
def get_sw_plot():
    """
    Endpoint untuk membuat dan menampilkan plot hasil kalkulasi Saturasi Air.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom hasil kalkulasi sebelumnya sudah ada
            required_cols = ['SW']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data SW belum lengkap. Jalankan modul SW Calculation terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting yang baru
            fig, fig_header = plot_sw_indo(
                df=df,
            )

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-rwa-calculation', methods=['POST', 'OPTIONS'])
def run_rwa_calculation():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Warning: Skipping well {well_name}, file not found.")
                    continue

                # 1. Read the entire well data file without filtering
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # --- KEY CHANGE ---
                # 2. Call the calculation function with the full DataFrame and filters
                df_updated = calculate_rwa(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )
                # --- END OF CHANGE ---

                # 3. Save the processed, complete DataFrame back to the file
                df_updated.to_csv(file_path, index=False)
                print(f"Hasil RWA untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi RWA berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-rwa-plot', methods=['POST', 'OPTIONS'])
def get_rwa_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            required_cols = ['RWA_FULL', 'RWA_SIMPLE', 'RWA_TAR']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data RWA belum dihitung. Jalankan modul RWA Calculation terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting RWA
            fig, fig_header = plot_rwa_indo(
                df=df,
            )

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-vsh-dn-calculation', methods=['POST', 'OPTIONS'])
def run_vsh_dn_calculation():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Warning: Skipping well {well_name}, file not found.")
                    continue

                # 1. Read the entire well data file without filtering
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # --- KEY CHANGE ---
                # 2. Call the calculation function with the full DataFrame and filters
                df_updated = calculate_vsh_dn(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )
                # --- END OF CHANGE ---

                # 3. Save the processed, complete DataFrame back to the file
                df_updated.to_csv(file_path, index=False)
                print(
                    f"Hasil VSH-DN untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi VSH (D-N) berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-module2-plot', methods=['POST', 'OPTIONS'])
def get_module2_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            df = df.rename(
                columns={'VSH_GR': 'VSH_LINEAR', 'RWA_SIMPLE': 'RWA'})

            # required_cols = ['RWA', 'IQUAL', 'PHIE', 'VSH_LINEAR', 'SW']
            # if not all(col in df.columns for col in required_cols):
            #     return jsonify({"error": "Data Module 2 belum lengkap. Jalankan semua Module 2 Calculation terlebih dahulu."}), 400

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting RWA
            fig, fig_header = plot_module_2(
                df=df,
            )

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-iqual-calculation', methods=['POST', 'OPTIONS'])
def run_iqual_calculation():
    """
    Endpoint untuk menjalankan kalkulasi IQUAL pada sumur yang dipilih.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            full_path = payload.get('full_path', '')
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            selected_zones = payload.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(full_path, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # Panggil fungsi kalkulasi dengan DataFrame lengkap dan semua parameter/filter
                df_updated = calculate_iqual(
                    df_well,
                    params,
                    target_intervals=selected_intervals,
                    target_zones=selected_zones
                )

                df_updated.to_csv(file_path, index=False)
                print(f"Hasil IQUAL untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi IQUAL berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-gwd', methods=['POST', 'OPTIONS'])
def gwd_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            gwd_path = os.path.join(GWD_DIR, 'gwd-data.csv')

            if not os.path.exists(gwd_path):
                return jsonify({"error": f"File tidak ditemukan: {gwd_path}"}), 404
            # 1. Baca data dari file CSV
            gwd_data = pd.read_csv(gwd_path, on_bad_lines='warn')

            # 2. Panggil fungsi plotting dengan data yang sudah diolah
            fig_result = plot_gwd(
                df=gwd_data
            )

            # 3. Kirim plot yang sudah jadi sebagai JSON
            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-iqual', methods=['POST', 'OPTIONS'])
def get_iqual_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            df = df.rename(
                columns={'RWA_SIMPLE': 'RWA'})

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting IQUAL yang baru
            fig_result = plot_iqual(df)

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/structures', methods=['GET'])
def get_structures_overview():
    """
    Get list of all available fields.
    """
    try:
        data = get_fields_list()
        return jsonify(data), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/structures/fields', methods=['GET'])
def get_fields_list_route():
    """
    Get list of all available fields.
    """
    try:
        data = get_fields_list()
        return jsonify(data), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/structures/field/<field_name>', methods=['GET'])
def get_field_info(field_name):
    """
    Get all structures for a specific field.
    """
    try:
        data = get_field_structures(field_name)
        return jsonify(data), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/structures/structure/<field_name>/<structure_name>', methods=['GET'])
def get_structure_info(field_name, structure_name):
    """
    Get detailed information for a specific structure.
    """
    try:
        data = get_structure_details(field_name, structure_name)
        return jsonify(data), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/structures/well/<well_name>', methods=['GET'])
def get_well_info(well_name):
    """
    Get detailed information for a specific well across all structures.
    """
    try:
        data = get_well_details(well_name)
        return jsonify(data), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/structures/wells', methods=['GET'])
def get_all_wells():
    """
    Get list of all wells across all structures (simplified).
    """
    try:
        # Get all fields first
        fields_data = get_fields_list()
        all_wells = set()

        # Collect wells from all fields
        for field in fields_data['fields']:
            try:
                field_structures = get_field_structures(field['field_name'])
                all_wells.update(field_structures['total_wells'])
            except Exception as e:
                print(
                    f"Error processing field {field['field_name']}: {str(e)}")
                continue

        wells = sorted(list(all_wells))
        return jsonify({
            'wells': wells,
            'total_wells': len(wells)
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/structures/search', methods=['POST'])
def search_structures():
    """
    Search for wells, fields, or structures based on query parameters.
    """
    try:
        request_data = request.get_json()
        # 'well', 'field', 'structure'
        search_type = request_data.get('type', 'well')
        query = request_data.get('query', '').strip().lower()

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        results = []

        if search_type == 'field':
            # Search for fields
            fields_data = get_fields_list()
            for field in fields_data['fields']:
                if query in field['field_name'].lower():
                    results.append(field)

        elif search_type == 'structure':
            # Search for structures across all fields
            fields_data = get_fields_list()
            for field in fields_data['fields']:
                try:
                    field_structures = get_field_structures(
                        field['field_name'])
                    for structure in field_structures['structures']:
                        if query in structure['structure_name'].lower():
                            results.append(structure)
                except Exception as e:
                    print(
                        f"Error searching in field {field['field_name']}: {str(e)}")
                    continue

        elif search_type == 'well':
            # Search for wells across all structures
            fields_data = get_fields_list()
            for field in fields_data['fields']:
                try:
                    field_structures = get_field_structures(
                        field['field_name'])
                    for structure in field_structures['structures']:
                        matching_wells = [
                            well for well in structure['wells'] if query in well.lower()]
                        for well in matching_wells:
                            results.append({
                                'well_name': well,
                                'field_name': structure['field_name'],
                                'structure_name': structure['structure_name']
                            })
                except Exception as e:
                    print(
                        f"Error searching wells in field {field['field_name']}: {str(e)}")
                    continue

        else:
            return jsonify({"error": "Invalid search type. Use 'well', 'field', or 'structure'"}), 400

        return jsonify({
            'search_type': search_type,
            'query': query,
            'results': results,
            'total_results': len(results)
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-splicing', methods=['POST', 'OPTIONS'])
def run_splicing():
    """
    Endpoint API untuk menjalankan proses splicing/merging logs dari file .csv.
    Output akan disimpan satu level di atas folder data dengan nama folder sebagai nama file.
    Automatically applies markers from marker file if available.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        # Import the marker functions
        from services.qc_service import read_marker_file, append_markers_to_dataframe

        # 1. Ambil data dari payload frontend
        payload = request.get_json()
        params = payload.get('params', {})
        field_name = payload.get('field_name', 'adera')
        structure_name = payload.get('structure_name', 'benuang')

        # Get file paths from frontend
        run1_file_path = payload.get('run1_file_path')
        run2_file_path = payload.get('run2_file_path')
        # run1_well = payload.get('run1_well')
        # run2_well = payload.get('run2_well')

        print("Payload diterima:", payload)

        # Validate required parameters
        if not run1_file_path or not run2_file_path:
            return jsonify({"error": "run1_file_path and run2_file_path are required"}), 400

        # if not run1_well or not run2_well:
        #     return jsonify({"error": "run1_well and run2_well are required"}), 400

        # 2. Use file paths from frontend directly
        path_run1 = run1_file_path
        path_run2 = run2_file_path

        print(f"Run 1 (data atas) dari file: {path_run1}")
        print(f"Run 2 (data bawah) dari file: {path_run2}")

        if not os.path.exists(path_run1) or not os.path.exists(path_run2):
            return jsonify({"error": f"Satu atau kedua file .csv tidak ditemukan."}), 404

        # 3. Baca file .csv dan konversi ke DataFrame
        df_run1 = pd.read_csv(path_run1, on_bad_lines='warn')
        df_run2 = pd.read_csv(path_run2, on_bad_lines='warn')

        # Rename DEPT to DEPTH if needed
        if 'DEPT' in df_run1.columns and 'DEPTH' not in df_run1.columns:
            df_run1 = df_run1.rename(columns={'DEPT': 'DEPTH'})
        if 'DEPT' in df_run2.columns and 'DEPTH' not in df_run2.columns:
            df_run2 = df_run2.rename(columns={'DEPT': 'DEPTH'})

        print("File .csv berhasil dimuat dan diubah ke DataFrame.")

        # Opsi data cleaning
        if 'RHOZ' in df_run1.columns:
            df_run1['RHOZ'] /= 1000
        if 'RHOZ' in df_run2.columns:
            df_run2['RHOZ'] /= 1000

        # 4. Panggil Logika Inti untuk Memproses Data
        processed_df = splice_and_flag_logs(df_run1, df_run2, params)

        # 5. Generate output path: one directory up, using folder name as filename
        # Get current directory (will be like .../BNG-057/)
        current_dir = os.path.dirname(path_run1)
        # Get parent directory (will be like .../benuang/)
        parent_dir = os.path.dirname(current_dir)
        # Get folder name (will be like BNG-057)
        folder_name = os.path.basename(current_dir)
        # Create output filename from folder name
        output_filename = f"{folder_name}.csv"
        # Full output path in the parent directory
        output_path = os.path.join(parent_dir, output_filename)

    # 6. NEW: Find and apply marker file before saving
        marker_applied = False
        marker_file_path = None
        zone_applied = False

        try:
            # First apply zones for BNG wells
            if 'BNG' in folder_name.upper():
                processed_df = append_zones_to_dataframe(
                    processed_df, folder_name)
                zone_applied = True
                print(f"Applied zone classification to {folder_name}")

            # Look for any file containing "MARKER" in its name in the parent directory
            marker_file_found = False
            for filename in os.listdir(parent_dir):
                if "MARKER" in filename.upper() and filename.endswith('.csv'):
                    marker_file_path = os.path.join(parent_dir, filename)
                    print(f"Found marker file: {marker_file_path}")

                    try:
                        # Read marker file
                        marker_df = read_marker_file(marker_file_path)
                        print(
                            f"Successfully read marker file with {len(marker_df)} rows")

                        # Apply markers to the spliced dataframe
                        # Use the folder name as the well identifier for marker matching
                        processed_df_with_markers = append_markers_to_dataframe(
                            processed_df, marker_df, folder_name
                        )

                        # If markers were successfully applied, use the updated dataframe
                        if 'MARKER' in processed_df_with_markers.columns:
                            processed_df = processed_df_with_markers
                            marker_applied = True
                            print(
                                f"Successfully applied markers to {folder_name}")
                        else:
                            print(
                                f"No markers found for well {folder_name} in marker file")

                        marker_file_found = True
                        break  # Stop after finding and processing the first marker file

                    except Exception as marker_error:
                        print(
                            f"Error processing marker file {filename}: {marker_error}")
                        continue

            if not marker_file_found:
                print(f"No marker file found in {parent_dir}")

        except Exception as marker_search_error:
            print(f"Error searching for marker files: {marker_search_error}")

        # Add this code after marker and zone processing is complete

        # 7. Handle existing file (preserve columns not in current operation)
        if os.path.exists(output_path):
            # Read existing file
            existing_df = pd.read_csv(output_path, on_bad_lines='warn')
            print(
                f"File {output_filename} already exists - updating specific columns only")

            # Identify columns to update (only in processed_df)
            columns_to_update = processed_df.columns.tolist()

            # Create a new DataFrame with all columns from existing file
            merged_df = existing_df.copy()

            # Update DEPTH index for matching
            merged_df.set_index('DEPTH', inplace=True)
            processed_df.set_index('DEPTH', inplace=True)

            # Update only the columns that exist in processed_df
            for col in columns_to_update:
                if col != 'DEPTH':  # Skip the index column
                    merged_df[col] = processed_df[col]

            # Reset index to get DEPTH back as a column
            merged_df.reset_index(inplace=True)
            processed_df.reset_index(inplace=True)

            # Save the merged result
            merged_df.to_csv(output_path, index=False)
            print(f"Updated columns {columns_to_update} in existing file")

        else:
            # If file doesn't exist, save the processed_df directly
            processed_df.to_csv(output_path, index=False)
            print(f"Created new file {output_filename}")
        # Update response to include zone information
        response = {
            "message": f"Splicing berhasil! Hasil disimpan sebagai '{output_filename}' di direktori '{parent_dir}'.",
            "output_file_path": output_path,
            "output_filename": output_filename,
            "folder_name": folder_name,
            "field_name": field_name,
            "structure_name": structure_name,
            "marker_applied": marker_applied,
            "marker_file_path": marker_file_path,
            "zone_applied": zone_applied
        }

        # Add marker-specific information if markers were applied
        if marker_applied:
            response[
                "message"] += f" Markers have been applied from {os.path.basename(marker_file_path)}."

            # Count unique markers applied
            if 'MARKER' in processed_df.columns:
                unique_markers = processed_df['MARKER'].dropna(
                ).unique().tolist()
                response["markers_applied"] = unique_markers
                response["total_markers"] = len(unique_markers)

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-splicing-plot', methods=['POST', 'OPTIONS'])
def get_splicing_plot():
    """
    Endpoint untuk membuat dan menampilkan plot hasil splicing.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            # Get file path from frontend
            payload = request.get_json()
            file_path = payload.get('file_path')

            if not file_path:
                return jsonify({"error": "file_path is required"}), 400

            print(f"Reading splicing results from: {file_path}")

            # Check if file exists
            if not os.path.exists(file_path):
                return jsonify({"error": f"Output file not found: {file_path}"}), 404

            df = pd.read_csv(file_path, on_bad_lines='warn')

            fig, fig_header = plot_splicing(df=df)
           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/structure-folders/<field_name>/<structure_name>', methods=['GET'])
def get_structure_folders(field_name: str, structure_name: str):
    """
    Get CSV files and folder names from a specific structure directory.

    Args:
        field_name: Name of the field (e.g., 'adera')
        structure_name: Name of the structure (e.g., 'benuang')

    Returns:
        JSON response containing folder names and CSV files list
    """
    try:
        # Use the folder navigation service to get structure contents
        contents = get_structure_wells_folders(field_name, structure_name)

        # Extract folder names only
        folder_names = [folder['name'] for folder in contents['folders']]

        # Extract CSV file names from the files list (filter by extension)
        csv_file_names = [file['name']
                          for file in contents['files'] if file['extension'] == '.csv']

        return jsonify({
            'field_name': field_name,
            'structure_name': structure_name,
            'folder_names': folder_names,
            'csv_files': csv_file_names,
            'total_folders': len(folder_names),
            'total_csv_files': len(csv_file_names),
            'structure_path': contents['structure_path']
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/well-folder-files/<field_name>/<structure_name>/<well_folder>', methods=['GET'])
def get_well_folder_files_route(field_name: str, structure_name: str, well_folder: str):
    """
    Get files inside a specific well folder within a structure.

    Args:
        field_name: Name of the field (e.g., 'adera')
        structure_name: Name of the structure (e.g., 'benuang')
        well_folder: Name of the well folder (e.g., 'BNG-057')

    Returns:
        JSON response containing categorized files information
    """
    try:
        # Use the folder navigation service to get well folder contents
        contents = get_well_folder_files(
            field_name, structure_name, well_folder)

        # Extract just the CSV file names for cleaner response
        csv_file_names = [csv_file['name']
                          for csv_file in contents['csv_files']]

        return jsonify({
            'field_name': contents['field_name'],
            'structure_name': contents['structure_name'],
            'well_folder': contents['well_folder'],
            'well_path': contents['well_path'],
            'csv_files': csv_file_names,
            'total_csv_files': contents['csv_count'],
            'total_files': contents['total_files'],
            # Include full CSV file details
            'csv_files_details': contents['csv_files']
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-module1-plot', methods=['POST', 'OPTIONS'])
def get_module1_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            # Gunakan .get() tanpa default value agar bisa None
            file_path = request_data.get('file_path')
            selected_wells = request_data.get('selected_wells', [])
            # selected_intervals = request_data.get('selected_intervals', []) # Belum digunakan, tapi biarkan saja

            # 1. Inisialisasi df sebagai None
            df = None

            # 2. Gunakan struktur if/elif/else yang jelas
            if file_path:
                print(f"Mode: File Path. Path: {file_path}")
                if not os.path.exists(file_path):
                    return jsonify({"error": f"File tidak ditemukan: {file_path}"}), 404
                df = pd.read_csv(file_path, on_bad_lines='warn')

            elif selected_wells:
                print(f"Mode: Selected Wells. Wells: {selected_wells}")
                # 3. Implementasikan logika untuk menangani multiple wells
                # CONTOH: Baca semua file dan gabungkan menjadi satu DataFrame
                list_of_dfs = []
                for well_path in selected_wells:
                    if os.path.exists(well_path):
                        well_df = pd.read_csv(well_path, on_bad_lines='warn')
                        list_of_dfs.append(well_df)

                if not list_of_dfs:
                    return jsonify({"error": "Tidak ada file valid yang ditemukan dari selected_wells"}), 404

                # Gabungkan semua dataframe menjadi satu
                df = pd.concat(list_of_dfs, ignore_index=True)

            else:
                # Kasus jika tidak ada file_path atau selected_wells yang diberikan
                return jsonify({"error": "Request harus menyertakan 'file_path' atau 'selected_wells'"}), 400

            # 4. Hanya panggil plot jika df berhasil dibuat
            if df is not None and not df.empty:
                fig, fig_header = plot_module1(df=df)
                if fig is None or fig_header is None:
                    print(
                        "Error: plot_log_default() failed to return one or both figures.")
                    return jsonify({"error": "Failed to generate plot figures."}), 500

                response_data = {
                    "main_plot": json.loads(fig.to_json()),
                    "header_plot": json.loads(fig_header.to_json())
                }

                return jsonify(response_data)
            else:
                return jsonify({"error": "Gagal memproses data atau data kosong."}), 500

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-normalization-prep-plot', methods=['POST', 'OPTIONS'])
def get_normalization_prep_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()

            # Support both file_path (for DirectorySidebar) and selected_wells (for Dashboard)
            file_path = request_data.get('file_path')
            # selected_wells = request_data.get('selected_wells', [])
            # selected_intervals = request_data.get('selected_intervals', [])

            if file_path:
                # DirectorySidebar mode - single file
                if not os.path.exists(file_path):
                    return jsonify({"error": f"File tidak ditemukan: {file_path}"}), 404
                df = pd.read_csv(file_path, on_bad_lines='warn')

            # Call plotting function with processed data
            fig, fig_header = plot_norm_prep(df=df)

            # Send finished plot as JSON
           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-smoothing-prep-plot', methods=['POST', 'OPTIONS'])
def get_smoothing_prep_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()

            # Support both file_path (for DirectorySidebar) and selected_wells (for Dashboard)
            file_path = request_data.get('file_path')
            # selected_wells = request_data.get('selected_wells', [])
            # selected_intervals = request_data.get('selected_intervals', [])

            if file_path:
                # DirectorySidebar mode - single file
                if not os.path.exists(file_path):
                    return jsonify({"error": f"File tidak ditemukan: {file_path}"}), 404
                df = pd.read_csv(file_path, on_bad_lines='warn')

            # Call plotting function with processed data
            fig_result = plot_smoothing_prep(df=df)

            # Send finished plot as JSON
            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

# ====================================================================
# FILL MISSING API
# ====================================================================


def resolve_paths(payload):
    """
    Helper function to resolve file paths from a request payload.
    Handles two cases:
    1. Data Prep: A 'file_paths' key with a list of relative paths.
    2. Dashboard: 'selected_wells' (names) and 'full_path' (directory).
    Returns a list of absolute, validated file paths.
    """
    paths_to_process = []

    # CASE 1: Data Prep (satu file)
    if 'file_paths' in payload and payload.get('file_paths'):
        print("Path Resolution: Data Prep flow detected.")
        for rel_path in payload['file_paths']:
            # Bangun path absolut dari project root
            abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, rel_path))
            if abs_path.startswith(DATA_ROOT) and os.path.exists(abs_path):
                paths_to_process.append(abs_path)
            else:
                print(
                    f"Warning: Path invalid or access denied for '{rel_path}'")

    # CASE 2: Dashboard (banyak sumur)
    elif 'selected_wells' in payload and 'full_path' in payload:
        print("Path Resolution: Dashboard flow detected.")
        base_dir_rel = payload['full_path']
        selected_wells = payload['selected_wells']

        # Bangun path direktori absolut dari project root
        base_dir_abs = os.path.abspath(
            os.path.join(PROJECT_ROOT, base_dir_rel))

        if not base_dir_abs.startswith(DATA_ROOT):
            print(
                f"Security Warning: Access to directory '{base_dir_rel}' is denied.")
            return []  # Kembalikan list kosong jika direktori dasar tidak aman

        for well_name in selected_wells:
            abs_path = os.path.join(base_dir_abs, f"{well_name}.csv")
            if os.path.exists(abs_path):
                paths_to_process.append(abs_path)
            else:
                print(
                    f"Warning: File not found for well '{well_name}' at '{abs_path}'")

    return paths_to_process


@app.route('/api/flag-missing', methods=['POST'])
def flag_missing_route():
    """
    Endpoint untuk Tahap 1: Menandai nilai yang hilang.
    Sekarang menggunakan helper untuk menangani path.
    """
    try:
        payload = request.get_json()
        logs_to_check = payload.get('logs_to_check', [])

        if not logs_to_check:
            return jsonify({"error": "Log harus dipilih."}), 400

        # Gunakan helper untuk mendapatkan daftar path yang valid
        file_paths = resolve_paths(payload)

        if not file_paths:
            return jsonify({"error": "Tidak ada file valid yang ditemukan untuk diproses."}), 400

        for path in file_paths:
            df = pd.read_csv(path)
            df_flagged = flag_missing_values(df, logs_to_check)
            df_flagged.to_csv(path, index=False)

        return jsonify({
            "message": f"Flagging missing values complete for {len(file_paths)} file(s).",
            "flag_columns": ["MISSING_FLAG"]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/api/fill-flagged-missing', methods=['POST'])
# def fill_flagged_route():
#     """
#     Endpoint untuk Tahap 2: Mengisi nilai yang sudah ditandai.
#     Sekarang menggunakan helper untuk menangani path.
#     """
#     try:
#         payload = request.get_json()
#         logs_to_fill = payload.get('logs_to_fill', [])
#         max_consecutive = payload.get('max_consecutive_nan', 3)
#         isDataPrep = payload.get('isDataPrep', True)

#         if not logs_to_fill:
#             return jsonify({"error": "Log harus dipilih."}), 400

#         # Gunakan helper yang sama untuk mendapatkan daftar path
#         file_paths = resolve_paths(payload)

#         if not file_paths:
#             return jsonify({"error": "Tidak ada file valid yang ditemukan untuk diproses."}), 400

#         for path in file_paths:
#             df = pd.read_csv(path)
#             df_filled = fill_flagged_values(
#                 df, logs_to_fill, max_consecutive, isDataPrep)
#             df_filled.to_csv(path, index=False)

#         return jsonify({"message": f"Fill missing process complete for {len(file_paths)} file(s)."}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/api/fill-flagged-missing', methods=['POST'])
def fill_flagged_route():
    """
    Endpoint untuk Tahap 2: Mengisi nilai yang hilang menggunakan imputasi Machine Learning.
    """
    try:
        payload = request.get_json()
        full_path = payload.get('full_path')
        selected_wells = payload.get('selected_wells', [])
        logs_to_fill = payload.get('logs_to_fill', [])

        # Anda bisa menambahkan parameter well_count jika ingin fleksibel dari frontend
        neighbor_count = payload.get('neighbor_count', 5)

        if not all([full_path, selected_wells, logs_to_fill]):
            return jsonify({"error": "Parameter 'full_path', 'selected_wells', dan 'logs_to_fill' dibutuhkan."}), 400

        # --- Integrasi Logika ML ---

        # 1. Tentukan path dan muat data koordinat
        base_dir = os.path.abspath(os.path.join(
            'data', full_path.replace('data/', '')))
        coord_path = os.path.join('data', 'coord.csv')

        if not os.path.exists(coord_path):
            return jsonify({"error": "File 'data/coord.csv' tidak ditemukan."}), 500

        df_coord = pd.read_csv(coord_path)

        df_coord['LAT'] = pd.to_numeric(df_coord['LAT'], errors='coerce')
        df_coord['LONG'] = pd.to_numeric(df_coord['LONG'], errors='coerce')

        df_coord.dropna(subset=['LAT', 'LONG'], inplace=True)

        processed_count = 0
        for well_filename in selected_wells:
            target_path = os.path.join(base_dir, (well_filename + '.csv'))

            if not os.path.exists(target_path):
                print(
                    f"Peringatan: File {target_path} tidak ditemukan. Melewati...")
                continue

            # Panggil fungsi utama dari ml_imputer
            df_filled = run_ml_imputation_for_well(
                target_well_path=target_path,
                logs_to_fill=logs_to_fill,
                coord_df=df_coord,
                well_count=neighbor_count
            )

            df_filled.to_csv(target_path, index=False)
            processed_count += 1

        if processed_count == 0:
            return jsonify({"error": "Tidak ada file yang berhasil diproses."}), 404

        return jsonify({
            "message": f"Proses imputasi ML selesai untuk {processed_count} sumur. Kolom baru dengan sufiks '_FM' telah ditambahkan ke file asli."
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-fill-missing-plot', methods=['POST'])
def get_fill_missing_plot():
    """
    Endpoint untuk membuat dan mengambil plot hasil proses Fill Missing.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()

            # Support both file_path (for DirectorySidebar) and selected_wells (for Dashboard)
            file_path = request_data.get('file_path')
            # selected_wells = request_data.get('selected_wells', [])
            # selected_intervals = request_data.get('selected_intervals', [])

            if file_path:
                # DirectorySidebar mode - single file
                if not os.path.exists(file_path):
                    return jsonify({"error": f"File tidak ditemukan: {file_path}"}), 404
                df = pd.read_csv(file_path, on_bad_lines='warn')

            # Call plotting function with processed data
            fig, fig_header = plot_fill_missing(df, title="Fill Missing Plot")

            # Send finished plot as JSON
           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-module3-plot', methods=['POST', 'OPTIONS'])
def get_module3_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', '')
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data
            df_list = [pd.read_csv(os.path.join(
                full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # df = df.rename(columns={'VSH_GR': 'VSH_LINEAR', 'RWA_FULL': 'RWA'})

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            # Panggil fungsi plotting RWA
            fig, fig_header = plot_module_3(
                df=df,
            )

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-custom-plot', methods=['POST', 'OPTIONS'])
def get_custom_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            request_data = request.get_json()
            full_path = request_data.get('full_path', [])
            file_path = request_data.get('file_path', '')
            selected_wells = request_data.get('selected_wells', [])
            selected_intervals = request_data.get('selected_intervals', [])
            custom_columns = request_data.get('custom_columns', [])
            selected_zones = request_data.get('selected_zones', [])

            if not selected_wells:
                df_list = [pd.read_csv(file_path, on_bad_lines='warn')]
                df = pd.concat(df_list, ignore_index=True)
                # return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400
            else:
                # Baca dan  gabungkan data dari sumur yang dipilih
                df_list = [pd.read_csv(os.path.join(
                    full_path, f"{well}.csv"), on_bad_lines='warn') for well in selected_wells]
                df = pd.concat(df_list, ignore_index=True)

            if selected_intervals:
                if 'MARKER' in df.columns:
                    df = df[df['MARKER'].isin(selected_intervals)]
                else:
                    print(
                        "Warning: 'MARKER' column not found, cannot filter by interval.")

            if selected_zones:
                if 'ZONE' in df.columns:
                    df = df[df['ZONE'].isin(selected_zones)]
                else:
                    print(
                        "Warning: 'ZONE' column not found, cannot filter by zone.")

            if df.empty:
                return jsonify({"error": "No data available for the selected wells and intervals."}), 404

            fig, fig_header = plot_custom(df, custom_columns)

           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/get-trimming-plot', methods=['POST'])
def get_trimming_plot():
    """
    Endpoint untuk membuat dan mengambil plot hasil proses Fill Missing.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()

            # Support both file_path (for DirectorySidebar) and selected_wells (for Dashboard)
            file_path = request_data.get('file_path')
            # selected_wells = request_data.get('selected_wells', [])
            # selected_intervals = request_data.get('selected_intervals', [])

            if file_path:
                # DirectorySidebar mode - single file
                if not os.path.exists(file_path):
                    return jsonify({"error": f"File tidak ditemukan: {file_path}"}), 404
                df = pd.read_csv(file_path, on_bad_lines='warn')

            # Call plotting function with processed data
            fig, fig_header = plot_trimming(df)

            # Send finished plot as JSON
           # 2. Check if the plots were created successfully
            if fig is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            # 3. Create the response object with two distinct keys
            response_data = {
                "main_plot": json.loads(fig.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            # 4. Return the combined JSON object
            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


# --- NEW: Define a constant for the base directory where LAS files are stored ---
LAS_FILES_BASE_DIR = 'data/las'


@app.route('/api/get-las-logs', methods=['POST'])
def get_las_curves():
    """
    Gets the list of curve mnemonics (logs) from a specified LAS file.
    Expects a JSON payload with a 'las_path' key containing only the filename.
    Example: {"las_path": "BNG-07_Property&Minsol.las"}
    """
    try:
        # 1. Get the filename from the incoming JSON request
        data = request.get_json()
        if not data or 'las_path' not in data:
            return jsonify({"error": "Missing 'las_path' in request body."}), 400

        filename = data['las_path']

        # --- MODIFIED: Construct the full, safe path on the server ---
        # This joins the base directory with the filename from the request.
        full_path = os.path.join(LAS_FILES_BASE_DIR, filename)

        # Security check: ensure the path is still within the intended directory
        safe_path = os.path.abspath(full_path)
        if not safe_path.startswith(os.path.abspath(LAS_FILES_BASE_DIR)):
            return jsonify({"error": "Invalid file path."}), 400

        # 3. File Existence Check
        if not os.path.exists(safe_path):
            return jsonify({"error": f"LAS file not found at path: {full_path}"}), 404

        # 4. Read the LAS file using lasio
        print(f"Reading LAS file: {safe_path}")
        las = lasio.read(safe_path)

        # 5. Extract the curve mnemonics into a list
        curves = [curve.mnemonic for curve in las.curves]

        # 6. Return the list of curves as a JSON response
        return jsonify(curves)

    except lasio.las.LASHeaderError as e:
        return jsonify({'error': f"Failed to parse LAS file header: {str(e)}"}), 500
    except Exception as e:
        # General error handler for any other unexpected issues
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/api/save-las-curve', methods=['POST'])
def save_las_curve():
    """
    Reads a selected curve from a LAS file and merges it into a target CSV well file
    based on the 'DEPTH' column.
    """
    try:
        data = request.get_json()
        required_keys = ['las_filename', 'full_path',
                         'source_log', 'output_log_name']
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required data in request."}), 400

        las_filename = data['las_filename']
        target_well_csv = data['full_path']
        source_log = data['source_log']
        output_log_name = data['output_log_name']

        # 1. Construct full paths for both files
        las_full_path = os.path.join(LAS_FILES_BASE_DIR, las_filename)
        csv_full_path = target_well_csv

        # 2. Read the LAS file and convert to a DataFrame
        if not os.path.exists(las_full_path):
            return jsonify({"error": f"Source LAS file not found: {las_filename}"}), 404
        las = lasio.read(las_full_path)
        las_df = las.df().reset_index()

        # Ensure DEPTH and the source log exist
        if 'DEPTH' not in las_df.columns or source_log not in las_df.columns:
            return jsonify({"error": f"Required columns ('DEPTH', '{source_log}') not in LAS file."}), 400

        # FIXED: Rename the LAS column to the desired output name BEFORE merging
        # This prevents pandas from creating _x and _y suffixes
        las_df_subset = las_df[['DEPTH', source_log]].copy()
        las_df_subset.rename(
            columns={source_log: output_log_name}, inplace=True)

        # 3. Read the target CSV file
        if not os.path.exists(csv_full_path):
            return jsonify({"error": f"Target CSV file not found: {target_well_csv}"}), 404
        csv_df = pd.read_csv(csv_full_path)

        if 'DEPTH' not in csv_df.columns:
            return jsonify({"error": "'DEPTH' column not found in target CSV file."}), 400

        # FIXED: Remove existing column if it exists (corrected syntax)
        if output_log_name in csv_df.columns:
            csv_df = csv_df.drop(columns=[output_log_name])

        # 4. Merge the two DataFrames - now no naming conflicts will occur
        merged_df = pd.merge_asof(
            csv_df.sort_values("DEPTH"),
            las_df_subset.sort_values("DEPTH"),
            on="DEPTH",
            direction="nearest"
        )

        # 5. No need to rename anymore since we already did it before merging

        # 6. Save the updated DataFrame back to the CSV file
        merged_df.to_csv(csv_full_path, index=False)

        return jsonify({"message": f"Successfully saved '{output_log_name}' to '{target_well_csv}'."})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred during save: {str(e)}"}), 500


@app.route('/api/get-depth-matching-prep-plot', methods=['POST', 'OPTIONS'])
def depth_matching_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            file_path = request_data.get('file_path')

            if file_path:
                if not os.path.exists(file_path):
                    return jsonify({"error": f"File tidak ditemukan: {file_path}"}), 404
                df = pd.read_csv(file_path, on_bad_lines='warn')
            else:
                return jsonify({"error": "No file_path provided in the request"}), 400

            fig_result, fig_header = plot_depth_matching(df)

            if fig_result is None or fig_header is None:
                print("Error: plot_log_default() failed to return one or both figures.")
                return jsonify({"error": "Failed to generate plot figures."}), 500

            print(fig_header)

            response_data = {
                "main_plot": json.loads(fig_result.to_json()),
                "header_plot": json.loads(fig_header.to_json())
            }

            return jsonify(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # DEPTH MATCHING DUMMY
    return jsonify({"error": "Method not allowed"}), 405


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Menggabungkan base path dengan subfolder untuk membentuk path data yang dinamis
DM_DIR = os.path.join(BASE_DIR, 'data', 'depth-matching')


# @app.route('/api/get-matching-plot', methods=['POST', 'OPTIONS'])
# def get_matching_plot_endpoint():
#     if request.method == 'OPTIONS':
#         return jsonify({'status': 'ok'}), 200
#     try:
#         # payload = request.get_json()
#         lwd_path = 'data/structures/adera/benuang/BNG-056/'
#         # if not lwd_path:
#         #     return jsonify({"error": "Payload harus menyertakan 'lwd_path'."}), 400

#         matching_file_path = os.path.join(
#             os.path.dirname(lwd_path), "MATCHING.csv")
#         if not os.path.exists(matching_file_path):
#             return jsonify({"error": "'MATCHING.csv' tidak ditemukan."}), 404

#         df = pd.read_csv(matching_file_path)
#         fig_result, summary_result = create_before_after_plot_and_summary(df)

#         return jsonify({"plot_data": fig_result.to_json(), "summary_data": summary_result})
#     except Exception as e:
#         logging.error(
#             f"Error di get-full-matching-analysis: {e}", exc_info=True)
#         return jsonify({"error": str(e)}), 500


@app.route('/api/get-structures-summary', methods=['GET'])
def get_structures_summary():
    """
    Memindai struktur folder 'data/structures' dan mengembalikan ringkasan
    dalam format yang diharapkan oleh frontend StructuresDashboard.
    """
    try:
        # Tentukan path dasar. Sesuaikan jika struktur proyek Anda berbeda.
        base_path = os.path.join(os.path.dirname(
            __file__), 'data', 'structures')

        if not os.path.exists(base_path):
            return jsonify({"error": f"Direktori dasar tidak ditemukan: {base_path}"}), 404

        all_fields_data = []
        summary = {
            "total_fields": 0,
            "total_structures": 0,
            "total_wells": 0,
            # Catatan: total_records & columns memerlukan pembacaan file,
            # yang bisa lambat. Untuk saat ini, kita berikan nilai default.
            "total_records": "N/A"
        }

        # 1. Iterasi melalui Fields (e.g., Adera, Limau)
        for field_name in sorted(os.listdir(base_path)):
            field_path = os.path.join(base_path, field_name)
            if os.path.isdir(field_path):
                summary["total_fields"] += 1
                field_data = {
                    "field_name": field_name.capitalize(),
                    "structures_count": 0,
                    "structures": []
                }

                # 2. Iterasi melalui Structures (e.g., Abab, Benuang)
                structures_in_field = []
                for structure_name in sorted(os.listdir(field_path)):
                    structure_path = os.path.join(field_path, structure_name)
                    if os.path.isdir(structure_path):
                        summary["total_structures"] += 1

                        # 3. Iterasi melalui Wells (folder di dalam structure)
                        wells_list = []
                        for well_name in sorted(os.listdir(structure_path)):
                            well_path = os.path.join(structure_path, well_name)
                            # Asumsikan setiap item di dalam folder structure adalah well
                            # Atau cek jika file .csv/.las
                            if os.path.isdir(well_path):
                                summary["total_wells"] += 1
                                wells_list.append(well_name)

                        structure_data = {
                            "structure_name": structure_name.replace('_', ' ').title(),
                            "field_name": field_name.capitalize(),
                            # Path ini hanya sebagai referensi, sesuaikan jika perlu
                            "file_path": f"/data/structures/{field_name}/{structure_name}",
                            "wells_count": len(wells_list),
                            "wells": wells_list,
                            # Nilai ini tidak bisa didapat tanpa membaca file, jadi kita beri placeholder
                            "total_records": "N/A",
                            "columns": ["N/A"]
                        }
                        structures_in_field.append(structure_data)

                field_data["structures"] = structures_in_field
                field_data["structures_count"] = len(structures_in_field)
                all_fields_data.append(field_data)

        # Gabungkan semua struktur dari semua field untuk list utama
        all_structures = [
            structure for field in all_fields_data for structure in field["structures"]]

        response_data = {
            "fields": all_fields_data,
            "structures": all_structures,
            "wells": [],  # Wells akan di-load saat structure dipilih
            "summary": summary
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(
            f"Error saat memindai struktur folder: {e}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan internal saat memproses struktur direktori."}), 500


# This is for local development testing, Vercel will use its own server
if __name__ == '__main__':
    app.run(debug=True, port=5001)
