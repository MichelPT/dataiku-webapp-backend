# /api/app.py
from app.services.vsh_dn import calculate_vsh_dn
from app.services.rwa import calculate_rwa
from app.services.sw import calculate_sw
from app.services.crossplot import generate_crossplot
from app.services.histogram import plot_histogram
from app.services.ngsa import process_all_wells_ngsa
from app.services.dgsa import process_all_wells_dgsa
from app.services.rgsa import process_all_wells_rgsa
from app.services.depth_matching import depth_matching, plot_depth_matching_results
from app.services.porosity import calculate_porosity
from app.routes.qc_routes import qc_bp
from app.services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_gsa_main,
    plot_log_default,
    plot_smoothing,
    plot_normalization,
    plot_phie_den,
    plot_gsa_main,
    plot_vsh_linear,
    plot_sw_indo,
    plot_rwa_indo
)
from services.porosity import calculate_porosity
from services.depth_matching import depth_matching, plot_depth_matching_results
from services.rgsa import process_all_wells_rgsa
from services.dgsa import process_all_wells_dgsa
from services.ngsa import process_all_wells_ngsa
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
from services.crossplot import generate_crossplot
from services.histogram import plot_histogram

from typing import Optional
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import logging
from app.services.vsh_calculation import calculate_vsh_from_gr
from app.services.data_processing import fill_null_values_in_marker_range, handle_null_values, selective_normalize_handler, smoothing, trim_data_depth
from app.services.qc_service import run_full_qc_pipeline
from services.vsh_calculation import calculate_vsh_from_gr
from services.data_processing import handle_null_values, fill_null_values_in_marker_range, min_max_normalize, selective_normalize_handler, smoothing, trim_data_auto, trim_data_depth
from services.qc_service import run_full_qc_pipeline
from flask import Flask, request, jsonify, Response


app = Flask(__name__)

# Izinkan permintaan dari semua sumber (penting untuk development)
CORS(app)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
app.register_blueprint(qc_bp, url_prefix='/api/qc')


@app.route('/')
def home():
    # A simple route to confirm the API is running
    return "Flask backend is running!"


@app.route('/api/run-qc', methods=['POST'])
def qc_route():
    """
    Receives a list of files, runs the QC process, and returns the results.
    """
    app.logger.info("Received request for /api/run-qc")
    try:
        # The frontend will send a JSON object with a 'files' key
        # files is a list of {'name': '...', 'content': '...'}
        data = request.get_json()
        files_data = data.get('files')

        if not files_data or not isinstance(files_data, list):
            return jsonify({"error": "Invalid input: 'files' key with a list of file objects is required."}), 400

        # Call the refactored logic, passing the app's logger
        results = run_full_qc_pipeline(files_data, app.logger)

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
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'pass_qc.csv')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WELLS_DIR = os.path.join(PROJECT_ROOT, 'data', 'wells')
LAS_DIR = os.path.join(PROJECT_ROOT, 'data', 'depth-matching')


@app.route('/api/fill-null-marker', methods=['POST'])
def fill_null_marker():
    try:
        payload = request.get_json()
        selected_wells = payload.get('selected_wells', [])
        selected_logs = payload.get('selected_logs', [])

        if not selected_wells:
            return jsonify({'error': 'selected_wells wajib diisi'}), 400

        results = []

        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                results.append(
                    {'well': well_name, 'status': 'File tidak ditemukan'})
                continue

            df = pd.read_csv(file_path)

            struktur = df['STRUKTUR'].iloc[0] if 'STRUKTUR' in df.columns else 'UNKNOWN'

            df_filled = fill_null_values_in_marker_range(
                df, selected_logs)
            df_filled.to_csv(file_path, index=False)

            results.append({'well': well_name, 'rows': len(
                df_filled), 'status': 'Berhasil diproses'})

        return jsonify({'message': 'Pengisian nilai null selesai', 'results': results}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-plot', methods=['POST'])
def get_plot():
    try:
        # 1. Terima daftar nama sumur dari frontend
        request_data = request.get_json()
        selected_wells = request_data.get('selected_wells')

        if not selected_wells or len(selected_wells) == 0:
            return jsonify({"error": "Tidak ada sumur yang dipilih"}), 400

        print(f"Menerima permintaan untuk memproses sumur: {selected_wells}")

        list_of_dataframes = []
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if os.path.exists(file_path):
                list_of_dataframes.append(pd.read_csv(file_path))

        if not list_of_dataframes:
            return jsonify({"error": "Tidak ada data yang valid untuk sumur yang dipilih"}), 404

        # Gabungkan semua dataframe menjadi satu dataframe besar
        df = pd.concat(list_of_dataframes, ignore_index=True)

        # 3. PROSES DATA (seperti sebelumnya, tapi pada dataframe gabungan)
        df_marker = extract_markers_with_mean_depth(df)
        df = normalize_xover(df, 'NPHI', 'RHOB')
        df = normalize_xover(df, 'RT', 'RHOB')

        # 4. GENERATE PLOT
        fig = plot_log_default(
            df=df,
            df_marker=df_marker,
            df_well_marker=df
        )

        # 5. KIRIM HASIL
        return jsonify(fig.to_json())

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-normalization-plot', methods=['POST', 'OPTIONS'])
def get_normalization_plot():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            request_data = request.get_json()
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih"}), 400

            # Baca dan gabungkan HANYA data dari sumur yang dipilih
            df_list = []
            for well_name in selected_wells:
                file_path = os.path.join(
                    WELLS_DIR, f"{well_name}.csv")
                if os.path.exists(file_path):
                    df_list.append(pd.read_csv(file_path))

            if not df_list:
                return jsonify({"error": "Data untuk sumur yang dipilih tidak ditemukan."}), 404

            df = pd.concat(df_list, ignore_index=True)

            log_in_col = 'GR'
            log_out_col = 'GR_NORM'

            # Validasi kolom hasil normalisasi
            if log_out_col not in df.columns or df[log_out_col].isnull().all():
                return jsonify({"error": f"Tidak ada data normalisasi yang valid untuk sumur yang dipilih. Jalankan proses pada interval yang benar."}), 400

            fig_result = plot_normalization(
                df=df,
            )

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/list-intervals', methods=['GET'])
def list_intervals():
    """
    Membaca file data utama, menemukan semua nilai unik di kolom 'MARKER',
    dan mengembalikannya sebagai daftar JSON.
    """
    try:
        if not os.path.exists(WELLS_DIR):
            return jsonify({"error": f"File data utama tidak ditemukan di {WELLS_DIR}"}), 404

        files = [f for f in os.listdir(WELLS_DIR) if f.endswith('.csv')]
        if not files:
            return jsonify({"error": "Tidak ada file CSV ditemukan di folder 'wells'."}), 404

        data = []

        for filename in files:
            file_path = os.path.join(WELLS_DIR, filename)
            try:
                df_temp = pd.read_csv(file_path)
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
        unique_markers.sort()

        print(f"Mengirim {len(unique_markers)} interval unik ke frontend.")

        return jsonify(unique_markers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/list-wells', methods=['GET'])
def list_wells():
    try:
        if not os.path.exists(WELLS_DIR):
            return jsonify({"error": "Folder 'wells' tidak ditemukan."}), 404

        # Ambil semua file .csv, hapus ekstensinya
        well_files = [f.replace('.csv', '') for f in os.listdir(
            WELLS_DIR) if f.endswith('.csv')]
        well_files.sort()  # Urutkan nama sumur

        return jsonify(well_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-well-columns', methods=['POST'])
def get_well_columns():
    try:
        data = request.get_json()
        wells = data.get('wells', [])
        result = {}

        for well in wells:
            file_path = os.path.join(WELLS_DIR, f"{well}.csv")
            if os.path.exists(file_path):
                # Hanya baca baris pertama
                df = pd.read_csv(file_path, nrows=1)
                result[well] = df.columns.tolist()
            else:
                result[well] = []

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/run-interval-normalization', methods=['POST', 'OPTIONS'])
def run_interval_normalization():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        # Ambil data dari frontend
        payload = request.get_json()
        params = payload.get('params', {})
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])

        if not selected_wells or not selected_intervals:
            return jsonify({"error": "Sumur dan interval harus dipilih."}), 400

        print(f"Mulai normalisasi untuk {len(selected_wells)} sumur...")

        # Ambil parameter normalisasi
        log_in_col = params.get('LOG_IN', 'GR')
        low_ref = float(params.get('LOW_REF', 40))
        high_ref = float(params.get('HIGH_REF', 140))
        low_in = int(params.get('LOW_IN', 5))
        high_in = int(params.get('HIGH_IN', 95))
        cutoff_min = float(params.get('CUTOFF_MIN', 0.0))
        cutoff_max = float(params.get('CUTOFF_MAX', 250.0))

        processed_dfs = []

        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                print(f"Peringatan: File untuk {well_name} tidak ditemukan.")
                continue

            df = pd.read_csv(file_path)

            # Jalankan handler normalisasi untuk marker terpilih
            df_norm = selective_normalize_handler(
                df=df,
                log_column=log_in_col,
                marker_column='MARKER',
                target_markers=selected_intervals,
                low_ref=low_ref,
                high_ref=high_ref,
                low_in=low_in,
                high_in=high_in,
                cutoff_min=cutoff_min,
                cutoff_max=cutoff_max
            )

            # Simpan kembali ke file
            df_norm.to_csv(file_path, index=False)
            processed_dfs.append(df_norm)

            print(f"Normalisasi selesai untuk {well_name}")

        if not processed_dfs:
            return jsonify({"error": "Tidak ada file yang berhasil diproses."}), 400

        # Gabungkan semua hasil jika diperlukan
        final_df = pd.concat(processed_dfs, ignore_index=True)
        result_json = final_df.to_json(orient='records')

        return jsonify({
            "message": f"Normalisasi selesai untuk {len(processed_dfs)} sumur.",
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
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])

        if not selected_wells or not selected_intervals:
            return jsonify({"error": "Sumur dan interval harus dipilih."}), 400

        print(f"Mulai smoothing untuk {len(selected_wells)} sumur...")

        processed_dfs = []

        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                print(f"Peringatan: File untuk {well_name} tidak ditemukan.")
                continue

            df = pd.read_csv(file_path)

            df_smooth = smoothing(df)

            # Simpan kembali ke file
            df_smooth.to_csv(file_path, index=False)
            processed_dfs.append(df_smooth)

            print(f"Smoothing selesai untuk {well_name}")

        if not processed_dfs:
            return jsonify({"error": "Tidak ada file yang berhasil diproses."}), 400

        # Gabungkan semua hasil jika diperlukan
        final_df = pd.concat(processed_dfs, ignore_index=True)
        result_json = final_df.to_json(orient='records')

        return jsonify({
            "message": f"Smoothing selesai untuk {len(processed_dfs)} sumur.",
            "data": result_json
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-plot', methods=['POST'])
def get_plot():
    """
    Handles requests to generate a default well log plot for multiple wells.
    """
    try:
        request_data = request.get_json()
        selected_wells = request_data.get('selected_wells')

        if not selected_wells:
            return jsonify({"error": "No wells were selected."}), 400

        print(f"Request received to plot wells: {selected_wells}")

        df_list = []
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if os.path.exists(file_path):
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                df_list.append(df_well)

        if not df_list:
            return jsonify({"error": "No valid data could be found for the selected wells."}), 404

        df = pd.concat(df_list, ignore_index=True)

        df_marker = extract_markers_with_mean_depth(df)
        df = normalize_xover(df, 'NPHI', 'RHOB')
        df = normalize_xover(df, 'RT', 'RHOB')

        fig = plot_log_default(
            df=df,
            df_marker=df_marker,
            df_well_marker=df
        )

        return jsonify(fig.to_json())

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
            request_data = request.get_json()
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih"}), 400

            # Baca dan gabungkan HANYA data dari sumur yang dipilih
            df_list = []
            for well_name in selected_wells:
                file_path = os.path.join(
                    WELLS_DIR, f"{well_name}.csv")
                if os.path.exists(file_path):
                    df_list.append(pd.read_csv(file_path))

            if not df_list:
                return jsonify({"error": "Data untuk sumur yang dipilih tidak ditemukan."}), 404

            df = pd.concat(df_list, ignore_index=True)

            log_in_col = 'GR'
            log_out_col = 'GR_NORM'

            # Validasi kolom hasil normalisasi
            if log_out_col not in df.columns or df[log_out_col].isnull().all():
                return jsonify({"error": f"Tidak ada data normalisasi yang valid untuk sumur yang dipilih. Jalankan proses pada interval yang benar."}), 400

            # Siapkan data marker dari DataFrame gabungan
            df_marker_info = extract_markers_with_mean_depth(df)

            # ==========================================================
            # FIX: Panggil `plot_normalization` dengan argumen yang benar
            # ==========================================================
            fig_result = plot_normalization(
                df=df,                 # DataFrame lengkap dengan semua data log
                df_marker=df_marker_info,      # DataFrame khusus untuk teks marker
                df_well_marker=df      # DataFrame lengkap untuk plot latar belakang marker
            )

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/list-wells', methods=['GET'])
def list_wells():
    try:
        if not os.path.exists(WELLS_DIR):
            # This might happen on the very first run before any files are there
            return jsonify({"error": 'no file found'}), 200

        well_files = [f.replace('.csv', '')
                      for f in os.listdir(WELLS_DIR) if f.endswith('.csv')]
        well_files.sort()
        return jsonify(well_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-interval-normalization', methods=['POST', 'OPTIONS'])
def run_interval_normalization():

    if request.method == 'OPTIONS':
        # Respons ini sudah cukup untuk memberitahu browser bahwa permintaan POST diizinkan
        return jsonify({'status': 'ok'}), 200

    try:
        # 1. Terima semua data dari frontend: params, sumur, dan interval
        payload = request.get_json()
        params = payload.get('params', {})
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])

        if not selected_wells or not selected_intervals:
            return jsonify({"error": "Sumur dan Interval harus dipilih."}), 400

        print(
            f"Memulai normalisasi untuk {len(selected_wells)} sumur pada {len(selected_intervals)} interval...")

        # Ekstrak parameter normalisasi dari payload
        log_in_col = params.get('LOG_IN', 'GR')
        log_out_col = params.get('LOG_OUT', 'GR_NORM')
        low_ref = float(params.get('LOW_REF', 40))
        high_ref = float(params.get('HIGH_REF', 140))
        low_in = int(params.get('LOW_IN', 3))
        high_in = int(params.get('HIGH_IN', 97))
        cutoff_min = float(params.get('CUTOFF_MIN', 0.0))
        cutoff_max = float(params.get('CUTOFF_MAX', 250.0))

        processed_dfs = []

        # 2. LOOPING PERTAMA: Iterasi untuk setiap sumur yang dipilih
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                print(
                    f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                continue

            df_well = pd.read_csv(file_path)
            # Buat kolom output baru berisi NaN (Not a Number)
            df_well[log_out_col] = np.nan

            # 3. LOOPING KEDUA: Iterasi untuk setiap interval di dalam sumur saat ini
            for interval in selected_intervals:
                # Filter dataframe untuk mendapatkan baris yang sesuai dengan interval ini
                interval_mask = df_well['MARKER'] == interval

                # Jika tidak ada data untuk interval ini di sumur saat ini, lewati
                if interval_mask.sum() == 0:
                    continue

                # Ambil data log HANYA dari subset interval ini
                log_to_normalize = df_well.loc[interval_mask, log_in_col].dropna(
                ).values

                if len(log_to_normalize) == 0:
                    continue

                # 4. JALANKAN NORMALISASI pada data subset
                normalized_values = min_max_normalize(
                    log_in=log_to_normalize,
                    low_ref=low_ref,
                    high_ref=high_ref,
                    low_in=low_in,
                    high_in=high_in,
                    cutoff_min=cutoff_min,
                    cutoff_max=cutoff_max
                )

                # 5. SIMPAN HASIL kembali ke dataframe utama pada baris yang benar
                df_well.loc[interval_mask, log_out_col] = normalized_values

            processed_dfs.append(df_well)
            df_well.to_csv(file_path, index=False)
            print(
                f"Hasil normalisasi untuk sumur '{well_name}' telah disimpan ke {file_path}")

        # 6. Gabungkan semua dataframe yang sudah diproses
        if not processed_dfs:
            return jsonify({"error": "Tidak ada data yang berhasil diproses."}), 400

        final_df = pd.concat(processed_dfs, ignore_index=True)

        # 7. Kembalikan data yang sudah dinormalisasi sebagai JSON
        # Format 'records' mudah dibaca oleh JavaScript
        result_json = final_df.to_json(orient='records')

        return jsonify({
            "message": f"Normalisasi selesai untuk {len(processed_dfs)} sumur.",
            "data": result_json
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-depth-matching', methods=['POST', 'OPTIONS'])
def run_depth_matching_endpoint():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            ref_las_path = os.path.join(LAS_DIR, 'ref.las')
            lwd_las_path = os.path.join(LAS_DIR, 'lwd.las')

            if not os.path.exists(ref_las_path):
                return jsonify({"error": f"File tidak ditemukan: {ref_las_path}"}), 404
            if not os.path.exists(lwd_las_path):
                return jsonify({"error": f"File tidak ditemukan: {lwd_las_path}"}), 404

            # 1. Panggil fungsi logika untuk mendapatkan data
            ref_data, lwd_data, aligned_data = depth_matching(
                ref_las_path=ref_las_path,
                lwd_las_path=lwd_las_path,
                num_chunks=8
            )

            if aligned_data is None:
                raise ValueError("Proses komputasi Depth Matching gagal.")

            # 2. Panggil fungsi plotting dengan data yang sudah diolah
            fig_result = plot_depth_matching_results(
                ref_df=ref_data,
                lwd_df=lwd_data,
                final_df=aligned_data
            )

            # 3. Kirim plot yang sudah jadi sebagai JSON
            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
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
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"Memulai kalkulasi VSH untuk {len(selected_wells)} sumur...")

            # Ekstrak parameter dari frontend, dengan nilai default
            gr_ma = float(params.get('GR_MA', 30))
            gr_sh = float(params.get('GR_SH', 120))
            input_log = params.get('input_log', 'GR')
            output_log = params.get('output_log', 'VSH_GR')

            # Loop melalui setiap sumur yang dipilih
            for well_name in selected_wells:
                file_path = os.path.join(
                    WELLS_DIR, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # Baca data sumur
                df_well = pd.read_csv(file_path)

                # Panggil fungsi logika untuk menghitung VSH
                df_updated = calculate_vsh_from_gr(

                    df_well, input_log, gr_ma, gr_sh, output_log,
                    marker_column='MARKER',  # atau sesuaikan jika kolom marker berbeda
                    target_markers=selected_intervals
                )

                # Simpan (overwrite) file CSV dengan data yang sudah diperbarui
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
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Loop melalui setiap sumur yang dipilih
            for well_name in selected_wells:
                file_path = os.path.join(
                    WELLS_DIR, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    continue

                df_well = pd.read_csv(file_path)

                # Panggil fungsi logika untuk menghitung Porositas
                df_updated = calculate_porosity(
                    df_well, params,
                    marker_column='MARKER',
                    target_markers=selected_intervals
                )

                # Simpan (overwrite) file CSV dengan data yang sudah diperbarui
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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom hasil kalkulasi sebelumnya (VSH, PHIE) sudah ada
            required_cols = ['VSH', 'PHIE', 'PHIT', 'PHIE_DEN', 'PHIT_DEN']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi VSH dan Porosity terlebih dahulu."}), 400

            df_marker_info = extract_markers_with_mean_depth(df)

            # Panggil fungsi plotting yang baru
            fig_result = plot_phie_den(
                df=df,
                df_marker=df_marker_info,
                df_well_marker=df
            )

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-gsa-calculation', methods=['POST', 'OPTIONS'])
def run_gsa_calculation():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            selected_wells = payload.get('selected_wells', [])
            selected_intervals = payload.get('selected_intervals', [])
            print(selected_intervals)

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(
                    WELLS_DIR, f"{well_name}.csv")

                df_well = pd.read_csv(file_path)

                # Panggil fungsi orkestrator GSA
                df_rgsa = process_all_wells_rgsa(
                    df_well, params, selected_intervals)
                df_ngsa = process_all_wells_ngsa(
                    df_rgsa, params, selected_intervals)
                df_dgsa = process_all_wells_dgsa(
                    df_ngsa, params, selected_intervals)

                # Validasi kolom penting
                required_cols = ['GR', 'RT', 'NPHI',
                                 'RHOB', 'RGSA', 'NGSA', 'DGSA']
                df_dgsa = df_dgsa.dropna(subset=required_cols)
                # df_dgsa = df_dgsa.dropna(subset=required_cols)

                # Hitung anomali
                df_dgsa['RGSA_ANOM'] = df_dgsa['RT'] > df_dgsa['RGSA']
                df_dgsa['NGSA_ANOM'] = df_dgsa['NPHI'] < df_dgsa['NGSA']
                df_dgsa['DGSA_ANOM'] = df_dgsa['RHOB'] < df_dgsa['DGSA']

                # Skoring
                df_dgsa['SCORE'] = df_dgsa[['RGSA_ANOM',
                                            'NGSA_ANOM', 'DGSA_ANOM']].sum(axis=1)

                # Klasifikasi zona
                def classify_zone(score):
                    if score == 3:
                        return 'Zona Prospek Kuat'
                    elif score == 2:
                        return 'Zona Menarik'
                    elif score == 1:
                        return 'Zona Lemah'
                    else:
                        return 'Non Prospek'

                df_dgsa['ZONA'] = df_dgsa['SCORE'].apply(classify_zone)

                # Simpan kembali file CSV dengan kolom GSA baru
                df_dgsa.to_csv(file_path, index=False)
                print(f"Hasil GSA untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi GSA berhasil untuk {len(selected_wells)} sumur."}), 200

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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Panggil fungsi plotting GSA yang baru
            fig_result = plot_gsa_main(df)

            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/trim-data', methods=['POST'])
def run_trim_well_log():
    try:
        data = request.get_json()

        well_names = data.get('selected_wells', [])
        params = data.get('params', {})
        trim_mode = params.get('TRIM_MODE', 'AUTO')
        top_depth = params.get('TOP_DEPTH')
        bottom_depth = params.get('BOTTOM_DEPTH')
        required_columns = data.get(
            'required_columns', ['GR', 'RT', 'NPHI', 'RHOB'])

        if not well_names:
            return jsonify({'error': 'Daftar well_name wajib diisi'}), 400

        responses = []

        for well_name in well_names:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                return jsonify({'error': f"File {well_name}.csv tidak ditemukan."}), 404

            df = pd.read_csv(file_path)

            # Penentuan batas trimming tergantung mode
            if trim_mode == 'AUTO':
                trimmed_df = trim_well_log(
                    df, None, None, required_columns, trim_mode)

            elif trim_mode == 'UNTIL_TOP':
                if not bottom_depth:
                    responses.append(
                        {'error': 'BOTTOM_DEPTH harus diisi untuk mode UNTIL_TOP', 'well': well_name})
                    continue
                trimmed_df = trim_well_log(
                    df, bottom_depth, None, required_columns, trim_mode)

            elif trim_mode == 'UNTIL_BOTTOM':
                if not top_depth:
                    return jsonify({'error': 'TOP_DEPTH harus diisi untuk mode UNTIL_BOTTOM'}), 400
                trimmed_df = trim_well_log(
                    df, None, top_depth, required_columns, trim_mode)

            elif trim_mode == 'CUSTOM':
                if not top_depth or not bottom_depth:
                    return jsonify({'error': 'TOP_DEPTH dan BOTTOM_DEPTH harus diisi untuk mode CUSTOM'}), 400
                trimmed_df = trim_well_log(
                    df, bottom_depth, top_depth, required_columns, trim_mode)

            else:
                return jsonify({'error': f'Mode tidak dikenali: {trim_mode}'}), 400
            if 'DEPTH' not in df.columns:
                return jsonify({'error': f"Kolom DEPTH tidak ditemukan di {well_name}"}), 400

            df.set_index('DEPTH', inplace=True)

            if trim_mode == 'AUTO':
                trim_data_auto = trim_data_auto(df, required_columns)

            else:
                # Trim selain AUTO menggunakan helper `trim()`
                above_flag = 1 if top_depth else 0
                below_flag = 1 if bottom_depth else 0

                if above_flag and top_depth is None:
                    return jsonify({'error': 'top_depth harus diisi'}), 400
                if below_flag and bottom_depth is None:
                    return jsonify({'error': 'bottom_depth harus diisi'}), 400

                trimmed_df = trim_data_depth(
                    df.copy(),
                    top_depth=top_depth or 0,
                    bottom_depth=bottom_depth or 0,
                    above=above_flag,
                    below=below_flag,
                    mode=trim_mode
                )

            # Reset index agar DEPTH kembali sebagai kolom
            trimmed_df.reset_index(inplace=True)

            # Simpan hasil
            trimmed_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            trimmed_df.to_csv(trimmed_path, index=False)

            responses.append({
                'well': well_name,
                'rows': len(trimmed_df),
                'file_saved': f'{well_name}.csv'
            })

        return jsonify({'message': 'Trimming berhasil', 'results': responses}), 200

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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom hasil kalkulasi sebelumnya (VSH, PHIE) sudah ada
            required_cols = ['GR', 'GR_MovingAvg_5', 'GR_MovingAvg_10']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi Smoothing GR terlebih dahulu."}), 400

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
    Endpoint for running RGBE-RPBE calculations on selected wells
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                print(
                    "[ERROR] /api/run-rgbe-rpbe: No wells selected (selected_wells is empty)")
                return jsonify({"error": "Tidak ada sumur yang dipilih.", "detail": "selected_wells array is empty"}), 400

            print(
                f"[INFO] /api/run-rgbe-rpbe: Starting calculation for {len(selected_wells)} wells: {selected_wells}")

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # Read well data
                df_well = pd.read_csv(file_path)

                # Process RGBE-RPBE calculations
                df_processed = process_rgbe_rpbe(df_well, params)

                # Save back to CSV
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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            df_list = []
            for well in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well}.csv")
                # ✨ FIX APPLIED HERE: Handle bad lines gracefully
                # This will read the file, warn about bad lines in the console,
                # and skip them instead of crashing.
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                df_list.append(df_well)

            df = pd.concat(df_list, ignore_index=True)

            required_cols = ['DEPTH', 'GR', 'RT', 'NPHI',
                             'RHOB', 'VSH', 'IQUAL', 'RGBE', 'RPBE']
            if not all(col in df.columns for col in required_cols):
                # This error will now be correctly shown if the calculation hasn't been run
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi RGBE-RPBE terlebih dahulu."}), 400

            fig_result = plot_rgbe_rpbe(df)
            return jsonify(fig_result.to_json())

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-rt-r0', methods=['POST', 'OPTIONS'])
def run_rt_r0_calculation():
    """
    Endpoint for running RT-R0 calculations on selected wells
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"Memulai kalkulasi RT-R0 untuk {len(selected_wells)} sumur...")

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # Read well data
                df_well = pd.read_csv(file_path)

                # Process RT-R0 calculations
                df_processed = process_rt_r0(df_well, params)

                # Save back to CSV
                df_processed.to_csv(file_path, index=False)
                print(f"Hasil RT-R0 untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi RT-R0 berhasil untuk {len(selected_wells)} sumur."}), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


@app.route('/api/run-swgrad', methods=['POST', 'OPTIONS'])
def run_swgrad_calculation():
    """Endpoint for running SWGRAD calculations on selected wells."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    continue

                # 1. Read data robustly
                df_well = pd.read_csv(file_path, on_bad_lines='warn')

                # 2. Make the process idempotent by dropping old results
                cols_to_drop = ['SWGRAD'] + \
                    [f'SWARRAY_{i}' for i in range(1, 26)]
                df_well.drop(columns=df_well.columns.intersection(
                    cols_to_drop), inplace=True)

                # 3. Process SWGRAD calculations
                df_processed = process_swgrad(df_well)

                # 4. Save back to CSV
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
    Endpoint for running DNS-DNSV calculations on selected wells
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        try:
            payload = request.get_json()
            params = payload.get('params', {})
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"Memulai kalkulasi DNS-DNSV untuk {len(selected_wells)} sumur...")

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(
                        f"Peringatan: Melewatkan sumur {well_name}, file tidak ditemukan.")
                    continue

                # Read well data
                df_well = pd.read_csv(file_path)

                # Process DNS-DNSV calculations
                df_processed = process_dns_dnsv(df_well, params)

                # Save back to CSV
                df_processed.to_csv(file_path, index=False)
                print(
                    f"Hasil DNS-DNSV untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi DNS-DNSV berhasil untuk {len(selected_wells)} sumur."}), 200

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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Read and combine data from selected wells
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validate required columns
            required_cols = ['DEPTH', 'GR', 'RT', 'NPHI',
                             'RHOB', 'VSH', 'IQUAL', 'R0', 'RTR0', 'RWA']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi RT-R0 terlebih dahulu."}), 400

            # Generate plot
            fig_result = plot_rt_r0(df)

            return jsonify(fig_result.to_json())

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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            df_list = []
            for well in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well}.csv")
                # 1. Read data robustly
                df_well = pd.read_csv(file_path, on_bad_lines='warn')
                df_list.append(df_well)

            df = pd.concat(df_list, ignore_index=True)

            # 2. Add the normalization step before plotting
            df = normalize_xover(df, 'NPHI', 'RHOB')

            # 3. Validate that all required columns now exist
            required_cols = ['DEPTH', 'GR', 'RT', 'VSH', 'NPHI',
                             'RHOB', 'SWGRAD'] + [f'SWARRAY_{i}' for i in range(1, 26)]
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi SWGRAD terlebih dahulu."}), 400

            # 4. Generate plot
            fig_result = plot_swgrad(df)
            return jsonify(fig_result.to_json())

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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Read and combine data from selected wells
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validate required columns
            required_cols = ['DEPTH', 'GR', 'RHOB',
                             'NPHI', 'VSH', 'DNS', 'DNSV']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data belum lengkap. Jalankan kalkulasi DNS-DNSV terlebih dahulu."}), 400

            # Generate plot
            fig_result = plot_dns_dnsv(df)

            return jsonify(fig_result.to_json())

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
            selected_wells = payload.get('selected_wells', [])
            log_to_plot = payload.get('log_column')
            num_bins = int(payload.get('bins', 50))

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400
            if not log_to_plot:
                return jsonify({"error": "Tidak ada log yang dipilih untuk dianalisis."}), 400

            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

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
        selected_wells = payload.get('selected_wells', [])
        x_col = payload.get('x_col', 'NPHI')
        y_col = payload.get('y_col', 'RHOB')
        gr_ma = float(payload.get('GR_MA', 30))
        gr_sh = float(payload.get('GR_SH', 120))
        rho_ma = float(payload.get('RHO_MA', 2.645))
        rho_sh = float(payload.get('RHO_SH', 2.61))
        nphi_ma = float(payload.get('NPHI_MA', -0.02))
        nphi_sh = float(payload.get('NPHI_SH', 0.398))

        if not selected_wells:
            return jsonify({'error': 'Well belum dipilih'}), 400

        df_list = [pd.read_csv(os.path.join(
            WELLS_DIR, f"{w}.csv")) for w in selected_wells]
        df = pd.concat(df_list, ignore_index=True)

        fig = generate_crossplot(
            df, x_col, y_col, gr_ma, gr_sh, rho_ma, rho_sh, nphi_ma, nphi_sh)
        return jsonify(fig.to_dict())

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom VSH sudah ada
            if 'VSH_LINEAR' not in df.columns:
                # Jika menggunakan 'VSH' sebagai nama kolom utama
                if 'VSH_GR' in df.columns:
                    df.rename(columns={'VSH_GR': 'VSH_LINEAR'}, inplace=True)
                else:
                    return jsonify({"error": "Data VSH_GR belum dihitung. Jalankan modul VSH_GR Calculation terlebih dahulu."}), 400

            df_marker_info = extract_markers_with_mean_depth(df)

            fig_result = plot_vsh_linear(
                df=df,
                df_marker=df_marker_info,
                df_well_marker=df
            )

            return jsonify(fig_result.to_json())

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
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    continue

                df_well = pd.read_csv(file_path)

                df_updated = calculate_sw(df_well, params)

                df_updated.to_csv(file_path, index=False)
                print(f"Hasil SW untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi Saturasi Air berhasil untuk {len(selected_wells)} sumur."})

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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data dari sumur yang dipilih
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            # Validasi: Pastikan kolom hasil kalkulasi sebelumnya sudah ada
            required_cols = ['SWE_INDO']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data SW belum lengkap. Jalankan modul SW Calculation terlebih dahulu."}), 400

            df_marker_info = extract_markers_with_mean_depth(df)

            # Panggil fungsi plotting yang baru
            fig_result = plot_sw_indo(
                df=df,
                df_marker=df_marker_info,
                df_well_marker=df
            )

            return jsonify(fig_result.to_json())

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
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    continue

                df_well = pd.read_csv(file_path)

                # Panggil fungsi logika untuk menghitung RWA
                df_updated = calculate_rwa(df_well, params)

                # Simpan kembali file CSV dengan kolom RWA baru
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
            selected_wells = request_data.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            # Baca dan gabungkan data
            df_list = [pd.read_csv(os.path.join(
                WELLS_DIR, f"{well}.csv")) for well in selected_wells]
            df = pd.concat(df_list, ignore_index=True)

            required_cols = ['RWA_FULL', 'RWA_SIMPLE', 'RWA_TAR']
            if not all(col in df.columns for col in required_cols):
                return jsonify({"error": "Data RWA belum dihitung. Jalankan modul RWA Calculation terlebih dahulu."}), 400

            df_marker_info = extract_markers_with_mean_depth(df)

            # Panggil fungsi plotting RWA
            fig_result = plot_rwa_indo(
                df=df,
                df_marker=df_marker_info,
                df_well_marker=df
            )

            return jsonify(fig_result.to_json())

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
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            for well_name in selected_wells:
                file_path = os.path.join(
                    WELLS_DIR, f"{well_name}.csv")
                if not os.path.exists(file_path):
                    continue

                df_well = pd.read_csv(file_path)
                df_updated = calculate_vsh_dn(df_well, params)
                df_updated.to_csv(file_path, index=False)
                print(
                    f"Hasil VSH-DN untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi VSH (D-N) berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


# This is for local development testing, Vercel will use its own server
if __name__ == '__main__':
    app.run(debug=True, port=5001)
