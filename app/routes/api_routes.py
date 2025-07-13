from flask import Blueprint, request, jsonify, current_app, Response
from ..services import qc_service, plotting_service, data_processing_service
import pandas as pd
import io
import os # <-- Tambahkan import os

api_bp = Blueprint('api', __name__)

# --- Path untuk data lokal ---
# Mendefinisikan path ke direktori data sampel Anda secara dinamis
# Ini mengasumsikan folder 'data/wells' berada di level yang sama dengan folder 'api' Anda
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WELLS_DIR = os.path.join(PROJECT_ROOT, 'data', 'wells')

# --- Rute untuk Quality Control (Tetap sama, sudah benar) ---
@api_bp.route('/run-qc', methods=['POST'])
def qc_route():
    data = request.get_json()
    files_data = data.get('files')
    if not files_data: return jsonify({"error": "Files are required."}), 400
    # Menggunakan qc_service yang belum diimpor di file ini, asumsikan ada di services.
    # Untuk sementara kita gunakan placeholder jika belum ada.
    # results = qc_service.run_full_qc_pipeline(files_data, current_app.logger)
    results = {"message": "QC route hit, logic to be implemented in qc_service."}
    return jsonify(results)

@api_bp.route('/handle-nulls', methods=['POST'])
def handle_nulls_route():
    csv_content = request.get_data(as_text=True)
    if not csv_content: return jsonify({"error": "Request body is empty."}), 400
    cleaned_csv = data_processing_service.handle_null_values(csv_content)
    return Response(cleaned_csv, mimetype="text/csv")


# --- Rute untuk List Sumur (DIKEMBALIKAN) ---
@api_bp.route('/list-wells', methods=['GET'])
def list_wells():
    """Mengambil daftar nama sumur dari direktori lokal."""
    try:
        if not os.path.exists(WELLS_DIR):
            return jsonify({"error": f"Direktori 'wells' tidak ditemukan di: {WELLS_DIR}"}), 404
        well_files = [f.replace('.csv', '') for f in os.listdir(WELLS_DIR) if f.endswith('.csv')]
        well_files.sort()
        return jsonify(well_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Rute untuk Plotting (DIKEMBALIKAN KE VERSI LOKAL) ---
@api_bp.route('/get-plot', methods=['POST'])
def get_plot():
    """
    FIX: Menerima NAMA sumur, lalu MEMBACA file dari direktori lokal.
    """
    try:
        payload = request.get_json()
        selected_wells = payload.get('selected_wells') # Sekarang kita harapkan nama file
        if not selected_wells:
            return jsonify({"error": "Tidak ada sumur yang dipilih"}), 400

        list_of_dataframes = []
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if os.path.exists(file_path):
                list_of_dataframes.append(pd.read_csv(file_path))

        if not list_of_dataframes:
            return jsonify({"error": "Tidak ada data yang valid untuk sumur yang dipilih"}), 404

        df = pd.concat(list_of_dataframes, ignore_index=True)
        
        df_marker = plotting_service.extract_markers_with_mean_depth(df)
        df = plotting_service.normalize_xover(df, 'NPHI', 'RHOB')
        df = plotting_service.normalize_xover(df, 'RT', 'RHOB')
        
        fig = plotting_service.plot_log_default(df=df, df_marker=df_marker, df_well_marker=df)
        
        return jsonify(fig.to_dict())

    except Exception as e:
        current_app.logger.error(f"Error di get-plot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route('/get-normalization-plot', methods=['POST'])
def get_normalization_plot():
    """
    FIX: Menerima NAMA sumur, lalu MEMBACA file dari direktori lokal.
    """
    try:
        payload = request.get_json()
        selected_wells = payload.get('selected_wells')
        if not selected_wells: return jsonify({"error": "Tidak ada sumur yang dipilih"}), 400

        df_list = []
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if os.path.exists(file_path):
                df_list.append(pd.read_csv(file_path))

        if not df_list: return jsonify({"error": "Data untuk sumur yang dipilih tidak ditemukan."}), 404
        
        df = pd.concat(df_list, ignore_index=True)
        
        log_out_col = 'GR_NORM' # Contoh
        if log_out_col not in df.columns or df[log_out_col].isnull().all():
            return jsonify({"error": "Tidak ada data normalisasi yang valid."}), 400
            
        df_marker_info = plotting_service.extract_markers_with_mean_depth(df)
        fig_result = plotting_service.plot_normalization(df=df, df_marker=df_marker_info, df_well_marker=df)
        
        return jsonify(fig_result.to_dict())
    except Exception as e:
        current_app.logger.error(f"Error di get-normalization-plot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# --- Rute untuk Normalisasi Interval (DIKEMBALIKAN KE VERSI LOKAL) ---
@api_bp.route('/run-interval-normalization', methods=['POST'])
def run_interval_normalization():
    """
    FIX: Menerima NAMA file, MEMBACA dan MENULIS kembali ke sistem file lokal.
    PERINGATAN: Ini tidak akan berfungsi di Vercel.
    """
    try:
        payload = request.get_json()
        params = payload.get('params', {})
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])

        if not selected_wells or not selected_intervals:
            return jsonify({"error": "Konten file dan Interval harus dipilih."}), 400

        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                current_app.logger.warning(f"Melewatkan {well_name}, file tidak ditemukan.")
                continue

            df_well = pd.read_csv(file_path)
            processed_df = data_processing_service.run_interval_normalization(df_well, params, selected_intervals)
            # Menulis kembali ke file lokal
            processed_df.to_csv(file_path, index=False)
            current_app.logger.info(f"Normalisasi untuk {well_name} telah disimpan kembali ke {file_path}")

        return jsonify({"message": "Normalisasi interval selesai untuk sumur yang dipilih."})

    except Exception as e:
        current_app.logger.error(f"Error di run-interval-normalization: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500