# FILE 3: api/app/routes/qc_routes.py
# File ini khusus menangani rute-rute yang berhubungan dengan Quality Control.
from flask import Blueprint, request, jsonify, current_app, Response
from ..services import qc_service  # Impor service yang relevan

# Buat Blueprint. 'qc' adalah nama blueprint.
qc_bp = Blueprint('qc', __name__)


@qc_bp.route('/run', methods=['POST'])
def run_qc_endpoint():
    """Endpoint untuk menjalankan pipeline QC lengkap."""
    current_app.logger.info("Menerima permintaan untuk /api/qc/run")
    try:
        data = request.get_json()
        files_to_process = data.get('files')

        if not files_to_process or not isinstance(files_to_process, list):
            return jsonify({"error": "Input tidak valid: 'files' dibutuhkan."}), 400

        # Panggil fungsi dari service untuk melakukan pekerjaan berat
        # FIX: Panggil fungsi yang benar, `run_quality_control`, dan teruskan logger.
        results = qc_service.run_full_qc_pipeline(
            files_to_process, current_app.logger)
        return jsonify(results)

    except Exception as e:
        current_app.logger.error(f"Error di /api/qc/run: {e}", exc_info=True)
        return jsonify({"error": "Terjadi error internal."}), 500


@qc_bp.route('/handle-nulls', methods=['POST'])
def handle_nulls_endpoint():
    """Endpoint khusus untuk menangani nilai null pada konten CSV."""
    current_app.logger.info("Menerima permintaan untuk /api/qc/handle-nulls")
    try:
        csv_content = request.get_data(as_text=True)
        if not csv_content:
            return jsonify({"error": "Request body tidak boleh kosong."}), 400

        cleaned_csv = qc_service.handle_null_values(csv_content)
        return Response(
            cleaned_csv,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=cleaned_data.csv"}
        )
    except Exception as e:
        current_app.logger.error(
            f"Error di /api/qc/handle-nulls: {e}", exc_info=True)
        return jsonify({"error": "Gagal memproses data CSV."}), 500
