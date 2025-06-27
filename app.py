from flask import Flask, request, jsonify, Response
from app.services.qc_service import run_full_qc_pipeline
from app.services.data_processing import handle_null_values, min_max_normalize
from app.services.vsh_calculation import calculate_vsh_from_gr
import logging
import os
from flask_cors import CORS
import pandas as pd
import numpy as np
from app.services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_log_default,
    plot_normalization,
    plot_phie_den
)
from app.services.porosity import calculate_porosity
from app.services.depth_matching import depth_matching, plot_depth_matching_results

app = Flask(__name__)

# Allow CORS for all domains (important for deployment too)
CORS(app)

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Define reliable paths to data files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WELLS_DIR = os.path.join(PROJECT_ROOT, 'sample_data', 'wells')
LAS_DIR = os.path.join(PROJECT_ROOT, 'sample_data', 'depth-matching')

@app.route('/')
def home():
    return "Flask backend is running!"

@app.route('/api/run-qc', methods=['POST'])
def qc_route():
    app.logger.info("Received request for /api/run-qc")
    try:
        data = request.get_json()
        files_data = data.get('files')

        if not files_data or not isinstance(files_data, list):
            return jsonify({"error": "Invalid input: 'files' key with a list of file objects is required."}), 400

        results = run_full_qc_pipeline(files_data, app.logger)
        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Error in /api/run-qc: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/api/handle-nulls', methods=['POST'])
def handle_nulls_route():
    app.logger.info("Received request for /api/handle-nulls")
    try:
        csv_content = request.get_data(as_text=True)
        if not csv_content:
            return jsonify({"error": "Request body cannot be empty."}), 400

        cleaned_csv = handle_null_values(csv_content)
        return Response(
            cleaned_csv,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=cleaned_data.csv"}
        )
    except Exception as e:
        app.logger.error(f"Error in /api/handle-nulls: {e}", exc_info=True)
        return jsonify({"error": "Failed to process CSV data."}), 500

@app.route('/api/get-plot', methods=['POST'])
def get_plot():
    try:
        request_data = request.get_json()
        selected_wells = request_data.get('selected_wells')

        if not selected_wells or len(selected_wells) == 0:
            return jsonify({"error": "No wells selected"}), 400

        app.logger.info(f"Processing wells: {selected_wells}")
        list_of_dataframes = []
        
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if os.path.exists(file_path):
                list_of_dataframes.append(pd.read_csv(file_path))

        if not list_of_dataframes:
            return jsonify({"error": "No valid data for selected wells"}), 404

        df = pd.concat(list_of_dataframes, ignore_index=True)
        df_marker = extract_markers_with_mean_depth(df)
        df = normalize_xover(df, 'NPHI', 'RHOB')
        df = normalize_xover(df, 'RT', 'RHOB')

        fig = plot_log_default(df=df, df_marker=df_marker, df_well_marker=df)
        return jsonify(fig.to_json())

    except Exception as e:
        app.logger.error(f"Error in /api/get-plot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-normalization-plot', methods=['POST'])
def get_normalization_plot():
    try:
        request_data = request.get_json()
        selected_wells = request_data.get('selected_wells', [])

        if not selected_wells:
            return jsonify({"error": "No wells selected"}), 400

        df_list = []
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if os.path.exists(file_path):
                df_list.append(pd.read_csv(file_path))

        if not df_list:
            return jsonify({"error": "No data found for selected wells"}), 404

        df = pd.concat(df_list, ignore_index=True)
        log_out_col = 'GR_NORM'

        if log_out_col not in df.columns or df[log_out_col].isnull().all():
            return jsonify({"error": "No valid normalization data found"}), 400

        df_marker_info = extract_markers_with_mean_depth(df)
        fig_result = plot_normalization(
            df=df,
            df_marker=df_marker_info,
            df_well_marker=df
        )
        return jsonify(fig_result.to_json())

    except Exception as e:
        app.logger.error(f"Error in /api/get-normalization-plot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/list-wells', methods=['GET'])
def list_wells():
    try:
        if not os.path.exists(WELLS_DIR):
            return jsonify({"error": "Wells directory not found"}), 404

        well_files = [f.replace('.csv', '') for f in os.listdir(WELLS_DIR) 
                     if f.endswith('.csv')]
        well_files.sort()
        return jsonify(well_files)
    except Exception as e:
        app.logger.error(f"Error in /api/list-wells: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-interval-normalization', methods=['POST'])
def run_interval_normalization():
    try:
        payload = request.get_json()
        params = payload.get('params', {})
        selected_wells = payload.get('selected_wells', [])
        selected_intervals = payload.get('selected_intervals', [])

        if not selected_wells or not selected_intervals:
            return jsonify({"error": "Wells and intervals must be selected"}), 400

        app.logger.info(f"Normalizing {len(selected_wells)} wells across {len(selected_intervals)} intervals")
        log_in_col = params.get('LOG_IN', 'GR')
        log_out_col = params.get('LOG_OUT', 'GR_NORM')
        calib_min = float(params.get('CALIB_MIN', 40))
        calib_max = float(params.get('CALIB_MAX', 140))
        pct_min = int(params.get('PCT_MIN', 3))
        pct_max = int(params.get('PCT_MAX', 97))
        cutoff_min = float(params.get('CUTOFF_MIN', 0.0))
        cutoff_max = float(params.get('CUTOFF_MAX', 250.0))

        processed_dfs = []
        
        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                app.logger.warning(f"Skipping missing well: {well_name}")
                continue

            df_well = pd.read_csv(file_path)
            df_well[log_out_col] = np.nan

            for interval in selected_intervals:
                interval_mask = df_well['MARKER'] == interval
                if not interval_mask.any():
                    continue
                    
                log_to_normalize = df_well.loc[interval_mask, log_in_col].dropna().values
                if len(log_to_normalize) == 0:
                    continue

                normalized_values = min_max_normalize(
                    log_in=log_to_normalize,
                    calib_min=calib_min,
                    calib_max=calib_max,
                    pct_min=pct_min,
                    pct_max=pct_max,
                    cutoff_min=cutoff_min,
                    cutoff_max=cutoff_max
                )
                df_well.loc[interval_mask, log_out_col] = normalized_values

            processed_dfs.append(df_well)
            df_well.to_csv(file_path, index=False)
            app.logger.info(f"Saved normalized data for {well_name}")

        if not processed_dfs:
            return jsonify({"error": "No data processed"}), 400

        final_df = pd.concat(processed_dfs, ignore_index=True)
        return jsonify({
            "message": f"Normalization complete for {len(processed_dfs)} wells",
            "data": final_df.to_json(orient='records')
        })

    except Exception as e:
        app.logger.error(f"Error in /api/run-interval-normalization: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-depth-matching', methods=['POST'])
def run_depth_matching_endpoint():
    try:
        ref_las_path = os.path.join(LAS_DIR, 'ref.las')
        lwd_las_path = os.path.join(LAS_DIR, 'lwd.las')

        if not os.path.exists(ref_las_path):
            return jsonify({"error": f"File not found: {ref_las_path}"}), 404
        if not os.path.exists(lwd_las_path):
            return jsonify({"error": f"File not found: {lwd_las_path}"}), 404

        ref_data, lwd_data, aligned_data = depth_matching(
            ref_las_path=ref_las_path,
            lwd_las_path=lwd_las_path,
            num_chunks=8
        )

        if aligned_data is None:
            raise ValueError("Depth matching computation failed")

        fig_result = plot_depth_matching_results(
            ref_df=ref_data,
            lwd_df=lwd_data,
            final_df=aligned_data
        )
        return jsonify(fig_result.to_json())

    except Exception as e:
        app.logger.error(f"Error in /api/run-depth-matching: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-vsh-calculation', methods=['POST'])
def run_vsh_calculation():
    try:
        payload = request.get_json()
        params = payload.get('params', {})
        selected_wells = payload.get('selected_wells', [])

        if not selected_wells:
            return jsonify({"error": "No wells selected"}), 400

        app.logger.info(f"Calculating VSH for {len(selected_wells)} wells")
        gr_ma = float(params.get('gr_ma', 30))
        gr_sh = float(params.get('gr_sh', 120))
        input_log = params.get('input_log', 'GR')
        output_log = params.get('output_log', 'VSH_GR')

        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                app.logger.warning(f"Skipping missing well: {well_name}")
                continue

            df_well = pd.read_csv(file_path)
            df_updated = calculate_vsh_from_gr(
                df_well, input_log, gr_ma, gr_sh, output_log)
            df_updated.to_csv(file_path, index=False)
            app.logger.info(f"Saved VSH data for {well_name}")

        return jsonify({"message": f"VSH calculation completed for {len(selected_wells)} wells"})

    except Exception as e:
        app.logger.error(f"Error in /api/run-vsh-calculation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-porosity-calculation', methods=['POST'])
def run_porosity_calculation():
    try:
        payload = request.get_json()
        params = payload.get('params', {})
        selected_wells = payload.get('selected_wells', [])

        if not selected_wells:
            return jsonify({"error": "No wells selected"}), 400

        for well_name in selected_wells:
            file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")
            if not os.path.exists(file_path):
                continue

            df_well = pd.read_csv(file_path)
            df_updated = calculate_porosity(df_well, params)
            df_updated.to_csv(file_path, index=False)
            app.logger.info(f"Saved porosity data for {well_name}")

        return jsonify({"message": f"Porosity calculation completed for {len(selected_wells)} wells"})

    except Exception as e:
        app.logger.error(f"Error in /api/run-porosity-calculation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-porosity-plot', methods=['POST'])
def get_porosity_plot():
    try:
        request_data = request.get_json()
        selected_wells = request_data.get('selected_wells', [])

        if not selected_wells:
            return jsonify({"error": "No wells selected"}), 400

        df_list = [pd.read_csv(os.path.join(WELLS_DIR, f"{well}.csv")) 
                  for well in selected_wells if os.path.exists(os.path.join(WELLS_DIR, f"{well}.csv"))]
        
        if not df_list:
            return jsonify({"error": "No valid well data found"}), 404
            
        df = pd.concat(df_list, ignore_index=True)
        required_cols = ['VSH', 'PHIE', 'PHIT', 'PHIE_DEN', 'PHIT_DEN', 'RESERVOIR_CLASS']
        
        if not all(col in df.columns for col in required_cols):
            return jsonify({"error": "Required columns missing. Run VSH and porosity calculations first"}), 400

        df_marker_info = extract_markers_with_mean_depth(df)
        fig_result = plot_phie_den(
            df=df,
            df_marker=df_marker_info,
            df_well_marker=df
        )
        return jsonify(fig_result.to_json())

    except Exception as e:
        app.logger.error(f"Error in /api/get-porosity-plot: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# This will run when executed directly (for local development)
if __name__ == '__main__':
    app.run()