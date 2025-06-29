# my backend api/app.py 

from flask import Flask, request, jsonify, current_app, Response
from flask_cors import CORS
import pandas as pd
import lasio
import zipfile
import io
import os
from werkzeug.utils import secure_filename
import numpy as np
import logging

# --- Basic App Setup ---
app = Flask(__name__)
CORS(app)

# Configure logging to DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
app.logger.setLevel(logging.DEBUG)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import base64

# Add these helper functions
def encode_base64(content: str) -> str:
    """Encode string to Base64"""
    return base64.b64encode(content.encode('utf-8')).decode('ascii')

def decode_base64(content: str) -> str:
    """Decode Base64 to string"""
    return base64.b64decode(content).decode('utf-8', errors='ignore')

# --- Global Error Handler ---
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unhandled exception:")
    return jsonify({"error": "An internal error occurred."}), 500


# --- Parsing Functions with Debug-Level Error Logging ---
def parse_las(file_stream):
    app.logger.debug("parse_las: Starting LAS parse")
    try:
        raw = file_stream.read().decode('utf-8', errors='ignore')
        las = lasio.read(raw)
        df = las.df().reset_index()
        df.columns = [str(c) for c in df.columns]
        df = df.where(pd.notnull(df), None)
        headers = df.columns.tolist()
        data = df.to_dict(orient='records')
        app.logger.debug(f"parse_las: Parsed {len(data)} rows with {len(headers)} columns")
        return {"headers": headers, "data": data}
    except Exception as e:
        app.logger.exception("parse_las: LAS parsing error")
        return {"error": f"LAS parsing error: {str(e)}", "headers": [], "data": []}


def parse_csv(file_stream):
    app.logger.debug("parse_csv: Starting CSV parse")
    try:
        df = pd.read_csv(file_stream)
        df = df.where(pd.notnull(df), None)
        headers = df.columns.tolist()
        data = df.to_dict(orient='records')
        app.logger.debug(f"parse_csv: Parsed {len(data)} rows with {len(headers)} columns")
        return {"headers": headers, "data": data}
    except Exception as e:
        app.logger.exception("parse_csv: CSV parsing error")
        return {"error": f"CSV parsing error: {str(e)}", "headers": [], "data": []}


def parse_xlsx(file_stream):
    app.logger.debug("parse_xlsx: Starting XLSX parse")
    try:
        df = pd.read_excel(file_stream)
        df = df.where(pd.notnull(df), None)
        headers = df.columns.tolist()
        data = df.to_dict(orient='records')
        app.logger.debug(f"parse_xlsx: Parsed {len(data)} rows with {len(headers)} columns")
        return {"headers": headers, "data": data}
    except Exception as e:
        app.logger.exception("parse_xlsx: XLSX parsing error")
        return {"error": f"XLSX parsing error: {str(e)}", "headers": [], "data": []}


# --- QC Service Functions with Debug Logging ---
def add_markers_to_df(df, well_name, all_markers_df, logger):
    df['Marker'] = None
    well_name_cleaned = well_name.strip()
    logger.debug(f"[Markers] Starting marker search for Well: '{well_name_cleaned}'")
    if all_markers_df.empty:
        logger.warning("[Markers] Marker data is empty.")
        return False
    try:
        well_markers = all_markers_df[
            all_markers_df['Well identifier_cleaned'] == well_name_cleaned.upper()
        ].copy()
        if well_markers.empty:
            logger.warning(f"[Markers] No markers found for well '{well_name_cleaned}'.")
            return False

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

        logger.debug("[Markers] Marker tagging complete.")
        return True
    except Exception as e:
        logger.exception("[Markers] An error occurred during marker assignment")
        return False


def check_extreme_values(df, column):
    try:
        if pd.api.types.is_numeric_dtype(df[column]) and not df[column].isna().all():
            mean, std = df[column].mean(), df[column].std()
            if std == 0:
                return False
            mask = (df[column] > mean + 3 * std) | (df[column] < mean - 3 * std)
            return mask.any()
    except Exception as e:
        app.logger.exception(f"check_extreme_values: Error checking extremes in column '{column}'")
    return False


def run_quality_control(files_data: list, logger):
    logger.debug("run_quality_control: Starting QC pipeline")
    qc_results = []
    output_files = {}
    required_logs = ['GR', 'NPHI', 'RT', 'RHOB']
    skip_files_lower = {'abb-032.las', 'abb-033.las', 'abb-059.las'}
    all_markers_df = pd.DataFrame()

    # First, collect marker files
    for file_info in files_data:
        name_lower = file_info['name'].lower()
        if name_lower.endswith('.csv') and 'marker' in name_lower:
            logger.debug(f"run_quality_control: Processing marker file '{file_info['name']}'")
            try:
                marker_df = pd.read_csv(
                    io.StringIO(file_info['content']),
                    sep='[;,]',
                    engine='python',
                    on_bad_lines='skip'
                )
                if set(['Well identifier', 'MD', 'Surface']).issubset(marker_df.columns):
                    all_markers_df = pd.concat([all_markers_df, marker_df], ignore_index=True)
            except Exception as e:
                logger.warning(f"Could not read marker file '{file_info['name']}': {e}")

    # Clean marker data
    if not all_markers_df.empty:
        logger.debug("[Markers] Cleaning and preparing marker data...")
        try:
            all_markers_df['Well identifier_cleaned'] = (
                all_markers_df['Well identifier'].str.strip().str.upper()
            )
            all_markers_df['MD'] = pd.to_numeric(
                all_markers_df['MD'].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            )
            all_markers_df.dropna(subset=['MD', 'Well identifier_cleaned'], inplace=True)
            all_markers_df['Surface'] = all_markers_df['Surface'].astype(str)
            logger.debug(f"[Markers] Marker data clean. {len(all_markers_df)} rows loaded.")
        except Exception as e:
            logger.exception("[Markers] Error cleaning marker data")

    # Process each LAS
    las_files = [f for f in files_data if f['name'].lower().endswith('.las')]
    for file_info in las_files:
        filename = file_info['name']
        logger.info(f"--- [Processing] START: {filename} ---")
        if filename.lower() in skip_files_lower:
            logger.info(f"--- SKIPPING: {filename} ---")
            continue

        well_name = os.path.splitext(filename)[0]
        status = "PASS"
        details = {}

        try:
            las = lasio.read(io.StringIO(file_info['content']))
            df = las.df().reset_index()
            df.rename(columns=lambda c: c.upper(), inplace=True)

            # Standardize columns
            mapping = {
                'DEPT': 'DEPTH', 'ILD': 'RT', 'LLD': 'RT', 'RESD': 'RT',
                'RHOZ': 'RHOB', 'DENS': 'RHOB', 'TNPH': 'NPHI', 'GR_CAL': 'GR'
            }
            df.rename(columns=mapping, inplace=True)

            if 'DEPTH' not in df.columns:
                raise ValueError("DEPTH column not found.")
            df['DEPTH'] = pd.to_numeric(df['DEPTH'], errors='coerce')
            df.dropna(subset=['DEPTH'], inplace=True)

            # Missing logs?
            missing = [log for log in required_logs if log not in df.columns]
            if missing:
                status = "MISSING_LOGS"
                details['missing_columns'] = missing
                raise ValueError(f"Missing logs: {missing}")

            # Replace placeholders
            for col in required_logs:
                df[col] = df[col].replace([-999.0, -999.25], np.nan)

            # Add markers
            has_markers = add_markers_to_df(df, well_name, all_markers_df, logger)
            zone_df = df.dropna(subset=['Marker']) if has_markers else df

            # Null check
            nulls = [log for log in required_logs if zone_df[log].isna().any()]
            if nulls:
                status = "HAS_NULL"
                details['null_columns'] = nulls
                raise ValueError(f"Null values in: {nulls}")

            # Extreme values check
            extremes = [log for log in required_logs if check_extreme_values(zone_df, log)]
            if extremes:
                status = "EXTREME_VALUES"
                details['extreme_columns'] = extremes
                raise ValueError(f"Extreme values in: {extremes}")

            details['message'] = "All checks passed"
            logger.debug(f"{well_name}: QC passed")

        except Exception as e:
            logger.warning(f"{well_name}: QC status {status} — {e}")
            # even on exception, we save the result and the file
        finally:
            raw_details = details.get(
                'message',
                details.get('missing_columns') or
                details.get('null_columns') or
                details.get('extreme_columns') or
                str(e)
            )

            final_details_string = ""
            if isinstance(raw_details, list):
                final_details_string = ", ".join(raw_details)
            else:
                final_details_string = str(raw_details)

            qc_results.append({
                'well_name': well_name,
                'status': status,
                'details': final_details_string # Use the guaranteed string
            })
            # The df.to_csv part can remain the same
            if 'df' in locals():
                output_files[f"{well_name}_{status}.csv"] = df.to_csv(index=False)

    logger.debug("run_quality_control: QC pipeline complete")
    return {'qc_summary': qc_results, 'output_files': output_files}


def handle_null_values(csv_content: str) -> str:
    current_app.logger.debug("handle_null_values: Starting null filling")
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        numeric_cols = df.select_dtypes(include='number').columns
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].interpolate(
                method='linear', limit_direction='both', axis=0
            )
        df.fillna('NA', inplace=True)
        current_app.logger.debug("handle_null_values: Nulls filled and NAs set")
        return df.to_csv(index=False)
    except Exception as e:
        current_app.logger.exception("handle_null_values: Error processing CSV content")
        raise


def extract_markers_with_mean_depth(df):
    """
    Membuat dataframe baru yang berisi nilai unik dari marker (sebagai 'surface')
    dan rata-rata depth untuk setiap marker.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi kolom 'MARKER' dan 'DEPTH'

    Returns:
    --------
    pandas.DataFrame
        DataFrame baru dengan kolom 'surface' (nama marker) dan 'mean_depth'
    """
    # Pastikan df memiliki kolom 'MARKER' dan 'DEPTH'
    required_cols = ['MARKER', 'DEPTH']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame tidak memiliki kolom '{col}'")

    # Mengelompokkan berdasarkan MARKER dan menghitung rata-rata DEPTH
    markers_mean_depth = df.groupby('MARKER')['DEPTH'].mean().reset_index()

    # Mengganti nama kolom
    markers_mean_depth.columns = ['Surface', 'Mean Depth']

    return markers_mean_depth



def normalize_xover(df_well, log_1, log_2):

    # Salin DataFrame untuk menghindari modifikasi pada original
    df = df_well.copy()
    log_merge = log_1 + '_' + log_2
    log_1_norm = log_1 + '_NORM'
    log_2_norm = log_2 + '_NORM_' + log_1

    # Range untuk visualisasi
    log_1_range = range_col[log_merge][0]
    log_2_range = range_col[log_merge][1]

    # 1. Normalisasi NPHI agar sesuai dengan rentang visualisasi
    # Nilai NPHI biasanya dalam desimal (misalnya 0.3 untuk 30% porositas)
    # NPHI_NORM tetap dalam skala aslinya
    df[log_1_norm] = df[log_1]

    # 2. Konversi RHOB ke skala NPHI untuk visualisasi crossover
    # Ini membuat RHOB sesuai dengan skala NPHI agar crossover terlihat
    min_log_2 = log_2_range[0]  # 1.71
    max_log_2 = log_2_range[1]  # 2.71
    min_log_1 = log_1_range[0]  # 0.6
    max_log_1 = log_1_range[1]  # 0

    # Normalisasi RHOB dengan rumus interpolasi linier untuk pemetaan rentang
    df[log_2_norm] = min_log_1 + \
        (df[log_2] - min_log_2) * (max_log_1 -
                                   min_log_1) / (max_log_2 - min_log_2)

    return df



def plot_log_default(df, df_marker, df_well_marker):
    sequence = ['MARKER', 'GR', 'RT_RHOB', 'NPHI_RHOB']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}
    print(plot_sequence)

    ratio_plots_seq = []
    for key in plot_sequence.values():
        ratio_plots_seq.append(ratio_plots[key])

    subplot_col = len(plot_sequence.keys())

    fig = make_subplots(
        rows=1, cols=subplot_col,
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.0
    )

    counter = 0
    axes = {}
    for i in plot_sequence.values():
        axes[i] = []

    for n_seq, col in plot_sequence.items():
        if col == 'GR':
            fig, axes = plot_line(
                df, fig, axes, base_key='GR', n_seq=n_seq, col=col, label=col)
        elif col == 'RT':
            fig, axes = plot_line(
                df, fig, axes, base_key='RT', n_seq=n_seq, col=col, label=col)
        elif col == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                       y_color='rgba(0,0,0,0)', n_color='yellow', type=2, exclude_crossover=False)
        elif col == 'RT_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                       y_color='limegreen', n_color='lightgray', type=1, exclude_crossover=False)
        elif col in ['X_RT_RO', 'X_RWA_RW', 'X_RT_F', 'X_RT_RHOB']:
            fig, axes, counter = plot_xover_thres(
                df, fig, axes, col, n_seq, counter=counter)
        elif col == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, col, n_seq)
            fig, axes = plot_texts_marker(
                df_marker, df_well_marker['DEPTH'].max(), fig, axes, col, n_seq)

    fig = layout_range_all_axis(fig, axes, plot_sequence)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), height=1500,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified', hoverdistance=-1,
        title_text="Well Log Selected",
        title_x=0.5,
        modebar_remove=['lasso', 'autoscale', 'zoom',
                        'zoomin', 'zoomout', 'pan', 'select']
    )

    fig.update_yaxes(showspikes=True,  # tickangle=90,
                     range=[df[depth].max(), df[depth].min()])
    fig.update_traces(yaxis='y')

    fig = layout_draw_lines(fig, ratio_plots_seq, df, xgrid_intv=0)

    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)
    return fig


# --- API Routes with Error Logging ---
@app.route('/api/upload', methods=['POST'])
def upload_file():
    app.logger.info("Received /api/upload request")
    try:
        if 'files' not in request.files:
            app.logger.debug("upload_file: No 'files' part in request")
            return jsonify({"error": "No file part"}), 400

        processed = []
        for file in request.files.getlist('files'):
            if not file.filename:
                app.logger.debug("upload_file: Skipping empty filename")
                continue

            filename = secure_filename(file.filename)
            content_bytes = file.read()
            file.seek(0)
            info = {
                "id": f"{os.path.splitext(filename)[0]}_{pd.Timestamp.now().timestamp()}",
                "name": filename,
                "size": len(content_bytes),
                "originalFileType": file.content_type,
                "lastModified": pd.Timestamp.now().timestamp() * 1000,
                "isStructureFromZip": False,
                "content": [], "headers": [], "rawContentString": content_bytes.decode('utf-8', errors='ignore')
            }

            # ZIP handling
            if filename.lower().endswith('.zip'):
                app.logger.debug(f"upload_file: Handling ZIP '{filename}'")
                info["isStructureFromZip"] = True
                info["lasFiles"] = []
                info["csvFiles"] = []
                try:
                    with zipfile.ZipFile(io.BytesIO(content_bytes), 'r') as z:
                        for member in z.namelist():
                            if member.endswith('/'):
                                continue
                            data = z.read(member)
                            stream = io.BytesIO(data)
                            parsed = None

                            if member.lower().endswith('.las'):
                                parsed = parse_las(stream)
                                type_ = 'las'
                            elif member.lower().endswith('.csv'):
                                parsed = parse_csv(io.TextIOWrapper(stream, encoding='utf-8', errors='ignore'))
                                type_ = 'csv'

                            if parsed and not parsed.get("error"):
                                sub = {
                                    "id": f"{os.path.splitext(member)[0]}_{pd.Timestamp.now().timestamp()}",
                                    "name": os.path.basename(member),
                                    "type": type_,
                                    "content": parsed["data"],
                                    "headers": parsed["headers"],
                                    "rawContentString": data.decode('utf-8', errors='ignore')
                                }
                                (info["lasFiles"] if type_ == 'las' else info["csvFiles"]).append(sub)
                except Exception as e:
                    app.logger.exception(f"upload_file: Error unpacking ZIP '{filename}'")
                    info["error"] = str(e)

            # Non-ZIP files
            else:
                try:
                    stream = io.BytesIO(content_bytes)
                    if filename.lower().endswith('.csv'):
                        parsed = parse_csv(io.TextIOWrapper(stream, encoding='utf-8', errors='ignore'))
                    elif filename.lower().endswith('.las'):
                        parsed = parse_las(stream)
                    elif filename.lower().endswith(('.xls', '.xlsx')):
                        parsed = parse_xlsx(stream)
                    else:
                        parsed = {"error": "Unsupported file type"}

                    if parsed.get("error"):
                        info["error"] = parsed["error"]
                    else:
                        info["content"] = parsed["data"]
                        info["headers"] = parsed["headers"]
                except Exception as e:
                    app.logger.exception(f"upload_file: Error parsing '{filename}'")
                    info["error"] = str(e)

            processed.append(info)

        return jsonify(processed)
    except Exception as e:
        app.logger.exception("upload_file: Unexpected error")
        return jsonify({"error": "An internal error occurred."}), 500


@app.route('/api/qc/run', methods=['POST'])
def run_qc_endpoint():
    app.logger.info("Received /api/run-qc request")
    try:
        payload = request.get_json(force=True)
        files = payload.get('files')
        if not isinstance(files, list):
            app.logger.debug("run_qc_endpoint: 'files' missing or not a list")
            return jsonify({"error": "Invalid input: 'files' is required and must be a list."}), 400

        results = run_quality_control(files, app.logger)
        return jsonify(results)
    except Exception as e:
        app.logger.exception("run_qc_endpoint: Error running QC")
        return jsonify({"error": "An internal error occurred."}), 500


@app.route('/api/qc/handle-nulls', methods=['POST'])
def handle_nulls_endpoint():
    app.logger.info("Received /api/qc/handle-nulls request")
    try:
        text = request.get_data(as_text=True)
        if not text:
            app.logger.debug("handle_nulls_endpoint: Empty request body")
            return jsonify({"error": "Request body cannot be empty."}), 400

        cleaned = handle_null_values(text)
        return Response(
            cleaned,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=cleaned_data.csv"}
        )
    except Exception as e:
        app.logger.exception("handle_nulls_endpoint: Failed to process CSV data")
        return jsonify({"error": "Failed to process CSV data."}), 500
    

@app.route('/api/get-plot', methods=['POST'])
def get_plot():
    try:
        request_data = request.get_json()
        well_data_map = request_data.get('well_data')

        if not well_data_map:
            return jsonify({"error": "No well data provided"}), 400

        print(f"Received request to plot data for wells: {list(well_data_map.keys())}")

        list_of_dataframes = []
        for well_name, csv_content in well_data_map.items():
            df = pd.read_csv(io.StringIO(csv_content))
            if 'WELL_NAME' not in df.columns:
                df['WELL_NAME'] = well_name
            list_of_dataframes.append(df)
            
        if not list_of_dataframes:
            return jsonify({"error": "No valid data could be processed"}), 404

        # The rest of your logic is the same!
        df = pd.concat(list_of_dataframes, ignore_index=True)
        df_marker = extract_markers_with_mean_depth(df)
        df = normalize_xover(df, 'NPHI', 'RHOB')
        df = normalize_xover(df, 'RT', 'RHOB')
        fig = plot_log_default(df=df, df_marker=df_marker, df_well_marker=df)
        
        return jsonify(fig.to_json())

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "Flask backend is running!"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
