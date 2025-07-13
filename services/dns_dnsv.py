# app.py

import os
import pandas as pd
from flask import Flask, request, jsonify
from services.dns_dnsv_plot import plot_dns_dnsv # Assuming your plot function is here

# --- Configuration ---
app = Flask(__name__)
# IMPORTANT: Make sure this path is correct for your project structure
WELLS_DIR = os.path.join('..', 'data', 'wells')

# --- Calculation Functions (Your existing code) ---
def dns(rhob_in, nphi_in):
    """Calculate DNS (Density-Neutron Separation)"""
    return ((2.71 - rhob_in) / 1.71) - nphi_in

def dnsv(rhob_in, nphi_in, rhob_sh, nphi_sh, vsh):
    """Calculate DNSV (Density-Neutron Separation corrected for shale Volume)"""
    rhob_corv = rhob_in + vsh * (2.65 - rhob_sh)
    nphi_corv = nphi_in + vsh * (0 - nphi_sh)
    return ((2.71 - rhob_corv) / 1.71) - nphi_corv

def process_dns_dnsv(df, params=None):
    """Main function to process DNS-DNSV analysis"""
    if params is None:
        params = {}

    try:
        # Safely get parameters with defaults
        rhob_sh = float(params.get('RHOB_SH', 2.528)) 
        nphi_sh = float(params.get('NPHI_SH', 0.35))

        # First rename VSH_LINEAR to VSH if it exists
        if 'VSH_LINEAR' in df.columns and 'VSH' not in df.columns:
            df['VSH'] = df['VSH_LINEAR']

        # Ensure required columns are numeric, coercing errors
        for col in ['RHOB', 'NPHI', 'VSH']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential data is missing after coercion
        df.dropna(subset=['RHOB', 'NPHI', 'VSH'], inplace=True)

        # Calculate DNS and DNSV
        df['DNS'] = dns(df['RHOB'], df['NPHI'])
        df['DNSV'] = dnsv(df['RHOB'], df['NPHI'], rhob_sh, nphi_sh, df['VSH'])

        return df

    except Exception as e:
        print(f"Error in process_dns_dnsv: {str(e)}")
        raise e

# --- API Endpoints ---

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
            selected_wells = payload.get('selected_wells', [])

            if not selected_wells:
                return jsonify({"error": "No wells selected."}), 400

            print(f"Starting DNS-DNSV calculation for {len(selected_wells)} wells...")

            for well_name in selected_wells:
                file_path = os.path.join(WELLS_DIR, f"{well_name}.csv")

                if not os.path.exists(file_path):
                    print(f"Warning: Skipping well {well_name}, file not found.")
                    continue

                # --- FIX APPLIED HERE ---
                # Read the CSV file while handling potential malformed lines.
                # 'warn': Will print a warning for each bad line but continue processing.
                # 'skip': Will silently skip bad lines. 'warn' is better for debugging.
                try:
                    df_well = pd.read_csv(file_path, on_bad_lines='warn')
                except Exception as e:
                    print(f"Could not process file for well {well_name}. Error: {e}")
                    continue
                # --- END OF FIX ---

                columns_to_drop = ['DNS', 'DNSV']
                df_well.drop(columns=[col for col in columns_to_drop if col in df_well.columns], inplace=True)

                df_processed = process_dns_dnsv(df_well, params)

                df_processed.to_csv(file_path, index=False)
                print(f"DNS-DNSV results for well '{well_name}' have been saved.")

            return jsonify({"message": f"DNS-DNSV calculation successful for {len(selected_wells)} wells."}), 200

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

@app.route('/api/get-dns-dnsv-plot', methods=['POST', 'OPTIONS'])
def get_dns_dnsv_plot():
    """
    Endpoint for generating and returning the DNS-DNSV plot for selected wells.
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        request_data = request.get_json()
        selected_wells = request_data.get('selected_wells', [])

        if not selected_wells:
            return jsonify({"error": "No wells selected."}), 400

        # Read and combine data from the selected wells
        df_list = [pd.read_csv(os.path.join(WELLS_DIR, f"{well}.csv")) for well in selected_wells]
        df_combined = pd.concat(df_list, ignore_index=True)

        # Generate the plot using your existing function
        fig = plot_dns_dnsv(df_combined)

        # Return the plot as a JSON object
        return jsonify(fig.to_json())

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Your plotting function (plot_dns_dnsv) and its helpers 
# from services.plotting_service would be defined here or imported.