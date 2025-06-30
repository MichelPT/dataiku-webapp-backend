# my backend api/app.py 

from sklearn.linear_model import LinearRegression
from typing import Optional
from flask import Flask, request, jsonify, current_app, Response
from flask_cors import CORS
import pandas as pd
import lasio
import zipfile
import random
import io
import os
from werkzeug.utils import secure_filename
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from dtaidistance import dtw

# --- Basic App Setup ---
app = Flask(__name__)
CORS(app)

PERSISTENT_DATA_DIR = os.getenv('DATA_DIR', './data')
WELLS_DIR = os.path.join(PERSISTENT_DATA_DIR, 'wells')
LAS_DIR = os.path.join(PERSISTENT_DATA_DIR, 'las')

os.makedirs(WELLS_DIR, exist_ok=True)
os.makedirs(LAS_DIR, exist_ok=True)


# Configure logging to DEBUG
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
app.logger.setLevel(logging.DEBUG)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import base64

colors = px.colors.qualitative.G10
colors_dict = {
    'blue': 'royalblue',
    'red': 'tomato',
    'orange': colors[2],
    'green': colors[3],
    'purple': colors[4],
    'cyan': colors[5],
    'magenta': colors[6],
    'sage': colors[7],
    'maroon': colors[8],
    'navy': colors[9],
    'gray': 'gray',
    'lightgray': 'lightgray',
    'black': 'rgba(62, 62, 62,1)'
}

legends = ["legend"]
for i in range(2, 17):
    legends.append("legend"+str(i))

axes = ['xaxis', 'yaxis']
for i in range(16):
    axes.append('xaxis'+str(i+1))
    axes.append('yaxis'+str(i+1))

# inisiasi kolom data
depth = "DEPTH"


data_col = {
    'DNS': ['DNS'],
    'MARKER': ['MARKER'],
    'GR': ['GR'],
    'GR_NORM': ['GR_NORM'],
    'GR_DUAL': ['GR', 'GR_NORM'],
    'RT': ['RT'],
    'RT_RO': ['RT', 'RO'],
    'X_RT_RO': ['RT_RO'],
    'NPHI_RHOB_NON_NORM': ['NPHI', 'RHOB'],
    'RHOB': ['RHOB'],
    'NPHI_RHOB': ['NPHI', 'RHOB', 'NPHI_NORM', 'RHOB_NORM_NPHI'],
    'SW': ['SW'],
    'PHIE_PHIT': ['PHIE', 'PHIT'],
    'PERM': ['PERM'],
    'VCL': ['VCL'],
    'RWAPP_RW': ['RWAPP', 'RW'],
    'X_RWA_RW': ['RWA_RW'],
    'RT_F': ['RT', 'F'],
    'X_RT_F': ['RT_F'],
    'RT_RHOB': ['RT', 'RHOB', 'RT_NORM', 'RHOB_NORM_RT'],
    'X_RT_RHOB': ['RT_RHOB'],
    'TEST': ['TEST'],
    'XPT': ['XPT'],
    'RT_RGSA': ['RT', 'RGSA'],
    'NPHI_NGSA': ['NPHI', 'NGSA'],
    'RHOB_DGSA': ['RHOB', 'DGSA'],
    'ZONA': ['ZONA'],
    'VSH': ['VSH'],
    'SP': ['SP'],
    'VSH_LINEAR': ['VSH_LINEAR'],
    'VSH_DN': ['VSH_DN'],
    'VSH_SP': ['VSH_SP'],
    'PHIE_DEN': ['PHIE', 'PHIE_DEN'],
    'PHIT_DEN': ['PHIT', 'PHIT_DEN'],
    'RESERVOIR_CLASS': ['RESERVOIR_CLASS'],
    'RWA': ['RWA_FULL', 'RWA_SIMPLE', 'RWA_TAR'],
    'PHIE': ['PHIE'],
    'RT_GR': ['RT', 'GR', 'RT_NORM', 'GR_NORM_RT'],
    # 'RT_PHIE':['RT','PHIE','RT_NORM', 'PHIE_NORM_RT'],
    'RT_PHIE': ['RT', 'PHIE'],
    'RGBE': ['RGBE'],
    'RPBE': ['RPBE'],
    'RGBE_TEXT': ['RGBE'],
    'RPBE_TEXT': ['RPBE'],
    'IQUAL': ['IQUAL'],
    'SWARRAY': ['SWARRAY_10', 'SWARRAY_15', 'SWARRAY_20', 'SWARRAY_25'],
    'SWGRAD': ['SWGRAD'],
    'DNS': ['DNS'],
    'DNSV': ['DNSV'],
}


unit_col = {
    'DNS': [''],
    'MARKER': [''],
    'GR_NORM': ['GAPI'],
    'GR': ['GAPI'],
    'GR_DUAL': ['GAPI', 'GAPI'],
    'RT': ['OHMM'],
    'RT_RO': ['OHMM', 'OHMM'],
    'X_RT_RO': ['V/V'],
    'NPHI_RHOB_NON_NORM': ['V/V', 'G/C3'],
    'NPHI_RHOB': ['V/V', 'G/C3', 'V/V', 'G/C3'],
    'RHOB': ['G/C3'],
    'SW': ['DEC'],
    'PHIE_PHIT': ['V/V', 'V/V'],
    'PERM': ['mD'],
    'VCL': ['V/V'],
    'RWAPP_RW': ['OHMM', 'OHMM'],
    'X_RWA_RW': ['V/V'],
    'RT_F': ['OHMM', 'V/V'],
    'X_RT_F': ['V/V'],
    'RT_RHOB': ['OHMM', 'G/C3', 'OHMM', 'G/C3'],
    'X_RT_RHOB': ['V/V'],
    'TEST': ['V/V'],
    'CLASS': ['V/V'],
    'CTC': ['V/V'],
    'XPT': [''],
    'RT_RGSA': ['OHMM', ''],
    'NPHI_NGSA': ['V/V', ''],
    'RHOB_DGSA': ['G/C3', ''],
    'ZONA': [''],
    'VSH': ['V/V'],
    'SP': ['MV'],
    'VSH_LINEAR': ['V/V'],
    'VSH_DN': ['V/V'],
    'VSH_SP': ['V/V'],
    'PHIE_DEN': ['', ''],
    'PHIT_DEN': ['', ''],
    'RESERVOIR_CLASS': [''],
    'RWA': ['OHMM', 'OHMM', 'OHMM'],
    'PHIE': [''],
    'RT_GR': ['OHMM', 'GAPI', 'OHMM', 'GAPI'],
    # 'RT_PHIE':['OHMM','','OHMM',''],
    'RT_PHIE': ['OHMM', ''],
    'RGBE': [''],
    'RPBE': [''],
    'RGBE_TEXT': [''],
    'RPBE_TEXT': [''],
    'IQUAL': [''],
    'SWARRAY': ['V/V', 'V/V', 'V/V', 'V/V'],
    'SWGRAD': ['V/V'],
    'DNSV': [''],
    'DNS': [''],
}


color_col = {
    'DNS': ['darkgreen'],
    'MARKER': [colors_dict['black']],
    'GR_NORM': ['orange'],
    'GR_DUAL': ['darkgreen', 'orange'],
    'GR': ['darkgreen'],
    'RT': [colors_dict['red']],
    'RT_RO': [colors_dict['red'], colors_dict['purple']],
    'X_RT_RO': [colors_dict['black']],
    'NPHI_RHOB_NON_NORM': [colors_dict['blue'], colors_dict['red']],
    'NPHI_RHOB': [colors_dict['blue'], colors_dict['red'], colors_dict['blue'], colors_dict['red'],],
    'RHOB': [colors_dict['red']],
    'SW': [colors_dict['blue']],
    'PHIE_PHIT': ['darkblue', colors_dict['cyan']],
    'PERM': [colors_dict['blue']],
    'VCL': [colors_dict['black']],
    'RWAPP_RW': [colors_dict['black'], colors_dict['blue']],
    'X_RWA_RW': [colors_dict['black']],
    'RT_F': [colors_dict['red'], colors_dict['cyan']],
    'X_RT_F': [colors_dict['black']],
    'RT_RHOB': [colors_dict['red'], colors_dict['black'], colors_dict['red'], colors_dict['green']],
    'X_RT_RHOB': [colors_dict['black']],
    'TEST': [colors_dict['black']],
    'CLASS': [colors_dict['black']],
    'CTC': [colors_dict['black']],
    'XPT': [colors_dict['black']],
    'RT_RGSA': [colors_dict['red'], colors_dict['blue']],
    'NPHI_NGSA': [colors_dict['red'], colors_dict['green']],
    'RHOB_DGSA': [colors_dict['red'], colors_dict['green']],
    'ZONA': [colors_dict['black']],
    'VSH': ['darkblue'],
    'SP': ['darkblue'],
    'VSH_LINEAR': ['darkblue'],
    'VSH_DN': ['darkblue'],
    'VSH_SP': ['darkblue'],
    'PHIE_DEN': ['darkblue', colors_dict['blue']],
    'PHIT_DEN': [colors_dict['red'], colors_dict['orange']],
    'RESERVOIR_CLASS': [colors_dict['black']],
    'RWA': ['darkblue', 'darkgreen', colors_dict['red']],
    'PHIE': ['darkblue'],
    'RT_GR': [colors_dict['red'], 'darkgreen', colors_dict['red'], 'darkgreen'],
    # 'RT_PHIE':[colors_dict['red'],'darkblue',colors_dict['red'],'darkblue'],
    'RT_PHIE': [colors_dict['red'], 'darkblue'],
    'RGBE': [colors_dict['black']],
    'RPBE': [colors_dict['black']],
    'RGBE_TEXT': [colors_dict['black']],
    'RPBE_TEXT': [colors_dict['black']],
    'IQUAL': [colors_dict['black']],
    'SWARRAY': ['darkblue', 'orange', 'red', 'green'],
    'SWGRAD': ['darkgreen'],
    'DNS': [colors_dict['black']],
    'DNSV': [colors_dict['black']],
}

flag_color = {
    "TEST": {
        0: 'rgba(0,0,0,0)',
        1: colors_dict['cyan'],
        3: colors_dict['green']
    },
    "CLASS": {
        0: '#d9d9d9',
        1: '#00bfff',
        2: '#ffb6c1',
        3: '#a020f0',
        4: '#ffa600',
        5: '#8b1a1a',
        6: '#000000'
    },
    "ZONA": {
        3: colors_dict['red'],
        2: colors_dict['orange'],
        1: 'yellow',
        0: colors_dict['black'],
    },
    "RESERVOIR_CLASS": {
        4: 'green',
        3: 'yellow',
        2: 'orange',
        1: 'black',
        0: 'gray'
    },
    "IQUAL": {
        1: 'green',
    }

}

range_col = {
    'GR': [[0, 250]],
    'GR_NORM': [[0, 250]],
    'GR_DUAL': [[0, 250], [0, 250]],
    'RT': [[0.02, 2000]],
    'RT_RO': [[0.02, 2000], [0.02, 2000]],
    'X_RT_RO': [[0, 4]],
    'NPHI_RHOB_NON_NORM': [[0.6, 0], [1.71, 2.71]],
    'NPHI_RHOB': [[0.6, 0], [1.71, 2.71], [1, 0], [1, 0]],
    'RHOB': [[1.71, 2.71]],
    'SW': [[1, 0]],
    'PHIE_PHIT': [[0.5, 0], [0.5, 0]],
    'PERM': [[0.02, 2000]],
    'VCL': [[0, 1]],
    'RWAPP_RW': [[0.01, 1000], [0.01, 1000]],
    'X_RWA_RW': [[0, 4]],
    'RT_F': [[0.02, 2000], [0.02, 2000]],
    'X_RT_F': [[0, 2]],
    'RT_RHOB': [[0.01, 1000], [1.71, 2.71], [0, 1], [0, 1]],
    'X_RT_RHOB': [[-0.5, 0.5]],
    'XPT': [[0, 1]],
    'RT_RGSA': [[0.02, 2000], [0.02, 2000]],
    'NPHI_NGSA': [[0.6, 0], [0.6, 0]],
    'RHOB_DGSA': [[1.71, 2.71], [1.71, 2.71]],
    'VSH': [[0, 1]],
    'SP': [[-160, 40]],
    'VSH_LINEAR': [[0, 1]],
    'VSH_DN': [[0, 1]],
    'VSH_SP': [[0, 1]],
    'PHIE_DEN': [[0, 1], [0, 1]],
    'PHIT_DEN': [[0, 1], [0, 1]],
    'RWA': [[0, 60], [0, 60], [0, 60]],
    'PHIE': [[0.6, 0]],  # perbaiki seluruh PHIE dan PHIT
    'RT_GR': [[0.02, 2000], [0, 250], [0.02, 2000], [0, 250]],
    # 'RT_PHIE':[[0.02, 2000],[0.6,0],[0.02, 2000],[0.6,0]],
    'RT_PHIE': [[0.02, 2000], [0.6, 0]],
    'SWARRAY': [[1, 0], [1, 0], [1, 0], [1, 0]],
    'SWGRAD': [[0, 0.1]],
    'DNS': [[-1, 1]],
    'DNSV': [[-1, 1]],
}

ratio_plots = {
    'MARKER': 0.5,
    'GR': 1,
    'GR_NORM': 1,
    'GR_DUAL': 1,
    'RT': 0.5,
    'RT_RO': 1,
    'X_RT_RO': 0.5,
    'NPHI_RHOB_NON_NORM': 1,
    'NPHI_RHOB': 1,
    'RHOB': 1,
    'SW': 1,
    'PHIE_PHIT': 1,
    'PERM': 1,
    'VCL': 1,
    'RWAPP_RW': 1,
    'X_RWA_RW': 0.5,
    'RT_F': 1,
    'X_RT_F': 0.5,
    'RT_RHOB': 1,
    'X_RT_RHOB': 0.5,
    'TEST': 0.5,
    'CLASS': 0.5,
    'CTC': 0.5,
    'XPT': 1,
    'RT_RGSA': 1,
    'NPHI_NGSA': 1,
    'RHOB_DGSA': 1,
    'ZONA': 1,
    'VSH': 1,
    'SP': 1,
    'VSH_LINEAR': 1,
    'VSH_DN': 1,
    'VSH_SP': 1,
    'PHIE_DEN': 1,
    'PHIT_DEN': 1,
    'RESERVOIR_CLASS': 0.5,
    'RWA': 1,
    'PHIE': 1,
    'RT_GR': 1,
    'RT_PHIE': 1,
    'RGBE': 0.5,
    'RPBE': 0.5,
    'RGBE_TEXT': 0.5,
    'RPBE_TEXT': 0.5,
    'IQUAL': 0.5,
    'SWARRAY': 1,
    'SWGRAD': 0.5,
    'DNS': 1,
    'DNSV': 1,
}

flags_name = {
    'TEST': {
        0: "",
        1: 'Water',
        3: 'Gas'
    },
    'CLASS': {
        0: 'Non Reservoir',
        1: 'Water',
        2: 'LRLC-Potential',
        3: 'LRLC-Proven',
        4: 'LC-Res',
        5: 'Non-LCRes',
        6: 'Coal'
    },
    'ZONA': {
        0: 'Zona Prospek Kuat',
        1: 'Zona Menarik',
        2: 'Zona Lemah',
        3: 'Non Prospek',
    },
    'RESERVOIR_CLASS': {
        0: 'Zona Prospek Kuat',
        1: 'Zona Menarik',
        2: 'Zona Lemah',
        3: 'Non Prospek',
        4: 'No Data'
    },
    'IQUAL': {
        1: '1'
    }
}

thres = {
    'X_RT_RO': 1,
    'X_RWA_RW': 1.4,
    'X_RT_F': 0.7,
    'X_RT_RHOB': 0.02
}

line_width = 0.9


def plot_flag(df_well, fig, axes, key, n_seq):
    col = data_col[key][0]
    if key == 'TEST':
        flag_colors = flag_color[key]
        flags_names = flags_name[key]
        max_val = 3
    elif key == 'CLASS':
        flag_colors = flag_color[key]
        flags_names = flags_name[key]
        max_val = 6
    elif key == 'ZONA':
        flag_colors = flag_color[key]
        flags_names = flags_name[key]
        max_val = 4
    elif key == 'RESERVOIR_CLASS':
        flag_colors = flag_color[key]
        flags_names = flags_name[key]
        max_val = 4
    elif key == 'IQUAL':
        flag_colors = flag_color[key]
        flags_names = flags_name[key]
        max_val = 1
    elif key == 'CTC':
        flag_colors = flag_color[key]
        flags_names = flags_name[key]
        max_val = 6
    elif key in ['MARKER', 'RGBE', 'RPBE']:
        df_well, flags_names = encode_with_nan(df_well, key)
        max_val = len(flags_names.keys())
        flag_colors = {}
        for i in range(max_val):
            flag_colors[int(i)] = generate_new_color(
                flag_colors, pastel_factor=0)

        for i in range(max_val):
            flag_colors[int(i)] = rgb_to_hex(flag_colors[int(i)])

        flag_colors[0] = 'rgba(0,0,0,0)'

    ones = np.ones((len(df_well[depth]), 1))
    arr = np.array(df_well[col]/max_val).reshape(-1, 1)
    fill = np.multiply(ones, arr)

    bvals = []
    for i in range(1, len(flag_colors.values())+2):
        bvals.append(i)
    colors = list(flag_colors.values())
    colorscale = discrete_colorscale(bvals, colors)

    custom_data = []
    flag_names = df_well[col].map(flags_names.get)
    for i in flag_names:
        custom_data.append([i]*int(max_val+1))

    fig.add_trace(
        go.Heatmap(z=fill, zmin=0, zmax=1, y=df_well[depth], name=col,
                   customdata=custom_data, colorscale=colorscale, showscale=False, hovertemplate="%{customdata}"),
        row=1, col=n_seq, )

    xaxis = "xaxis"+str(n_seq)
    xaxis = "xaxis"+str(n_seq)
    fig.update_layout(
        **{xaxis: dict(
            side="top",
            showticklabels=False,
        )}
    )

    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    return fig, axes


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError(
            'len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0])
             for v in bvals]  # normalized values

    dcolorscale = []  # discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale

def encode_with_nan(df, col):
    encoding_dict = {}
    df_encoded = df.copy()

    unique_vals = df[col].dropna().unique()
    col_map = {val: i+1 for i, val in enumerate(unique_vals)}
    flag_map = {i+1: val for i, val in enumerate(unique_vals)}

    encoding_dict[col] = flag_map

    col_map[None] = 0
    col_map[pd.NA] = 0

    df_encoded[col] = df[col].map(col_map).fillna(0).astype(int)

    col_map.pop(None, None)
    col_map.pop(pd.NA, None)

    encoding_dict[col].update({0: ''})

    return df_encoded, encoding_dict[col]

def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c)
                            for c in existing_colors.values()])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def get_random_color(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


def rgb_to_hex(rgb):
    """Convert RGB (0-1 range) to HEX."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

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

def layout_draw_lines(fig, ratio_plots, df_well, xgrid_intv):
    # Menambahkan garis pembatas
    ratio_plots = np.array(ratio_plots)
    line_pos = []
    for i in ratio_plots:
        line_pos.append(
            i*(1/(ratio_plots/len(ratio_plots)).sum())/len(ratio_plots))
    shapes = []

    shapes.append(
        dict(
            type='line', xref='paper', yref='paper', x0=0, x1=0, y0=1, y1=0,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    x = 0
    for pos in line_pos:
        x += pos
        shapes.append(
            dict(
                type='line', xref='paper', yref='paper', x0=x, x1=x, y0=1, y1=0,
                line=dict(color='black', width=1, dash='solid')
            )
        )

    for i in range(2):
        shapes.append(
            dict(
                type='line', xref='paper', yref='paper', x0=0, x1=1, y0=i, y1=i,
                line=dict(color='black', width=1, dash='solid')
            )
        )

    shapes.append(
        dict(
            type='line', xref='paper', yref='paper', x0=0, x1=1, y0=0.9, y1=0.9,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    # plot grid
    if xgrid_intv is not None and xgrid_intv != 0:
        shapes = shapes + [dict(layer='below',
                                type="line",
                                x0=0, x1=1,
                                xref="paper",
                                y0=y, y1=y,
                                line=dict(color="gainsboro", width=1)) for y in range(0, int(df_well[depth].max()), xgrid_intv)]  # Setiap 2 satuan

        fig.update_layout(shapes=shapes, yaxis=dict(showgrid=False))
    else:
        fig.update_layout(shapes=shapes)

    return fig

def layout_axis(fig, axes, ratio_plots, plot_sequence):
    fig.add_annotation(
        dict(font=dict(color='black', size=12),
             x=-0.001,
             y=0.97,
             xanchor="right",
             yanchor="top",
             showarrow=False,
             text=depth+' (m)',
             textangle=-90,
             xref='paper',
             yref="paper"
             )
    )
    pos_x_c = 0
    ratio_plots = np.array(ratio_plots)
    line_pos = []
    for i in ratio_plots:
        line_pos.append(
            i*(1/(ratio_plots/len(ratio_plots)).sum())/len(ratio_plots))

    pos_x_t = 0
    for i, key in enumerate(axes.keys()):
        # key = plot_sequence[i]
        pos_x = line_pos[i]
        # pos_y = 0.85
        pos_y = 0.92
        pos_x_c += 0.5*pos_x

        # Ganti dengan key yang butuh semua axis (feature di datacol)
        if key in ['SWARRAY']:
            axis_range = axes[key][1:]  # Semua axis
        else:
            axis_range = axes[key][1:3]  # Hanya 2 axis pertama

        for j, axis in enumerate(axis_range):
            # print(f'{i}:{j}')
            fig.update_layout(
                **{axis: dict(
                    tickfont=dict(color=color_col[key][j], size=9),
                    anchor="free",
                    showline=True,
                    position=pos_y,
                    showticklabels=False,
                    linewidth=1.5,
                    linecolor=color_col[key][j],
                )}
            )

            # Add Text Parameter
            fig.add_annotation(
                dict(font=dict(color=color_col[key][j], size=12),
                     # x=x_loc,
                     x=pos_x_c,
                     y=pos_y,
                     xanchor="center",
                     yanchor="bottom",
                     showarrow=False,
                     text=data_col[key][j],
                     textangle=0,
                     xref='paper',
                     yref="paper"
                     )
            )

            # Add Text Unit
            fig.add_annotation(
                dict(font=dict(color=color_col[key][j], size=10),
                     x=pos_x_c,
                     y=pos_y,
                     xanchor="center",
                     yanchor="top",
                     showarrow=False,
                     text=unit_col[key][j],
                     textangle=0,
                     xref='paper',
                     yref="paper"
                     )
            )

            # Add Text Min Max Range
            if key not in ['CLASS', 'TEST', 'XPT', 'MARKER', 'ZONA', 'RESERVOIR_CLASS', 'RGBE', 'RPBE', 'IQUAL', 'RGBE_TEXT', 'RPBE_TEXT']:
                fig.add_annotation(
                    dict(font=dict(color=color_col[key][j], size=10),
                         x=pos_x_t,
                         y=pos_y,
                         xanchor="left",
                         yanchor="top",
                         showarrow=False,
                         text=range_col[key][j][0],
                         textangle=0,
                         xref='paper',
                         yref="paper"
                         )
                )

                fig.add_annotation(
                    dict(font=dict(color=color_col[key][j], size=10),
                         # x=x_loc,
                         x=pos_x_t+pos_x,
                         y=pos_y,
                         xanchor="right",
                         yanchor="top",
                         showarrow=False,
                         text=range_col[key][j][1],
                         textangle=0,
                         xref='paper',
                         yref="paper"
                         )
                )

            pos_y += 0.03
            pos_y = min(pos_y, 1.0)

        pos_x_t += pos_x
        pos_x_c += 0.5*pos_x

    return fig

def plot_line(df_well, fig, axes, base_key, n_seq, type=None, col=None, label=None):
    """
    Plot a line curve on the well log plot.

    Parameters:
    -----------
    df_well : pandas DataFrame
        DataFrame containing well log data
    fig : plotly Figure object
        Figure to add trace to
    axes : dict
        Dictionary with axes information
    key : str
        Key for display settings (colors, ranges, units)
    n_seq : int
        Sequence number for the plot
    type : str, optional
        Plot type, if 'log' then logarithmic scale
    col : str, optional
        Column name in df_well to plot (if None, uses data_col[key][0])
    label : str, optional
        Label to display for the curve (if None, uses col)

    Returns:
    --------
    fig : plotly Figure object
        Updated figure
    axes : dict
        Updated axes dictionary
    """
    # If col is not provided, use the default column for the key
    if col is None:
        col = data_col[base_key][0]

    # If label is not provided, use the column name
    if label is None:
        label = col

    # Add trace to figure
    fig.add_trace(
        go.Scattergl(
            x=df_well[col],
            y=df_well[depth],
            line=dict(color=color_col[base_key][0], width=line_width),
            name=label,  # Use the provided label
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_seq),
            yaxis='y'+str(n_seq),
        ),
    )

    # Update x-axis layout
    xaxis = "xaxis"+str(n_seq)
    if type is None:
        fig.update_layout(
            **{xaxis: dict(
                side="top",
                range=range_col[base_key][0]
            )}
        )
    else:
        fig.update_layout(
            **{xaxis: dict(
                side="top",
                type="log",
                range=[np.log10(range_col[base_key][0][0]),
                       np.log10(range_col[base_key][0][1])]
            )}
        )

    # Update axes dictionary
    axes[col].append('yaxis'+str(n_seq))
    axes[col].append('xaxis'+str(n_seq))

    return fig, axes


def plot_fill_x_to_int(df_well, fig, axes, key, n_seq, index):
    col = data_col[key][index]
    t_g = range_col[key][index][1]

    x_g = [t_g for x in df_well[col]]
    fig.add_trace(
        go.Scatter(x=x_g, y=df_well[depth],
                   line=dict(color='rgba(0,0,0,0)', width=0),
                   showlegend=False,
                   name='dummy'+col,
                   hoverinfo="skip"),
        row=1, col=n_seq)

    fig.add_trace(
        go.Scatter(
            x=df_well[col],
            y=df_well[depth],
            line=dict(color=color_col[key][index], width=line_width),
            name=col,
            legend=legends[n_seq-1],
            showlegend=True,
            fill='tonextx',
            xaxis='x'+str(n_seq),
        ),
    )

    xaxis = "xaxis"+str(n_seq)
    fig.update_layout(
        **{xaxis: dict(
            side="top",
            range=range_col[key][index]
        )}
    )

    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    return fig, axes


def plot_dual_gr(df_well, fig, axes, key, n_seq, counter, n_plots):
    """
    Plot dua kurva GR dan GR_NORM dalam satu plot
    """
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Plot kurva pertama (GR)
    col1 = data_col[key][0]  # 'GR'
    fig.add_trace(
        go.Scattergl(
            x=df_well[col1],
            y=df_well[depth],
            line=dict(color=color_col[key][0], width=line_width),
            name=col1,
            legend=legends[n_seq-1],
            showlegend=True,
        ),
        row=1, col=n_seq
    )

    # Plot kurva kedua (GR_NORM) - menggunakan pola yang sama dengan fungsi lain
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))
    col2 = data_col[key][1]  # 'GR_NORM'

    fig.add_trace(
        go.Scattergl(
            x=df_well[col2],
            y=df_well[depth],
            line=dict(color=color_col[key][1], width=line_width),
            name=col2,
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_plots+counter),
            yaxis='y'+str(n_seq),
        ),
    )

    # Konfigurasi axis pertama
    xaxis1 = "xaxis"+str(n_seq)
    fig.update_layout(
        **{xaxis1: dict(
            side="top",
            range=range_col[key][0]
        )}
    )

    # Konfigurasi axis kedua (overlay)
    xaxis2 = "xaxis"+str(n_plots+counter)
    fig.update_layout(
        **{xaxis2: dict(
            overlaying='x'+str(n_seq),
            side="top",
            range=range_col[key][1]
        )}
    )

    return fig, axes, counter


def plot_gsa_crossover(df_well, fig, axes, key, n_seq, counter, n_plots, fill_color_red='red', fill_color_blue=colors_dict['blue']):
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Tentukan kondisi fill berdasarkan jenis GSA
    if key == 'RT_RGSA':
        condition_red = df_well[data_col[key][0]
                                ] > df_well[data_col[key][1]]  # RT > RGSA (MERAH)
        condition_blue = df_well[data_col[key][0]
                                 ] < df_well[data_col[key][1]]  # RT < RGSA (BIRU)
        log_scale = True
    elif key == 'NPHI_NGSA':
        # NPHI < NGSA (MERAH)
        condition_red = df_well[data_col[key][0]] < df_well[data_col[key][1]]
        # NPHI > NGSA (BIRU)
        condition_blue = df_well[data_col[key][0]] > df_well[data_col[key][1]]
        log_scale = False
    elif key == 'RHOB_DGSA':
        # RHOB < DGSA (MERAH)
        condition_red = df_well[data_col[key][0]] < df_well[data_col[key][1]]
        # RHOB > DGSA (BIRU)
        condition_blue = df_well[data_col[key][0]] > df_well[data_col[key][1]]
        log_scale = False

    # Plot kurva utama terlebih dahulu
    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][0]],
            y=df_well[depth],
            line=dict(color=color_col[key][0], width=line_width),
            name=data_col[key][0],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_seq),
            yaxis='y'+str(n_seq),
        )
    )

    # Setup axis kedua untuk kurva baseline
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))

    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][1]],
            y=df_well[depth],
            line=dict(color=color_col[key][1], width=line_width),
            name=data_col[key][1],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_plots+counter),
            yaxis='y'+str(n_seq),
        )
    )

    # Setup axis ketiga untuk crossover fill RED
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))

    # Setup axis keempat untuk crossover fill BLUE
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))

    # Buat DataFrame untuk crossover RED
    xover_red_df = pd.DataFrame({
        data_col[key][0]: df_well[data_col[key][0]],
        data_col[key][1]: df_well[data_col[key][1]],
        depth: df_well[depth],
        'label_red': condition_red.astype(int)
    })

    # Buat DataFrame untuk crossover BLUE
    xover_blue_df = pd.DataFrame({
        data_col[key][0]: df_well[data_col[key][0]],
        data_col[key][1]: df_well[data_col[key][1]],
        depth: df_well[depth],
        'label_blue': condition_blue.astype(int)
    })

    # Group berdasarkan perubahan label RED
    xover_red_df['group'] = xover_red_df['label_red'].ne(
        xover_red_df['label_red'].shift()).cumsum()
    xover_red_groups = xover_red_df.groupby('group')

    # Plot area fill RED untuk setiap group yang memenuhi kondisi
    for _, group_data in xover_red_groups:
        if group_data['label_red'].iloc[0] == 1:
            # Baseline (invisible line)
            fig.add_trace(
                go.Scatter(
                    x=group_data[data_col[key][0]],
                    y=group_data[depth],
                    name='baseline_red',
                    showlegend=False,
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    xaxis='x'+str(n_plots+counter-1),  # GUNAKAN AXIS KETIGA
                    yaxis='y'+str(n_seq),
                    hoverinfo="skip"
                )
            )

            # Fill area RED (tonextx)
            fig.add_trace(
                go.Scatter(
                    x=group_data[data_col[key][1]],
                    y=group_data[depth],
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    name='fill_area_red',
                    fill='tonextx',
                    showlegend=False,
                    fillcolor=fill_color_red,
                    # GUNAKAN AXIS KETIGA YANG SAMA
                    xaxis='x'+str(n_plots+counter-1),
                    yaxis='y'+str(n_seq),
                    hoverinfo="skip"
                )
            )

    # Group berdasarkan perubahan label BLUE
    xover_blue_df['group'] = xover_blue_df['label_blue'].ne(
        xover_blue_df['label_blue'].shift()).cumsum()
    xover_blue_groups = xover_blue_df.groupby('group')

    # Plot area fill BLUE untuk setiap group yang memenuhi kondisi
    for _, group_data in xover_blue_groups:
        if group_data['label_blue'].iloc[0] == 1:
            # Baseline (invisible line)
            fig.add_trace(
                go.Scatter(
                    x=group_data[data_col[key][0]],
                    y=group_data[depth],
                    name='baseline_blue',
                    showlegend=False,
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    xaxis='x'+str(n_plots+counter),  # GUNAKAN AXIS KEEMPAT
                    yaxis='y'+str(n_seq),
                    hoverinfo="skip"
                )
            )

            # Fill area BLUE (tonextx)
            fig.add_trace(
                go.Scatter(
                    x=group_data[data_col[key][1]],
                    y=group_data[depth],
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    name='fill_area_blue',
                    fill='tonextx',
                    showlegend=False,
                    fillcolor=fill_color_blue,
                    # GUNAKAN AXIS KEEMPAT YANG SAMA
                    xaxis='x'+str(n_plots+counter),
                    yaxis='y'+str(n_seq),
                    hoverinfo="skip"
                )
            )

    # Update axis layout untuk axis pertama
    xaxis1 = "xaxis"+str(n_seq)
    if log_scale:
        fig.update_layout(
            **{xaxis1: dict(
                side="top",
                type="log",
                range=[np.log10(range_col[key][0][0]),
                       np.log10(range_col[key][0][1])]
            )}
        )
    else:
        fig.update_layout(
            **{xaxis1: dict(
                side="top",
                range=range_col[key][0]
            )}
        )

    # Update axis layout untuk axis kedua (baseline curve)
    xaxis2 = "xaxis"+str(n_plots+counter-2)
    if log_scale and len(range_col[key]) > 1:
        fig.update_layout(
            **{xaxis2: dict(
                overlaying='x'+str(n_seq),
                side="top",
                type="log",
                range=[np.log10(range_col[key][1][0]),
                       np.log10(range_col[key][1][1])]
            )}
        )
    elif len(range_col[key]) > 1:
        fig.update_layout(
            **{xaxis2: dict(
                overlaying='x'+str(n_seq),
                side="top",
                range=range_col[key][1]
            )}
        )

    # Update axis layout untuk axis ketiga (crossover fill RED) - INVISIBLE
    xaxis3 = "xaxis"+str(n_plots+counter-1)
    # Gunakan range yang sesuai untuk fill area
    fill_range = range_col[key][0] if len(range_col[key]) > 0 else [0, 1]

    if log_scale:
        fig.update_layout(
            **{xaxis3: dict(
                visible=False,
                overlaying='x'+str(n_seq),
                side="top",
                type="log",
                range=[np.log10(fill_range[0]), np.log10(fill_range[1])]
            )}
        )
    else:
        fig.update_layout(
            **{xaxis3: dict(
                visible=False,
                overlaying='x'+str(n_seq),
                side="top",
                range=fill_range
            )}
        )

    # Update axis layout untuk axis keempat (crossover fill BLUE) - INVISIBLE
    xaxis4 = "xaxis"+str(n_plots+counter)

    if log_scale:
        fig.update_layout(
            **{xaxis4: dict(
                visible=False,
                overlaying='x'+str(n_seq),
                side="top",
                type="log",
                range=[np.log10(fill_range[0]), np.log10(fill_range[1])]
            )}
        )
    else:
        fig.update_layout(
            **{xaxis4: dict(
                visible=False,
                overlaying='x'+str(n_seq),
                side="top",
                range=fill_range
            )}
        )

    return fig, axes, counter

def min_max_normalize(log_in,
                      calib_min=40, calib_max=140,
                      pct_min=3, pct_max=97,
                      cutoff_min=0, cutoff_max=250):
    """
    Geolog-style MIN-MAX normalization using percentiles
    """
    log = np.array(log_in, dtype=float)

    if cutoff_min is not None:
        log[log < cutoff_min] = np.nan
    if cutoff_max is not None:
        log[log > cutoff_max] = np.nan

    min_pnt = np.nanpercentile(log, pct_min)
    max_pnt = np.nanpercentile(log, pct_max)

    # Hindari pembagian dengan nol jika semua data sama
    if max_pnt == min_pnt:
        return np.full_like(log, calib_min)

    m = (calib_max - calib_min) / (max_pnt - min_pnt)
    log_out = calib_min + m * (log - min_pnt)

    return log_out

def layout_range_all_axis(fig, axes, plot_sequence):
    for key, axess in axes.items():
        for axis in axess:
            # key = plot_sequence[i]
            if axis.startswith('yaxis'):
                fig.update_layout(
                    **{axis: dict(
                        domain=[0, 0.9],
                        gridcolor='gainsboro',
                        showspikes=True,
                        showgrid=True,
                        showticklabels=False if axis.startswith(
                            'xaxis') else True,
                    )}
                )
            elif key in ['RT_RO', 'PERM', 'RWAPP_RW', 'RT_F', 'RT_RHOB', 'RT_RGSA', 'RT', 'RT_GR', 'RT_PHIE']:
                a = range_col[key][0][0]
                b = range_col[key][0][1]
                n = int(np.log10(b/a))
                arr = list(np.arange(a, 0.1, 0.01))
                for j in range(n-1):
                    arr = arr + list(np.arange(10**(j-1), 10**j, 10**(j-1)))
                arr = arr + list(np.arange(1000, b+1, 1000))

                fig.update_layout(
                    **{axis: dict(
                        # gridcolor='rgba(0,0,0,0)',
                        tickvals=arr,
                        gridcolor='gainsboro',
                        side="top",
                        fixedrange=True,
                        showticklabels=False if axis.startswith(
                            'xaxis') else True,
                    )}
                )
            elif key in ['GR', 'SP', 'GR_NORM', 'GR_DUAL', 'RTRO', 'NPHI_RHOB', 'SW', 'PHIE_PHIT', 'VCL', 'X_RWA_RW', 'X_RT_F', 'X_RT_RHOB', 'NPHI_NGSA', 'RHOB_DGSA', 'VSH_LINEAR', 'VSH_DN', 'VSH_SP', 'RHOB', 'PHIE_DEN', 'PHIT_DEN', 'RWA', 'PHIE', 'DNS', 'DNSV', 'VSH']:
                fig.update_layout(
                    **{axis: dict(
                        # gridcolor='rgba(0,0,0,0)',
                        tickvals=list(np.linspace(
                            range_col[key][0][0], range_col[key][0][1], 5)),
                        gridcolor='gainsboro',
                        side="top",
                        fixedrange=True,
                        showticklabels=False if axis.startswith(
                            'xaxis') else True,
                    )}
                )
    return fig


def fillcol(label, yc='rgba(0,250,0,0.4)', nc='rgba(250,0,0,0)'):
    if label >= 1:
        return yc
    else:
        return nc

def xover_label_df(df_well, key, type=1):
    if key in ['X_RT_RO', 'X_RWA_RW', 'X_RT_F', 'X_RT_RHOB']:
        xover_df = pd.DataFrame(df_well[data_col[key]].copy())
        xover_df['thres'] = [thres[key]]*len(xover_df)
        xover_df['label'] = np.where(
            df_well[data_col[key][0]] > thres[key], 1, 0)

    elif key == 'NPHI_RHOB' or key == 'RT_RHOB':
        xover_df = pd.DataFrame(df_well[data_col[key]].copy())
        xover_df['label'] = np.where(
            xover_df[data_col[key][2]] > xover_df[data_col[key][3]], 1, 0)

    else:
        xover_df = pd.DataFrame(df_well[data_col[key]].copy())
        xover_df['label'] = np.where(
            xover_df[data_col[key][0]] > xover_df[data_col[key][1]], 1, 0)

    xover_df[depth] = df_well[depth]
    xover_df['group'] = xover_df['label'].ne(
        xover_df['label'].shift()).cumsum()
    xover_df = xover_df.groupby('group')
    xover_dfs = []

    for _, data in xover_df:
        if type == 1:
            if data['label'].reset_index(drop=True)[0]:
                xover_dfs.append(data)
            else:
                continue
        else:
            xover_dfs.append(data)
    return xover_dfs

def plot_xover_thres(df_well, fig, axes, key, n_seq, counter, y_color=colors_dict['red'], n_color='lightgray'):
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Plot Area Xover
    xover_dfs = xover_label_df(df_well, key, type=1)
    for xover_df in xover_dfs:
        fig.add_traces(
            go.Scatter(
                x=xover_df[data_col[key][0]],
                y=xover_df[depth],
                name='xover',
                showlegend=False,
                line=dict(color='rgba(0,0,0,0)', width=0),
                xaxis='x'+str(n_seq),
                yaxis='y'+str(n_seq),
                hoverinfo="skip"
            ),
        )

        fig.add_traces(
            go.Scatter(
                x=xover_df['thres'],
                y=xover_df[depth],
                line=dict(color='rgba(0,0,0,0)', width=0),
                name='xover',
                fill='tonextx',
                showlegend=False,
                fillcolor=fillcol(xover_df['label'].iloc[0], y_color, n_color),
                xaxis='x'+str(n_seq),
                yaxis='y'+str(n_seq),
                hoverinfo="skip"
            ),
        )

    # Plot Line
    col = data_col[key][0]
    fig.add_trace(
        go.Scattergl(
            x=df_well[col],
            y=df_well[depth],
            line=dict(color=color_col[key][0], width=line_width),
            name=col,
            legend=legends[n_seq-1],
            showlegend=True,
        ),
        row=1, col=n_seq,
    )

    fig.add_trace(
        go.Scattergl(
            x=[thres[key]]*len(df_well[depth]),
            y=df_well[depth],
            line=dict(color=colors_dict['red'], width=line_width),
            name="Threshold",
            legend=legends[n_seq-1],
            showlegend=True,
        ),
        row=1, col=n_seq,
    )

    axis = 'xaxis'+str(n_seq)
    fig.update_layout(
        **{axis: dict(
            side="top",
            range=range_col[key][0]
        )})

    return fig, axes, counter

def plot_depth_matching_results(ref_df, lwd_df, final_df):
    """
    Menerima 3 DataFrame dan membuat plot 4-panel yang komprehensif.
    """
    if ref_df is None or lwd_df is None or final_df is None:
        raise ValueError("Data input untuk plotting tidak boleh None.")

    # Buat 4 subplot dengan sumbu Y yang sama
    fig = make_subplots(
        rows=1, cols=4,
        shared_yaxes=True,
        subplot_titles=("Reference Log", "LWD Log (Original)",
                        "Before Alignment", "After Alignment (DTW)")
    )

    # --- Panel 1 (Paling Kiri): Reference Log ---
    fig.add_trace(go.Scattergl(
        x=ref_df["GR"], y=ref_df["Depth"], name='REF GR',
        line=dict(color='black')
    ), row=1, col=1)

    # --- Panel 2: LWD Log Asli ---
    fig.add_trace(go.Scattergl(
        x=lwd_df["DGRCC"], y=lwd_df["Depth"], name='LWD GR',
        line=dict(color='red')
    ), row=1, col=2)

    # --- Panel 3: Sebelum Alignment (Ditumpuk) ---
    fig.add_trace(go.Scattergl(
        x=ref_df["GR"], y=ref_df["Depth"], name='REF GR (Before)',
        line=dict(color='black'), legendgroup='before'
    ), row=1, col=3)
    fig.add_trace(go.Scattergl(
        x=lwd_df["DGRCC"], y=lwd_df["Depth"], name='LWD GR (Before)',
        line=dict(color='red', dash='dash'), legendgroup='before'
    ), row=1, col=3)

    # --- Panel 4 (Paling Kanan): Setelah Alignment (Ditumpuk) ---
    fig.add_trace(go.Scattergl(
        x=final_df["REF_GR"], y=final_df["Depth"], name='REF GR (After)',
        line=dict(color='black'), legendgroup='after'
    ), row=1, col=4)
    fig.add_trace(go.Scattergl(
        x=final_df["LWD_DGRCC_Aligned"], y=final_df[
            "Depth"], name='LWD Aligned (After)',
        line=dict(color='red', dash='dash'), legendgroup='after'
    ), row=1, col=4)

    fig.update_layout(
        title_text="Depth Matching Analysis",
        height=8600,
        showlegend=False,
        template="plotly_white",
        yaxis=dict(autorange='reversed', title_text="Depth (m)"),
        hovermode="y unified",
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='red',
        dtick=15,
        griddash='dot',
        showline=True,
        linewidth=1.5,
        linecolor='black',
        mirror=True
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor='black',
        mirror=True
    )

    return fig

def plot_xover_log_normal(df_well, fig, axes, key, n_seq, counter, n_plots, y_color='limegreen', n_color='lightgray', type=1, exclude_crossover=False):
    axes[key] = ['yaxis'+str(n_seq), 'xaxis'+str(n_seq)]  # Initialize
    col = data_col[key][0]
    range_type = 'log' if col == 'RT' else "-"
    range_axis = [np.log10(range_col[key][0][0]), np.log10(
        range_col[key][0][1])] if range_type == 'log' else range_col[key][0]
    fig.add_trace(go.Scattergl(x=df_well[col], y=df_well[depth], line=dict(
        color=color_col[key][0], width=line_width), name=col, legend=legends[n_seq-1], showlegend=False), row=1, col=n_seq)
    fig.update_layout(
        **{"xaxis"+str(n_seq): dict(side="top", type=range_type, range=range_axis)})

    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))
    col = data_col[key][1]
    range_type = 'log' if col == 'RT' else "-"
    range_axis = [np.log10(range_col[key][1][0]), np.log10(
        range_col[key][1][1])] if range_type == 'log' else range_col[key][1]
    fig.add_trace(go.Scattergl(x=df_well[col], y=df_well[depth], line=dict(color=color_col[key][1], width=line_width),
                  name=col, legend=legends[n_seq-1], showlegend=False, xaxis='x'+str(n_plots+counter), yaxis='y'+str(n_seq)))
    fig.update_layout(**{"xaxis"+str(n_plots+counter): dict(overlaying="x" +
                      str(n_seq), side="top", type=range_type, range=range_axis)})

    if not exclude_crossover:
        counter += 1
        axes[key].append('xaxis'+str(n_plots+counter))
        xover_dfs = xover_label_df(df_well, key, type=type)
        for xover_df in xover_dfs:
            fig.add_traces(go.Scatter(x=xover_df[data_col[key][2]], y=xover_df[depth], name='xover', showlegend=False, line=dict(
                color='rgba(0,0,0,0)'), xaxis='x'+str(n_plots+counter), yaxis='y'+str(n_seq), hoverinfo="skip"))
            fig.add_traces(go.Scatter(x=xover_df[data_col[key][3]], y=xover_df[depth], line=dict(color='rgba(0,0,0,0)'), name='xover', fill='tonextx', showlegend=False, fillcolor=fillcol(
                xover_df['label'].iloc[0], y_color, n_color), xaxis='x'+str(n_plots+counter), yaxis='y'+str(n_seq), hoverinfo="skip"))
        fig.update_layout(**{"xaxis"+str(n_plots+counter): dict(visible=False,
                          overlaying="x"+str(n_seq), side="top", range=range_col[key][0])})

    return fig, axes, counter


def plot_texts_marker(df_text, depth_btm, fig, axes, key, n_seq):
    if not df_text.empty:
        for index, row in df_text[['Mean Depth', 'Surface']].iterrows():
            x = 0
            y = row['Mean Depth']
            text = row['Surface'][:6]
            if y < depth_btm:
                fig.add_annotation(
                    x=x,
                    y=y,
                    xref="x"+str(n_seq),
                    yref="y",
                    xanchor='center',
                    yanchor='middle',
                    text=text,
                    showarrow=True,
                    font=dict(
                        size=10,
                        color="black"
                    ),
                    align="center",
                    bgcolor="white",
                    ax=0,
                    ay=0,
                )

    return fig, axes

def plot_normalization(df, df_marker, df_well_marker):
    sequence = ['MARKER', 'GR', 'GR_NORM', 'GR_DUAL']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}
    print(plot_sequence)

    ratio_plots_seq = []
    ratio_plots_seq.append(ratio_plots['MARKER'])
    sequence_keys = list(plot_sequence.values())
    for key in sequence_keys[1:]:
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

    # Plot Marker
    fig, axes = plot_flag(df_well_marker, fig, axes,
                          "MARKER", 1)  # n_seq=1 for marker
    fig, axes = plot_texts_marker(
        df_marker, df_well_marker['DEPTH'].max(), fig, axes, "MARKER", 1)

    # Plot GR - start from n_seq=2 which is 'GR' in your sequence
    for n_seq, col in plot_sequence.items():
        if n_seq > 1:
            if col == 'GR':  # Skip n_seq=1 which is 'MARKER'
                fig, axes = plot_line(
                    df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)
            elif col == 'GR_NORM':  # Skip n_seq=1 which is 'MARKER'
                fig, axes = plot_line(
                    df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)
            elif col == 'GR_DUAL':
                fig, axes, counter = plot_dual_gr(
                    df, fig, axes, col, n_seq, counter, subplot_col)

    fig = layout_range_all_axis(fig, axes, plot_sequence)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), height=1500,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified', hoverdistance=-1,
        title_text="Normalization",
        title_x=0.5,
        modebar_remove=['lasso', 'autoscale', 'zoom',
                        'zoomin', 'zoomout', 'pan', 'select']
    )

    fig.update_yaxes(showspikes=True,
                     range=[df[depth].max(), df[depth].min()])
    fig.update_traces(yaxis='y')

    fig = layout_draw_lines(fig, ratio_plots_seq, df, xgrid_intv=0)

    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    print(axes)

    return fig

def normalize(series):
    std_dev = np.std(series)
    if std_dev == 0:
        # Jika semua nilai sama, kembalikan array berisi nol
        return np.zeros_like(series)
    return (series - np.mean(series)) / std_dev

def calculate_vsh_from_gr(df: pd.DataFrame, gr_log: str, gr_ma: float, gr_sh: float, output_col: str) -> pd.DataFrame:
    """
    Menghitung VSH dari Gamma Ray menggunakan metode linear.

    Args:
        df (pd.DataFrame): DataFrame input yang berisi data log.
        gr_log (str): Nama kolom Gamma Ray yang akan digunakan.
        gr_ma (float): Nilai GR matriks (zona bersih).
        gr_sh (float): Nilai GR shale.
        output_col (str): Nama kolom baru untuk menyimpan hasil VSH.

    Returns:
        pd.DataFrame: DataFrame asli dengan tambahan kolom VSH.
    """
    # Pastikan kolom yang dibutuhkan ada
    if gr_log not in df.columns:
        print(
            f"Peringatan: Kolom '{gr_log}' tidak ditemukan. Melewatkan kalkulasi VSH.")
        df[output_col] = np.nan
        return df

    # Salin untuk menghindari SettingWithCopyWarning
    df_processed = df.copy()

    # Hitung VSH dengan rumus linear
    v_gr = (df_processed[gr_log] - gr_ma) / (gr_sh - gr_ma)

    # Batasi nilai antara 0 dan 1
    df_processed[output_col] = v_gr.clip(0, 1)

    return df_processed

def depth_matching(ref_las_path: str, lwd_las_path: str, num_chunks: int = 10):
    """
    Menjalankan logika DTW dan mengembalikan tiga DataFrame: 
    data referensi, data LWD asli, dan data hasil alignment.
    """
    try:
        ref_las = lasio.read(ref_las_path)
        lwd_las = lasio.read(lwd_las_path)

        ref_df = ref_las.df().reset_index()[["DEPTH", "GR_CAL"]].dropna()
        ref_df.columns = ["Depth", "GR"]

        lwd_df = lwd_las.df().reset_index()[["DEPTH", "DGRCC"]].dropna()
        lwd_df.columns = ["Depth", "DGRCC"]

        N_ref = len(ref_df)
        N_lwd = len(lwd_df)

        ref_chunk_size = N_ref // num_chunks
        lwd_chunk_size = N_lwd // num_chunks

        all_chunks = []

        for i in range(num_chunks):
            ref_start = i * ref_chunk_size
            ref_end = N_ref if i == num_chunks - \
                1 else (i + 1) * ref_chunk_size
            ref_chunk = ref_df.iloc[ref_start:ref_end]

            lwd_start = i * lwd_chunk_size
            lwd_end = N_lwd if i == num_chunks - \
                1 else (i + 1) * lwd_chunk_size
            lwd_chunk = lwd_df.iloc[lwd_start:lwd_end]

            if len(ref_chunk) < 2 or len(lwd_chunk) < 2:
                continue

            ref_signal = normalize(ref_chunk["GR"].values)
            lwd_signal = normalize(lwd_chunk["DGRCC"].values)

            path = dtw.warping_path(lwd_signal, ref_signal)

            aligned_depths = []
            aligned_dgrcc = []
            for lwd_idx, ref_idx in path:
                aligned_depths.append(ref_chunk.iloc[ref_idx]["Depth"])
                aligned_dgrcc.append(lwd_chunk.iloc[lwd_idx]["DGRCC"])

            aligned_df = pd.DataFrame({
                "Depth": aligned_depths,
                "Aligned_DGRCC": aligned_dgrcc
            }).sort_values(by="Depth")

            interp_func = interp1d(aligned_df["Depth"], aligned_df["Aligned_DGRCC"],
                                   kind='linear', bounds_error=False, fill_value="extrapolate")
            interp_dgrcc = interp_func(ref_chunk["Depth"].values)

            chunk_result = pd.DataFrame({
                "Depth": ref_chunk["Depth"].values,
                "REF_GR": ref_chunk["GR"].values,
                "LWD_DGRCC_Aligned": interp_dgrcc
            })

            all_chunks.append(chunk_result)

            final_df = pd.concat(all_chunks, ignore_index=True)
            final_df = final_df.drop_duplicates(
                subset="Depth").sort_values(by="Depth")

    except Exception as e:
        print(f"Error di dalam depth_matching_logic: {e}")
        return None, None, None

    return ref_df, lwd_df, final_df


import shutil

@app.route('/api/seed-data-volume', methods=['POST'])
def seed_data_volume():
    """
    Receives a single ZIP file, clears the existing data in the persistent volume,
    then unpacks the ZIP and saves the entire structure into the volume.
    """
    app.logger.info("Received request to seed data volume.")
    
    # Simple security check to prevent unauthorized use.
    if request.headers.get('X-Seed-Auth') != 'your-secret-key-123':
        app.logger.warning("Seed attempt without valid auth header.")
        return jsonify({"error": "Unauthorized"}), 403

    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.zip'):
        filename = secure_filename(file.filename)
        app.logger.info(f"Processing seed file: {filename}")

        try:
            # === NEW: Clear the persistent volume before extracting ===
            if os.path.exists(PERSISTENT_DATA_DIR):
                app.logger.info(f"Clearing existing data in {PERSISTENT_DATA_DIR}")
                for item in os.listdir(PERSISTENT_DATA_DIR):
                    item_path = os.path.join(PERSISTENT_DATA_DIR, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        app.logger.info(f"Deleted directory: {item_path}")
                    else:
                        os.remove(item_path)
                        app.logger.info(f"Deleted file: {item_path}")

            # Read the ZIP into memory
            zip_content = io.BytesIO(file.read())
            
            with zipfile.ZipFile(zip_content, 'r') as z:
                for member in z.infolist():
                    target_path = os.path.join(PERSISTENT_DATA_DIR, member.filename)
                    # Prevent zip slip
                    if not os.path.realpath(target_path).startswith(os.path.realpath(PERSISTENT_DATA_DIR)):
                        app.logger.warning(f"Skipping potentially malicious file path: {member.filename}")
                        continue
                    if member.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                        app.logger.info(f"Created directory: {target_path}")
                    else:
                        parent_dir = os.path.dirname(target_path)
                        os.makedirs(parent_dir, exist_ok=True)
                        with open(target_path, "wb") as f_out:
                            f_out.write(z.read(member.filename))
                        app.logger.info(f"Extracted file: {target_path}")

            return jsonify({"message": f"Successfully CLEARED and seeded volume with contents of {filename}"}), 200

        except Exception as e:
            app.logger.exception("Failed to process seed ZIP file.")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type. Please upload a ZIP file."}), 400

# --- THIS IS THE NEW, ONE-TIME-USE ENDPOINT ---
# @app.route('/api/seed-data-volume', methods=['POST'])
# def seed_data_volume():
#     """
#     Receives a single ZIP file, unpacks it, and saves the entire
#     directory structure and its contents into the persistent volume.
#     This is intended for one-time setup.
#     """
#     app.logger.info("Received request to seed data volume.")
    
#     # Simple security check to prevent unauthorized use.
#     # You must send this header from your frontend.
#     if request.headers.get('X-Seed-Auth') != 'your-secret-key-123':
#         app.logger.warning("Seed attempt without valid auth header.")
#         return jsonify({"error": "Unauthorized"}), 403

#     if 'file' not in request.files:
#         return jsonify({"error": "No 'file' part in the request"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and file.filename.lower().endswith('.zip'):
#         filename = secure_filename(file.filename)
#         app.logger.info(f"Processing seed file: {filename}")
        
#         try:
#             # Read the entire ZIP file into memory
#             zip_content = io.BytesIO(file.read())
            
#             with zipfile.ZipFile(zip_content, 'r') as z:
#                 # Get the list of all files and directories in the zip
#                 member_list = z.infolist()
#                 for member in member_list:
#                     # Construct the full extraction path inside the volume
#                     # e.g., /data/sample_data/wells/ABB-035.csv
#                     target_path = os.path.join(PERSISTENT_DATA_DIR, member.filename)
                    
#                     # This is crucial for security, prevents 'zip slip' vulnerabilities
#                     if not os.path.realpath(target_path).startswith(os.path.realpath(PERSISTENT_DATA_DIR)):
#                         app.logger.warning(f"Skipping potentially malicious file path: {member.filename}")
#                         continue

#                     # If it's a directory, create it
#                     if member.is_dir():
#                         os.makedirs(target_path, exist_ok=True)
#                         app.logger.info(f"Created directory: {target_path}")
#                     # If it's a file, extract it
#                     else:
#                         # Ensure the parent directory exists before extracting
#                         parent_dir = os.path.dirname(target_path)
#                         os.makedirs(parent_dir, exist_ok=True)
#                         with open(target_path, "wb") as f_out:
#                             f_out.write(z.read(member.filename))
#                         app.logger.info(f"Extracted file: {target_path}")
                        
#             return jsonify({"message": f"Successfully seeded volume with contents of {filename}"}), 200

#         except Exception as e:
#             app.logger.exception("Failed to process seed ZIP file.")
#             return jsonify({"error": str(e)}), 500

#     return jsonify({"error": "Invalid file type. Please upload a ZIP file."}), 400


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
    
@app.route('/api/list-wells', methods=['GET'])
def list_wells():
    try:
        if not os.path.exists(WELLS_DIR):
            # This might happen on the very first run before any files are there
            return jsonify({"error": 'no file found'}), 200

        well_files = [f.replace('.csv', '') for f in os.listdir(WELLS_DIR) if f.endswith('.csv')]
        well_files.sort()
        return jsonify(well_files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

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


# @app.route('/api/list-wells', methods=['GET'])
# def list_wells():
#     try:
#         if not os.path.exists(WELLS_DIR):
#             return jsonify({"error": "Folder 'wells' tidak ditemukan."}), 404

#         # Ambil semua file .csv, hapus ekstensinya
#         well_files = [f.replace('.csv', '') for f in os.listdir(
#             WELLS_DIR) if f.endswith('.csv')]
#         well_files.sort()  # Urutkan nama sumur

#         return jsonify(well_files)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


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
        calib_min = float(params.get('CALIB_MIN', 40))
        calib_max = float(params.get('CALIB_MAX', 140))
        pct_min = int(params.get('PCT_MIN', 3))
        pct_max = int(params.get('PCT_MAX', 97))
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
                    calib_min=calib_min,
                    calib_max=calib_max,
                    pct_min=pct_min,
                    pct_max=pct_max,
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

            if not selected_wells:
                return jsonify({"error": "Tidak ada sumur yang dipilih."}), 400

            print(
                f"Memulai kalkulasi VSH untuk {len(selected_wells)} sumur...")

            # Ekstrak parameter dari frontend, dengan nilai default
            gr_ma = float(params.get('gr_ma', 30))
            gr_sh = float(params.get('gr_sh', 120))
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
                    df_well, input_log, gr_ma, gr_sh, output_log)

                # Simpan (overwrite) file CSV dengan data yang sudah diperbarui
                df_updated.to_csv(file_path, index=False)
                print(f"Hasil VSH untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi VSH berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

def dn_xplot(rho0, nphi0, rho_ma, rho_max, rho_fl):
    """(Internal) Menghitung porositas dan densitas matriks dari crossplot D-N."""
    try:
        phid = (rho_ma - rho0 * 1000) / (rho_ma - rho_fl)
        if nphi0 >= phid:
            pda = (rho_ma - rho_max) / (rho_ma - rho_fl)
            pna = 0.7 - 10 ** (-5 * nphi0 - 0.16)
        else:
            pda = 1.0
            pna = -2.06 * nphi0 - 1.17 + 10 ** (-16 * nphi0 - 0.4)

        denom = pda - pna
        if np.isclose(denom, 0) or np.isnan(denom):
            return np.nan, np.nan

        phix = (pda * nphi0 - phid * pna) / denom
        if np.isclose(1 - phix, 0) or np.isnan(phix):
            return np.nan, np.nan

        rma = (rho0 * 1000 - phix * rho_fl) / (1 - phix)
        return phix, rma
    except Exception:
        return np.nan, np.nan


def _klasifikasi_reservoir_numeric(phie):
    """(Internal) Memberikan KODE kelas reservoir berdasarkan nilai PHIE."""
    if pd.isna(phie):
        return 0  # NoData
    elif phie >= 0.20:
        return 4  # Prospek Kuat
    elif phie >= 0.15:
        return 3  # Zona Menarik
    elif phie >= 0.10:
        return 2  # Zona Lemah
    else:
        return 1  # Non Prospek


def calculate_porosity(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Menghitung berbagai jenis porositas berdasarkan parameter yang diberikan.
    """
    df_processed = df.copy()

    # Ekstrak parameter dengan nilai default
    RHO_FL = params.get('rho_fl', 1.00)
    RHO_W = params.get('rho_w', 1.00)
    RHO_SH = params.get('rho_sh', 2.45)
    RHO_DSH = params.get('rho_dsh', 2.60)
    NPHI_SH = params.get('nphi_sh', 0.35)
    PHIE_MAX = params.get('phie_max', 0.3)
    RHO_MA_BASE = params.get('rho_ma_base', 2.71) * 1000
    RHO_MAX = params.get('rho_max', 4.00) * 1000

    # Pastikan kolom VSH ada, jika tidak, hitung terlebih dahulu
    if 'VSH' not in df_processed.columns:
        # Asumsi VSH_GR adalah representasi VSH utama
        if 'VSH_GR' in df_processed.columns:
            df_processed['VSH'] = df_processed['VSH_GR']
        else:
            raise ValueError("Kolom VSH atau VSH_GR tidak ditemukan.")

    # Perhitungan
    PHIT_SH = (RHO_DSH - RHO_SH) / (RHO_DSH - RHO_W)
    df_processed["RHOB_SR"] = (
        df_processed["RHOB"] - df_processed["VSH"] * RHO_SH) / (1 - df_processed["VSH"])
    df_processed["NPHI_SR"] = (
        df_processed["NPHI"] - df_processed["VSH"] * NPHI_SH) / (1 - df_processed["VSH"])
    df_processed["NPHI_SR"] = df_processed["NPHI_SR"].clip(
        lower=-0.015, upper=1)

    phix_vals, rma_vals = [], []
    for i, row in df_processed.iterrows():
        if pd.notna(row["RHOB_SR"]) and pd.notna(row["NPHI_SR"]):
            phix, rma = dn_xplot(
                row["RHOB_SR"], row["NPHI_SR"], RHO_MA_BASE, RHO_MAX, RHO_FL * 1000)
        else:
            phix, rma = np.nan, np.nan
        phix_vals.append(phix)
        rma_vals.append(rma)

    df_processed["PHIE_DN"] = np.array(phix_vals) * (1 - df_processed["VSH"])
    df_processed["PHIT_DN"] = df_processed["PHIE_DN"] + \
        df_processed["VSH"] * PHIT_SH
    df_processed["PHIE"] = df_processed["PHIE_DN"].clip(
        lower=0, upper=PHIE_MAX * (1 - df_processed["VSH"]))
    df_processed["PHIT"] = df_processed["PHIE"] + df_processed["VSH"] * PHIT_SH
    df_processed["RHO_MAT"] = np.array(rma_vals) / 1000

    df_processed.rename(
        columns={"PHIE_DN": "PHIE_DEN", "PHIT_DN": "PHIT_DEN"}, inplace=True)

    df_processed["RESERVOIR_CLASS"] = df_processed["PHIE"].apply(
        _klasifikasi_reservoir_numeric)

    print("Kolom Porosity baru telah ditambahkan: PHIE, PHIT, PHIE_DN, RHO_MAT")
    return df_processed


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
                df_updated = calculate_porosity(df_well, params)

                # Simpan (overwrite) file CSV dengan data yang sudah diperbarui
                df_updated.to_csv(file_path, index=False)
                print(
                    f"Hasil Porositas untuk sumur '{well_name}' telah disimpan.")

            return jsonify({"message": f"Kalkulasi Porositas berhasil untuk {len(selected_wells)} sumur."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

def plot_two_features_simple(df_well, fig, axes, key, n_seq, counter, n_plots, log_scale=False):
    """
    Plot dua feature dengan dual x-axis untuk range yang berbeda.
    Feature kedua menggunakan x-axis overlay dengan garis putus-putus.
    """

    # Tambahkan axis info ke dictionary
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Plot kurva pertama (solid line) - menggunakan axis pertama
    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][0]],
            y=df_well[depth],
            line=dict(
                color=color_col[key][0],
                width=line_width
            ),
            name=data_col[key][0],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_seq),
            yaxis='y'+str(n_seq),
        )
    )

    # PERBAIKAN: Increment counter dan tambah axis kedua ke dictionary
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))

    # Plot kurva kedua (dashed line) - menggunakan axis kedua
    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][1]],
            y=df_well[depth],
            line=dict(
                color=color_col[key][1],
                width=line_width,
                # dash='dash'
            ),
            name=data_col[key][1],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_plots+counter),  # PERBAIKAN: Gunakan axis kedua
            yaxis='y'+str(n_seq),
        )
    )

    # Update axis layout untuk axis pertama
    xaxis1 = "xaxis"+str(n_seq)
    if log_scale:
        fig.update_layout(
            **{xaxis1: dict(
                side="top",
                type="log",
                range=[np.log10(range_col[key][0][0]),
                       np.log10(range_col[key][0][1])]
            )}
        )
    else:
        fig.update_layout(
            **{xaxis1: dict(
                side="top",
                range=range_col[key][0]
            )}
        )

    # Update axis layout untuk axis kedua (overlay)
    xaxis2 = "xaxis"+str(n_plots+counter)
    if log_scale and len(range_col[key]) > 1:
        fig.update_layout(
            **{xaxis2: dict(
                overlaying='x'+str(n_seq),
                side="top",
                type="log",
                range=[np.log10(range_col[key][1][0]),
                       np.log10(range_col[key][1][1])]
            )}
        )
    elif len(range_col[key]) > 1:
        fig.update_layout(
            **{xaxis2: dict(
                overlaying='x'+str(n_seq),
                side="top",
                range=range_col[key][1]
            )}
        )

    return fig, axes, counter


def plot_three_features_simple(df_well, fig, axes, key, n_seq, counter, n_plots, log_scale=False):
    """
    Plot tiga feature dengan triple x-axis untuk range yang berbeda.
    Feature kedua dan ketiga menggunakan x-axis overlay dengan garis putus-putus dan titik-titik.
    """

    # Tambahkan axis info ke dictionary untuk feature pertama
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Plot kurva pertama (solid line) - menggunakan axis pertama
    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][0]],
            y=df_well[depth],
            line=dict(
                color=color_col[key][0],
                width=line_width
            ),
            name=data_col[key][0],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_seq),
            yaxis='y'+str(n_seq),
        )
    )

    # PERBAIKAN: Increment counter dan tambah axis kedua ke dictionary
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))

    # Plot kurva kedua (dashed line) - menggunakan axis kedua
    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][1]],
            y=df_well[depth],
            line=dict(
                color=color_col[key][1],
                width=line_width,
                dash='dash'
            ),
            name=data_col[key][1],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_plots+counter),
            yaxis='y'+str(n_seq),
        )
    )

    # PERBAIKAN: Increment counter lagi dan tambah axis ketiga ke dictionary
    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))

    # Plot kurva ketiga (dotted line) - menggunakan axis ketiga
    fig.add_trace(
        go.Scattergl(
            x=df_well[data_col[key][2]],
            y=df_well[depth],
            line=dict(
                color=color_col[key][2],
                width=line_width,
                dash='dot'
            ),
            name=data_col[key][2],
            legend=legends[n_seq-1],
            showlegend=True,
            xaxis='x'+str(n_plots+counter),
            yaxis='y'+str(n_seq),
        )
    )

    # Update axis layout untuk axis pertama
    xaxis1 = "xaxis"+str(n_seq)
    if log_scale:
        fig.update_layout(
            **{xaxis1: dict(
                side="top",
                type="log",
                range=[np.log10(range_col[key][0][0]),
                       np.log10(range_col[key][0][1])]
            )}
        )
    else:
        fig.update_layout(
            **{xaxis1: dict(
                side="top",
                range=range_col[key][0]
            )}
        )

    # Update axis layout untuk axis kedua (overlay)
    # counter-1 karena sudah di-increment
    xaxis2 = "xaxis"+str(n_plots+counter-1)
    if log_scale and len(range_col[key]) > 1:
        fig.update_layout(
            **{xaxis2: dict(
                overlaying='x'+str(n_seq),
                side="top",
                type="log",
                range=[np.log10(range_col[key][1][0]),
                       np.log10(range_col[key][1][1])]
            )}
        )
    elif len(range_col[key]) > 1:
        fig.update_layout(
            **{xaxis2: dict(
                overlaying='x'+str(n_seq),
                side="top",
                range=range_col[key][1]
            )}
        )

    # Update axis layout untuk axis ketiga (overlay)
    xaxis3 = "xaxis"+str(n_plots+counter)
    if log_scale and len(range_col[key]) > 2:
        fig.update_layout(
            **{xaxis3: dict(
                overlaying='x'+str(n_seq),
                side="top",
                type="log",
                range=[np.log10(range_col[key][2][0]),
                       np.log10(range_col[key][2][1])]
            )}
        )
    elif len(range_col[key]) > 2:
        fig.update_layout(
            **{xaxis3: dict(
                overlaying='x'+str(n_seq),
                side="top",
                range=range_col[key][2]
            )}
        )

    return fig, axes, counter


def plot_four_features_simple(df_well, fig, axes, key, n_seq, counter, n_plots, log_scale=False):
    """
    Plot empat feature dengan semua garis solid (tidak putus-putus).
    Semua menggunakan overlay x-axis dengan range yang berbeda.
    """

    # Tambahkan axis info ke dictionary
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Plot semua 4 kurva dengan loop
    for i in range(4):
        # Plot kurva
        fig.add_trace(
            go.Scattergl(
                x=df_well[data_col[key][i]],
                y=df_well[depth],
                line=dict(
                    color=color_col[key][i],
                    width=line_width
                ),
                name=data_col[key][i],
                legend=legends[n_seq-1],
                showlegend=True,
                xaxis='x'+str(n_seq if i == 0 else n_plots+counter+i),
                yaxis='y'+str(n_seq),
            )
        )

        # Tambah axis ke dictionary (kecuali yang pertama, sudah ditambah di atas)
        if i > 0:
            axes[key].append('xaxis'+str(n_plots+counter+i))

    # Setup axis layout untuk semua 4 axis
    for i in range(4):
        if i == 0:
            # Axis pertama (main axis)
            xaxis_name = "xaxis"+str(n_seq)
            axis_config = dict(
                side="top",
                range=range_col[key][0]
            )
        else:
            # Axis overlay
            xaxis_name = "xaxis"+str(n_plots+counter+i)
            axis_config = dict(
                overlaying='x'+str(n_seq),
                side="top",
                range=range_col[key][i] if len(
                    range_col[key]) > i else range_col[key][0]
            )

        # Apply log scale jika diperlukan
        if log_scale and len(range_col[key]) > i:
            axis_config.update({
                "type": "log",
                "range": [np.log10(range_col[key][i][0]), np.log10(range_col[key][i][1])]
            })

        fig.update_layout(**{xaxis_name: axis_config})

    # Update counter
    # Tambah 3 karena menambah 3 axis baru (yang pertama sudah ada)
    counter += 3

    return fig, axes, counter

def plot_phie_den(df, df_marker, df_well_marker):
    """
    Membuat plot multi-panel untuk visualisasi hasil kalkulasi Porositas.
    """
    # FIX: Urutan plot disesuaikan dan RESERVOIR_CLASS dihapus
    sequence = ['MARKER', 'GR', 'RHOB',
                'PHIE_DEN', 'PHIT_DEN', 'RESERVOIR_CLASS']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}

    ratio_plots_seq = [ratio_plots.get(key, 1)
                       for key in plot_sequence.values()]
    subplot_col = len(plot_sequence)

    fig = make_subplots(
        rows=1, cols=subplot_col,
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.01
    )

    counter = 0
    axes = {key: [] for key in plot_sequence.values()}

    # Loop untuk memanggil plotter yang sesuai
    for n_seq, key in plot_sequence.items():
        if key == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, key, n_seq)
            fig, axes = plot_texts_marker(
                df_marker, df_well_marker['DEPTH'].max(), fig, axes, key, n_seq)
        elif key in ['GR', 'RHOB']:
            fig, axes = plot_line(
                df, fig, axes, base_key=key, n_seq=n_seq, col=key, label=key)
        elif key in ['PHIE_DEN', 'PHIT_DEN']:
            fig, axes, counter = plot_two_features_simple(
                df, fig, axes, key, n_seq, counter, n_plots=subplot_col)
        elif key == 'RESERVOIR_CLASS':
            fig, axes = plot_flag(df, fig, axes, key, n_seq)

    # Panggil fungsi-fungsi layout akhir
    fig = layout_range_all_axis(fig, axes, plot_sequence)
    fig = layout_draw_lines(fig, ratio_plots_seq, df,
                            xgrid_intv=50)  # Memberi grid interval 50
    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    # Atur layout global
    fig.update_layout(
        margin=dict(l=40, r=20, t=80, b=40),
        height=1600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified',
        title_text="Porosity from Density (RHOB)",
        title_x=0.5,
    )
    fig.update_yaxes(autorange='reversed', range=[
                     df[depth].max(), df[depth].min()])
    fig.update_traces(yaxis='y')

    return fig

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
            required_cols = ['VSH', 'PHIE', 'PHIT',
                             'PHIE_DEN', 'PHIT_DEN', 'RESERVOIR_CLASS']
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

def process_rgsa_for_well(df_well: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
    """
    Memproses RGSA untuk satu DataFrame sumur menggunakan parameter dinamis.
    """
    # --- STEP 1: SIAPKAN DATA & VALIDASI ---
    required_cols = ['DEPTH', 'GR', 'RT']
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom {required_cols} tidak lengkap.")
        return None

    df_rgsa = df_well[required_cols].dropna().copy()

    if len(df_rgsa) < 100:
        print(
            f"Peringatan: Data terlalu sedikit untuk regresi RGSA (hanya {len(df_rgsa)} baris)")
        return None

    # --- STEP 2: SLIDING WINDOW REGRESI DENGAN PARAMETER DINAMIS ---
    # Ambil nilai dari dictionary params, gunakan default jika tidak ada
    window_size = int(params.get('window_size', 106))
    step = int(params.get('step', 20))
    min_points_in_window = int(params.get('min_points_in_window', 30))
    # Ambil filter dari dictionary params
    filters = params.get('filters', {})
    gr_filter = filters.get('GR', (5, 180))
    rt_filter = filters.get('RT', (0.1, 1000))

    coeffs = []

    for start in range(0, len(df_rgsa) - window_size, step):
        window = df_rgsa.iloc[start:start+window_size]
        gr = window['GR'].values
        rt = window['RT'].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            rt > rt_filter[0]) & (rt < rt_filter[1])
        gr_filtered = gr[mask]
        rt_filtered = rt[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered
        log_rt = np.log10(rt_filtered)

        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = log_rt

        try:
            model = LinearRegression().fit(X, y)
            if hasattr(model, 'coef_') and len(model.coef_) == 3:
                coeffs.append({
                    'DEPTH': window['DEPTH'].mean(),
                    'b0': model.intercept_, 'b1': model.coef_[0],
                    'b2': model.coef_[1], 'b3': model.coef_[2]
                })
        except Exception as e:
            print(
                f"Peringatan: Gagal melakukan regresi pada kedalaman sekitar {window['DEPTH'].mean()}: {e}")
            continue

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return None

    coeff_df = pd.DataFrame(coeffs)

    # --- STEP 3: INTERPOLASI & HITUNG RGSA ---
    def interpolate_coeffs(depth):
        if depth <= coeff_df['DEPTH'].min():
            return coeff_df.iloc[0]
        if depth >= coeff_df['DEPTH'].max():
            return coeff_df.iloc[-1]
        lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
        upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]
        if upper['DEPTH'] == lower['DEPTH']:
            return lower
        weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
        return lower + weight * (upper - lower)

    rgsa_list = []
    for _, row in df_rgsa.iterrows():
        depth, gr = row['DEPTH'], row['GR']

        if not (gr_filter[0] < gr < gr_filter[1]):
            rgsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = interpolate_coeffs(
            depth)[['b0', 'b1', 'b2', 'b3']].values
        grfix = 0.01 * gr
        log_rgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        rgsa = 10**log_rgsa
        rgsa_list.append(rgsa)

    df_rgsa['RGSA'] = rgsa_list

    # --- STEP 4: GABUNGKAN HASIL DAN ANALISIS ---
    df_merged = pd.merge(
        df_well, df_rgsa[['DEPTH', 'RGSA']], on='DEPTH', how='left')

    if 'RT' in df_merged and 'RGSA' in df_merged:
        df_merged['GAS_EFFECT_RT'] = (df_merged['RT'] > df_merged['RGSA'])
        df_merged['RT_RATIO'] = df_merged['RT'] / df_merged['RGSA']
        df_merged['RT_DIFF'] = df_merged['RT'] - df_merged['RGSA']

    return df_merged

def process_all_wells_rgsa(df_well: pd.DataFrame, params: dict):
    """
    Fungsi orkestrator utama: memproses RGSA untuk semua sumur dengan parameter kustom.
    """

    unique_wells = df_well['WELL_NAME'].unique()
    print(f"📊 Memproses RGSA untuk {len(unique_wells)} sumur...")

    processed_wells = []
    failed_wells = []

    for i, well_name in enumerate(unique_wells, 1):
        print(f"\n🔍 Memproses sumur {i}/{len(unique_wells)}: {well_name}")
        df_well = df_well[df_well['WELL_NAME'] ==
                          well_name].copy().reset_index(drop=True)

        # Teruskan dictionary `params` ke fungsi pemrosesan
        result_df = process_rgsa_for_well(df_well, params)

        if result_df is not None:
            processed_wells.append(result_df)
            print(f"✅ {well_name} - RGSA berhasil dihitung.")
        else:
            failed_wells.append(well_name)
            print(f"❌ {well_name} - RGSA gagal dihitung.")

    if processed_wells:
        df_final = pd.concat(processed_wells, ignore_index=True)
        print(
            f"\n✅ Proses selesai. Hasil gabungan pada file asli.")

        # print_summary_statistics(df_final, len(processed_wells), failed_wells)
    else:
        print("\n❌ Tidak ada sumur yang berhasil diproses!")

    return df_final

def _interpolate_coeffs(depth, coeff_df):
    """(Internal) Melakukan interpolasi linear pada koefisien regresi."""
    if coeff_df.empty:
        return np.array([np.nan] * 4)

    target_cols = ['b0', 'b1', 'b2', 'b3']

    if depth <= coeff_df['DEPTH'].min():
        return coeff_df.iloc[0][target_cols].values
    if depth >= coeff_df['DEPTH'].max():
        return coeff_df.iloc[-1][target_cols].values

    lower = coeff_df[coeff_df['DEPTH'] <= depth].iloc[-1]
    upper = coeff_df[coeff_df['DEPTH'] > depth].iloc[0]

    if upper['DEPTH'] == lower['DEPTH']:
        return lower[target_cols].values

    weight = (depth - lower['DEPTH']) / (upper['DEPTH'] - lower['DEPTH'])
    interpolated = lower[target_cols] + weight * \
        (upper[target_cols] - lower[target_cols])

    return interpolated.values

def process_ngsa_for_well(df_well: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
    """
    Memproses NGSA untuk satu DataFrame sumur menggunakan parameter dinamis.
    """
    # --- STEP 1: SIAPKAN DATA & VALIDASI ---
    required_cols = ['DEPTH', 'GR', 'NPHI']
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom {required_cols} tidak lengkap.")
        return None

    df_ngsa = df_well[required_cols].dropna().copy()

    if len(df_ngsa) < 100:
        print(
            f"Peringatan: Data terlalu sedikit untuk regresi NGSA (hanya {len(df_ngsa)} baris)")
        return None

    # --- STEP 2: SLIDING WINDOW REGRESI DENGAN PARAMETER DINAMIS ---
    window_size = int(params.get('window_size', 106))
    step = int(params.get('step', 20))
    min_points_in_window = int(params.get('min_points_in_window', 30))
    filters = params.get('filters', {})
    gr_filter = filters.get('GR', (5, 180))
    nphi_filter = filters.get('NPHI', (0.05, 0.6))

    coeffs = []

    for start in range(0, len(df_ngsa) - window_size, step):
        window = df_ngsa.iloc[start:start+window_size]
        gr = window['GR'].values
        nphi = window['NPHI'].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            nphi > nphi_filter[0]) & (nphi < nphi_filter[1])
        gr_filtered = gr[mask]
        nphi_filtered = nphi[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered

        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        # Untuk NGSA, y adalah nilai NPHI langsung (bukan log)
        y = nphi_filtered

        try:
            model = LinearRegression().fit(X, y)
            if hasattr(model, 'coef_') and len(model.coef_) == 3:
                coeffs.append({
                    'DEPTH': window['DEPTH'].mean(),
                    'b0': model.intercept_, 'b1': model.coef_[0],
                    'b2': model.coef_[1], 'b3': model.coef_[2]
                })
        except Exception as e:
            print(
                f"Peringatan: Gagal melakukan regresi pada kedalaman sekitar {window['DEPTH'].mean()}: {e}")
            continue

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return None

    coeff_df = pd.DataFrame(coeffs)

    # --- STEP 3: INTERPOLASI & HITUNG NGSA ---
    ngsa_list = []
    for _, row in df_ngsa.iterrows():
        depth, gr = row['DEPTH'], row['GR']

        if not (gr_filter[0] < gr < gr_filter[1]):
            ngsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        ngsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        ngsa_list.append(ngsa)

    df_ngsa['NGSA'] = ngsa_list

    # --- STEP 4: GABUNGKAN HASIL DAN ANALISIS ---
    df_merged = pd.merge(
        df_well, df_ngsa[['DEPTH', 'NGSA']], on='DEPTH', how='left')

    if 'NPHI' in df_merged and 'NGSA' in df_merged:
        df_merged['GAS_EFFECT_NPHI'] = (df_merged['NPHI'] < df_merged['NGSA'])
        df_merged['NPHI_DIFF'] = df_merged['NGSA'] - df_merged['NPHI']

    return df_merged


def process_all_wells_ngsa(df_well: pd.DataFrame, params: dict):
    """
    Fungsi orkestrator utama: memproses NGSA untuk semua sumur dengan parameter kustom.
    """
    unique_wells = df_well['WELL_NAME'].unique()
    print(f"📊 Memproses NGSA untuk {len(unique_wells)} sumur...")

    processed_wells = []
    failed_wells = []

    for i, well_name in enumerate(unique_wells, 1):
        print(f"\n🔍 Memproses sumur {i}/{len(unique_wells)}: {well_name}")
        df_well = df_well[df_well['WELL_NAME'] ==
                          well_name].copy().reset_index(drop=True)

        result_df = process_ngsa_for_well(df_well, params)

        if result_df is not None:
            processed_wells.append(result_df)
            print(f"✅ {well_name} - NGSA berhasil dihitung.")
        else:
            failed_wells.append(well_name)
            print(f"❌ {well_name} - NGSA gagal dihitung.")

    if processed_wells:
        df_final = pd.concat(processed_wells, ignore_index=True)

        print(
            f"\n✅ Proses selesai. Hasil gabungan pada file asli.")

        # Menampilkan ringkasan statistik (opsional)
        # print_summary_statistics_ngsa(df_final, len(processed_wells), failed_wells)
    else:
        print("\n❌ Tidak ada sumur yang berhasil diproses!")

    return df_final

def process_all_wells_ngsa(df_well: pd.DataFrame, params: dict):
    """
    Fungsi orkestrator utama: memproses NGSA untuk semua sumur dengan parameter kustom.
    """
    unique_wells = df_well['WELL_NAME'].unique()
    print(f"📊 Memproses NGSA untuk {len(unique_wells)} sumur...")

    processed_wells = []
    failed_wells = []

    for i, well_name in enumerate(unique_wells, 1):
        print(f"\n🔍 Memproses sumur {i}/{len(unique_wells)}: {well_name}")
        df_well = df_well[df_well['WELL_NAME'] ==
                          well_name].copy().reset_index(drop=True)

        result_df = process_ngsa_for_well(df_well, params)

        if result_df is not None:
            processed_wells.append(result_df)
            print(f"✅ {well_name} - NGSA berhasil dihitung.")
        else:
            failed_wells.append(well_name)
            print(f"❌ {well_name} - NGSA gagal dihitung.")

    if processed_wells:
        df_final = pd.concat(processed_wells, ignore_index=True)

        print(
            f"\n✅ Proses selesai. Hasil gabungan pada file asli.")

        # Menampilkan ringkasan statistik (opsional)
        # print_summary_statistics_ngsa(df_final, len(processed_wells), failed_wells)
    else:
        print("\n❌ Tidak ada sumur yang berhasil diproses!")

    return df_final

def process_dgsa_for_well(df_well: pd.DataFrame, params: dict) -> Optional[pd.DataFrame]:
    """
    Memproses DGSA untuk satu DataFrame sumur menggunakan parameter dinamis.
    """
    # --- STEP 1: SIAPKAN DATA & VALIDASI ---
    required_cols = ['DEPTH', 'GR', 'RHOB']
    if not all(col in df_well.columns for col in required_cols):
        print(
            f"Peringatan: Melewatkan sumur karena kolom {required_cols} tidak lengkap.")
        return None

    df_dgsa = df_well[required_cols].dropna().copy()

    if len(df_dgsa) < 100:
        print(
            f"Peringatan: Data terlalu sedikit untuk regresi DGSA (hanya {len(df_dgsa)} baris)")
        return None

    # --- STEP 2: SLIDING WINDOW REGRESI DENGAN PARAMETER DINAMIS ---
    window_size = int(params.get('window_size', 106))
    step = int(params.get('step', 20))
    min_points_in_window = int(params.get('min_points_in_window', 30))
    filters = params.get('filters', {})
    gr_filter = filters.get('GR', (5, 180))
    rhob_filter = filters.get('RHOB', (1.5, 3.0))

    coeffs = []

    for start in range(0, len(df_dgsa) - window_size, step):
        window = df_dgsa.iloc[start:start+window_size]
        gr = window['GR'].values
        dens = window['RHOB'].values

        mask = (gr > gr_filter[0]) & (gr < gr_filter[1]) & (
            dens > rhob_filter[0]) & (dens < rhob_filter[1])
        gr_filtered = gr[mask]
        dens_filtered = dens[mask]

        if len(gr_filtered) < min_points_in_window:
            continue

        gr_scaled = 0.01 * gr_filtered

        X = np.vstack([gr_scaled, gr_scaled**2, gr_scaled**3]).T
        y = dens_filtered

        try:
            model = LinearRegression().fit(X, y)
            if hasattr(model, 'coef_') and len(model.coef_) == 3:
                coeffs.append({
                    'DEPTH': window['DEPTH'].mean(),
                    'b0': model.intercept_, 'b1': model.coef_[0],
                    'b2': model.coef_[1], 'b3': model.coef_[2]
                })
        except Exception as e:
            print(
                f"Peringatan: Gagal melakukan regresi pada kedalaman sekitar {window['DEPTH'].mean()}: {e}")
            continue

    if not coeffs:
        print("Peringatan: Tidak ada koefisien regresi yang berhasil dihitung.")
        return None

    coeff_df = pd.DataFrame(coeffs)

    # --- STEP 3: INTERPOLASI & HITUNG DGSA ---
    dgsa_list = []
    for _, row in df_dgsa.iterrows():
        depth, gr = row['DEPTH'], row['GR']

        if not (gr_filter[0] < gr < gr_filter[1]):
            dgsa_list.append(np.nan)
            continue

        b0, b1, b2, b3 = _interpolate_coeffs(depth, coeff_df)
        grfix = 0.01 * gr
        dgsa = b0 + b1*grfix + b2*grfix**2 + b3*grfix**3
        dgsa_list.append(dgsa)

    df_dgsa['DGSA'] = dgsa_list

    # --- STEP 4: GABUNGKAN HASIL DAN ANALISIS ---
    df_merged = pd.merge(
        df_well, df_dgsa[['DEPTH', 'DGSA']], on='DEPTH', how='left')

    if 'RHOB' in df_merged and 'DGSA' in df_merged:
        df_merged['GAS_EFFECT_RHOB'] = (df_merged['RHOB'] < df_merged['DGSA'])
        df_merged['DENS_DIFF'] = df_merged['DGSA'] - df_merged['RHOB']

    return df_merged

def process_all_wells_dgsa(df_well: str, params: dict):
    """
    Fungsi orkestrator utama: memproses DGSA untuk semua sumur dengan parameter kustom.
    """

    unique_wells = df_well['WELL_NAME'].unique()
    print(f"📊 Memproses DGSA untuk {len(unique_wells)} sumur...")

    processed_wells = []
    failed_wells = []

    for i, well_name in enumerate(unique_wells, 1):
        print(f"\n🔍 Memproses sumur {i}/{len(unique_wells)}: {well_name}")
        df_well = df_well[df_well['WELL_NAME'] ==
                          well_name].copy().reset_index(drop=True)

        result_df = process_dgsa_for_well(df_well, params)

        if result_df is not None:
            processed_wells.append(result_df)
            print(f"✅ {well_name} - DGSA berhasil dihitung.")
        else:
            failed_wells.append(well_name)
            print(f"❌ {well_name} - DGSA gagal dihitung.")

    if processed_wells:
        df_final = pd.concat(processed_wells, ignore_index=True)

        print(
            f"\n✅ Proses selesai. Hasil gabungan pada file asli.")

        # Menampilkan ringkasan statistik (opsional)
        # print_summary_statistics_dgsa(df_final, len(processed_wells), failed_wells)
    else:
        print("\n❌ Tidak ada sumur yang berhasil diproses!")

    return df_final

@app.route('/api/run-gsa-calculation', methods=['POST', 'OPTIONS'])
def run_gsa_calculation():
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

                df_well = pd.read_csv(file_path)

                # Panggil fungsi orkestrator GSA
                df_rgsa = process_all_wells_rgsa(df_well, params)
                df_ngsa = process_all_wells_ngsa(df_rgsa, params)
                df_dgsa = process_all_wells_dgsa(df_ngsa, params)

                # Validasi kolom penting
                required_cols = ['GR', 'RT', 'NPHI',
                                 'RHOB', 'RGSA', 'NGSA', 'DGSA']
                df_dgsa = df_dgsa.dropna(subset=required_cols)

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

def plot_gsa_main(df_well):
    """
    Fungsi utama untuk membuat plot komprehensif Gas Show Anomaly.
    """
    # Lakukan pra-pemrosesan data yang diperlukan untuk plot ini
    df_well = normalize_xover(df_well, 'NPHI', 'RHOB')
    df_well = normalize_xover(df_well, 'RT', 'RHOB')
    df_well = normalize_xover(df_well, 'RT', 'RGSA')
    df_well = normalize_xover(df_well, 'NPHI', 'NGSA')
    df_well = normalize_xover(df_well, 'RHOB', 'DGSA')

    zona_mapping = {
        'Zona Prospek Kuat': 3,
        'Zona Menarik': 2,
        'Zona Lemah': 1,
        'Non Prospek': 0
    }

    df_well['ZONA'] = df_well['ZONA'].map(zona_mapping)
    df_marker = extract_markers_with_mean_depth(df_well)
    df_well_marker = df_well.copy()

    # Definisikan urutan track untuk plot GSA
    sequence = ['MARKER', 'GR', 'RT_RHOB', 'NPHI_RHOB',
                'RT_RGSA', 'NPHI_NGSA', 'RHOB_DGSA', 'ZONA']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}

    ratio_plots_seq = [ratio_plots.get(key, 1)
                       for key in plot_sequence.values()]
    subplot_col = len(plot_sequence)

    fig = make_subplots(
        rows=1, cols=subplot_col,
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.01
    )

    counter = 0
    axes = {key: [] for key in plot_sequence.values()}

    # Loop untuk memanggil plotter yang sesuai untuk setiap track
    for n_seq, key in plot_sequence.items():
        if key == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, key, n_seq)
            fig, axes = plot_texts_marker(
                df_marker, df_well_marker['DEPTH'].max(), fig, axes, key, n_seq)
        elif key == 'GR':
            fig, axes = plot_line(df_well, fig, axes, key, n_seq)
        elif key in ['NPHI_RHOB', 'RT_RHOB']:
            fig, axes, counter = plot_xover_log_normal(
                df_well, fig, axes, key, n_seq, counter, subplot_col)
        elif key in ['RT_RGSA', 'NPHI_NGSA', 'RHOB_DGSA']:
            fig, axes, counter = plot_gsa_crossover(
                df_well, fig, axes, key, n_seq, counter, subplot_col)
        elif key == 'ZONA':
            fig, axes = plot_flag(df_well, fig, axes, key, n_seq)

    # Finalisasi Layout
    fig = layout_range_all_axis(fig, axes, plot_sequence)
    fig = layout_draw_lines(fig, ratio_plots_seq, df_well, xgrid_intv=50)
    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    fig.update_layout(
        title_text="Gas Show Anomaly (GSA) Analysis",
        yaxis=dict(autorange='reversed', range=[
                   df_well[depth].max(), df_well[depth].min()]),
        hovermode='y unified',
        template='plotly_white',
        showlegend=False,
        height=1600,
    )

    return fig

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

def trim_well_log(df, bottom_depth=None, top_depth=None, required_columns=None, mode='AUTO'):
    """
    Trim data well log berdasarkan validitas log dan (opsional) batas DEPTH dari user.

    Parameters:
    - df: DataFrame sumur
    - bottom_depth: batas bawah DEPTH (opsional)
    - top_depth: batas atas DEPTH (opsional)
    - required_columns: list kolom yang dicek validitasnya (default ['GR', 'RT', 'NPHI', 'RHOB'])

    Returns:
    - DataFrame yang sudah di-trim
    """
    if required_columns is None:
        required_columns = ['GR', 'RT', 'NPHI', 'RHOB']

    # Validasi data berdasarkan required_columns
    valid_index = {}
    for col in required_columns:
        valid_index[col] = df[(df[col] != -999.0) & (df[col].isna())].index

    # Hitung batas default jika tidak ada input user
    start_idx = min(idx.min()
                    for idx in valid_index.values())
    end_idx = max(idx.max()
                  for idx in valid_index.values())

    depth_min = df['DEPTH'].min()
    depth_max = df['DEPTH'].max()

    # Konversi input ke float jika diberikan
    try:
        if bottom_depth is not None:
            bottom_depth = float(bottom_depth)
        if top_depth is not None:
            top_depth = float(top_depth)
    except Exception as e:
        raise ValueError(f"Gagal mengonversi DEPTH input ke float: {e}")

    # Logika trimming berdasarkan mode
    if mode == 'AUTO':
        df = df.loc[start_idx:end_idx].copy()

    elif mode == 'UNTIL_TOP':
        if bottom_depth is None:
            raise ValueError("BOTTOM_DEPTH harus diisi pada mode UNTIL_TOP")
        df = df[~((df['DEPTH'] >= depth_min) & (
            df['DEPTH'] <= bottom_depth))].copy()

    elif mode == 'UNTIL_BOTTOM':
        if top_depth is None:
            raise ValueError("TOP_DEPTH harus diisi pada mode UNTIL_BOTTOM")
        df = df[~((df['DEPTH'] >= top_depth) & (
            df['DEPTH'] <= depth_max))].copy()

    elif mode == 'CUSTOM':
        if top_depth is None or bottom_depth is None:
            raise ValueError(
                "TOP_DEPTH dan BOTTOM_DEPTH harus diisi untuk mode CUSTOM")
        if top_depth > bottom_depth:
            raise ValueError(
                "TOP_DEPTH tidak boleh lebih besar dari BOTTOM_DEPTH")
        df = df[~((df['DEPTH'] >= top_depth) & (
            df['DEPTH'] <= bottom_depth))].copy()

    else:
        raise ValueError(f"Mode tidak dikenali: {mode}")

    return df

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
                    return jsonify({'error': 'BOTTOM_DEPTH harus diisi untuk mode UNTIL_TOP'}), 400
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


@app.route('/')
def home():
    return "Flask backend is running!"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
