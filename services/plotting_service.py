# @title
import pandas as pd
import numpy as np
import random
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from services.iqual import calculate_iqual

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
    'ZONE': ['ZONE'],
    'GR': ['GR'],
    'GR_NORM': ['GR_NORM'],
    'GR_DUAL': ['GR', 'GR_NORM'],
    'GR_DUAL_2': ['GR_RAW', 'GR_NORM'],
    'GR_RAW_NORM': ['GR_RAW_NORM'],
    'GR_SM': ['GR_SM'],
    'GR_MovingAvg_5': ['GR_MovingAvg_5'],
    'GR_MovingAvg_10': ['GR_MovingAvg_10'],
    'RT': ['RT'],
    'RT_RO': ['RT', 'RO'],
    'X_RT_RO': ['RT_RO'],
    'NPHI_RHOB_NON_NORM': ['NPHI', 'RHOB'],
    'RHOB': ['RHOB'],
    'NPHI_RHOB': ['NPHI', 'RHOB', 'NPHI_NORM', 'RHOB_NORM_NPHI'],
    'SW': ['SW'],
    'SW_SIMANDOUX': ['SW_SIMANDOUX'],
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
    'VSH_GR_DN': ['VSH_LINEAR', 'VSH_DN'],
    'PHIE_DEN': ['PHIE', 'PHIE_DEN'],
    'PHIT_DEN': ['PHIT', 'PHIT_DEN'],
    'PHIE_PHIT': ['PHIE', 'PHIT'],
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
    'TGC': ['C3', 'C2', 'C1', 'TG'],
    'TG_SUMC': ['TG_SUMC'],
    'C3_C1': ['C3_C1'],
    'C3_C1_BASELINE': ['C3_C1_BASELINE'],
    'GR_CAL': ['GR_CAL'],
    'RLA5': ['RLA5'],
    'R39PC': ['R39PC'],
    'RHOZ': ['RHOZ'],
    'TNPH': ['TNPH'],
    'A40H': ['A40H'],
    'ROBB': ['ROBB'],
    'DGRCC': ['DGRCC'],
    'ARM48PC': ['ARM48PC'],
    'ALCDLC': ['ALCDLC'],
    'TNPL': ['TNPL'],
    # Normalization (_NO)
    'GR_CAL_NO': ['GR_CAL_NO'], 'DGRCC_NO': ['DGRCC_NO'],
    # Trimmed (_TR)
    'GR_CAL_TR': ['GR_CAL_TR'], 'DGRCC_TR': ['DGRCC_TR'], 'RLA5_TR': ['RLA5_TR'],
    'R39PC_TR': ['R39PC_TR'], 'A40H_TR': ['A40H_TR'], 'ARM48PC_TR': ['ARM48PC_TR'],
    'RHOZ_TR': ['RHOZ_TR'], 'ALCDLC_TR': ['ALCDLC_TR'], 'ROBB_TR': ['ROBB_TR'],
    'TNPH_TR': ['TNPH_TR'], 'TNPL_TR': ['TNPL_TR'],
    # Smoothed (_SM)
    'GR_CAL_SM': ['GR_CAL_SM'], 'DGRCC_SM': ['DGRCC_SM'], 'RLA5_SM': ['RLA5_SM'],
    'R39PC_SM': ['R39PC_SM'], 'A40H_SM': ['A40H_SM'], 'ARM48PC_SM': ['ARM48PC_SM'],
    'RHOZ_SM': ['RHOZ_SM'], 'ALCDLC_SM': ['ALCDLC_SM'], 'ROBB_SM': ['ROBB_SM'],
    'TNPH_SM': ['TNPH_SM'], 'TNPL_SM': ['TNPL_SM'],
    # Filled Missing (_FM)
    'GR_CAL_FM': ['GR_CAL_FM'], 'DGRCC_FM': ['DGRCC_FM'], 'RLA5_FM': ['RLA5_FM'],
    'R39PC_FM': ['R39PC_FM'], 'A40H_FM': ['A40H_FM'], 'ARM48PC_FM': ['ARM48PC_FM'],
    'RHOZ_FM': ['RHOZ_FM'], 'ALCDLC_FM': ['ALCDLC_FM'], 'ROBB_FM': ['ROBB_FM'],
    'TNPH_FM': ['TNPH_FM'], 'TNPL_FM': ['TNPL_FM'],
}


unit_col = {
    'DNS': [''],
    'MARKER': [''],
    'ZONE': [''],
    'GR_NORM': ['GAPI'],
    'GR': ['GAPI'],
    'GR_DUAL': ['GAPI', 'GAPI'],
    'GR_DUAL_2': ['GAPI', 'GAPI'],
    'GR_RAW_NORM': ['GAPI'],
    'GR_SM': ['GAPI'],
    'GR_MovingAvg_5': ['GAPI'],
    'GR_MovingAvg_10': ['GAPI'],
    'RT': ['OHMM'],
    'RT_RO': ['OHMM', 'OHMM'],
    'X_RT_RO': ['V/V'],
    'NPHI_RHOB_NON_NORM': ['V/V', 'G/C3'],
    'NPHI_RHOB': ['V/V', 'G/C3', 'V/V', 'G/C3'],
    'RHOB': ['G/C3'],
    'SW': ['V/V'],
    'SW': ['V/V'],
    'SW_SIMANDOUX': ['V/V'],
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
    'VSH_GR_DN': ['V/V', 'V/V'],
    'PHIE_DEN': ['', ''],
    'PHIT_DEN': ['', ''],
    'PHIE_PHIT': ['', ''],
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
    'TGC': ['PPM', 'PPM', 'PPM', 'PPM'],
    'TG_SUMC': ['PPM'],
    'C3_C1': ['PPM'],
    'C3_C1_BASELINE': ['PPM'],
    'GR_CAL': ['GAPI'],        # Sama seperti GR
    'DGRCC': ['GAPI'],         # Sama seperti GR
    'RLA5': ['OHMM'],          # Sama seperti RT
    'A40H': ['OHMM'],          # Sama seperti RT
    'ARM48PC': ['OHMM'],       # Sama seperti RT
    'R39PC': ['OHMM'],       # Sama seperti RT
    'RHOZ': ['G/C3'],          # Sama seperti RHOB
    'ALCDLC': ['G/C3'],        # Sama seperti RHOB
    'ROBB': ['G/C3'],          # Sama seperti RHOB
    'TNPL': ['V/V'],           # Sama seperti NPHI
    'TNPH': ['V/V'],           # Sama seperti NPHI
    # Normalization (_NO)
    'GR_CAL_NO': ['GAPI'], 'DGRCC_NO': ['GAPI'],
    # Trimmed (_TR)
    'GR_CAL_TR': ['GAPI'], 'DGRCC_TR': ['GAPI'], 'RLA5_TR': ['OHMM'], 'R39PC_TR': ['OHMM'],
    'A40H_TR': ['OHMM'], 'ARM48PC_TR': ['OHMM'], 'RHOZ_TR': ['G/C3'], 'ALCDLC_TR': ['G/C3'],
    'ROBB_TR': ['G/C3'], 'TNPH_TR': ['V/V'], 'TNPL_TR': ['V/V'],
    # Smoothed (_SM)
    'GR_CAL_SM': ['GAPI'], 'DGRCC_SM': ['GAPI'], 'RLA5_SM': ['OHMM'], 'R39PC_SM': ['OHMM'],
    'A40H_SM': ['OHMM'], 'ARM48PC_SM': ['OHMM'], 'RHOZ_SM': ['G/C3'], 'ALCDLC_SM': ['G/C3'],
    'ROBB_SM': ['G/C3'], 'TNPH_SM': ['V/V'], 'TNPL_SM': ['V/V'],
    # Filled Missing (_FM)
    'GR_CAL_FM': ['GAPI'], 'DGRCC_FM': ['GAPI'], 'RLA5_FM': ['OHMM'], 'R39PC_FM': ['OHMM'],
    'A40H_FM': ['OHMM'], 'ARM48PC_FM': ['OHMM'], 'RHOZ_FM': ['G/C3'], 'ALCDLC_FM': ['G/C3'],
    'ROBB_FM': ['G/C3'], 'TNPH_FM': ['V/V'], 'TNPL_FM': ['V/V'],
}


color_col = {
    'DNS': ['darkgreen'],
    'MARKER': [colors_dict['black']],
    'ZONE': [colors_dict['black']],
    'GR_NORM': ['orange'],
    'GR_DUAL': ['darkgreen', 'orange'],
    'GR_DUAL_2': ['darkgreen', 'orange'],
    'GR_RAW_NORM': ['orange'],
    'GR_SM': ['darkgreen'],
    'GR_MovingAvg_5': ['darkgreen'],
    'GR_MovingAvg_10': ['darkgreen'],
    'GR': ['darkblue'],
    'RT': [colors_dict['red']],
    'RT_RO': [colors_dict['red'], colors_dict['purple']],
    'X_RT_RO': [colors_dict['black']],
    'NPHI_RHOB_NON_NORM': ['darkgreen', colors_dict['red']],
    'NPHI_RHOB': ['darkgreen', colors_dict['red'], colors_dict['blue'], colors_dict['red'],],
    'RHOB': [colors_dict['red']],
    'SW': ['darkgreen'],
    'SW_SIMANDOUX': ['darkgreen'],
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
    'VSH_LINEAR': ['darkgreen'],
    'VSH_DN': ['darkgreen'],
    'VSH_SP': ['darkgreen'],
    'VSH_GR_DN': ['darkgreen', 'red'],
    'PHIE_DEN': ['darkblue', colors_dict['blue']],
    'PHIT_DEN': [colors_dict['red'], colors_dict['orange']],
    'PHIE_PHIT': ['darkblue', colors_dict['red']],
    'RESERVOIR_CLASS': [colors_dict['black']],
    'RWA': ['darkblue', 'darkgreen', colors_dict['red']],
    'PHIE': ['darkgreen'],
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
    'TGC': ['lightgreen', 'blue', 'red', colors_dict['black']],
    'TG_SUMC': ['red'],
    'C3_C1': ['blue'],
    'C3_C1_BASELINE': [colors_dict['black']],
    'GR_CAL': ['darkblue'],         # Sama seperti GR
    'DGRCC': ['darkblue'],          # Sama seperti GR
    'RLA5': [colors_dict['red']],   # Sama seperti RT
    'A40H': [colors_dict['red']],   # Sama seperti RT
    'ARM48PC': [colors_dict['red']],  # Sama seperti RT
    'R39PC': [colors_dict['red']],  # Sama seperti RT
    'RHOZ': [colors_dict['red']],   # Sama seperti RHOB
    'ALCDLC': [colors_dict['red']],  # Sama seperti RHOB
    'ROBB': [colors_dict['red']],   # Sama seperti RHOB
    'TNPL': ['darkgreen'],          # Sama seperti NPHI
    'TNPH': ['darkgreen'],          # Sama seperti NPHI
    # Normalization (_NO) - Oranye
    'GR_CAL_NO': ['orange'], 'DGRCC_NO': ['orange'],
    # Trimmed (_TR) - Coklat
    'GR_CAL_TR': ['saddlebrown'], 'DGRCC_TR': ['saddlebrown'], 'RLA5_TR': ['saddlebrown'], 'R39PC_TR': ['saddlebrown'],
    'A40H_TR': ['saddlebrown'], 'ARM48PC_TR': ['saddlebrown'], 'RHOZ_TR': ['saddlebrown'], 'ALCDLC_TR': ['saddlebrown'],
    'ROBB_TR': ['saddlebrown'], 'TNPH_TR': ['saddlebrown'], 'TNPL_TR': ['saddlebrown'],
    # Smoothed (_SM) - Ungu
    'GR_CAL_SM': ['purple'], 'DGRCC_SM': ['purple'], 'RLA5_SM': ['purple'], 'R39PC_SM': ['purple'],
    'A40H_SM': ['purple'], 'ARM48PC_SM': ['purple'], 'RHOZ_SM': ['purple'], 'ALCDLC_SM': ['purple'],
    'ROBB_SM': ['purple'], 'TNPH_SM': ['purple'], 'TNPL_SM': ['purple'],
    # Filled Missing (_FM) - Magenta
    'GR_CAL_FM': ['magenta'], 'DGRCC_FM': ['magenta'], 'RLA5_FM': ['magenta'], 'R39PC_FM': ['magenta'],
    'A40H_FM': ['magenta'], 'ARM48PC_FM': ['magenta'], 'RHOZ_FM': ['magenta'], 'ALCDLC_FM': ['magenta'],
    'ROBB_FM': ['magenta'], 'TNPH_FM': ['magenta'], 'TNPL_FM': ['magenta'],
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
        1: 'orange',
        1: 'orange',
    }

}

range_col = {
    'GR': [[0, 250]],
    'GR_NORM': [[0, 250]],
    'GR_DUAL': [[0, 250], [0, 250]],
    'GR_DUAL_2': [[0, 250], [0, 250]],
    'GR_RAW_NORM': [[0, 250]],
    'GR_SM': [[0, 250]],
    'GR_MovingAvg_5': [[0, 250]],
    'GR_MovingAvg_10': [[0, 250]],
    'RT': [[0.2, 2000]],
    'RT_RO': [[0.02, 2000], [0.02, 2000]],
    'X_RT_RO': [[0, 4]],
    'NPHI_RHOB_NON_NORM': [[0.6, 0], [1.71, 2.71]],
    'NPHI_RHOB': [[0.6, 0], [1.71, 2.71], [1, 0], [1, 0]],
    'RHOB': [[1.71, 2.71]],
    'SW': [[1, 0]],
    'SW_SIMANDOUX': [[1, 0]],
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
    'VSH_GR_DN': [[0, 1], [0, 1]],
    'PHIE_DEN': [[0, 1], [0, 1]],
    'PHIT_DEN': [[0, 1], [0, 1]],
    'PHIE_PHIT': [[0, 1], [0, 1]],
    'RWA': [[0.02, 100], [0.02, 100], [0.02, 100]],
    'PHIE': [[0.5, 0]],  # perbaiki seluruh PHIE dan PHIT
    'RT_GR': [[0.02, 2000], [0, 250], [0.02, 2000], [0, 250]],
    # 'RT_PHIE':[[0.02, 2000],[0.6,0],[0.02, 2000],[0.6,0]],
    'RT_PHIE': [[0.02, 2000], [0.6, 0]],
    'SWARRAY': [[1, 0], [1, 0], [1, 0], [1, 0]],
    'SWGRAD': [[0, 0.1]],
    'DNS': [[-0.5, 0.5]],
    'DNSV': [[-1, 1]],
    'RGBE': [[10, -10]],
    'RPBE': [[-10, 10]],
    'TGC': [[100, 100000], [100, 100000], [100, 100000], [100, 100000]],
    'TG_SUMC': [[0, 2]],
    'C3_C1': [[0.00001, 0.08]],
    'C3_C1_BASELINE': [[0.00001, 0.08]],
    'IQUAL': [[0, 1]],
    'GR_CAL': [[0, 250]],         # Sama seperti GR
    'DGRCC': [[0, 250]],          # Sama seperti GR
    'RLA5': [[0.02, 100]],        # Sama seperti RT
    'R39PC': [[0.02, 100]],        # Sama seperti RT
    'A40H': [[0.02, 100]],        # Sama seperti RT
    'ARM48PC': [[0.02, 100]],    # Sama seperti RT
    'RHOZ': [[1.71, 2.71]],      # Sama seperti RHOB
    'ALCDLC': [[1.71, 2.71]],     # Sama seperti RHOB
    'ROBB': [[1.71, 2.71]],       # Sama seperti RHOB
    'TNPL': [[0.6, 0]],           # Sama seperti NPHI
    'TNPH': [[0.6, 0]],           # Sama seperti NPHI
    'GR_CAL_NO': [[0, 250]], 'DGRCC_NO': [[0, 250]],
    'GR_CAL_TR': [[0, 250]], 'DGRCC_TR': [[0, 250]], 'RLA5_TR': [[0.2, 2000]],
    'R39PC_TR': [[0.2, 2000]], 'A40H_TR': [[0.2, 2000]], 'ARM48PC_TR': [[0.2, 2000]],
    'RHOZ_TR': [[1.65, 2.65]], 'ALCDLC_TR': [[1.65, 2.65]], 'ROBB_TR': [[1.65, 2.65]],
    'TNPH_TR': [[0.6, -0.1]], 'TNPL_TR': [[0.6, -0.1]],
    'GR_CAL_SM': [[0, 250]], 'DGRCC_SM': [[0, 250]], 'RLA5_SM': [[0.2, 2000]],
    'R39PC_SM': [[0.2, 2000]], 'A40H_SM': [[0.2, 2000]], 'ARM48PC_SM': [[0.2, 2000]],
    'RHOZ_SM': [[1.65, 2.65]], 'ALCDLC_SM': [[1.65, 2.65]], 'ROBB_SM': [[1.65, 2.65]],
    'TNPH_SM': [[0.6, -0.1]], 'TNPL_SM': [[0.6, -0.1]],
    'GR_CAL_FM': [[0, 250]], 'DGRCC_FM': [[0, 250]], 'RLA5_FM': [[0.2, 2000]],
    'R39PC_FM': [[0.2, 2000]], 'A40H_FM': [[0.2, 2000]], 'ARM48PC_FM': [[0.2, 2000]],
    'RHOZ_FM': [[1.65, 2.65]], 'ALCDLC_FM': [[1.65, 2.65]], 'ROBB_FM': [[1.65, 2.65]],
    'TNPH_FM': [[0.6, -0.1]], 'TNPL_FM': [[0.6, -0.1]],
}

ratio_plots = {
    'MARKER': 0.2,
    'ZONE': 0.3,
    'GR': 1,
    'GR_NORM': 1,
    'GR_DUAL': 1,
    'GR_DUAL_2': 1,
    'GR_RAW_NORM': 1,
    'GR_SM': 1,
    'GR_MovingAvg_5': 1,
    'GR_MovingAvg_10': 1,
    'RT': 1,
    'RT': 1,
    'RT_RO': 1,
    'X_RT_RO': 0.5,
    'NPHI_RHOB_NON_NORM': 1,
    'NPHI_RHOB': 1,
    'RHOB': 1,
    'SW': 1,
    'SW_SIMANDOUX': 1,
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
    'VSH_GR_DN': 1,
    'PHIE_DEN': 1,
    'PHIT_DEN': 1,
    'PHIE_PHIT': 1,
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
    'TGC': 1,
    'TG_SUMC': 1,
    'C3_C1': 1,
    'C3_C1_BASELINE': 1,
    'GR_CAL': 1,        # Sama seperti GR
    'DGRCC': 1,         # Sama seperti GR
    'RLA5': 1,          # Sama seperti RT
    'R39PC': 1,          # Sama seperti RT
    'A40H': 1,          # Sama seperti RT
    'ARM48PC': 1,       # Sama seperti RT
    'RHOZ': 1,          # Sama seperti RHOB
    'ALCDLC': 1,        # Sama seperti RHOB
    'ROBB': 1,          # Sama seperti RHOB
    'TNPL': 1,           # Sama seperti NPHI
    'TNPH': 1,           # Sama seperti NPHI
    'GR_CAL_NO': 1, 'DGRCC_NO': 1,
    'GR_CAL_TR': 1, 'DGRCC_TR': 1, 'RLA5_TR': 1, 'R39PC_TR': 1, 'A40H_TR': 1,
    'ARM48PC_TR': 1, 'RHOZ_TR': 1, 'ALCDLC_TR': 1, 'ROBB_TR': 1, 'TNPH_TR': 1, 'TNPL_TR': 1,
    'GR_CAL_SM': 1, 'DGRCC_SM': 1, 'RLA5_SM': 1, 'R39PC_SM': 1, 'A40H_SM': 1,
    'ARM48PC_SM': 1, 'RHOZ_SM': 1, 'ALCDLC_SM': 1, 'ROBB_SM': 1, 'TNPH_SM': 1, 'TNPL_SM': 1,
    'GR_CAL_FM': 1, 'DGRCC_FM': 1, 'RLA5_FM': 1, 'R39PC_FM': 1, 'A40H_FM': 1,
    'ARM48PC_FM': 1, 'RHOZ_FM': 1, 'ALCDLC_FM': 1, 'ROBB_FM': 1, 'TNPH_FM': 1, 'TNPL_FM': 1,
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
    'X_RT_RHOB': 0.02,
    'VSH_LINEAR': 0.5,
    'VSH': 0.5,
    'PHIE': 0.1,
    'RGBE': 0,
    'RPBE': 0,
    'SW': 0.7,
}

line_width = 0.9

# ----------------------------- Plot Function ------------------------------


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


def fillcol(label, yc='rgba(0,250,0,0.4)', nc='rgba(250,0,0,0)'):
    if label >= 1:
        return yc
    else:
        return nc


def fillcol_dual(label, data_value, threshold, above_color='green', below_color='yellow', nc='rgba(250,0,0,0)'):
    """
    Modified fillcol function to handle dual threshold colors based on original fillcol pattern

    Parameters:
    - label: original label (same as original fillcol)
    - data_value: current data value to compare with threshold
    - threshold: threshold value for comparison
    - above_color: color for values above threshold (yc equivalent)
    - below_color: color for values below threshold (yc equivalent)
    - nc: neutral color for label < 1 (same as original)
    """
    if label == 1:  # Above threshold
        return above_color
    elif label == 0:  # Below threshold
        return below_color
    else:
        return nc


def xover_label_df(df_well, key, type=1):
    if key in ['X_RT_RO', 'X_RWA_RW', 'X_RT_F', 'X_RT_RHOB', 'VSH_LINEAR', 'PHIE', 'RGBE', 'RPBE', 'SW', 'VSH']:
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
    # Sebelum return xover_dfs

    return xover_dfs


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
    if log_scale:
        # Periksa dan perbaiki rentang untuk sumbu pertama
        if range_col[key][0][0] <= 0:
            range_col[key][0][0] = 0.01  # Ganti 0 dengan nilai positif kecil

        # Periksa dan perbaiki rentang untuk sumbu kedua (jika ada)
        if len(range_col[key]) > 1 and range_col[key][1][0] <= 0:
            range_col[key][1][0] = 0.01

        # Periksa dan perbaiki rentang untuk sumbu ketiga (jika ada)
        if len(range_col[key]) > 2 and range_col[key][2][0] <= 0:
            range_col[key][2][0] = 0.01

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


def plot_xover(df_well, fig, axes, key, n_seq, counter, n_plots, y_color='limegreen', n_color='lightgray'):
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
                x=xover_df[data_col[key][1]],
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

    col = data_col[key][1]
    fig.add_trace(
        go.Scattergl(
            x=df_well[col],
            y=df_well[depth],
            line=dict(color=color_col[key][1], width=line_width),
            name=col,
            legend=legends[n_seq-1],
            showlegend=True,
        ),
        row=1, col=n_seq,
    )

    axis = 'xaxis'+str(n_seq)
    fig.update_layout(
        **{axis: dict(
            side="top",
            type="log",
            range=[np.log10(range_col[key][0][0]),
                   np.log10(range_col[key][0][1])]
        )})

    counter += 1
    axes[key].append('xaxis'+str(n_plots+counter))
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            line=dict(color="rgba(0,0,0,0)", width=0),
            name=col,
            legend=legends[n_seq-1],
            showlegend=False,
            hoverinfo="skip",
            xaxis='x'+str(n_plots+counter),
            yaxis='y'+str(n_seq),
        ),
    )

    axis = 'xaxis'+str(n_plots+counter)
    fig.update_layout(
        **{axis: dict(
            overlaying='x'+str(n_seq),
            side="top",
            type="log",
            range=[np.log10(range_col[key][0][0]),
                   np.log10(range_col[key][0][1])]
        )})

    return fig, axes, counter


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


def plot_xover_thres_dual(df_well, fig, axes, key, n_seq, counter,
                          above_thres_color='green',
                          below_thres_color='yellow',
                          n_color='rgba(250,0,0,0)'):
    """
    Plot function with dual colors for values above and below threshold

    Parameters:
    - above_thres_color: color for values above threshold
    - below_thres_color: color for values below threshold
    - n_color: neutral color (existing parameter)
    """
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Plot Area Xover with dual threshold colors using fillcol_dual
    xover_dfs = xover_label_df(df_well, key, type=0)

    for xover_df in xover_dfs:
        # Plot base trace (invisible)
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

        # Plot with fillcol_dual for dual threshold coloring
        fig.add_traces(
            go.Scatter(
                x=xover_df['thres'],
                y=xover_df[depth],
                line=dict(color='rgba(0,0,0,0)', width=0),
                name='xover',
                fill='tonextx',
                showlegend=False,
                fillcolor=fillcol_dual(
                    xover_df['label'].iloc[0],
                    xover_df[data_col[key][0]].iloc[0],
                    thres[key],
                    above_thres_color,
                    below_thres_color,
                    n_color
                ),
                xaxis='x'+str(n_seq),
                yaxis='y'+str(n_seq),
                hoverinfo="skip"
            ),
        )

    # Plot main data line
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

    # Plot threshold line
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

    # Update axis layout
    axis = 'xaxis'+str(n_seq)
    fig.update_layout(
        **{axis: dict(
            side="top",
            range=range_col[key][0]
        )}
    )

    return fig, axes, counter


def plot_xover_bar_horizontal(df_well, fig, axes, key, n_seq, counter,
                              above_thres_color='darkgreen',
                              below_thres_color='lightblue'):
    """
    Plot horizontal bar chart style for RGBE / RPBE values with working hover.
    """
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    # Ambil data valid
    xover_df = df_well[[depth, data_col[key][0]]].copy().dropna()
    xover_df['value'] = xover_df[data_col[key][0]]
    xover_df['label'] = np.where(xover_df['value'] > thres[key], 1, 0)
    xover_df['color'] = np.where(
        xover_df['label'] == 1, above_thres_color, below_thres_color)

    # Buat bar chart horizontal
    fig.add_trace(
        go.Bar(
            x=xover_df['value'],
            y=xover_df[depth],
            orientation='h',
            marker=dict(color=xover_df['color']),
            # width=1.2,
            text=None,
            hovertemplate=f"{key}: %{{x:.2f}}<br>Depth: %{{y}}<extra></extra>",
            showlegend=False
        ),
        row=1, col=n_seq,
    )

    # Garis threshold
    fig.add_trace(
        go.Scatter(
            x=[thres[key]]*2,
            y=[df_well[depth].min(), df_well[depth].max()],
            mode="lines",
            line=dict(color="red", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=n_seq,
    )

    fig.update_layout(
        **{f"xaxis{n_seq}": dict(range=range_col[key][0], side="top")},
    )

    return fig, axes, counter


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


def plot_fill_x_to_zero(df_well, fig, axes, key, n_seq, index):
    col = data_col[key][index]
    fig.add_trace(
        go.Scatter(
            x=df_well[col],
            y=df_well[depth],
            line=dict(color=color_col[key][index], width=line_width),
            name=col,
            legend=legends[n_seq-1],
            showlegend=True,
            fill='tozerox',
            fillcolor='lightgray',
            xaxis='x'+str(n_seq),
            yaxis='y'+str(n_seq),
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


def plot_n_fill_x_to_zero(df_well, fig, axes, key, n_seq, counter, n_plots):
    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))
    for ind, col in enumerate(data_col[key]):
        fig.add_trace(
            go.Scatter(
                x=df_well[col],
                y=df_well[depth],
                line=dict(color=color_col[key][ind], width=line_width),
                name=col,
                legend=legends[n_seq-1],
                showlegend=True,
                fill='tonextx',
            ),
            row=1, col=n_seq,
        )

    xaxis = "xaxis"+str(n_seq)
    fig.update_layout(
        **{xaxis: dict(
            side="top",
            range=range_col[key][0]
        )}
    )

    for j in range(1, len(data_col[key])):
        counter += 1
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                line=dict(color="rgba(0,0,0,0)", width=0),
                name=col,
                legend=legends[n_seq-1],
                showlegend=False,
                hoverinfo="skip",
                xaxis='x'+str(n_plots+counter),
                yaxis='y'+str(n_seq),
            ),
        )
        xaxis = "xaxis"+str(n_plots+counter)
        fig.update_layout(
            **{xaxis: dict(
                side="top",
                overlaying='x'+str(n_seq),
                range=range_col[key][0]
            )}
        )
        axes[key].append('xaxis'+str(n_plots+counter))

    return fig, axes, counter


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
    elif key in ['MARKER', 'RGBE', 'RPBE', 'ZONE']:
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


def plot_xpt(df_well, fig, axes, key, n_seq):
    fig.add_trace(
        go.Scattergl(
            x=[1]*len(df_well[depth]),
            y=df_well[depth],
            line=dict(color="rgba(0,0,0,0)", width=0),
            # name=col.split("_",1)[-1],
            # legend=legends[key-1],
            showlegend=False,
            xaxis='x'+str(n_seq),
            yaxis='y'+str(n_seq),
        ),
        # row = 1, col = key,
    )

    xaxis = "xaxis"+str(n_seq)
    fig.update_layout(
        **{xaxis: dict(
            side="top",
            range=[0, 1]
        )}
    )

    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    return fig, axes


def plot_texts_xpt(df_text, fig, axes, key, n_seq):
    if not df_text.empty:
        for index, row in df_text[['Depth (m)', 'Note']].iterrows():
            x = 0
            y = row['Depth (m)']
            text = row['Note'][:20]
            fig.add_annotation(
                x=x,
                y=y,
                xref="x"+str(n_seq),
                yref="y",
                xanchor='left',
                yanchor='middle',
                text=text,
                showarrow=True,
                font=dict(
                    size=10,
                    color="black"
                ),
                align="left",
                arrowhead=0,
                arrowsize=5,
                arrowwidth=1,
                arrowcolor="black",
                ax=7,
                ay=0,
            )

    return fig, axes


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


def plot_text_values(df_text, depth_btm, fig, axes, key, n_seq):

    if not df_text.empty:
        # Membuat trace kosong untuk mendefinisikan axes
        # Menggunakan scatter plot kosong sebagai placeholder
        fig.add_trace(
            go.Scatter(
                x=[0, 1],  # range x sederhana
                y=[df_text['Mean Depth'].min(), df_text['Mean Depth'].max()
                   ],  # range y berdasarkan data
                mode='markers',
                # marker transparan
                marker=dict(size=0, color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=n_seq
        )

        # Menambahkan annotation untuk setiap teks
        for index, row in df_text[['Mean Depth', 'Surface']].iterrows():
            x = 0.5  # posisi x di tengah
            y = row['Mean Depth']
            text = row['Surface'][:6]  # mengambil 6 karakter pertama

            if y < depth_btm:
                fig.add_annotation(
                    x=x,
                    y=y,
                    xref="x"+str(n_seq),
                    # menggunakan yref yang sesuai dengan n_seq
                    yref="y"+str(n_seq),
                    xanchor='center',
                    yanchor='middle',
                    text=text,
                    showarrow=True,
                    font=dict(
                        size=10,
                        color="black"
                    ),
                    align="center",
                    ax=0,
                    ay=0,
                )

    # Mengatur xaxis dan yaxis
    xaxis = "xaxis"+str(n_seq)
    yaxis = "yaxis"+str(n_seq)

    fig.update_layout(
        **{xaxis: dict(
            side="top",
            showticklabels=False,
        ),
            yaxis: dict(
            showticklabels=False,
            visible=False
        )}
    )

    axes[key].append('yaxis'+str(n_seq))
    axes[key].append('xaxis'+str(n_seq))

    return fig, axes

# ---------------------------- Layout FUnction -----------------------------


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
            elif key in ['RT_RO', 'PERM', 'RWAPP_RW', 'RT_F', 'RT_RHOB', 'RT_RGSA', 'RT', 'RT_GR', 'RT_PHIE', 'TGC', 'RWA']:
                a = range_col[key][0][0]
                b = range_col[key][0][1]
                arr = log_tickvals(a, b)
                fig.update_layout(
                    **{axis: dict(
                        type="log",  # <--- THIS IS CRUCIAL
                        tickvals=arr,
                        gridcolor='gainsboro',
                        side="top",
                        fixedrange=True,
                        showticklabels=False if axis.startswith('xaxis') else True,
                    )}
                )
            elif key in ['GR', 'SP', 'GR_NORM', 'GR_DUAL', 'GR_RAW_NORM', 'GR_DUAL_2', 'GR_MovingAvg_5', 'GR_MovingAvg_10', 'RTRO', 'NPHI_RHOB', 'SW', 'PHIE_PHIT', 'VCL', 'X_RWA_RW', 'X_RT_F', 'X_RT_RHOB', 'NPHI_NGSA', 'RHOB_DGSA', 'VSH_LINEAR', 'VSH_DN', 'VSH_SP', 'RHOB', 'PHIE_DEN', 'PHIT_DEN', 'PHIE_PHIT', 'PHIE', 'DNS', 'DNSV', 'VSH', 'VSH_GR_DN', 'RGBE', 'RPBE', 'TG_SUMC', 'C3_C1', 'C3_C1_BASELINE']:
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

def log_tickvals(a, b):
    ticks = []
    exp_min = int(np.floor(np.log10(a)))
    exp_max = int(np.ceil(np.log10(b)))
    for exp in range(exp_min, exp_max + 1):
        for mult in range(1, 10):
            val = mult * 10 ** exp
            if a <= val <= b:
                ticks.append(val)
    return ticks

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

# ---panggil layout axis


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
        if key in ['SWARRAY', 'TGC']:
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
            if key not in ['CLASS', 'TEST', 'XPT', 'MARKER', 'ZONA', 'RESERVOIR_CLASS', 'IQUAL', 'RGBE_TEXT', 'RPBE_TEXT', 'ZONE']:
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


def layout_draw_header_lines(fig, ratio_plots):
    """
    Fungsi untuk menggambar garis-garis pada area header saja
    """
    ratio_plots = np.array(ratio_plots)
    line_pos = []
    for i in ratio_plots:
        line_pos.append(
            i*(1/(ratio_plots/len(ratio_plots)).sum())/len(ratio_plots))

    shapes = []

    # Garis vertikal kiri (x=0) - hanya untuk area header (y dari 0.8 ke 1.0)
    shapes.append(
        dict(
            type='line', xref='paper', yref='paper',
            x0=0, x1=0, y0=1, y1=0.8,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    # Garis vertikal pembatas antar kolom - hanya untuk area header
    x = 0
    for pos in line_pos:
        x += pos
        shapes.append(
            dict(
                type='line', xref='paper', yref='paper',
                x0=x, x1=x, y0=1, y1=0.8,
                line=dict(color='black', width=1, dash='solid')
            )
        )

    # Garis horizontal atas header (y=1.0)
    shapes.append(
        dict(
            type='line', xref='paper', yref='paper',
            x0=0, x1=1, y0=1, y1=1,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    # Garis horizontal bawah header / pembatas dengan main (y=0.8)
    shapes.append(
        dict(
            type='line', xref='paper', yref='paper',
            x0=0, x1=1, y0=0.8, y1=0.8,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    fig.update_layout(shapes=shapes)
    return fig


def layout_draw_main_lines(fig, ratio_plots, df_well, xgrid_intv):
    """
    Fungsi untuk menggambar garis-garis pada area main saja (tanpa header)
    """
    ratio_plots = np.array(ratio_plots)
    line_pos = []
    for i in ratio_plots:
        line_pos.append(
            i*(1/(ratio_plots/len(ratio_plots)).sum())/len(ratio_plots))

    shapes = []

    # Garis vertikal kiri (x=0) - hanya untuk area main (y dari 0 ke 0.8)
    shapes.append(
        dict(
            type='line', xref='paper', yref='paper',
            x0=0, x1=0, y0=0.9, y1=0,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    # Garis vertikal pembatas antar kolom - hanya untuk area main
    x = 0
    for pos in line_pos:
        x += pos
        shapes.append(
            dict(
                type='line', xref='paper', yref='paper',
                x0=x, x1=x, y0=0.9, y1=0,
                line=dict(color='black', width=1, dash='solid')
            )
        )

    # Garis horizontal bawah main (y=0)
    shapes.append(
        dict(
            type='line', xref='paper', yref='paper',
            x0=0, x1=1, y0=0, y1=0,
            line=dict(color='black', width=1, dash='solid')
        )
    )

    # Plot grid horizontal (jika diperlukan)
    if xgrid_intv is not None and xgrid_intv != 0:
        # Asumsi 'depth' adalah nama kolom, sesuaikan dengan variabel yang tepat
        grid_lines = [dict(
            layer='below',
            type="line",
            x0=0, x1=1,
            xref="paper",
            # Normalisasi ke area main (0-0.9)
            y0=y * 0.9 / int(df_well['depth'].max()), y1=y * 0.9 / int(df_well['depth'].max()),
            line=dict(color="gainsboro", width=1)
        ) for y in range(0, int(df_well['depth'].max()), xgrid_intv)]

        shapes = shapes + grid_lines
        fig.update_layout(shapes=shapes, yaxis=dict(showgrid=False))
    else:
        fig.update_layout(shapes=shapes)

    return fig


def layout_axis_header(fig_main, axes, ratio_plots, plot_sequence):
    """
    Membuat figure header yang terpisah dari main plot
    Returns: fig_header - figure untuk header
    """
    import plotly.graph_objects as go

    # Buat figure baru untuk header
    fig_header = go.Figure()

    # Atur ukuran dan margin untuk header
    fig_header.update_layout(
        height=200,  # Tinggi header yang lebih kecil
        margin=dict(l=50, r=50, t=20, b=20),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Tambahkan annotation untuk depth label di header
    fig_header.add_annotation(
        dict(font=dict(color='black', size=12),
             x=-0.001,
             y=0.5,  # Posisi di tengah header
             xanchor="right",
             yanchor="middle",
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
        pos_x = line_pos[i]
        pos_y = 0.2  # Posisi untuk header (lebih rendah dari 0.85)
        pos_x_c += 0.5*pos_x

        # Ganti dengan key yang butuh semua axis (feature di datacol)
        if key in ['SWARRAY']:
            axis_range = axes[key][1:]  # Semua axis
        else:
            axis_range = axes[key][1:3]  # Hanya 2 axis pertama

        for j, axis in enumerate(axis_range):
            # UPDATE LAYOUT AXIS - SAMA SEPERTI KODE ASLI
            fig_header.update_layout(
                **{axis: dict(
                    tickfont=dict(color=color_col[key][j], size=9),
                    anchor="free",
                    showline=True,        # INI YANG BIKIN GARIS BERWARNA
                    position=pos_y,
                    showticklabels=False,
                    linewidth=1.5,
                    linecolor=color_col[key][j],  # WARNA GARIS
                )}
            )

            # Add Text Parameter
            fig_header.add_annotation(
                dict(font=dict(color=color_col[key][j], size=12),
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
            fig_header.add_annotation(
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
                fig_header.add_annotation(
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

                fig_header.add_annotation(
                    dict(font=dict(color=color_col[key][j], size=10),
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

            # Increment posisi y untuk axis selanjutnya (seperti kode asli)
            pos_y += 0.35  # Lebih besar dari 0.04 karena ruang header lebih terbatas
            pos_y = min(pos_y, 1.0)

        pos_x_t += pos_x
        pos_x_c += 0.5*pos_x

    # Tambahkan garis pembatas vertikal di header
    shapes = []
    x = 0
    for pos in line_pos:
        x += pos
        shapes.append(
            dict(
                type='line', xref='paper', yref='paper',
                x0=x, x1=x, y0=0, y1=1,
                line=dict(color='black', width=1, dash='solid')
            )
        )

    # Garis border header
    shapes.extend([
        # Garis atas
        dict(type='line', xref='paper', yref='paper',
             x0=0, x1=1, y0=1, y1=1,
             line=dict(color='black', width=1, dash='solid')),
        # Garis bawah
        dict(type='line', xref='paper', yref='paper',
             x0=0, x1=1, y0=0, y1=0,
             line=dict(color='black', width=1, dash='solid')),
        # Garis kiri
        dict(type='line', xref='paper', yref='paper',
             x0=0, x1=0, y0=0, y1=1,
             line=dict(color='black', width=1, dash='solid')),
        # Garis kanan
        dict(type='line', xref='paper', yref='paper',
             x0=1, x1=1, y0=0, y1=1,
             line=dict(color='black', width=1, dash='solid'))
    ])

    fig_header.update_layout(shapes=shapes)

    # TIDAK hilangkan axis - biarkan axis tetap visible untuk menampilkan garis berwarna
    # Hanya hilangkan ticklabels dan grid
    fig_header.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig_header
# ------------------------------ Other Func --------------------------------


def convert_depth_ft_to_m(df, column='DEPTH'):
    """
    Mengonversi nilai pada kolom 'DEPTH' dari feet ke meter.

    Args:
        df (pd.DataFrame): DataFrame yang memiliki kolom 'DEPTH'.
        column (str): Nama kolom yang akan dikonversi. Default 'DEPTH'.

    Returns:
        pd.DataFrame: DataFrame dengan kolom 'DEPTH' dalam satuan meter.
    """
    df[column] = df[column] * 0.3048  # 1 ft = 0.3048 m
    return df


def update_plot_sequence(plot_sequence, exclude_keys):
    filtered_items = [v for k, v in plot_sequence.items()
                      if v not in exclude_keys]
    return {i + 1: v for i, v in enumerate(filtered_items)}


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


def get_random_color(pastel_factor=0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]


def color_distance(c1, c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1, c2)])


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


def rgb_to_hex(rgb):
    """Convert RGB (0-1 range) to HEX."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

# ------------------------------- Main Plot --------------------------------
# @title


def main_plot(df, sequence=[], title="", height_plot=1600):
    # Zona RGSA-NGSA-DGSA
    zona_mapping = {
        'Zona Prospek Kuat': 3,
        'Zona Menarik': 2,
        'Zona Lemah': 1,
        'Non Prospek': 0
    }

    for seq in sequence:
        if seq == 'MARKER':
            df_marker = extract_markers_with_mean_depth(df)
            df_well_marker = df.copy()
        elif seq == 'ZONE':
            df_zone = extract_markers_customize(df, 'ZONE')
            df_well_zone = df.copy()
        elif seq == 'NPHI_RHOB':
            df = normalize_xover(df, 'NPHI', 'RHOB')
        elif seq == 'RT_RHOB':
            df = normalize_xover(df, 'RT', 'RHOB')
        elif seq == 'RT_GR':
            df = normalize_xover(df, 'RT', 'GR')
        elif seq == 'PHIE_PHIT':
            df.rename(columns={'PHIE_DN': 'PHIE_DEN',
                      'PHIT_DN': 'PHIT_DEN'}, inplace=True)
        elif seq == 'SW':
            df = df.rename(columns={'SWE_INDO': 'SW'})
        # elif seq == 'RT_RGSA':
        #     df['ZONA'] = df['ZONA'].map(zona_mapping)
        elif seq == 'RGBE':
            df['RGBE'] = round(df['RGBE'])
            df_marker_rgbe = extract_markers_customize(df, 'RGBE')
            df_well_marker_rgbe = df.copy()
        elif seq == 'RPBE':
            df['RPBE'] = round(df['RPBE'])
            df_marker_rpbe = extract_markers_customize(df, 'RPBE')
            df_well_marker_rpbe = df.copy()
        elif seq == 'IQUAL':
            df['IQUAL'] = df['IQUAL'].replace(0, np.nan)
            df_well_marker_iqual = df.copy()
            df_marker_iqual = extract_markers_customize(df, 'IQUAL')
        elif seq == 'SWGRAD':
            df['SWGRAD'] = df['SWGRAD'].abs()
        elif seq == 'RT_RO':
            df = df.rename(columns={'R0': 'RO'})

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
                df, fig, axes, base_key='RT', n_seq=n_seq, type="log", col=col, label=col)
        elif col == 'SP':
            fig, axes = plot_line(
                df, fig, axes, base_key='SP', n_seq=n_seq, col=col, label=col)
        elif col == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, y_color='rgba(0,0,0,0)', n_color='yellow', type=2, exclude_crossover=False)
        elif col == 'RT_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                       y_color='limegreen', n_color='lightgray', type=1, exclude_crossover=False)
        elif col in ['X_RT_RO', 'X_RWA_RW', 'X_RT_F', 'X_RT_RHOB']:
            fig, axes, counter = plot_xover_thres(
                df, fig, axes, col, n_seq, counter=counter)

        # VSHALE
        elif col == 'VSH_LINEAR':
            # fig, axes = plot_line(df, fig, axes, base_key='VSH_LINEAR', n_seq=n_seq, col=col, label=col)
            fig, axes, counter = plot_xover_thres_dual(
                df, fig, axes, col, n_seq, counter)

        elif col == 'VSH_GR_DN':
            fig, axes, counter = plot_two_features_simple(df, fig, axes, 'VSH_GR_DN', n_seq,
                                                          counter, n_plots=subplot_col, log_scale=False)
        elif col == 'VSH':
            # fig, axes = plot_line(df, fig, axes, base_key='VSH', n_seq=n_seq, col=col, label=col)
            fig, axes, counter = plot_xover_thres_dual(
                df, fig, axes, col, n_seq, counter)

        # POROSITY
        elif col == 'PHIE':
            # fig, axes = plot_line(df, fig, axes, base_key='PHIE', n_seq=n_seq, col=col, label=col)
            fig, axes, counter = plot_xover_thres_dual(
                df, fig, axes, col, n_seq, counter, above_thres_color="yellow", below_thres_color="green")
        elif col == 'PHIE_PHIT':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq,
                                                       counter, n_plots=subplot_col,
                                                       y_color='limegreen', n_color='lightgray', type=1, exclude_crossover=False)

        # SWE INDONESIA
        elif col == 'SW':
            # fig, axes = plot_line(df, fig, axes, base_key='SW', n_seq=n_seq, col=col, label=col)
            fig, axes, counter = plot_xover_thres_dual(
                df, fig, axes, col, n_seq, counter, above_thres_color="rgba(250,0,0,0)", below_thres_color="lightgrey")

        # RWA
        elif col == 'RWA':
            fig, axes, counter = plot_three_features_simple(
                df, fig, axes, col, n_seq, counter, subplot_col, log_scale=True)

        # RGSA-NGSA-DGSA
        elif col == 'RT_RGSA':
            fig, axes, counter = plot_gsa_crossover(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, fill_color_red='red', fill_color_blue=colors_dict['blue'])
        elif col == 'NPHI_NGSA':
            fig, axes, counter = plot_gsa_crossover(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, fill_color_red='red', fill_color_blue='darkgreen')
        elif col == 'RHOB_DGSA':
            fig, axes, counter = plot_gsa_crossover(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, fill_color_red='red', fill_color_blue='darkgreen')
        elif col == 'ZONA':
            fig, axes = plot_flag(df, fig, axes, col, n_seq)

        # RGBE RPBE
        elif col == 'RT_GR':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                       y_color='limegreen', n_color='lightgray', type=1, exclude_crossover=False)
        elif col == 'RT_PHIE':
            fig, axes, counter = plot_two_features_simple(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, log_scale=True)
        elif col == 'RGBE':
            # fig,axes = plot_flag(df_well_marker_rgbe,fig,axes,col,n_seq)
            # fig, axes, counter = plot_xover_thres_dual(df, fig, axes, col, n_seq, counter,above_thres_color="darkgreen", below_thres_color="lightblue")
            fig, axes, counter = plot_xover_bar_horizontal(
                df, fig, axes, col, n_seq, counter)
        elif col == 'RPBE':
            # fig,axes = plot_flag(df_well_marker_rpbe,fig,axes,col,n_seq)
            # fig, axes, counter = plot_xover_thres_dual(df, fig, axes, col, n_seq, counter,above_thres_color="lightblue", below_thres_color="darkgreen")
            fig, axes, counter = plot_xover_bar_horizontal(
                df, fig, axes, col, n_seq, counter)

        elif col == 'RGBE_TEXT':
            fig, axes = plot_text_values(
                df_marker_rgbe, df_well_marker_rgbe['DEPTH'].max(), fig, axes, col, n_seq)
        elif col == 'RPBE_TEXT':
            fig, axes = plot_text_values(
                df_marker_rpbe, df_well_marker_rpbe['DEPTH'].max(), fig, axes, col, n_seq)

        # SWGRAD
        elif col == 'SWGRAD':
            fig, axes = plot_line(
                df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)
        elif col == 'SWARRAY':
            fig, axes, counter = plot_four_features_simple(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, log_scale=False)

        # DNS DNSV
        elif col == 'DNS':
            fig, axes = plot_line(df, fig, axes, col, n_seq)
        elif col == 'DNSV':
            fig, axes = plot_line(df, fig, axes, col, n_seq)

        # RT R0
        elif col == 'RT_RO':
            fig, axes, counter = plot_xover(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, y_color='limegreen', n_color='lightgray')

        # GWD ANALYSIS
        elif col == 'TGC':
            fig, axes, counter = plot_four_features_simple(
                df, fig, axes, col, n_seq, counter, n_plots=subplot_col, log_scale=True)
        elif col == 'TG_SUMC':
            fig, axes = plot_line(
                df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)
        elif col == 'C3_C1':
            fig, axes = plot_line(
                df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)
        elif col == 'C3_C1_BASELINE':
            fig, axes = plot_line(
                df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)

        # FLAG
        elif col == 'IQUAL':
            fig, axes = plot_flag(df_well_marker_iqual,
                                  fig, axes, 'IQUAL', n_seq)
            fig, axes = plot_texts_marker(
                df_marker_iqual, df_well_marker_iqual['DEPTH'].max(), fig, axes, col, n_seq)
        elif col == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, col, n_seq)
            fig, axes = plot_texts_marker(
                df_marker, df_well_marker['DEPTH'].max(), fig, axes, col, n_seq)
        elif col == 'ZONE':
            fig, axes = plot_flag(df_well_zone, fig, axes, col, n_seq)
            fig, axes = plot_texts_marker(
                df_zone, df_well_zone['DEPTH'].max(), fig, axes, col, n_seq)

        elif col in [
            'GR_CAL', 'DGRCC',
            'GR_CAL_NO', 'DGRCC_NO',
            'GR_CAL_TR', 'DGRCC_TR',
            'GR_CAL_SM', 'DGRCC_SM',
            'GR_CAL_FM', 'DGRCC_FM'
        ]:
            fig, axes = plot_line(
                df, fig, axes, base_key='GR', n_seq=n_seq, col=col, label=col)

        # Group untuk semua log berbasis RT (asli dan turunan, dengan skala logaritmik)
        elif col in [
            'RLA5', 'A40H', 'ARM48PC', 'R39PC',
            'RLA5_TR', 'A40H_TR', 'ARM48PC_TR', 'R39PC_TR',
            'RLA5_SM', 'A40H_SM', 'ARM48PC_SM', 'R39PC_SM',
            'RLA5_FM', 'A40H_FM', 'ARM48PC_FM', 'R39PC_FM'
        ]:
            fig, axes = plot_line(
                df, fig, axes, base_key='RT', n_seq=n_seq, type="log", col=col, label=col)

        # Group untuk semua log berbasis RHOB (asli dan turunan)
        elif col in [
            'RHOZ', 'ALCDLC', 'ROBB',
            'RHOZ_TR', 'ALCDLC_TR', 'ROBB_TR',
            'RHOZ_SM', 'ALCDLC_SM', 'ROBB_SM',
            'RHOZ_FM', 'ALCDLC_FM', 'ROBB_FM'
        ]:
            fig, axes = plot_line(
                df, fig, axes, base_key='RHOB', n_seq=n_seq, col=col, label=col)

        # Group untuk semua log berbasis NPHI (asli dan turunan)
        elif col in [
            'TNPL', 'TNPH',
            'TNPL_TR', 'TNPH_TR',
            'TNPL_SM', 'TNPH_SM',
            'TNPL_FM', 'TNPH_FM'
        ]:
            # Catatan: Menggunakan 'NPHI_RHOB_NON_NORM' sebagai base_key
            # untuk mendapatkan properti NPHI yang benar dari dictionary Anda.
            fig, axes = plot_line(
                df, fig, axes, base_key='NPHI_RHOB_NON_NORM', n_seq=n_seq, col=col, label=col)
    print(axes)

    fig = layout_range_all_axis(fig, axes, plot_sequence)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=height_plot,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified', hoverdistance=-1,
        title_text=title,
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

# @title


def extract_markers_customize(df, key):
    required_cols = [key, 'DEPTH']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame tidak memiliki kolom '{col}'")

    # Mengelompokkan berdasarkan MARKER dan menghitung rata-rata DEPTH
    markers_mean_depth = df.groupby(key)['DEPTH'].mean().reset_index()

    # Mengganti nama kolom
    markers_mean_depth.columns = ['Surface', 'Mean Depth']

    # PERBAIKAN: Convert kolom Surface ke string untuk menghindari IndexError saat slicing
    markers_mean_depth['Surface'] = markers_mean_depth['Surface'].astype(str)

    return markers_mean_depth

# @title


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


"""# Module 1"""

"""## Log ABB-036"""

# @title


def plot_log_default(df):
    """
    Creates a default well log plot that dynamically includes or excludes ZONE based on data availability.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing well log data

    Returns:
    --------
    plotly.graph_objects.Figure
        The generated well log plot
    """
    # Define the base sequence with all potential tracks
    marker_zone_sequence = ['ZONE', 'MARKER']

    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]

    # Create a flat list by extending filtered_sequence with other track names
    sequence_default = filtered_sequence + ['GR', 'RT', 'NPHI_RHOB']

    # Create the plot with the filtered sequence
    fig = main_plot(df, sequence=sequence_default,
                    title="Plot Well Log Selected", height_plot=1600)
    return fig


def plot_normalization(df):
    df_marker = extract_markers_with_mean_depth(df)
    df_well_marker = df.copy()
    sequence = ['MARKER', 'GR', 'GR_DUAL_2', 'GR_DUAL', 'GR_RAW_NORM']
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
            elif col == 'GR_DUAL_2':
                fig, axes, counter = plot_dual_gr(
                    df, fig, axes, col, n_seq, counter, subplot_col)
            elif col == 'GR_DUAL':
                fig, axes, counter = plot_dual_gr(
                    df, fig, axes, col, n_seq, counter, subplot_col)
            elif col == 'GR_RAW_NORM':
                fig, axes = plot_line(
                    df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)

    fig = layout_range_all_axis(fig, axes, plot_sequence)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), height=1300,
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


def plot_phie_den(df):
    """
    Membuat plot multi-panel untuk visualisasi hasil kalkulasi Porositas.
    """
    sequence_phie = ['MARKER', 'GR',
                     'RT', 'NPHI_RHOB', 'VSH', 'PHIE_PHIT']
    fig = main_plot(df, sequence_phie, title="Porosity Bateman/Konen")

    return fig


def plot_gsa_main(df):
    """
    Fungsi utama untuk membuat plot komprehensif Gas Show Anomaly.
    """
    # # Lakukan pra-pemrosesan data yang diperlukan untuk plot ini
    # df_well = normalize_xover(df_well, 'NPHI', 'RHOB')
    # df_well = normalize_xover(df_well, 'RT', 'RHOB')
    # df_well = normalize_xover(df_well, 'RT', 'RGSA')
    # df_well = normalize_xover(df_well, 'NPHI', 'NGSA')
    # df_well = normalize_xover(df_well, 'RHOB', 'DGSA')

    # zona_mapping = {
    #     'Zona Prospek Kuat': 3,
    #     'Zona Menarik': 2,
    #     'Zona Lemah': 1,
    #     'Non Prospek': 0
    # }

    # df_well['ZONA'] = df_well['ZONA'].map(zona_mapping)
    # df_marker = extract_markers_with_mean_depth(df_well)
    # df_well_marker = df_well.copy()

    # # Definisikan urutan track untuk plot GSA
    # sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB',
    #             'RT_RGSA', 'NPHI_NGSA', 'RHOB_DGSA', 'ZONA']
    # plot_sequence = {i+1: v for i, v in enumerate(sequence)}

    # ratio_plots_seq = [ratio_plots.get(key, 1)
    #                    for key in plot_sequence.values()]
    # subplot_col = len(plot_sequence)

    # fig = make_subplots(
    #     rows=1, cols=subplot_col,
    #     shared_yaxes=True,
    #     column_widths=ratio_plots_seq,
    #     horizontal_spacing=0.01
    # )

    # counter = 0
    # axes = {key: [] for key in plot_sequence.values()}

    # # Loop untuk memanggil plotter yang sesuai untuk setiap track
    # for n_seq, key in plot_sequence.items():
    #     if key == 'MARKER':
    #         fig, axes = plot_flag(df_well_marker, fig, axes, key, n_seq)
    #         fig, axes = plot_texts_marker(
    #             df_marker, df_well_marker['DEPTH'].max(), fig, axes, key, n_seq)
    #     elif key == 'GR':
    #         fig, axes = plot_line(df_well, fig, axes, key, n_seq)
    #     elif key in ['NPHI_RHOB', 'RT']:
    #         fig, axes, counter = plot_xover_log_normal(
    #             df_well, fig, axes, key, n_seq, counter, subplot_col)
    #     elif key in ['RT_RGSA', 'NPHI_NGSA', 'RHOB_DGSA']:
    #         fig, axes, counter = plot_gsa_crossover(
    #             df_well, fig, axes, key, n_seq, counter, subplot_col)
    #     elif key == 'ZONA':
    #         fig, axes = plot_flag(df_well, fig, axes, key, n_seq)

    # # Finalisasi Layout
    # fig = layout_range_all_axis(fig, axes, plot_sequence)
    # fig = layout_draw_lines(fig, ratio_plots_seq, df_well, xgrid_intv=50)
    # fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    # fig.update_layout(
    #     title_text="Gas Show Anomaly (GSA) Analysis",
    #     yaxis=dict(range=[
    #                df_well[depth].max(), df_well[depth].min()]),
    #     hovermode='y unified',
    #     template='plotly_white',
    #     showlegend=False,
    #     height=1600,
    # )

    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence_rgsa = filtered_sequence + ['GR', 'RT', 'NPHI_RHOB',
                                         'RT_RGSA', 'NPHI_NGSA', 'RHOB_DGSA']
    fig = main_plot(df, sequence_rgsa, title="Gas Show Anomaly Analysis")

    return fig


def plot_vsh_linear(df):
    """
    Membuat plot multi-panel untuk visualisasi hasil kalkulasi VSH.
    """
    marker_zone_sequence = ['ZONE', 'MARKER']

    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence_vsh = filtered_sequence + ['GR',
                                        'RT', 'NPHI_RHOB', 'VSH_GR_DN']
    fig = main_plot(df, sequence_vsh, title="Log VSH GR-DN")

    return fig


def plot_sw_indo(df):
    """
    Membuat plot multi-panel untuk visualisasi hasil kalkulasi Saturasi Air (Indonesia).
    """
    marker_zone_sequence = ['ZONE', 'MARKER']

    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence_swe = filtered_sequence + ['GR', 'RT',
                                        'NPHI_RHOB', 'VSH', 'PHIE_PHIT', 'SW']
    fig = main_plot(df, sequence_swe, title="Water Saturation")
    return fig


def plot_rwa_indo(df):
    """
    Membuat plot multi-panel untuk visualisasi hasil kalkulasi RWA.
    """
    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence_rwa = filtered_sequence + ['GR',
                                        'RT', 'NPHI_RHOB', 'VSH', 'PHIE', 'RWA']
    fig = main_plot(df, sequence_rwa, title="Water Resistivity")
    return fig


def plot_sw_simandoux(df):
    """
    Membuat plot multi-panel untuk visualisasi hasil kalkulasi Water Saturation (Simandoux).
    """
    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence_sw_sim = filtered_sequence + ['GR', 'RT',
                                           'NPHI_RHOB', 'VSH', 'PHIE_PHIT', 'SW_SIMANDOUX', 'RESERVOIR_CLASS']
    fig = main_plot(df, sequence_sw_sim,
                    title="Water Saturation (Modified Simandoux)")
    return fig


def plot_smoothing(df, df_marker, df_well_marker):
    sequence = ['MARKER', 'GR', 'GR_SM']
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
        elif col == 'GR_SM':
            fig, axes = plot_line(
                df, fig, axes, base_key='GR_SM', n_seq=n_seq, col=col, label=col)
        elif col == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, col, n_seq)
            fig, axes = plot_texts_marker(
                df_marker, df_well_marker['DEPTH'].max(), fig, axes, col, n_seq)

    fig = layout_range_all_axis(fig, axes, plot_sequence)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), height=1300,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified', hoverdistance=-1,
        title_text="Smoothing",
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


def plot_module_2(df):
    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    seq_module_2 = filtered_sequence + ['GR', 'RT',
                                        'NPHI_RHOB', 'PHIE', 'VSH_LINEAR', 'SW', 'IQUAL']
    fig = main_plot(df, seq_module_2, title="Log Interpretation Selected Well")
    return fig


def plot_gwd(df):
    sequence = ['TGC', 'TG_SUMC', 'C3_C1', 'C3_C1_BASELINE']
    fig = main_plot(df, sequence, title="GWD Analysis")
    return fig


def plot_iqual(df):
    """
    Membuat plot IQUAL.
    """
    df = calculate_iqual(df)
    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence_iqual = filtered_sequence + ['GR', 'RT',
                                          'NPHI_RHOB', 'PHIE', 'VSH_LINEAR', 'IQUAL']
    fig = main_plot(df, sequence_iqual, title="IQUAL")

    return fig


def plot_splicing(df):

    sequence = ['GR', 'RT', 'NPHI_RHOB']
    fig = main_plot(df, sequence, title="Splicing BNG-057")
    return fig


def plot_module1(df):
    """
    Membuat plot Module1 dengan auto-detection LWD vs WL.
    """
    # Reset index seperti di colab code
    df = df.reset_index()

    # Ensure DEPTH column exists (rename DEPT to DEPTH if needed)
    if 'DEPT' in df.columns and 'DEPTH' not in df.columns:
        df = df.rename(columns={'DEPT': 'DEPTH'})

    # Auto-detect LWD vs WL berdasarkan kolom yang tersedia
    lwd_sequence = ['DGRCC', 'ALCDLC', 'TNPL', 'R39PC']
    wl_sequence = ['GR_CAL', 'RHOZ', 'RLA5', 'TNPH']

    # Check which type of data we have
    lwd_available = sum(1 for col in lwd_sequence if col in df.columns)
    wl_available = sum(1 for col in wl_sequence if col in df.columns)

    if lwd_available >= wl_available:
        # Use LWD sequence
        sequence = lwd_sequence
        print(f"Available sequence LWD for Module1: {sequence}")
        title = 'LWD'
    else:
        # Use WL sequence and scale RHOZ if available
        if 'RHOZ' in df.columns:
            df['RHOZ'] = df['RHOZ'] / 1000
        sequence = wl_sequence
        print(f"Available sequence WL for Module1: {sequence}")
        title = 'WL'

    # Filter sequence to only include available columns
    available_sequence = [col for col in sequence if col in df.columns]
    print(f"Available sequence for Module1: {available_sequence}")

    if not available_sequence:
        raise ValueError("No valid columns found for Module1 plot")

    fig = main_plot(df, available_sequence, title=title)

    return fig


def plot_norm_prep(df):
    """
    Membuat plot Normalization Preparation.
    """
    # Reset index seperti di colab code
    df = df.reset_index()

    # Ensure DEPTH column exists (rename DEPT to DEPTH if needed)
    if 'DEPT' in df.columns and 'DEPTH' not in df.columns:
        df = df.rename(columns={'DEPT': 'DEPTH'})

    # Auto-detect LWD vs WL berdasarkan kolom yang tersedia
    lwd_sequence = ['DGRCC', 'DGRCC_NO', 'ALCDLC', 'TNPL', 'R39PC']
    wl_sequence = ['GR_CAL', 'GR_CAL_NO', 'RHOZ', 'RLA5', 'TNPH']

    # Check which type of data we have
    lwd_available = sum(1 for col in lwd_sequence if col in df.columns)
    wl_available = sum(1 for col in wl_sequence if col in df.columns)

    if lwd_available >= wl_available:
        # Use LWD sequence
        sequence = lwd_sequence
        title = 'LWD'
    else:
        # Use WL sequence and scale RHOZ if available
        if 'RHOZ' in df.columns:
            df['RHOZ'] = df['RHOZ'] / 1000
        sequence = wl_sequence
        title = 'WL'

    # Filter sequence to only include available columns
    sequence = [col for col in sequence if col in df.columns]

    if not sequence:
        raise ValueError("No valid columns found for Module1 plot")

    fig = main_plot(df, sequence, title=title)

    return fig


def plot_smoothing_prep(df):
    """
    Membuat plot Normalization Preparation.
    """
    # Reset index seperti di colab code
    df = df.reset_index()

    # Ensure DEPTH column exists (rename DEPT to DEPTH if needed)
    if 'DEPT' in df.columns and 'DEPTH' not in df.columns:
        df = df.rename(columns={'DEPT': 'DEPTH'})

    # Auto-detect LWD vs WL berdasarkan kolom yang tersedia
    lwd_sequence = [
        'DGRCC', 'DGRCC_SM',
        'ALCDLC', 'ALCDLC_SM',
        'TNPL', 'TNPL_SM',
        'R39PC', 'R39PC_SM'
    ]

    wl_sequence = [
        'GR_CAL', 'GR_CAL_SM',
        'RHOZ', 'RHOZ_SM',
        'RLA5', 'RLA5_SM',
        'TNPH', 'TNPH_SM'
    ]

    # Check which type of data we have
    lwd_available = sum(1 for col in lwd_sequence if col in df.columns)
    wl_available = sum(1 for col in wl_sequence if col in df.columns)

    if lwd_available >= wl_available:
        # Use LWD sequence
        sequence = lwd_sequence
        title = 'LWD'
    else:
        # Use WL sequence and scale RHOZ if available
        if 'RHOZ' in df.columns:
            df['RHOZ'] = df['RHOZ'] / 1000
        sequence = wl_sequence
        title = 'WL'

    # Filter sequence to only include available columns
    sequence = [col for col in sequence if col in df.columns]

    if not sequence:
        raise ValueError("No valid columns found for Module1 plot")

    fig = main_plot(df, sequence, title=title)

    return fig


def plot_fill_missing(df, title="Fill Missing Plot"):
    """Membuat plot Fill Missing dengan sequence yang sudah ditentukan."""
    # Definisikan urutan track yang ingin ditampilkan
    sequence = ['GR', 'NPHI_RHOB', 'RT', 'MISSING_FLAG']

    # Panggil fungsi plotting utama
    fig = main_plot(df, sequence, title=title)

    return fig


def plot_module_3(df, title="Module 3 Plot"):
    """Membuat plot untuk Module 3 dengan sequence yang sudah ditentukan."""
    # Definisikan urutan track yang ingin ditampilkan
    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [
        col for col in marker_zone_sequence if col in df.columns]
    sequence = ['GR', 'RT', 'NPHI_RHOB', 'VSH', 'PHIE', 'IQUAL', 'RT_RGSA',
                'NPHI_NGSA', 'RHOB_DGSA', 'RGBE', 'RPBE', 'SWGRAD', 'DNS', 'DNSV', 'RT_RO']
    fig = main_plot(df, sequence, title=title)

    return fig

def plot_custom(df, sequence):
    """
    Membuat plot kustom berdasarkan urutan yang diberikan.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data untuk plotting.
    sequence : list
        Daftar kolom yang akan digunakan dalam plot.
    title : str
        Judul plot.
    height_plot : int
        Tinggi plot.

    Returns:
    --------
    plotly.graph_objects.Figure
        Objek Figure yang berisi plot.
    """
    fig = main_plot(df, sequence)
    return fig