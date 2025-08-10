from plotly.subplots import make_subplots
from services.plotting_service import (
    main_plot,
)
import numpy as np


def plot_rt_r0(df, title="RT-R0 Analysis"):

    sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB', 'IQUAL', 'RT_RO']
    fig = main_plot(df, sequence, title)

    return fig
