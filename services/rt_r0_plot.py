from plotly.subplots import make_subplots
from services.plotting_service import (
    main_plot,
)
import numpy as np


def plot_rt_r0(df, title="RT-R0 Analysis"):
    marker_zone_sequence = ['ZONE', 'MARKER']
    # Filter the sequence to include only columns that exist in the DataFrame
    filtered_sequence = [col for col in marker_zone_sequence if col in df.columns]
    sequence = filtered_sequence + ['GR', 'RT', 'NPHI_RHOB', 'IQUAL', 'RT_RO']
    fig = main_plot(df, sequence, title)

    return fig
