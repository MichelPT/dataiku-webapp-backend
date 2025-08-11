from narwhals import col
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services.plotting_service import (
    extract_markers_with_mean_depth,
    main_plot,
    normalize_xover,
    plot_line,
    plot_xover_log_normal,
    plot_four_features_simple,
    plot_flag,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis,
    ratio_plots
)


def plot_swgrad(df, title='SWGRAD Analysis'):
    """
    Creates a comprehensive SWGRAD visualization plot based on the working Colab logic.
    """
    sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB', 'SWARRAY', 'SWGRAD']
    fig = main_plot(df, sequence, title)

    return fig
