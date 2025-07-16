from plotly.subplots import make_subplots
from services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_line,
    plot_xover,
    plot_xover_log_normal,
    plot_two_features_simple,
    ratio_plots,
    plot_flag,
    plot_text_values,
    extract_markers_customize,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis,
    depth
)
import numpy as np


def plot_rt_r0(df, title="RT-R0 Analysis"):
    """
    Fungsi utama untuk membuat plot RT-R0 Analysis.
    """
    # --- 1. Pra-pemrosesan Data ---
    # Handle IQUAL jika ada
    if 'IQUAL' in df.columns:
        df.loc[df['IQUAL'] == 0, 'IQUAL'] = np.nan

    # Ekstraksi Marker
    df_marker = extract_markers_with_mean_depth(df)
    df_marker_iqual = extract_markers_customize(df, 'IQUAL')
    df_well_marker = df.copy()
    df_well_marker_iqual = df.copy()

    # Normalisasi Crossover
    df = normalize_xover(df, 'NPHI', 'RHOB')
    df = normalize_xover(df, 'RT', 'RHOB')
    df = normalize_xover(df, 'RT', 'GR')

    # --- 2. Konfigurasi Plot ---
    sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB',
                'VSH', 'PHIE', 'IQUAL', 'RT_RO']
    plot_sequence = {i + 1: v for i, v in enumerate(sequence)}

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

    # --- 3. Plot Setiap Track ---
    for n_seq, key in plot_sequence.items():
        # FIX: Ganti `col_to_plot` yang tidak terdefinisi dengan `key` dari loop
        if key == 'GR':
            fig, axes = plot_line(
                df, fig, axes, base_key=key, n_seq=n_seq, col=key, label=key)
        elif key == 'RT':
            fig, axes = plot_line(
                df, fig, axes, base_key=key, n_seq=n_seq, col=key, label=key)
        elif key == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(
                df, fig, axes, key, n_seq, counter, n_plots=subplot_col)
        elif key == 'VSH':
            fig, axes = plot_line(
                df, fig, axes, base_key=key, n_seq=n_seq, col=key, label=key)
        elif key == 'PHIE':
            fig, axes = plot_line(
                df, fig, axes, base_key=key, n_seq=n_seq, col=key, label=key)
        elif key == 'IQUAL':
            fig, axes = plot_flag(df_well_marker_iqual,
                                  fig, axes, 'IQUAL', n_seq)
            fig, axes = plot_texts_marker(
                df_marker_iqual, df_well_marker_iqual[depth].max(), fig, axes, key, n_seq)
        elif key == 'RT_RO':
            fig, axes, counter = plot_xover(
                df, fig, axes, key, n_seq, counter, n_plots=subplot_col, y_color='limegreen', n_color='lightgray')
        elif key == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, key, n_seq)
            fig, axes = plot_texts_marker(
                df_marker, df_well_marker[depth].max(), fig, axes, key, n_seq)

    fig = layout_range_all_axis(fig, axes, plot_sequence)
    fig = layout_draw_lines(fig, ratio_plots_seq, df, xgrid_intv=0)
    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(l=40, r=20, t=80, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified',
        height=1600
    )

    fig.update_yaxes(showspikes=True, range=[
        df[depth].max(), df[depth].min()])

    fig.update_traces(yaxis='y')

    return fig
