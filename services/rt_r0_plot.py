from plotly.subplots import make_subplots
from services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_line,
    plot_xover,
    plot_xover_log_normal,
    plot_two_features_simple,
    plot_flag,
    plot_text_values,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis
)


def plot_rt_r0(df, title="RT-R0 Analysis"):
    """
    RT-R0 plot logic matching Colab code: normalization, marker handling, plotting sequence, and options.
    """
    import numpy as np
    # 1. Handle IQUAL values
    if 'IQUAL' in df.columns:
        df.loc[df['IQUAL'] == 0, 'IQUAL'] = np.nan

    # 2. Marker extraction
    df_marker = extract_markers_with_mean_depth(df)
    try:
        extract_markers_customize = globals().get('extract_markers_customize')
    except Exception:
        extract_markers_customize = None
    if extract_markers_customize:
        df_marker_iqual = extract_markers_customize(df, 'IQUAL')
    else:
        df_marker_iqual = df_marker
    df_well_marker = df.copy()
    df_well_marker_iqual = df.copy()

    # 3. Normalization steps
    df = normalize_xover(df, 'NPHI', 'RHOB')
    df = normalize_xover(df, 'RT', 'RHOB')
    df = normalize_xover(df, 'RT', 'GR')

    # 4. Plot sequence and ratios
    sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB',
                'VSH', 'PHIE', 'IQUAL', 'RT_RO']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}
    ratio_plots = {
        'MARKER': 0.1, 'GR': 0.12, 'RT': 0.12, 'VSH': 0.12,
        'NPHI_RHOB': 0.18, 'IQUAL': 0.1, 'RO': 0.12,
        'RT_RO': 0.12, 'RWA': 0.12, 'PHIE': 0.12, 'RT_RO': 0.12
    }
    ratio_plots_seq = [ratio_plots.get(key, 0.1)
                       for key in plot_sequence.values()]
    subplot_col = len(plot_sequence.keys())

    # 5. Create subplots
    fig = make_subplots(
        rows=1, cols=subplot_col,
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.0
    )

    counter = 0
    axes = {i: [] for i in plot_sequence.values()}

    # 6. Plot each track
    for n_seq, col in plot_sequence.items():
        # Treat 'O' and '0' as equivalent in column names (e.g., RT_RO <-> RT_R0)
        col_variants = [col]
        # if 'O' in col:
        #     col_variants.append(col.replace('O', '0'))
        # if '0' in col:
        #     col_variants.append(col.replace('0', 'O'))

        # Find the first variant that exists in df.columns
        col_to_plot = next((c for c in col_variants if c in df.columns), None)

        # Only plot if the required column exists, or if it's a marker/flag/derived track
        if col_to_plot or col in ['MARKER', 'IQUAL', 'RT_RO', 'RT_O', 'NPHI_RHOB']:
            if col == 'GR' and col_to_plot:
                fig, axes = plot_line(
                    df, fig, axes, base_key=col_to_plot, n_seq=n_seq, col=col, label=col)
            elif col == 'RT' and col_to_plot:
                fig, axes = plot_line(
                    df, fig, axes, base_key=col_to_plot, n_seq=n_seq, col=col, label=col)
            elif col == 'NPHI_RHOB':
                fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                           y_color='rgba(0,0,0,0)', n_color='yellow', type=2, exclude_crossover=False)
            elif col == 'VSH' and col_to_plot:
                fig, axes = plot_line(
                    df, fig, axes, base_key=col_to_plot, n_seq=n_seq, col=col, label=col)
            elif col == 'PHIE' and col_to_plot:
                fig, axes = plot_line(
                    df, fig, axes, base_key=col_to_plot, n_seq=n_seq, col=col, label=col)
            elif col == 'IQUAL':
                fig, axes = plot_flag(df_well_marker_iqual,
                                      fig, axes, 'IQUAL', n_seq)
                fig, axes = plot_texts_marker(
                    df_marker_iqual, df_well_marker_iqual['DEPTH'].max(), fig, axes, col, n_seq)
            elif col in 'RT_RO' and (col_to_plot is not None):
                if 'plot_xover' in globals():
                    fig, axes, counter = plot_xover(
                        df, fig, axes, col_to_plot, n_seq, counter, n_plots=subplot_col, y_color='limegreen', n_color='lightgray')
                else:
                    fig, axes, counter = plot_xover_log_normal(df, fig, axes, col_to_plot, n_seq, counter, n_plots=subplot_col,
                                                               y_color='limegreen', n_color='lightgray', type=2, exclude_crossover=False)
            elif col == 'MARKER':
                fig, axes = plot_flag(df_well_marker, fig, axes, col, n_seq)
                fig, axes = plot_texts_marker(
                    df_marker, df_well_marker['DEPTH'].max(), fig, axes, col, n_seq)

    # 7. Finalize layout
    fig = layout_range_all_axis(fig, axes, plot_sequence)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=1800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified', hoverdistance=-1,
        title_text="RT RO",
        title_x=0.5,
        modebar_remove=['lasso', 'autoscale', 'zoom',
                        'zoomin', 'zoomout', 'pan', 'select']
    )
    if 'DEPTH' in df.columns:
        fig.update_yaxes(showspikes=True,
                         range=[df['DEPTH'].max(), df['DEPTH'].min()])
    fig.update_traces(yaxis='y')
    fig = layout_draw_lines(fig, ratio_plots_seq, df, xgrid_intv=0)
    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    return fig
