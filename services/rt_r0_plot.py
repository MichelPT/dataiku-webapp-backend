from plotly.subplots import make_subplots
from services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_line,
    plot_xover_log_normal,
    plot_two_features_simple,
    plot_flag,
    plot_text_values,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis
)

def plot_rt_r0(df):
    """Create RT-R0 visualization plot"""
    df_marker = extract_markers_with_mean_depth(df)
    df_well_marker = df.copy()
    
    # Define plot sequence
    sequence = ['MARKER', 'GR', 'RT', 'VSH', 'NPHI_RHOB', 'IQUAL', 'R0', 'RTR0', 'RWA']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}

    # Plot ratios
    ratio_plots = {
        'MARKER': 0.1,
        'GR': 0.2,
        'RT': 0.2,
        'VSH': 0.2,
        'NPHI_RHOB': 0.2,
        'IQUAL': 0.2,
        'R0': 0.2,
        'RTR0': 0.2,
        'RWA': 0.2
    }
    
    # Calculate ratios for subplot widths
    ratio_plots_seq = [ratio_plots[key] for key in plot_sequence.values()]

    # Create subplot
    subplot_col = len(plot_sequence.keys())
    fig = make_subplots(
        rows=1, cols=subplot_col,
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.0
    )

    # Initialize counters and axes
    counter = 0
    axes = {i: [] for i in plot_sequence.values()}

    # Plot each component
    for n_seq, col in plot_sequence.items():
        if col in ['GR', 'RT', 'VSH', 'R0', 'RTR0', 'RWA']:
            fig, axes = plot_line(df, fig, axes, base_key=col, n_seq=n_seq, col=col, label=col)
        elif col == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                     y_color='rgba(0,0,0,0)', n_color='yellow', type=2)
        elif col == 'IQUAL':
            fig, axes = plot_flag(df_well_marker, fig, axes, col, n_seq)
        elif col == 'MARKER':
            fig, axes = plot_flag(df_well_marker, fig, axes, col, n_seq)
            fig, axes = plot_texts_marker(df_marker, df_well_marker['DEPTH'].max(), fig, axes, col, n_seq)

    # Apply layouts
    fig = layout_range_all_axis(fig, axes, plot_sequence)

    # Update figure layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=1800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified',
        hoverdistance=-1,
        title_text="RT-R0 Analysis",
        title_x=0.5,
        modebar_remove=['lasso', 'autoscale', 'zoom', 'zoomin', 'zoomout', 'pan', 'select']
    )

    # Update axes
    fig.update_yaxes(
        showspikes=True,
        range=[df['DEPTH'].max(), df['DEPTH'].min()]
    )
    fig.update_traces(yaxis='y')

    # Apply final layouts
    fig = layout_draw_lines(fig, ratio_plots_seq, df, xgrid_intv=0)
    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    return fig
