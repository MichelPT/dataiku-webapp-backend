import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_line,
    plot_xover_log_normal,
    plot_four_features_simple, 
    plot_flag,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis
)

def plot_swgrad(df, title='SWGRAD Analysis'):
    """
    Creates a comprehensive SWGRAD visualization plot based on the working Colab logic.
    """
    # 1. Pre-process Data
    df = normalize_xover(df, 'NPHI', 'RHOB')
    df_marker = extract_markers_with_mean_depth(df)

    # 2. Define Plot Sequence and Layout
    sequence = ['MARKER', 'GR', 'RT', 'NPHI_RHOB', 'SWARRAY', 'SWGRAD']
    plot_sequence = {i + 1: v for i, v in enumerate(sequence)}

    # Define the layout ratios for the plot tracks
    ratio_plots = {
        'MARKER': 0.1,
        'GR': 0.15,
        'RT': 0.15,
        'NPHI_RHOB': 0.2,
        'SWARRAY': 0.25, # Give the main array track more space
        'SWGRAD': 0.15,
    }
    ratio_plots_seq = [ratio_plots[key] for key in plot_sequence.values()]
    
    # 3. Create Subplots
    fig = make_subplots(
        rows=1, cols=len(plot_sequence),
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.01
    )

    # 4. Plot Each Track According to its Type
    counter = 0
    axes = {val: [] for val in plot_sequence.values()}

    for n_seq, key in plot_sequence.items():
        if key == 'MARKER':
            fig, axes = plot_flag(df, fig, axes, key, n_seq)
            fig, axes = plot_texts_marker(df_marker, df['DEPTH'].max(), fig, axes, key, n_seq)
        
        elif key in ['GR', 'RT', 'SWGRAD']:
            fig, axes = plot_line(df, fig, axes, key, n_seq)
            
        elif key == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, key, n_seq, counter, n_plots=len(plot_sequence))
        
        # This is the key fix: Use the dedicated function for the SWARRAY track
        elif key == 'SWARRAY':
            # This assumes your 'plot_four_features_simple' function is designed
            # to find and plot the relevant SWARRAY_ curves from the dataframe.
            fig, axes, counter = plot_four_features_simple(df, fig, axes, key, n_seq, counter, n_plots=len(plot_sequence))

    # 5. Finalize Layout
    fig = layout_range_all_axis(fig, axes, plot_sequence)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=1800,
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        hovermode='y unified',
        title_text=title,
        title_x=0.5,
    )
    fig.update_yaxes(
        showspikes=True,
        range=[df['DEPTH'].max(), df['DEPTH'].min()],
        autorange=False 
    )
    fig.update_traces(yaxis='y')
    fig = layout_draw_lines(fig, ratio_plots_seq, df, xgrid_intv=50)
    fig = layout_axis(fig, axes, ratio_plots_seq, plot_sequence)

    return fig