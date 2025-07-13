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

def plot_rt_r0(df, title="RT-R0 Analysis"):
    """
    Creates a comprehensive RT-R0 visualization plot,
    incorporating the correct pre-processing from your Colab notebook.
    """
    # 1. Pre-process Data (from your working Colab code)
    # This is the missing step that creates the '_NORM' columns.
    df = normalize_xover(df, 'NPHI', 'RHOB')
    df = normalize_xover(df, 'RT', 'RHOB') # Also include other normalizations from your Colab version
    df_marker = extract_markers_with_mean_depth(df)
    
    # 2. Define Plot Sequence and Layout
    # Using the sequence from your backend code as it's more comprehensive
    sequence = ['MARKER','GR','RT','NPHI_RHOB','VSH','PHIE','IQUAL','RT_RO']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}

    ratio_plots = {
        'MARKER': 0.1, 'GR': 0.12, 'RT': 0.12, 'VSH': 0.12,
        'NPHI_RHOB': 0.18, 'IQUAL': 0.1, 'R0': 0.12,
        'RTR0': 0.12, 'RWA': 0.12
    }
    ratio_plots_seq = [ratio_plots.get(key, 0.1) for key in plot_sequence.values()]

    # 3. Create Subplots
    fig = make_subplots(
        rows=1, cols=len(plot_sequence),
        shared_yaxes=True,
        column_widths=ratio_plots_seq,
        horizontal_spacing=0.01
    )

    # 4. Plot Each Track
    counter = 0
    axes = {val: [] for val in plot_sequence.values()}

    for n_seq, key in plot_sequence.items():
        if key in ['GR', 'RT', 'VSH', 'R0', 'RTR0', 'RWA']:
            fig, axes = plot_line(df, fig, axes, key, n_seq)
        
        elif key == 'NPHI_RHOB':
            # This call will now succeed because the data has been normalized
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, key, n_seq, counter, n_plots=len(plot_sequence))
            
        elif key == 'IQUAL':
            fig, axes = plot_flag(df, fig, axes, key, n_seq)
            
        elif key == 'MARKER':
            fig, axes = plot_flag(df, fig, axes, key, n_seq)
            fig, axes = plot_texts_marker(df_marker, df['DEPTH'].max(), fig, axes, key, n_seq)

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