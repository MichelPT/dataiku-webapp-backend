# In your services/dns_dnsv_plot.py or equivalent file

from plotly.subplots import make_subplots
# Ensure all your helper functions are imported from your plotting service
from services.plotting_service import (
    extract_markers_with_mean_depth,
    normalize_xover,
    plot_line,
    plot_xover_log_normal,
    plot_flag,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis
)

def plot_dns_dnsv(df, title='DNS + DNSV Analysis'):
    """
    Creates a comprehensive DNS-DNSV plot based on the working Colab logic.
    """
    # 1. Pre-process Data (as in your Colab code)
    df = normalize_xover(df, 'NPHI', 'RHOB')
    df = normalize_xover(df, 'RT', 'RHOB') # Added based on your Colab code
    df_marker = extract_markers_with_mean_depth(df)

    # 2. Define Plot Sequence (using crossover tracks)
    sequence = ['MARKER', 'GR', 'RT_RHOB', 'NPHI_RHOB', 'VSH', 'DNS', 'DNSV']
    plot_sequence = {i + 1: v for i, v in enumerate(sequence)}

    # Define the layout ratios for the plot tracks
    ratio_plots = {
        'MARKER': 0.1,
        'GR': 0.15,
        'RT_RHOB': 0.2, # Crossover track
        'NPHI_RHOB': 0.2, # Crossover track
        'VSH': 0.1,
        'DNS': 0.15,
        'DNSV': 0.15,
    }
    ratio_plots_seq = [ratio_plots.get(key, 0.15) for key in plot_sequence.values()]
    
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
        if key in ['GR', 'VSH', 'DNS', 'DNSV']:
            fig, axes = plot_line(df, fig, axes, key, n_seq)
        
        elif key == 'MARKER':
            fig, axes = plot_flag(df, fig, axes, key, n_seq)
            fig, axes = plot_texts_marker(df_marker, df['DEPTH'].max(), fig, axes, key, n_seq)
        
        # ✨ This is the key fix: Use the dedicated crossover plot functions
        elif key == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, key, n_seq, counter, n_plots=len(plot_sequence))

        elif key == 'RT_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, key, n_seq, counter, n_plots=len(plot_sequence))


    # 5. Finalize Layout
    fig = layout_range_all_axis(fig, axes, plot_sequence)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=1400,
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