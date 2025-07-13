import numpy as np
import pandas as pd
from scipy.stats import linregress
from services.plotting_service import (
    extract_markers_with_mean_depth,
    extract_markers_customize,
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
from plotly.subplots import make_subplots

# Configuration for plot ratios (this should match your existing ratio_plots dictionary)
ratio_plots = {
    'MARKER': 0.1,
    'GR': 0.2,
    'RT': 0.2,
    'VSH': 0.2,
    'NPHI_RHOB': 0.2,
    'IQUAL': 0.2,
    'RT_GR': 0.2,
    'RGBE': 0.2,
    'RGBE_TEXT': 0.1,
    'RT_PHIE': 0.2,
    'RPBE': 0.2,
    'RPBE_TEXT': 0.1
}

def calculate_iqual(df):
    """
    Calculate IQUAL based on conditions:
    IF (PHIE>0.1)&&(VSH<0.5): IQUAL = 1
    else: IQUAL = 0
    """
    df = df.copy()
    df['IQUAL'] = np.where((df['PHIE'] > 0.1) & (df['VSH'] < 0.5), 1, 0)
    return df

def group_by_seq(df, seq_col):
    """
    Group data based on sequential changes in a specific column
    """
    diff = df[seq_col].diff()
    seq_change = diff != 0
    group_id = seq_change.cumsum()
    df['GROUP_ID'] = group_id
    return df

def process_single_well(well_df):
    """
    Process analysis for a single well's data
    """
    # Calculate IQUAL first
    well_df = calculate_iqual(well_df)

    # Data cleaning & grouping
    df_clean = well_df.dropna()

    if len(df_clean) == 0:
        return pd.DataFrame()

    df_grouped = group_by_seq(df_clean, 'IQUAL')

    # Calculate slope & r-squared
    results_fluid = []
    for group_id, group in df_grouped.groupby('GROUP_ID'):
        n = len(group)

        # Skip invalid groups
        if (group['GR'].nunique() == 1) | (group['PHIE'].nunique() == 1) | (n <= 1):
            continue

        # Linear regression for slope and r-squared
        slope_rgbe, _, r_rgbe, _, _ = linregress(group['GR'], group['RT'])
        slope_rpbe, _, r_rpbe, _, _ = linregress(group['PHIE'], group['RT'])

        # Store results with 1 decimal rounding
        results_fluid.append({
            'GROUP_ID': group_id,
            'RGBE': round(100 * slope_rgbe, 1),
            'R_RGBE': round(r_rgbe, 1),
            'RPBE': round(slope_rpbe, 1),
            'R_RPBE': round(r_rpbe, 1),
        })

    if not results_fluid:
        return pd.DataFrame()

    df_results_fluid = pd.DataFrame(results_fluid)

    # Merge results with grouped data
    df_results = df_grouped.merge(df_results_fluid, on='GROUP_ID', how='left')

    # Filter and select required columns
    df_results = df_results.query('IQUAL > 0').dropna()
    df_results = df_results[['DEPTH']]  # First select just DEPTH
    # Now add the columns from df_results_fluid that we know exist
    df_results['RGBE'] = df_results_fluid['RGBE']
    df_results['R_RGBE'] = df_results_fluid['R_RGBE']
    df_results['RPBE'] = df_results_fluid['RPBE']
    df_results['R_RPBE'] = df_results_fluid['R_RPBE']

    return df_results

def process_rgbe_rpbe(df, params=None):
    """
    Main function to process RGBE-RPBE analysis
    """
    try:
        # Calculate IQUAL first
        df = calculate_iqual(df)
        
        # Process well's data
        well_results = process_single_well(df)
        
        if not well_results.empty:
            # Merge results back to original dataframe
            # First initialize the new columns with NaN
            for col in ['RGBE', 'R_RGBE', 'RPBE', 'R_RPBE']:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Now do the merge safely
            if not well_results.empty:
                # Create a temporary merge result
                merge_cols = ['DEPTH'] + [col for col in ['RGBE', 'R_RGBE', 'RPBE', 'R_RPBE'] 
                                        if col in well_results.columns]
                merge_result = df.merge(
                    well_results[merge_cols],
                    on=['DEPTH'],
                    how='left',
                    suffixes=('', '_new')
                )
                
                # Update only the columns that came from well_results
                for col in ['RGBE', 'R_RGBE', 'RPBE', 'R_RPBE']:
                    if col in well_results.columns:
                        df[col] = merge_result[col + '_new'].fillna(df[col])
        
        # Ensure IQUAL column exists
        if 'IQUAL' not in df.columns:
            df['IQUAL'] = np.where((df['PHIE'] > 0.1) & (df['VSH'] < 0.5), 1, 0)
        
        return df
    
    except Exception as e:
        print(f"Error in process_rgbe_rpbe: {str(e)}")
        raise e

def plot_rgbe_rpbe(df):
    """
    Create RGBE-RPBE visualization plot
    """
    df_marker = extract_markers_with_mean_depth(df)
    df_well_marker = df.copy()
    df_marker_rgbe = extract_markers_customize(df, 'RGBE')
    df_marker_rpbe = extract_markers_customize(df, 'RPBE')
    df_well_marker_rgbe = df.copy()
    df_well_marker_rpbe = df.copy()
    df_well_marker_iqual = df.copy()
    df_marker_iqual = extract_markers_customize(df, 'IQUAL')
    
    # Normalize crossovers
    df = normalize_xover(df, 'NPHI', 'RHOB')
    df = normalize_xover(df, 'RT', 'RHOB')
    df = normalize_xover(df, 'RT', 'GR')

    # Define plot sequence
    sequence = ['MARKER', 'GR', 'RT', 'VSH', 'NPHI_RHOB', 'IQUAL', 'RT_GR', 'RGBE', 'RGBE_TEXT', 'RT_PHIE', 'RPBE', 'RPBE_TEXT']
    plot_sequence = {i+1: v for i, v in enumerate(sequence)}

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
        if col == 'GR':
            fig, axes = plot_line(df, fig, axes, base_key='GR', n_seq=n_seq, col=col, label=col)
        elif col == 'RT':
            fig, axes = plot_line(df, fig, axes, base_key='RT', n_seq=n_seq, col=col, label=col)
        elif col == 'VSH':
            fig, axes = plot_line(df, fig, axes, base_key='VSH', n_seq=n_seq, col=col, label=col)
        elif col == 'NPHI_RHOB':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                     y_color='rgba(0,0,0,0)', n_color='yellow', type=2, exclude_crossover=False)
        elif col == 'RT_GR':
            fig, axes, counter = plot_xover_log_normal(df, fig, axes, col, n_seq, counter, n_plots=subplot_col,
                                                     y_color='limegreen', n_color='lightgray', type=1, exclude_crossover=False)
        elif col == 'RT_PHIE':
            fig, axes, counter = plot_two_features_simple(df, fig, axes, col, n_seq, counter, n_plots=subplot_col, log_scale=True)
        elif col == 'RGBE':
            fig, axes = plot_flag(df_well_marker_rgbe, fig, axes, col, n_seq)
        elif col == 'RPBE':
            fig, axes = plot_flag(df_well_marker_rpbe, fig, axes, col, n_seq)
        elif col == 'RGBE_TEXT':
            fig, axes = plot_text_values(df_marker_rgbe, df_well_marker_rgbe['DEPTH'].max(), fig, axes, col, n_seq)
        elif col == 'RPBE_TEXT':
            fig, axes = plot_text_values(df_marker_rpbe, df_well_marker_rpbe['DEPTH'].max(), fig, axes, col, n_seq)
        elif col == 'IQUAL':
            fig, axes = plot_flag(df_well_marker_iqual, fig, axes, 'IQUAL', n_seq)
            fig, axes = plot_texts_marker(df_marker_iqual, df_well_marker_iqual['DEPTH'].max(), fig, axes, col, n_seq)
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
        title_text="RGBE - RPBE",
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
