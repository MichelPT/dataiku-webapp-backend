import numpy as np
import pandas as pd
from scipy.stats import linregress
from services.plotting_service import (
    extract_markers_with_mean_depth,
    extract_markers_customize,
    main_plot,
    normalize_xover,
    plot_line,
    plot_xover_log_normal,
    plot_two_features_simple,
    plot_flag,
    plot_text_values,
    plot_texts_marker,
    layout_range_all_axis,
    layout_draw_lines,
    layout_axis,
    ratio_plots
)
from plotly.subplots import make_subplots


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
            df['IQUAL'] = np.where((df['PHIE'] > 0.1) &
                                   (df['VSH'] < 0.5), 1, 0)

        return df

    except Exception as e:
        print(f"Error in process_rgbe_rpbe: {str(e)}")
        raise e


def plot_rgbe_rpbe(df):
    """
    Create RGBE-RPBE visualization plot
    """
    sequence_rgbe = ['GR', 'RT', 'NPHI_RHOB', 'VSH', 'PHIE',
                     'IQUAL', 'RGBE_TEXT', 'RGBE', 'RPBE_TEXT', 'RPBE']
    fig = main_plot(df, sequence_rgbe, title="RGBE Selected Well")

    return fig
