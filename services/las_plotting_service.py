"""
Service for plotting LAS files using the main plotting functions from plotting_service.py
"""
import os
import pandas as pd
import lasio
from typing import List, Dict, Any, Optional
from services.plotting_service import main_plot


def plot_las_file(file_path: str, sequence: List[str] = None, title: str = None) -> Dict[str, Any]:
    """
    Read a LAS file and create a plot using main_plot function.
    
    Args:
        file_path: Path to the LAS file
        sequence: List of curve names to plot (optional)
        title: Title for the plot (optional)
        
    Returns:
        Dict containing plot JSON and metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LAS file does not exist: {file_path}")
    
    if not file_path.lower().endswith('.las'):
        raise ValueError("File must be a LAS file (.las extension)")
    
    try:
        # Read LAS file
        las = lasio.read(file_path)
        
        # Convert to DataFrame and reset index
        df = las.df()
        df = df.reset_index()  # Make DEPT a column instead of index
        
        # Get available curves
        available_curves = df.columns.tolist()
        
        # Use provided sequence or default sequence
        if sequence is None:
            # Try to find common curves, fallback to first few available
            common_curves = ['DGRCC', 'ALCDLC', 'TNPL', 'R39PC', 'GR', 'RT', 'NPHI', 'RHOB']
            sequence = []
            for curve in common_curves:
                if curve in available_curves:
                    sequence.append(curve)
                if len(sequence) >= 4:  # Limit to 4 curves for readability
                    break
            
            # If no common curves found, use first available curves
            if not sequence:
                sequence = available_curves[1:min(5, len(available_curves))]  # Skip DEPT, take up to 4 curves
        
        # Filter sequence to only include available curves
        valid_sequence = [curve for curve in sequence if curve in available_curves]
        
        if not valid_sequence:
            raise ValueError(f"None of the requested curves {sequence} are available in the LAS file. Available curves: {available_curves}")
        
        # Set default title if not provided
        if title is None:
            file_name = os.path.basename(file_path)
            title = f"LAS Plot - {file_name}"
        
        # Create the plot using main_plot function
        fig = main_plot(df, valid_sequence, title=title)
        
        # Convert plot to JSON
        fig_json = fig.to_json()
        
        return {
            "success": True,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "title": title,
            "plotted_curves": valid_sequence,
            "available_curves": available_curves,
            "data_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "plot_json": fig_json
        }
        
    except Exception as e:
        raise Exception(f"Error plotting LAS file {file_path}: {str(e)}")


def plot_multiple_las_files(file_paths: List[str], sequence: List[str] = None, title: str = None) -> Dict[str, Any]:
    """
    Plot multiple LAS files in separate subplots or combined.
    
    Args:
        file_paths: List of paths to LAS files
        sequence: List of curve names to plot (optional)
        title: Title for the plot (optional)
        
    Returns:
        Dict containing plot JSON and metadata for all files
    """
    if not file_paths:
        raise ValueError("At least one file path must be provided")
    
    results = []
    combined_df = pd.DataFrame()
    
    for file_path in file_paths:
        try:
            # Read each LAS file
            las = lasio.read(file_path)
            df = las.df().reset_index()
            
            # Add file identifier column
            file_name = os.path.basename(file_path).replace('.las', '')
            df['FILE_SOURCE'] = file_name
            
            # Combine dataframes
            if combined_df.empty:
                combined_df = df
            else:
                # Align columns and combine
                common_cols = list(set(combined_df.columns) & set(df.columns))
                combined_df = pd.concat([combined_df[common_cols], df[common_cols]], ignore_index=True)
            
            results.append({
                "file_path": file_path,
                "file_name": file_name,
                "status": "success",
                "curves": df.columns.tolist(),
                "rows": len(df)
            })
            
        except Exception as e:
            results.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "status": "error",
                "error": str(e)
            })
    
    if combined_df.empty:
        raise Exception("No valid LAS files could be processed")
    
    # Get available curves from combined data
    available_curves = combined_df.columns.tolist()
    
    # Use provided sequence or default sequence
    if sequence is None:
        common_curves = ['DGRCC', 'ALCDLC', 'TNPL', 'R39PC', 'GR', 'RT', 'NPHI', 'RHOB']
        sequence = []
        for curve in common_curves:
            if curve in available_curves:
                sequence.append(curve)
            if len(sequence) >= 4:
                break
        
        if not sequence:
            sequence = [col for col in available_curves if col not in ['DEPT', 'FILE_SOURCE']][:4]
    
    # Filter sequence to only include available curves
    valid_sequence = [curve for curve in sequence if curve in available_curves]
    
    if not valid_sequence:
        raise ValueError(f"None of the requested curves {sequence} are available. Available curves: {available_curves}")
    
    # Set default title if not provided
    if title is None:
        title = f"Combined LAS Plot - {len(file_paths)} files"
    
    # Create the plot
    fig = main_plot(combined_df, valid_sequence, title=title)
    fig_json = fig.to_json()
    
    return {
        "success": True,
        "file_count": len(file_paths),
        "processed_files": results,
        "title": title,
        "plotted_curves": valid_sequence,
        "available_curves": available_curves,
        "combined_data_shape": {
            "rows": len(combined_df),
            "columns": len(combined_df.columns)
        },
        "plot_json": fig_json
    }


def get_las_curves_info(file_paths: List[str]) -> Dict[str, Any]:
    """
    Get curve information from multiple LAS files to help user select curves for plotting.
    
    Args:
        file_paths: List of paths to LAS files
        
    Returns:
        Dict containing curve information across all files
    """
    all_curves = {}
    file_info = []
    
    for file_path in file_paths:
        try:
            las = lasio.read(file_path)
            file_name = os.path.basename(file_path)
            
            file_curves = []
            for curve in las.curves:
                curve_info = {
                    "mnemonic": curve.mnemonic,
                    "unit": curve.unit,
                    "description": curve.descr
                }
                file_curves.append(curve_info)
                
                # Track all unique curves
                if curve.mnemonic not in all_curves:
                    all_curves[curve.mnemonic] = {
                        "unit": curve.unit,
                        "description": curve.descr,
                        "files": [file_name]
                    }
                else:
                    if file_name not in all_curves[curve.mnemonic]["files"]:
                        all_curves[curve.mnemonic]["files"].append(file_name)
            
            file_info.append({
                "file_path": file_path,
                "file_name": file_name,
                "curves": file_curves,
                "curve_count": len(file_curves)
            })
            
        except Exception as e:
            file_info.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "error": str(e),
                "curves": [],
                "curve_count": 0
            })
    
    # Suggest common curves for plotting
    common_curves = []
    total_files = len([f for f in file_info if "error" not in f])
    
    for curve, info in all_curves.items():
        if len(info["files"]) == total_files:  # Present in all files
            common_curves.append(curve)
    
    return {
        "total_files": len(file_paths),
        "processed_files": len([f for f in file_info if "error" not in f]),
        "failed_files": len([f for f in file_info if "error" in f]),
        "all_curves": all_curves,
        "common_curves": common_curves,
        "file_details": file_info,
        "suggested_sequence": common_curves[:4] if common_curves else list(all_curves.keys())[:4]
    }
