"""
Service for navigating folder structures and handling CSV files within well directories.
"""
import os
import pandas as pd
from typing import List, Dict, Any, Optional


def get_folder_contents(base_path: str) -> Dict[str, Any]:
    """
    Get contents of a folder, separating files and directories.
    
    Args:
        base_path: Path to the folder to explore
        
    Returns:
        Dict containing folders and files information
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Path does not exist: {base_path}")
    
    folders = []
    files = []
    
    try:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Count files inside the folder
                try:
                    file_count = len([f for f in os.listdir(item_path) 
                                    if os.path.isfile(os.path.join(item_path, f))])
                    folders.append({
                        "name": item,
                        "type": "folder",
                        "file_count": file_count,
                        "path": item_path
                    })
                except PermissionError:
                    folders.append({
                        "name": item,
                        "type": "folder", 
                        "file_count": 0,
                        "path": item_path
                    })
            else:
                # Get file info
                file_size = os.path.getsize(item_path)
                file_ext = os.path.splitext(item)[1].lower()
                files.append({
                    "name": item,
                    "type": "file",
                    "extension": file_ext,
                    "size_bytes": file_size,
                    "path": item_path
                })
    
    except PermissionError:
        raise PermissionError(f"No permission to read directory: {base_path}")
    
    return {
        "current_path": base_path,
        "folders": sorted(folders, key=lambda x: x["name"]),
        "files": sorted(files, key=lambda x: x["name"]),
        "total_folders": len(folders),
        "total_files": len(files)
    }


def get_structure_wells_folders(field_name: str, structure_name: str) -> Dict[str, Any]:
    """
    Get folder contents specifically for a structure's wells directory.
    
    Args:
        field_name: Name of the field (e.g., 'adera')
        structure_name: Name of the structure (e.g., 'benuang')
        
    Returns:
        Dict containing folder and file information for the structure
    """
    base_path = f"data/structures/{field_name}/{structure_name}"
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Structure path does not exist: {base_path}")
    
    contents = get_folder_contents(base_path)
    
    # Add structure-specific information
    contents.update({
        "field_name": field_name,
        "structure_name": structure_name,
        "structure_path": base_path
    })
    
    return contents


def get_well_folder_files(field_name: str, structure_name: str, well_folder: str) -> Dict[str, Any]:
    """
    Get CSV files inside a specific well folder within a structure.
    
    Args:
        field_name: Name of the field
        structure_name: Name of the structure
        well_folder: Name of the well folder
        
    Returns:
        Dict containing CSV files information for the well folder
    """
    folder_path = f"data/structures/{field_name}/{structure_name}/{well_folder}"
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Well folder does not exist: {folder_path}")
    
    contents = get_folder_contents(folder_path)
    
    # Add well-specific information
    contents.update({
        "field_name": field_name,
        "structure_name": structure_name,
        "well_folder": well_folder,
        "well_path": folder_path
    })
    
    # Only get CSV files
    csv_files = [f for f in contents["files"] if f["extension"] == ".csv"]
    
    contents.update({
        "csv_files": csv_files,
        "csv_count": len(csv_files),
        "total_files": len(csv_files)  # Only count CSV files
    })
    
    # Remove non-CSV file information
    contents.pop("files", None)
    contents.pop("folders", None)
    contents.pop("total_folders", None)
    
    return contents


def read_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Read a CSV file and return its data and metadata.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dict containing CSV file data and metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file does not exist: {file_path}")
    
    if not file_path.lower().endswith('.csv'):
        raise ValueError("File must be a CSV file (.csv extension)")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, on_bad_lines='warn')
        
        # Basic statistics
        stats = {}
        for col in df.select_dtypes(include=['number']).columns:
            stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum())
            }
        
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "data_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "column_names": df.columns.tolist(),
            "column_types": {col: str(df[col].dtype) for col in df.columns},
            "statistics": stats,
            "data": df.to_dict('records')  # Include actual data
        }
        
    except Exception as e:
        raise Exception(f"Error reading CSV file {file_path}: {str(e)}")


def get_csv_file_summary(file_path: str) -> Dict[str, Any]:
    """
    Get summary information about a CSV file without loading all data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dict containing CSV file summary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file does not exist: {file_path}")
    
    try:
        # Read just the header to get column info
        df_sample = pd.read_csv(file_path, nrows=5, on_bad_lines='warn')
        
        # Get total row count efficiently
        with open(file_path, 'r') as f:
            row_count = sum(1 for line in f) - 1  # subtract header row
        
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "column_names": df_sample.columns.tolist(),
            "column_types": {col: str(df_sample[col].dtype) for col in df_sample.columns},
            "total_columns": len(df_sample.columns),
            "data_shape": {
                "rows": row_count,
                "columns": len(df_sample.columns)
            },
            "sample_data": df_sample.to_dict('records')  # First 5 rows as sample
        }
        
    except Exception as e:
        raise Exception(f"Error reading CSV file summary {file_path}: {str(e)}")