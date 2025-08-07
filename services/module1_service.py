import pandas as pd
import os
from services.plotting_service import plot_module1

def get_module1_plot_data(file_path):
    """
    Service untuk mendapatkan plot Module1 dari single CSV file.
    Auto-detect antara LWD dan WL data berdasarkan kolom yang tersedia.
    
    Args:
        file_path: Full path to the CSV file
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if it's a CSV file
        if not file_path.lower().endswith('.csv'):
            raise ValueError(f"File is not a CSV file: {file_path}")
        
        # Read CSV file
        df = pd.read_csv(file_path, on_bad_lines='warn')
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Generate Module1 plot (auto-detects LWD vs WL inside plot_module1)
        fig = plot_module1(df)
        
        return {
            'success': True,
            'figure': fig
        }
        
    except FileNotFoundError as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': 'file_not_found'
        }
    except pd.errors.EmptyDataError:
        return {
            'success': False,
            'error': "CSV file is empty or corrupted",
            'error_type': 'empty_file'
        }
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': 'value_error'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': 'unexpected_error'
        }
        
    except FileNotFoundError as e:
        return {
            'success': False,
            'error': f"File not found: {str(e)}",
            'error_type': 'file_not_found'
        }
    except pd.errors.EmptyDataError:
        return {
            'success': False,
            'error': "One or more files are empty or corrupted",
            'error_type': 'empty_file'
        }
    except ValueError as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': 'value_error'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': 'unexpected_error'
        }
