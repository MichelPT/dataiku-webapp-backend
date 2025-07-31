# -*- coding: utf-8 -*-
"""
Dataiku DSS API wrapper for well log analysis operations.
This module provides a unified interface for well log analysis in Dataiku environment.
"""

import pandas as pd
import numpy as np
try:
    import dataiku
except ImportError:
    # For development/testing outside Dataiku environment
    dataiku = None

# Import all service modules
from services.data_processing import (
    extract_markers_with_mean_depth,
    normalize_xover
)
from services.plotting_service import (
    plot_log_default,
    plot_normalization,
    plot_vsh_linear,
    plot_phie_den,
    plot_gsa_main,
    plot_smoothing,
    plot_sw_indo,
    plot_rwa_indo
)
from services.qc_service import apply_qc_pass
from services.trim_data import trim_data_to_markers
from services.vsh_calculation import calculate_vsh_linear
from services.porosity import calculate_porosity_density
from services.sw import calculate_sw_indonesia
from services.rwa import calculate_rwa_indonesia
from services.dgsa import calculate_dgsa
from services.ngsa import calculate_ngsa
from services.rgsa import calculate_rgsa
from services.gsa import calculate_gsa
from services.depth_matching import depth_matching_main
from services.dns_dnsv import calculate_dns_dnsv
import services.data_processing as pdu


class WellLogAnalysis:
    """
    Main class for well log analysis operations in Dataiku DSS.
    
    This class provides methods for:
    - Data loading and saving using Dataiku datasets
    - Quality control operations
    - Log calculations (VSH, porosity, saturation, etc.)
    - Plotting and visualization
    - Gas show anomaly analysis
    """
    
    def __init__(self):
        """Initialize the WellLogAnalysis class."""
        if dataiku is None:
            print("Warning: Running outside Dataiku environment. Dataset operations will not work.")
        
    def get_well_data(self, dataset_name):
        """
        Load well data from a Dataiku dataset.
        
        Args:
            dataset_name (str): Name of the Dataiku dataset
            
        Returns:
            pandas.DataFrame: Well log data
        """
        if dataiku is None:
            raise RuntimeError("Dataiku not available. Cannot load dataset.")
        
        dataset = dataiku.Dataset(dataset_name)
        df = dataset.get_dataframe()
        return df
    
    def save_well_data(self, df, dataset_name):
        """
        Save well data to a Dataiku dataset.
        
        Args:
            df (pandas.DataFrame): Well log data to save
            dataset_name (str): Name of the target Dataiku dataset
        """
        if dataiku is None:
            raise RuntimeError("Dataiku not available. Cannot save dataset.")
        
        dataset = dataiku.Dataset(dataset_name)
        dataset.write_with_schema(df)
    
    def list_available_datasets(self):
        """
        List all available datasets in the current project.
        
        Returns:
            list: List of dataset names
        """
        if dataiku is None:
            raise RuntimeError("Dataiku not available. Cannot list datasets.")
        
        client = dataiku.api_client()
        project = client.get_default_project()
        datasets = project.list_datasets()
        return [ds['name'] for ds in datasets]
    
    # Quality Control Methods
    def apply_quality_control(self, input_dataset_name, output_dataset_name, qc_params=None):
        """Apply quality control to well data"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            if qc_params is None:
                qc_params = {
                    'GR_min': 0, 'GR_max': 250,
                    'RT_min': 0.1, 'RT_max': 1000,
                    'NPHI_min': -0.15, 'NPHI_max': 0.45,
                    'RHOB_min': 1.5, 'RHOB_max': 3.0
                }
            
            df_qc = apply_qc_pass(df, qc_params)
            self.save_well_data(df_qc, output_dataset_name)
            
            return {"status": "success", "message": "Quality control applied successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def trim_data_by_markers(self, input_dataset_name, output_dataset_name, markers_to_keep):
        """Trim data to specified markers"""
        try:
            df = self.get_well_data(input_dataset_name)
            df_trimmed = trim_data_to_markers(df, markers_to_keep)
            self.save_well_data(df_trimmed, output_dataset_name)
            
            return {"status": "success", "message": "Data trimmed successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # Calculation Methods
    def calculate_vsh(self, input_dataset_name, output_dataset_name, params=None):
        """Calculate volume of shale"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            if params is None:
                params = {'GR_clean': 40, 'GR_shale': 140}
            
            df_with_vsh = calculate_vsh_linear(df, params)
            self.save_well_data(df_with_vsh, output_dataset_name)
            
            return {"status": "success", "message": "VSH calculation completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_porosity(self, input_dataset_name, output_dataset_name, params=None):
        """Calculate porosity from density"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            if params is None:
                params = {
                    'rho_matrix': 2.65,
                    'rho_fluid': 1.0,
                    'rho_shale': 2.45
                }
            
            df_with_porosity = calculate_porosity_density(df, params)
            self.save_well_data(df_with_porosity, output_dataset_name)
            
            return {"status": "success", "message": "Porosity calculation completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_water_saturation(self, input_dataset_name, output_dataset_name, params=None):
        """Calculate water saturation using Indonesia method"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            if params is None:
                params = {
                    'a': 1.0,
                    'm': 2.0,
                    'n': 2.0,
                    'Rw': 0.1
                }
            
            df_with_sw = calculate_sw_indonesia(df, params)
            self.save_well_data(df_with_sw, output_dataset_name)
            
            return {"status": "success", "message": "Water saturation calculation completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_rwa(self, input_dataset_name, output_dataset_name, params=None):
        """Calculate apparent water resistivity"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            if params is None:
                params = {
                    'a': 1.0,
                    'm': 2.0
                }
            
            df_with_rwa = calculate_rwa_indonesia(df, params)
            self.save_well_data(df_with_rwa, output_dataset_name)
            
            return {"status": "success", "message": "RWA calculation completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_gas_show_anomaly(self, input_dataset_name, output_dataset_name, params=None):
        """Calculate gas show anomalies (GSA)"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Calculate all GSA components
            df = calculate_dgsa(df, params or {})
            df = calculate_ngsa(df, params or {})
            df = calculate_rgsa(df, params or {})
            df = calculate_gsa(df, params or {})
            
            self.save_well_data(df, output_dataset_name)
            
            return {"status": "success", "message": "GSA calculation completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def calculate_dns_dnsv(self, input_dataset_name, output_dataset_name, params=None):
        """Calculate DNS and DNSV"""
        try:
            df = self.get_well_data(input_dataset_name)
            df_with_dns = calculate_dns_dnsv(df, params or {})
            self.save_well_data(df_with_dns, output_dataset_name)
            
            return {"status": "success", "message": "DNS/DNSV calculation completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_depth_matching(self, lwd_dataset_name, ref_dataset_name, output_dataset_name, params=None):
        """Run depth matching between LWD and reference data"""
        try:
            lwd_df = self.get_well_data(lwd_dataset_name)
            ref_df = self.get_well_data(ref_dataset_name)
            
            matched_df = depth_matching_main(lwd_df, ref_df, params or {})
            self.save_well_data(matched_df, output_dataset_name)
            
            return {"status": "success", "message": "Depth matching completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_interval_normalization(self, input_dataset_name, output_dataset_name, params, intervals):
        """Run interval normalization"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            log_in_col = params.get('LOG_IN', 'GR')
            log_out_col = params.get('LOG_OUT', 'GR_NORM')
            
            # Initialize output column
            df[log_out_col] = np.nan
            
            # Process each interval
            for interval in intervals:
                interval_mask = df['MARKER'] == interval
                if interval_mask.sum() == 0:
                    continue
                    
                log_data = df.loc[interval_mask, log_in_col].dropna().values
                if len(log_data) == 0:
                    continue
                    
                # Normalize the interval
                low_ref = float(params.get('LOW_REF', 40))
                high_ref = float(params.get('HIGH_REF', 140))
                low_in = int(params.get('LOW_IN', 3))
                high_in = int(params.get('HIGH_IN', 97))
                cutoff_min = float(params.get('CUTOFF_MIN', 0.0))
                cutoff_max = float(params.get('CUTOFF_MAX', 250.0))
                
                normalized = pdu.normalize_numeric_array(
                    log_data,
                    low_ref=low_ref,
                    high_ref=high_ref,
                    low_in=low_in,
                    high_in=high_in,
                    cutoff_min=cutoff_min,
                    cutoff_max=cutoff_max
                )
                
                df.loc[interval_mask, log_out_col] = normalized
            
            # Save results
            self.save_well_data(df, output_dataset_name)
            return {"status": "success", "message": "Interval normalization completed"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Plotting Methods
    def create_default_log_plot(self, input_dataset_name):
        """Create default log plot"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Extract markers and normalize data
            df_marker = extract_markers_with_mean_depth(df)
            df = normalize_xover(df, 'NPHI', 'RHOB')
            df = normalize_xover(df, 'RT', 'RHOB')
            
            # Create plot
            fig = plot_log_default(
                df=df,
                df_marker=df_marker,
                df_well_marker=df
            )
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_normalization_plot(self, input_dataset_name):
        """Create normalization plot"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Validate normalization data
            if 'GR_NORM' not in df.columns or df['GR_NORM'].isnull().all():
                return {"status": "error", "message": "No valid normalization data found"}
            
            # Create plot
            fig = plot_normalization(df)
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_vsh_plot(self, input_dataset_name):
        """Create VSH plot with default logs: GR, RT, NPHI_RHOB, VSH_LINEAR"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Check for VSH data
            if 'VSH_LINEAR' not in df.columns:
                return {"status": "error", "message": "No VSH_LINEAR data found"}
            
            # Create plot
            df_marker = extract_markers_with_mean_depth(df)
            fig = plot_vsh_linear(
                df=df,
                df_marker=df_marker,
                df_well_marker=df
            )
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_porosity_plot(self, input_dataset_name):
        """Create porosity plot with default logs: GR, RT, NPHI_RHOB, PHIE, PHIT"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Check required columns
            required_cols = ['PHIE', 'PHIT']
            if not all(col in df.columns for col in required_cols):
                return {"status": "error", "message": "Missing required porosity data (PHIE, PHIT)"}
            
            # Create plot
            df_marker = extract_markers_with_mean_depth(df)
            fig = plot_phie_den(
                df=df,
                df_marker=df_marker,
                df_well_marker=df
            )
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_gsa_plot(self, input_dataset_name):
        """Create GSA plot"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Check required columns
            required_cols = ['GR', 'RT', 'NPHI', 'RHOB', 'RGSA', 'NGSA', 'DGSA']
            if not all(col in df.columns for col in required_cols):
                return {"status": "error", "message": "Missing required GSA data"}
            
            # Create plot
            fig = plot_gsa_main(df)
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_smoothing_plot(self, input_dataset_name):
        """Create smoothing plot"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Check required columns
            required_cols = ['GR', 'GR_MovingAvg_5', 'GR_MovingAvg_10']
            if not all(col in df.columns for col in required_cols):
                return {"status": "error", "message": "Missing smoothing data"}
            
            # Create plot
            df_marker = extract_markers_with_mean_depth(df)
            fig = plot_smoothing(
                df=df,
                df_marker=df_marker,
                df_well_marker=df
            )
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_sw_plot(self, input_dataset_name):
        """Create water saturation plot with default logs: GR, RT, NPHI_RHOB, SWE_INDO"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Check required columns
            if 'SWE_INDO' not in df.columns:
                return {"status": "error", "message": "Missing water saturation data (SWE_INDO)"}
            
            # Create plot
            df_marker = extract_markers_with_mean_depth(df)
            fig = plot_sw_indo(
                df=df,
                df_marker=df_marker,
                df_well_marker=df
            )
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_rwa_plot(self, input_dataset_name):
        """Create RWA plot with default logs: GR, RT, NPHI_RHOB, RWA_FULL, RWA_SIMPLE"""
        try:
            df = self.get_well_data(input_dataset_name)
            
            # Check required columns
            required_cols = ['RWA_FULL', 'RWA_SIMPLE']
            if not all(col in df.columns for col in required_cols):
                return {"status": "error", "message": "Missing RWA data (RWA_FULL, RWA_SIMPLE)"}
            
            # Create plot
            df_marker = extract_markers_with_mean_depth(df)
            fig = plot_rwa_indo(
                df=df,
                df_marker=df_marker,
                df_well_marker=df
            )
            
            return {"status": "success", "figure": fig.to_dict()}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Example usage functions for Dataiku notebooks
def example_basic_workflow():
    """
    Example of basic workflow for well log analysis in Dataiku.
    
    This function demonstrates how to use the WellLogAnalysis class
    in a Dataiku notebook or recipe.
    """
    
    # Initialize the analysis class
    analyzer = WellLogAnalysis()
    
    # Step 1: Load and apply QC
    qc_params = {
        'GR_min': 0, 'GR_max': 250,
        'RT_min': 0.1, 'RT_max': 1000,
        'NPHI_min': -0.15, 'NPHI_max': 0.45,
        'RHOB_min': 1.5, 'RHOB_max': 3.0
    }
    analyzer.apply_quality_control("raw_well_data", "qc_well_data", qc_params)
    
    # Step 2: Calculate VSH
    vsh_params = {'GR_clean': 40, 'GR_shale': 140}
    analyzer.calculate_vsh("qc_well_data", "vsh_well_data", vsh_params)
    
    # Step 3: Calculate porosity
    porosity_params = {
        'rho_matrix': 2.65,
        'rho_fluid': 1.0,
        'rho_shale': 2.45
    }
    analyzer.calculate_porosity("vsh_well_data", "porosity_well_data", porosity_params)
    
    # Step 4: Calculate water saturation
    sw_params = {
        'a': 1.0,
        'm': 2.0,
        'n': 2.0,
        'Rw': 0.1
    }
    analyzer.calculate_water_saturation("porosity_well_data", "final_well_data", sw_params)
    
    # Step 5: Create plots
    default_plot = analyzer.create_default_log_plot("final_well_data")
    vsh_plot = analyzer.create_vsh_plot("final_well_data")
    porosity_plot = analyzer.create_porosity_plot("final_well_data")
    sw_plot = analyzer.create_sw_plot("final_well_data")
    
    return {
        "default_plot": default_plot,
        "vsh_plot": vsh_plot,
        "porosity_plot": porosity_plot,
        "sw_plot": sw_plot
    }


def example_gsa_workflow():
    """
    Example workflow for gas show anomaly analysis.
    """
    
    analyzer = WellLogAnalysis()
    
    # Calculate GSA
    gsa_params = {
        'baseline_method': 'percentile',
        'percentile': 10
    }
    analyzer.calculate_gas_show_anomaly("input_data", "gsa_data", gsa_params)
    
    # Create GSA plot
    gsa_plot = analyzer.create_gsa_plot("gsa_data")
    
    return gsa_plot


# Initialize the main analysis object
# This can be imported and used directly in Dataiku recipes
well_log_analyzer = WellLogAnalysis()
