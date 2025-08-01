"""
Service for extracting fields and structures data from the structures folder.
"""
import os
import pandas as pd
from typing import Dict, List, Any


"""
Service for extracting fields and structures data from the structures folder.
"""
import os
import pandas as pd
from typing import Dict, List, Any


def get_fields_list() -> Dict[str, Any]:
    """
    Extract all available fields from data/structures folder.
    
    Returns:
        Dict containing list of fields with basic info
    """
    structures_dir = 'data/structures'
    
    if not os.path.exists(structures_dir):
        raise FileNotFoundError(f"Structures directory not found: {structures_dir}")
    
    # Get all field folders
    field_folders = [f for f in os.listdir(structures_dir) 
                    if os.path.isdir(os.path.join(structures_dir, f)) and not f.startswith('.')]
    
    fields = []
    for field_name in sorted(field_folders):
        field_path = os.path.join(structures_dir, field_name)
        
        # Count structure files in the field folder
        structure_files = [f for f in os.listdir(field_path) 
                          if f.endswith('.xlsx') and not f.startswith('~')]
        
        field_info = {
            'field_name': field_name,
            'structures_count': len(structure_files)
        }
        fields.append(field_info)
    
    return {
        'fields': fields,
        'total_fields': len(fields)
    }


def get_field_structures(field_name: str) -> Dict[str, Any]:
    """
    Get all structures for a specific field with detailed information.
    
    Args:
        field_name: Name of the field to get structures for
        
    Returns:
        Dict containing field information and its structures
    """
    structures_dir = 'data/structures'
    field_path = os.path.join(structures_dir, field_name)
    
    if not os.path.exists(field_path):
        raise FileNotFoundError(f"Field not found: {field_name}")
    
    # Get all Excel files in the field folder
    structure_files = [f for f in os.listdir(field_path) 
                      if f.endswith('.xlsx') and not f.startswith('~')]
    
    structures = []
    total_wells = set()
    total_records = 0
    
    for structure_file in sorted(structure_files):
        structure_name = structure_file.replace('.xlsx', '')
        structure_path = os.path.join(field_path, structure_file)
        
        try:
            # Read only the Well Name column for efficiency
            df_wells = pd.read_excel(structure_path, usecols=['Well Name'])
            
            # Extract unique wells from this structure
            wells = []
            if 'Well Name' in df_wells.columns:
                wells = df_wells['Well Name'].dropna().unique().tolist()
                total_wells.update(wells)
            
            # Get total record count
            record_count = len(df_wells)
            total_records += record_count
            
            # Read sample data for column info (just first few rows)
            df_sample = pd.read_excel(structure_path, nrows=5)
            columns = df_sample.columns.tolist() if not df_sample.empty else []
            
            structure_info = {
                'structure_name': structure_name,
                'field_name': field_name,
                'wells': sorted(wells),
                'wells_count': len(wells),
                'total_records': record_count,
                'columns': columns,
                'file_path': structure_path
            }
            
            structures.append(structure_info)
            
        except Exception as e:
            print(f"Error reading {structure_path}: {str(e)}")
            # Add structure with error info
            structure_info = {
                'structure_name': structure_name,
                'field_name': field_name,
                'wells': [],
                'wells_count': 0,
                'total_records': 0,
                'columns': [],
                'file_path': structure_path,
                'error': str(e)
            }
            structures.append(structure_info)
    
    return {
        'field_name': field_name,
        'structures': structures,
        'structures_count': len(structures),
        'total_wells': sorted(list(total_wells)),
        'total_wells_count': len(total_wells),
        'total_records': total_records
    }


def get_structure_details(field_name: str, structure_name: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific structure.
    
    Args:
        field_name: Name of the field
        structure_name: Name of the structure
        
    Returns:
        Dict containing detailed structure information
    """
    structures_dir = 'data/structures'
    structure_path = os.path.join(structures_dir, field_name, f"{structure_name}.xlsx")
    
    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    
    try:
        df = pd.read_excel(structure_path)
        
        # Extract wells
        wells = []
        if 'Well Name' in df.columns:
            wells = df['Well Name'].dropna().unique().tolist()
        
        # Get data statistics for numeric columns
        statistics = {}
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_columns:
            statistics[col] = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'count': int(df[col].count())
            }
        
        structure_details = {
            'field_name': field_name,
            'structure_name': structure_name,
            'file_path': structure_path,
            'wells': sorted(wells),
            'wells_count': len(wells),
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'statistics': statistics,
            'sample_data': df.head(10).to_dict('records'),  # First 10 rows
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        return structure_details
        
    except Exception as e:
        raise Exception(f"Error reading structure file: {str(e)}")


def get_well_details(well_name: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific well across all fields and structures.
    
    Args:
        well_name: Name of the well to search for
        
    Returns:
        Dict containing well information across all structures
    """
    # Get all fields first
    fields_data = get_fields_list()
    
    well_info = {
        'well_name': well_name,
        'found_in': [],
        'total_records': 0,
        'fields': [],
        'structures': []
    }
    
    # Search through all fields and structures
    for field in fields_data['fields']:
        field_name = field['field_name']
        
        try:
            # Get structures for this field
            field_structures = get_field_structures(field_name)
            
            # Check each structure for the well
            for structure in field_structures['structures']:
                if well_name in structure['wells']:
                    # Read the structure file to get well-specific data
                    try:
                        df = pd.read_excel(structure['file_path'])
                        well_data = df[df['Well Name'] == well_name]
                        
                        location_info = {
                            'field_name': structure['field_name'],
                            'structure_name': structure['structure_name'],
                            'records_count': len(well_data),
                            'sample_data': well_data.head(5).to_dict('records')
                        }
                        
                        well_info['found_in'].append(location_info)
                        well_info['total_records'] += len(well_data)
                        
                        if structure['field_name'] not in well_info['fields']:
                            well_info['fields'].append(structure['field_name'])
                        
                        well_info['structures'].append({
                            'field_name': structure['field_name'],
                            'structure_name': structure['structure_name']
                        })
                        
                    except Exception as e:
                        print(f"Error reading data for well {well_name} in {structure['file_path']}: {str(e)}")
                        
        except Exception as e:
            print(f"Error processing field {field_name}: {str(e)}")
            continue
    
    if not well_info['found_in']:
        raise FileNotFoundError(f"Well '{well_name}' not found in any structure")
    
    return well_info
