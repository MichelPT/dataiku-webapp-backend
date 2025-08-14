import pandas as pd
import lasio

# --- User Configuration ---
# Please update these file paths to match your files.

# Path to your Log ASCII Standard (.las) file
las_file_path = 'data/structures/adera/benuang/BNG-54_Property&Minsol.las' 

# Path to your existing Comma-Separated Values (.csv) file
csv_file_path = 'data/structures/adera/benuang/BNG-054.csv'

# Path for the new output file with the merged data
output_csv_path = 'data/structures/adera/benuang/BNG-054-minsol.csv'

# List of curve mnemonics (column names) to copy from the .las file.
# This script is case-insensitive, so 'sw' will match 'SW', 'Sw', etc.
curves_to_copy = ['PHIE', 'PHIT', 'SW', 'VCLAY', 'PERM-FZI']

# Optional: Define custom names for the output columns.
# The key should be the uppercase version of the name in 'curves_to_copy'.
# The value is the desired name in the final CSV file.
output_column_names = {
    'PERM-FZI': 'PERM',
    'VCLAY': 'VSH'
}

# --- Script Execution ---

def merge_las_to_csv(las_path, csv_path, output_path, curves, output_names={}):
    """
    Reads a .las file and a .csv file, merges data based on depth,
    and saves the result to a new .csv file. This process is case-insensitive
    and allows for custom column renaming and derived curve creation.
    """
    try:
        # 1. Load the existing CSV file into a pandas DataFrame
        print(f"Reading existing CSV file from: {csv_path}")
        existing_df = pd.read_csv(csv_path)
        
        # Check if the 'DEPTH' column exists in the CSV
        if 'DEPTH' not in existing_df.columns:
            print(f"Error: A 'DEPTH' column was not found in {csv_path}.")
            print("Please ensure your CSV has a column named 'DEPTH'.")
            return
            
        # 2. Load the .las file
        print(f"Reading LAS file from: {las_path}")
        las = lasio.read(las_path)
        
        # 3. Convert the .las file data to a pandas DataFrame
        las_df = las.df()
        las_df.reset_index(inplace=True)
        if 'DEPT' in las_df.columns:
            las_df.rename(columns={'DEPT': 'DEPTH'}, inplace=True)
        
        # 4. Handle derived curves and case-insensitivity
        processed_curves = list(curves)
        las_curves_map = {c.upper(): c for c in las_df.columns}

        # --- Logic to handle PHIE/PHIT transformation ---
        if 'PHIE' in las_curves_map and 'PHIT' in las_curves_map:
            print("Found PHIE and PHIT. Creating derived 'PHIE_PHIT' column (PHIT - PHIE).")
            
            phie_col = las_curves_map['PHIE']
            phit_col = las_curves_map['PHIT']
            
            # Calculate secondary porosity and add it as a new column to the LAS DataFrame
            las_df['PHIE_PHIT'] = las_df[phit_col] - las_df[phie_col]
            
            # Update the list of curves to merge: remove PHIT and add PHIE_PHIT
            processed_curves = [c for c in processed_curves if c.upper() != 'PHIT']
            processed_curves.append('PHIE_PHIT')
            
            print(f"Updated curves to merge: {processed_curves}")

        # Create a new map that includes the derived column if it was created
        las_curves_map_updated = {c.upper(): c for c in las_df.columns}
        
        # Convert the processed list of desired curves to uppercase for matching
        uppercase_curves_to_copy = [c.upper() for c in processed_curves]
        
        # Find which requested curves are available (case-insensitively)
        available_original_case = []
        missing_curves = []
        
        for uc_curve in uppercase_curves_to_copy:
            if uc_curve in las_curves_map_updated:
                available_original_case.append(las_curves_map_updated[uc_curve])
            else:
                missing_curves.append(uc_curve)
        
        if not available_original_case:
            print("Error: None of the requested curves were found in the LAS file (case-insensitive search).")
            print(f"Available curves are: {list(las_df.columns)}")
            return
            
        if missing_curves:
            print(f"Warning: The following curves were not found and will be skipped: {missing_curves}")

        # Create a subset DataFrame using the original case names
        las_subset_df = las_df[['DEPTH'] + available_original_case].copy()
        
        # Rename the columns in the subset for the final output
        rename_dict = {'DEPTH': 'DEPTH'}
        for original_case_name in available_original_case:
            uc_name = original_case_name.upper()
            # Check if a custom output name is defined, otherwise use the uppercase name
            rename_dict[original_case_name] = output_names.get(uc_name, uc_name)
        las_subset_df.rename(columns=rename_dict, inplace=True)
        
        # 5. Merge the DataFrames
        
        # --- NEW: Logic to handle replacing existing VSH column ---
        # Check if any of the new column names already exist in the original CSV
        # and drop them to ensure they are replaced by the new data.
        new_column_names = list(rename_dict.values())
        for col_name in new_column_names:
            if col_name in existing_df.columns and col_name != 'DEPTH':
                print(f"Found existing column '{col_name}' in CSV. It will be replaced with data from the LAS file.")
                existing_df.drop(columns=[col_name], inplace=True)
        # --- END NEW LOGIC ---

        # We use a 'left' merge to keep all rows from your original CSV.
        # Using pd.merge_asof for nearest-value matching, which is robust for log data.
        
        # Ensure both depth columns are sorted for merge_asof
        existing_df = existing_df.sort_values('DEPTH')
        las_subset_df = las_subset_df.sort_values('DEPTH')

        print("Merging data based on depth...")
        merged_df = pd.merge_asof(
            left=existing_df,
            right=las_subset_df,
            on='DEPTH',
            direction='nearest' # Finds the closest depth value in the LAS file
        )
        
        # 6. Save the merged DataFrame to a new CSV file
        merged_df.to_csv(output_path, index=False)
        print(f"Success! Merged data has been saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check your file paths.")
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    merge_las_to_csv(las_file_path, csv_file_path, output_csv_path, curves_to_copy, output_column_names)
