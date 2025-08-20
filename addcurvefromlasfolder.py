# import pandas as pd
# import lasio
# import os
# import re

# # --- User Configuration ---
# # Please update these folder paths to match your directory structure.

# # Folder containing the well data CSV files (e.g., 'BNG-054.csv')
# csv_folder = 'data/structures/adera/benuang'

# # Folder containing the Log ASCII Standard files (e.g., 'BNG-54_Property&Minsol.las')
# las_folder = 'data/BENUANG-minsol-las'

# # Folder where the modified well CSVs with the new log data will be saved.
# # This folder will be created if it doesn't exist.
# output_folder = 'data/BENUANG-minsol-merged'

# # --- NEW: Define curves and their possible aliases ---
# # The script will search for aliases in the order they are listed.
# # The key is the final column name you want in the CSV.
# curves_to_find = {
#     'VSH': ['VCLAY', 'VCL', 'VSH'],
#     'PERM': ['PERM-FZI', 'PERM-PZI', 'PERM'],
#     'PHIE': ['PHIE'],
#     'PHIT': ['PHIT'],
#     'SW': ['SW']
# }


# # --- Script Execution ---

# def normalize_well_name(filename):
#     """
#     Extracts a standardized well name from a filename for matching.
#     Handles variations like 'GNK-055', 'GNK-55', and 'GNK-55ST'.
#     e.g., 'GNK-055.csv' -> 'gnk55'
#     e.g., 'GNK-55ST_Property.las' -> 'gnk55'
#     """
#     # Convert to lowercase for consistent matching
#     name = filename.lower()
    
#     # Use regex to find the letter prefix and the number part of the well name.
#     match = re.search(r'([a-z]+)[_-]*(\d+)', name)
    
#     if match:
#         prefix = match.group(1)
#         # Convert number to integer to remove leading zeros (e.g., '055' -> 55)
#         number = int(match.group(2)) 
#         return f"{prefix}{number}"
#     else:
#         # Fallback for any names that do not match the expected pattern.
#         name = re.sub(r'[^a-z0-9]', '', name)
#         name = name.replace('propertyminsol', '')
#         name = name.replace('csv', '')
#         name = name.replace('las', '')
#         return name

# def merge_las_to_csv_folder(csv_dir, las_dir, output_dir, curve_aliases):
#     """
#     Main function to process all well files, match them with LAS files,
#     and merge specified log data using a list of possible aliases for each curve.
#     """
#     # 1. Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created output directory: {output_dir}")

#     # 2. Map normalized names to full paths for all LAS files
#     las_files_map = {}
#     try:
#         for f in os.listdir(las_dir):
#             if f.lower().endswith('.las'):
#                 norm_name = normalize_well_name(f)
#                 if norm_name: # Ensure a valid name was extracted
#                     las_files_map[norm_name] = os.path.join(las_dir, f)
#     except FileNotFoundError:
#         print(f"Error: The LAS folder was not found at '{las_dir}'. Please check the path.")
#         return

#     # 3. Iterate through each well CSV file and process it
#     try:
#         csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
#         if not csv_files:
#             print(f"No CSV files found in the folder: '{csv_dir}'")
#             return
            
#         print(f"\nFound {len(csv_files)} well CSV(s) to process...")

#         for csv_filename in csv_files:
#             csv_filepath = os.path.join(csv_dir, csv_filename)
#             norm_csv_name = normalize_well_name(csv_filename)
            
#             print(f"\n--- Processing: {csv_filename} ---")

#             # 4. Find the matching LAS file
#             if norm_csv_name and norm_csv_name in las_files_map:
#                 las_filepath = las_files_map[norm_csv_name]
#                 print(f"Found matching LAS file: {os.path.basename(las_filepath)}")

#                 try:
#                     # Load existing CSV
#                     existing_df = pd.read_csv(csv_filepath)
#                     if 'DEPTH' not in existing_df.columns:
#                         print(f"  - Skipping: 'DEPTH' column not found in {csv_filename}.")
#                         continue

#                     # Load LAS file
#                     las = lasio.read(las_filepath)
#                     las_df = las.df()
#                     las_df.reset_index(inplace=True)
#                     if 'DEPT' in las_df.columns:
#                         las_df.rename(columns={'DEPT': 'DEPTH'}, inplace=True)
                    
#                     # --- NEW: Find available curves using the alias dictionary ---
#                     las_curves_map = {c.upper(): c for c in las_df.columns}
#                     rename_dict = {'DEPTH': 'DEPTH'}
#                     available_original_case = []

#                     for target_name, aliases in curve_aliases.items():
#                         found_alias = False
#                         for alias in aliases:
#                             if alias.upper() in las_curves_map:
#                                 original_case_name = las_curves_map[alias.upper()]
#                                 available_original_case.append(original_case_name)
#                                 rename_dict[original_case_name] = target_name
#                                 print(f"  - Found '{original_case_name}' for target '{target_name}'.")
#                                 found_alias = True
#                                 break # Stop searching for this target once an alias is found
#                         if not found_alias:
#                              print(f"  - Warning: Could not find any alias for '{target_name}'.")
                    
#                     if not available_original_case:
#                         print(f"  - Skipping: None of the requested curves were found in {os.path.basename(las_filepath)}.")
#                         continue

#                     las_subset_df = las_df[['DEPTH'] + available_original_case].copy()
#                     las_subset_df.rename(columns=rename_dict, inplace=True)
                    
#                     # Drop existing columns from CSV to be replaced
#                     for col_name in rename_dict.values():
#                         if col_name in existing_df.columns and col_name != 'DEPTH':
#                             print(f"  - Replacing existing column '{col_name}'.")
#                             existing_df.drop(columns=[col_name], inplace=True)
                    
#                     # Sort and merge
#                     existing_df.sort_values('DEPTH', inplace=True)
#                     las_subset_df.sort_values('DEPTH', inplace=True)

#                     merged_df = pd.merge_asof(
#                         left=existing_df,
#                         right=las_subset_df,
#                         on='DEPTH',
#                         direction='nearest'
#                     )
                    
#                     # Save the result
#                     output_filename = os.path.splitext(csv_filename)[0] + '-minsol.csv'
#                     output_filepath = os.path.join(output_dir, output_filename)
#                     merged_df.to_csv(output_filepath, index=False)
#                     print(f"  - Success! Saved merged file to: {output_filepath}")

#                 except Exception as e:
#                     print(f"  - An error occurred while processing {csv_filename}: {e}")

#             else:
#                 print(f"  - Warning: No matching LAS file found for {csv_filename}.")
                
#     except FileNotFoundError:
#         print(f"Error: The CSV folder was not found at '{csv_dir}'. Please check the path.")
#         return

# # Run the main function
# if __name__ == "__main__":
#     merge_las_to_csv_folder(csv_folder, las_folder, output_folder, curves_to_find)

import pandas as pd
import lasio
import os
import re

# --- User Configuration ---
# Please update these folder paths to match your directory structure.

# Folder containing the well data CSV files (e.g., 'BNG-054.csv')
csv_folder = 'data/structures/adera/benuang'

# Folder containing the Log ASCII Standard files (e.g., 'BNG-54_Property&Minsol.las')
las_folder = 'data/BENUANG-minsol-las'

# Folder where the modified well CSVs with the new log data will be saved.
# This folder will be created if it doesn't exist.
output_folder = 'data/BENUANG-minsol-merged'

# --- NEW: Define curves and their possible aliases ---
# The script will search for aliases in the order they are listed.
# The key is the final column name you want in the CSV.
curves_to_find = {
    'VSH': ['VCLAY', 'VCL', 'VSH'],
    'PERM': ['PERM-FZI', 'PERM-PZI', 'PERM'],
    'PHIE': ['PHIE'],
    'PHIT': ['PHIT'],
    'SW': ['SW']
}


# --- Script Execution ---

def normalize_well_name(filename):
    """
    Extracts a standardized well name from a filename for matching.
    Handles variations like 'GNK-055', 'GNK-55', and 'GNK-55ST'.
    e.g., 'GNK-055.csv' -> 'gnk55'
    e.g., 'GNK-55ST_Property.las' -> 'gnk55'
    """
    # Convert to lowercase for consistent matching
    name = filename.lower()
    
    # Use regex to find the letter prefix and the number part of the well name.
    match = re.search(r'([a-z]+)[_-]*(\d+)', name)
    
    if match:
        prefix = match.group(1)
        # Convert number to integer to remove leading zeros (e.g., '055' -> 55)
        number = int(match.group(2)) 
        return f"{prefix}{number}"
    else:
        # Fallback for any names that do not match the expected pattern.
        name = re.sub(r'[^a-z0-9]', '', name)
        name = name.replace('propertyminsol', '')
        name = name.replace('csv', '')
        name = name.replace('las', '')
        return name

def merge_las_to_csv_folder(csv_dir, las_dir, output_dir, curve_aliases):
    """
    Main function to process all well files, match them with LAS files,
    and merge specified log data using a list of possible aliases for each curve.
    """
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Map normalized names to full paths for all LAS files, prioritizing 'Property&Minsol'
    las_file_groups = {}
    try:
        for f in os.listdir(las_dir):
            if f.lower().endswith('.las'):
                norm_name = normalize_well_name(f)
                if norm_name:
                    if norm_name not in las_file_groups:
                        las_file_groups[norm_name] = []
                    las_file_groups[norm_name].append(os.path.join(las_dir, f))
    except FileNotFoundError:
        print(f"Error: The LAS folder was not found at '{las_dir}'. Please check the path.")
        return

    las_files_map = {}
    for norm_name, file_list in las_file_groups.items():
        # Search for the prioritized file containing 'propertyminsol'
        priority_file = None
        for file_path in file_list:
            if 'propertyminsol' in os.path.basename(file_path).lower():
                priority_file = file_path
                break # Found it, no need to look further in this group
        
        if priority_file:
            las_files_map[norm_name] = priority_file
        elif file_list: # If no priority file was found, but the list is not empty
            las_files_map[norm_name] = file_list[0] # Pick the first one available

    # 3. Iterate through each well CSV file and process it
    try:
        csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in the folder: '{csv_dir}'")
            return
            
        print(f"\nFound {len(csv_files)} well CSV(s) to process...")

        for csv_filename in csv_files:
            csv_filepath = os.path.join(csv_dir, csv_filename)
            norm_csv_name = normalize_well_name(csv_filename)
            
            print(f"\n--- Processing: {csv_filename} ---")

            # 4. Find the matching LAS file
            if norm_csv_name and norm_csv_name in las_files_map:
                las_filepath = las_files_map[norm_csv_name]
                print(f"Found matching LAS file: {os.path.basename(las_filepath)}")

                try:
                    # Load existing CSV
                    existing_df = pd.read_csv(csv_filepath)
                    if 'DEPTH' not in existing_df.columns:
                        print(f"  - Skipping: 'DEPTH' column not found in {csv_filename}.")
                        continue

                    # Load LAS file
                    las = lasio.read(las_filepath)
                    las_df = las.df()
                    las_df.reset_index(inplace=True)
                    if 'DEPT' in las_df.columns:
                        las_df.rename(columns={'DEPT': 'DEPTH'}, inplace=True)
                    
                    # --- Find available curves using the alias dictionary ---
                    las_curves_map = {c.upper(): c for c in las_df.columns}
                    rename_dict = {'DEPTH': 'DEPTH'}
                    available_original_case = []

                    for target_name, aliases in curve_aliases.items():
                        found_alias = False
                        for alias in aliases:
                            if alias.upper() in las_curves_map:
                                original_case_name = las_curves_map[alias.upper()]
                                available_original_case.append(original_case_name)
                                rename_dict[original_case_name] = target_name
                                print(f"  - Found '{original_case_name}' for target '{target_name}'.")
                                found_alias = True
                                break # Stop searching for this target once an alias is found
                        if not found_alias:
                             print(f"  - Warning: Could not find any alias for '{target_name}'.")
                    
                    if not available_original_case:
                        print(f"  - Skipping: None of the requested curves were found in {os.path.basename(las_filepath)}.")
                        continue

                    las_subset_df = las_df[['DEPTH'] + available_original_case].copy()
                    las_subset_df.rename(columns=rename_dict, inplace=True)
                    
                    # Drop existing columns from CSV to be replaced
                    for col_name in rename_dict.values():
                        if col_name in existing_df.columns and col_name != 'DEPTH':
                            print(f"  - Replacing existing column '{col_name}'.")
                            existing_df.drop(columns=[col_name], inplace=True)
                    
                    # Sort and merge
                    existing_df.sort_values('DEPTH', inplace=True)
                    las_subset_df.sort_values('DEPTH', inplace=True)

                    merged_df = pd.merge_asof(
                        left=existing_df,
                        right=las_subset_df,
                        on='DEPTH',
                        direction='nearest'
                    )
                    
                    # Save the result
                    output_filename = os.path.splitext(csv_filename)[0] + '-minsol.csv'
                    output_filepath = os.path.join(output_dir, output_filename)
                    merged_df.to_csv(output_filepath, index=False)
                    print(f"  - Success! Saved merged file to: {output_filepath}")

                except Exception as e:
                    print(f"  - An error occurred while processing {csv_filename}: {e}")

            else:
                print(f"  - Warning: No matching LAS file found for {csv_filename}.")
                
    except FileNotFoundError:
        print(f"Error: The CSV folder was not found at '{csv_dir}'. Please check the path.")
        return

# Run the main function
if __name__ == "__main__":
    merge_las_to_csv_folder(csv_folder, las_folder, output_folder, curves_to_find)
