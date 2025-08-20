import pandas as pd
import os
import re

# --- User Configuration ---
# Please update these folder paths to match your directory structure.
                                              
# Folder containing the well data CSV files (e.g., 'ABB-007.csv')
wells_folder = 'data/structures/adera/abab'

# Folder containing the zone definition CSV files (e.g., 'abb-007_zone.csv')
zones_folder = 'data/ABAB_ZONE'

# Folder where the modified well CSVs with the new 'ZONE' column will be saved.
# This folder will be created if it doesn't exist.
output_folder = 'data/ababwithzone'

# --- Script Execution ---

def normalize_well_name(filename):
    """
    Extracts a standardized well name from a filename for matching.
    e.g., 'ABB-007.csv' -> 'abb-007'
    e.g., 'abb-007_zone.csv' -> 'abb-007'
    """
    # Convert to lowercase
    name = filename.lower()
    # Remove '_zone' if it exists
    name = name.replace('_zone', '')
    # Use regex to remove file extension like .csv, .txt, etc.
    name = re.sub(r'\.[^.]+$', '', name)
    return name

def annotate_wells_with_zones(wells_dir, zones_dir, output_dir):
    """
    Main function to process all well files, match them with zone files,
    and add a 'ZONE' column based on depth.
    """
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Map normalized names to full paths for all zone files
    zone_files_map = {}
    try:
        for f in os.listdir(zones_dir):
            if f.lower().endswith('.csv'):
                norm_name = normalize_well_name(f)
                zone_files_map[norm_name] = os.path.join(zones_dir, f)
    except FileNotFoundError:
        print(f"Error: The zones folder was not found at '{zones_dir}'. Please check the path.")
        return

    # 3. Iterate through each well file and process it
    try:
        well_files = [f for f in os.listdir(wells_dir) if f.lower().endswith('.csv')]
        if not well_files:
            print(f"No CSV files found in the wells folder: '{wells_dir}'")
            return
            
        print(f"\nFound {len(well_files)} well CSV(s) to process...")

        for well_filename in well_files:
            well_filepath = os.path.join(wells_dir, well_filename)
            norm_well_name = normalize_well_name(well_filename)
            
            print(f"\n--- Processing: {well_filename} ---")

            # 4. Find the matching zone file
            if norm_well_name in zone_files_map:
                zone_filepath = zone_files_map[norm_well_name]
                print(f"Found matching zone file: {os.path.basename(zone_filepath)}")

                try:
                    # 5. Load the well and zone data into pandas DataFrames
                    well_df = pd.read_csv(well_filepath)
                    
                    # Check for DEPTH column in well data
                    if 'DEPTH' not in well_df.columns:
                        print(f"  - Skipping: 'DEPTH' column not found in {well_filename}.")
                        continue

                    # Load the zone file, skipping the second row (units)
                    zone_df = pd.read_csv(zone_filepath, skiprows=[1])
                    
                    # Standardize column names for the zone file
                    zone_df.columns = ['DEPTH', 'ZONE']
                    
                    # Clean up zone data: drop rows with no depth and ensure correct data types
                    zone_df.dropna(subset=['DEPTH'], inplace=True)
                    zone_df['DEPTH'] = pd.to_numeric(zone_df['DEPTH'])

                    # 6. Sort both dataframes by DEPTH, which is required for the merge
                    well_df.sort_values('DEPTH', inplace=True)
                    zone_df.sort_values('DEPTH', inplace=True)
                    
                    # 7. Use merge_asof to efficiently map zones to depths.
                    # 'direction=backward' finds the last zone top at or before each depth value.
                    merged_df = pd.merge_asof(
                        left=well_df,
                        right=zone_df,
                        on='DEPTH',
                        direction='backward'
                    )
                    
                    # 8. Save the modified DataFrame to the output folder
                    output_filepath = os.path.join(output_dir, well_filename)
                    merged_df.to_csv(output_filepath, index=False)
                    print(f"  - Success! Saved modified file to: {output_filepath}")

                except Exception as e:
                    print(f"  - An error occurred while processing {well_filename}: {e}")

            else:
                print(f"  - Warning: No matching zone file found for {well_filename}.")
                
    except FileNotFoundError:
        print(f"Error: The wells folder was not found at '{wells_dir}'. Please check the path.")
        return

# Run the main function
if __name__ == "__main__":
    annotate_wells_with_zones(wells_folder, zones_folder, output_folder)

