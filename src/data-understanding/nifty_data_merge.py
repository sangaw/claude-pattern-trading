import os
import glob
import pandas as pd

# Add import for technical analysis library
def try_import_ta():
    try:
        import ta
        return ta
    except ImportError:
        print("The 'ta' library is not installed. Please install it with 'pip install ta'.")
        return None

# 1. Merge all CSVs in data/nifty/train/ into a single DataFrame
def merge_csvs(input_dir, output_file):
    """
    Merge CSV files and ensure a clean, date-sorted timeline for ML indicator calculation.
    """
    all_files = glob.glob(os.path.join(input_dir, '*.csv'))
    df_list = []
    
    for file in all_files:
        df = pd.read_csv(file)
        # Clean column names immediately upon reading
        df.columns = df.columns.str.strip().str.lower()
        df_list.append(df)
    
    # Concatenate all files
    merged_df = pd.concat(df_list, ignore_index=True)

    # --- FIX: Handle Duplicate 'date' Columns ---
    # This specifically addresses the "cannot assemble with duplicate keys" error
    if merged_df.columns.duplicated().any():
        # Keep only the first occurrence of each column name
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Ensure 'date' column exists
    if 'date' not in merged_df.columns:
        raise KeyError("The 'date' column was not found in the CSV files.")

    # Convert to datetime - Passing merged_df['date'] as a Series prevents assembly errors
    try:
        # We explicitly select the Series to avoid the 'assembly' logic
        merged_df['date'] = pd.to_datetime(merged_df['date'].astype(str), errors='coerce')
        
        # Drop rows where date parsing failed
        merged_df = merged_df.dropna(subset=['date'])
        
        # Handle Timezone
        if merged_df['date'].dt.tz is None:
            merged_df['date'] = merged_df['date'].dt.tz_localize('Asia/Kolkata')
        else:
            merged_df['date'] = merged_df['date'].dt.tz_convert('Asia/Kolkata')

    except Exception as e:
        print(f"Error during date processing: {e}")

    # Sort and remove duplicate rows (same date/price entries)
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    merged_df = merged_df.drop_duplicates(subset=['date'], keep='first')

    # Save output
    merged_df.to_csv(output_file, index=False)
    print(f"Merged successfully. Final row count: {len(merged_df)}")
    
    return merged_df

# 3. Main function to run the pipeline
def main():
    input_dir = 'data/raw-data/nifty'
    # input_dir = "data/raw-data/infosys"
    merged_csv = os.path.join(input_dir, 'merged.csv')

    # Step 1: Merge CSVs (now sorted by date)
    merged_df = merge_csvs(input_dir, merged_csv)

if __name__ == "__main__":
    main() 