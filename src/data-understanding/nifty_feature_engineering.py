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

# 2. Feature engineering: add new features to the DataFrame
def add_features(df):
    """
    Add new features to the DataFrame for ML analysis, including technical indicators.
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
    Returns:
        pd.DataFrame: DataFrame with new features
    """
    # Ensure columns are lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Sort by date if not already
    df = df.sort_values('date').reset_index(drop=True)

    # Nifty columns - date,open,high,low,close,shares traded,turnover (₹ cr)

    """
    # For price with comma, specific logic
    df['open'] = df['open'].str.replace(',', '', regex=False).astype(float)
    df['high'] = df['high'].str.replace(',', '', regex=False).astype(float)
    df['low'] = df['low'].str.replace(',', '', regex=False).astype(float)
    df['close'] = df['close'].str.replace(',', '', regex=False).astype(float)
    df['volume'] = df['shares traded'].str.replace(',', '', regex=False).astype(float)
    df['turnover'] = df['turnover (₹ cr)'].str.replace(',', '', regex=False).astype(float)
    

    # df['prev-close'] = df['prev-close'].str.replace(',', '', regex=False).astype(float)
    # df['ltp'] = df['ltp'].str.replace(',', '', regex=False).astype(float)
    # df['vwap'] = df['vwap'].str.replace(',', '', regex=False).astype(float)
    

    # df['52w-h'] = df['52w-h'].str.replace(',', '', regex=False).astype(float)
    # df['52w-l'] = df['52w-l'].str.replace(',', '', regex=False).astype(float)
    
    # df['volume'] = df['volume'].str.replace(',', '', regex=False).astype(float)
    # df['value'] = df['value'].str.replace(',', '', regex=False).astype(float)
    # df['no-of-trades'] = df['no-of-trades'].str.replace(',', '', regex=False).astype(float)
    """
    
    # Daily return (percentage change in close)
    df['daily_return'] = df['close'].pct_change(fill_method=None)

    # Log return
    import numpy as np
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Price range (high - low)
    df['price_range'] = df['high'] - df['low']

    # 5-day and 20-day moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()

    # 5-day and 20-day rolling volatility (std of returns)
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()

    df['Volume_Avg'] = df['shares traded'].rolling(window=20).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # --- Technical Indicators using 'ta' library ---
    ta = try_import_ta()
    if ta is not None:
        # RSI (Relative Strength Index, 14-period)
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['rsi_14'] = rsi_indicator.rsi()

        # MACD (12, 26, 9)
        macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_12_26'] = macd_indicator.macd()
        df['macd_signal_12_26'] = macd_indicator.macd_signal()
        df['macd_histogram_12_26'] = macd_indicator.macd_diff()
        # MACD signal strength: abs value of histogram
        df['macd_signal_strength'] = np.abs(df['macd_histogram_12_26'])

        # Stochastic Oscillator (14)
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3
        )
        # %K (stoch_14)
        df['stoch_14'] = stoch.stoch()
        # Smoothed %K (stoch_smoothk)
        df['stoch_smoothk'] = stoch.stoch_signal()
        # Smoothed %D (stoch_smoothd) - usually a further smoothing, here we use the signal line
        df['stoch_smoothd'] = df['stoch_smoothk'].rolling(window=3).mean()

        print("Added technical indicators: RSI (14), MACD (12,26,9), MACD Histogram, MACD Signal Strength, Stochastic (14), Stoch SmoothK, Stoch SmoothD")
    else:
        print("Skipping technical indicators (ta library not available)")

    print("Added new features: daily_return, log_return, price_range, ma_5, ma_20, volatility_5, volatility_20")
    return df

# 3. Main function to run the pipeline
def main():
    input_dir = 'data/input/'
    # input_dir = "data/raw-data/infosys"
    merged_csv_file = os.path.join(input_dir, 'merged.csv')

    # Step 1: Read Merge CSVs (now sorted by date)
    merged_df = pd.read_csv(merged_csv_file)

    # Step 2: Add features (including technical indicators)
    featured_df = add_features(merged_df)

    # Step 3: Save the feature-engineered data (also sorted by date)
    featured_df = featured_df.sort_values('date').reset_index(drop=True)
    featured_csv = os.path.join(input_dir, 'featured.csv')
    featured_df.to_csv(featured_csv, index=False)
    print(f"Saved feature-engineered data to {featured_csv} ({len(featured_df)} rows), sorted by date")
    

if __name__ == "__main__":
    main() 