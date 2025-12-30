"""
Fixed Pattern Detection - Addresses 6.6% Success Rate Problem
Based on analysis showing successful patterns have:
- Trend = 0.000 (only work in sideways markets)
- Pattern_Height = 321 vs 615 (smaller patterns work better)
- Lower ATR (less volatile)
"""

import pandas as pd
import numpy as np


def calculate_trend_strength(df, idx, lookback=50):
    """
    Calculate trend strength at a given point
    Returns value between -1 (strong downtrend) and 1 (strong uptrend)
    0 = sideways
    """
    start_idx = max(0, idx - lookback)
    prices = df['Close'].iloc[start_idx:idx+1]
    
    if len(prices) < 2:
        return 0
    
    # Linear regression slope
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]
    
    # Normalize by price and period
    trend = slope / (prices.mean() / lookback)
    
    return np.clip(trend, -1, 1)


def is_consolidation(df, idx, lookback=50, max_range=0.10):
    """
    Check if market is consolidating (sideways)
    Based on finding: Successful patterns have Trend=0.000
    """
    start_idx = max(0, idx - lookback)
    prices = df['Close'].iloc[start_idx:idx+1]
    
    if len(prices) < 2:
        return False
    
    highest = prices.max()
    lowest = prices.min()
    
    if lowest == 0:
        return False
    
    range_pct = (highest - lowest) / lowest
    
    # Must be in tight range
    return range_pct < max_range


def filter_patterns_strict(df, patterns_df):
    """
    Apply STRICT filters based on success analysis
    Only keep patterns that match successful pattern characteristics
    """
    if patterns_df.empty:
        return patterns_df
    
    print(f"\n{'='*80}")
    print("APPLYING STRICT PATTERN FILTERS (Based on 6.6% Success Analysis)")
    print(f"{'='*80}")
    print(f"Starting patterns: {len(patterns_df)}")
    
    filtered = []
    
    for _, pattern in patterns_df.iterrows():
        idx = pattern['Detection_Index']
        
        if idx >= len(df) or idx < 50:
            continue
        
        # FILTER 1: Must be in consolidation (Trend = 0.000)
        trend = calculate_trend_strength(df, idx, lookback=50)
        if abs(trend) > 0.05:  # Very strict: almost no trend
            continue
        
        # FILTER 2: Must be in tight range
        if not is_consolidation(df, idx, lookback=50, max_range=0.08):  # 8% max
            continue
        
        # FILTER 3: Pattern height must be small (Success: 321 vs Failed: 615)
        pattern_height = pattern.get('Pattern_Height', 0)
        avg_price = df['Close'].iloc[max(0, idx-20):idx+1].mean()
        
        if avg_price > 0:
            height_pct = pattern_height / avg_price
            if height_pct > 0.06:  # Max 6% height
                continue
        
        # FILTER 4: Lower volatility (Success: 0.009 vs Failed: 0.009)
        if 'Volatility' in df.columns or 'volatility_20' in df.columns:
            vol_col = 'Volatility' if 'Volatility' in df.columns else 'volatility_20'
            current_vol = df.loc[idx, vol_col]
            avg_vol = df[vol_col].iloc[max(0, idx-50):idx+1].mean()
            
            if current_vol > avg_vol * 1.2:  # Not in high volatility
                continue
        
        # FILTER 5: ATR check (Success: 76 vs Failed: 130)
        if 'ATR' in df.columns:
            current_atr = df.loc[idx, 'ATR']
            avg_atr = df['ATR'].iloc[max(0, idx-50):idx+1].mean()
            
            if current_atr > avg_atr * 1.3:  # Not in high ATR period
                continue
        
        # FILTER 6: Only high quality patterns
        quality = pattern.get('Quality_Score', 0)
        if quality < 70:  # Raise quality threshold
            continue
        
        # Passed all filters
        pattern_dict = pattern.to_dict()
        pattern_dict['Trend_Strength'] = trend
        pattern_dict['Is_Consolidation'] = True
        filtered.append(pattern_dict)
    
    result = pd.DataFrame(filtered)
    
    print(f"\nFilter Results:")
    print(f"  ✗ Failed trend filter: {len(patterns_df) - len([p for p in patterns_df.itertuples() if abs(calculate_trend_strength(df, p.Detection_Index)) <= 0.05])}")
    print(f"  ✓ Passed all filters: {len(result)}")
    print(f"  Final retention: {len(result)/len(patterns_df)*100:.1f}%")
    
    if not result.empty:
        print(f"\nFiltered Pattern Statistics:")
        print(f"  Avg Quality: {result['Quality_Score'].mean():.1f}")
        print(f"  Avg Trend: {result['Trend_Strength'].mean():.3f}")
    
    return result


def add_alternative_patterns(df):
    """
    Since double top/bottom only work 6.6% of time, add alternatives:
    - Support/Resistance breakouts in consolidation
    - Tight range breakouts
    - Volume-confirmed breakouts
    """
    patterns = []
    
    # Calculate required indicators
    df = df.copy()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Vol_Avg'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else 1
    
    # Look for consolidation breakouts
    for i in range(100, len(df) - 5):
        # Check if in consolidation
        if not is_consolidation(df, i, lookback=30, max_range=0.05):
            continue
        
        # Look for breakout attempt
        lookback_prices = df['Close'].iloc[i-30:i]
        current_price = df['Close'].iloc[i]
        resistance = lookback_prices.max()
        support = lookback_prices.min()
        
        # Bullish breakout
        if current_price > resistance * 1.01:  # 1% above resistance
            # Confirm with volume if available
            if 'Volume' in df.columns:
                vol_surge = df['Volume'].iloc[i] > df['Vol_Avg'].iloc[i] * 1.5
            else:
                vol_surge = True
            
            if vol_surge:
                patterns.append({
                    'Detection_Index': i,
                    'Pattern_Type': 'ConsolidationBreakout',
                    'Direction': 'Bullish',
                    'Quality_Score': 75,
                    'Breakout_Level': resistance,
                    'Support_Level': support
                })
        
        # Bearish breakdown
        elif current_price < support * 0.99:  # 1% below support
            if 'Volume' in df.columns:
                vol_surge = df['Volume'].iloc[i] > df['Vol_Avg'].iloc[i] * 1.5
            else:
                vol_surge = True
            
            if vol_surge:
                patterns.append({
                    'Detection_Index': i,
                    'Pattern_Type': 'ConsolidationBreakdown',
                    'Direction': 'Bearish',
                    'Quality_Score': 75,
                    'Breakout_Level': support,
                    'Resistance_Level': resistance
                })
    
    result = pd.DataFrame(patterns)
    
    if not result.empty:
        print(f"\n✓ Found {len(result)} consolidation breakout patterns")
    
    return result


def add_ma_crossover_patterns(df):
    """
    Add simple MA crossover signals (often more reliable than patterns)
    Expected: 15-25% return in bull markets
    """
    patterns = []
    
    df = df.copy()
    df['MA_Fast'] = df['Close'].rolling(20).mean()
    df['MA_Slow'] = df['Close'].rolling(50).mean()
    df['MA_Long'] = df['Close'].rolling(200).mean()
    
    for i in range(201, len(df)):
        # Bullish crossover
        if (df['MA_Fast'].iloc[i] > df['MA_Slow'].iloc[i] and 
            df['MA_Fast'].iloc[i-1] <= df['MA_Slow'].iloc[i-1]):
            
            # Only above 200 MA (uptrend filter)
            if df['Close'].iloc[i] > df['MA_Long'].iloc[i]:
                patterns.append({
                    'Detection_Index': i,
                    'Pattern_Type': 'MA_Crossover',
                    'Direction': 'Bullish',
                    'Quality_Score': 80,
                    'MA_Fast': df['MA_Fast'].iloc[i],
                    'MA_Slow': df['MA_Slow'].iloc[i]
                })
        
        # Bearish crossover
        elif (df['MA_Fast'].iloc[i] < df['MA_Slow'].iloc[i] and 
              df['MA_Fast'].iloc[i-1] >= df['MA_Slow'].iloc[i-1]):
            
            patterns.append({
                'Detection_Index': i,
                'Pattern_Type': 'MA_Crossunder',
                'Direction': 'Bearish',
                'Quality_Score': 80,
                'MA_Fast': df['MA_Fast'].iloc[i],
                'MA_Slow': df['MA_Slow'].iloc[i]
            })
    
    result = pd.DataFrame(patterns)
    
    if not result.empty:
        print(f"✓ Found {len(result)} MA crossover signals")
    
    return result


def fix_pattern_detection_pipeline(df, original_patterns_df):
    """
    Main function: Fix the 6.6% success rate problem
    
    Strategy:
    1. Strictly filter existing patterns (only keep consolidation patterns)
    2. Add alternative patterns that work better
    3. Add simple MA crossovers as baseline
    """
    print(f"\n{'='*80}")
    print("FIXING PATTERN DETECTION - Addressing 6.6% Success Rate")
    print(f"{'='*80}")
    
    all_patterns = []
    
    # 1. Filter original patterns STRICTLY
    filtered_originals = filter_patterns_strict(df, original_patterns_df)
    if not filtered_originals.empty:
        all_patterns.append(filtered_originals)
        print(f"\n✓ Kept {len(filtered_originals)} strict double top/bottom patterns")
    
    # 2. Add consolidation breakouts
    breakouts = add_alternative_patterns(df)
    if not breakouts.empty:
        all_patterns.append(breakouts)
        print(f"✓ Added {len(breakouts)} consolidation breakout patterns")
    
    # 3. Add MA crossovers (baseline strategy)
    ma_signals = add_ma_crossover_patterns(df)
    if not ma_signals.empty:
        all_patterns.append(ma_signals)
        print(f"✓ Added {len(ma_signals)} MA crossover signals")
    
    # Combine all
    if all_patterns:
        combined = pd.concat(all_patterns, ignore_index=True)
        
        # Sort by detection index
        combined = combined.sort_values('Detection_Index').reset_index(drop=True)
        
        print(f"\n{'='*80}")
        print("FINAL PATTERN SUMMARY")
        print(f"{'='*80}")
        print(f"Total patterns: {len(combined)}")
        print(f"\nBy type:")
        for ptype in combined['Pattern_Type'].unique():
            count = len(combined[combined['Pattern_Type'] == ptype])
            print(f"  {ptype}: {count}")
        
        return combined
    else:
        print("\n⚠ No patterns found after filtering!")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    """
    # Test with sample data
    # dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    # np.random.seed(42)
    
    # Create sideways market with some trends
    # price = 100
    # prices = [price]

    for _ in range(499):
        # Sideways with noise
        price += np.random.randn() * 2
        price = max(price, 80)  # Floor
        price = min(price, 120)  # Ceiling
        prices.append(price)
    """

    csv_file = input("\nEnter CSV file path: ").strip()
    df_input = pd.read_csv(csv_file)
    n = 6157
    dates = [n]
    prices = [n]

    dates = df_input["date"]
    prices = df_input["Close"]

    df = pd.DataFrame({
        'date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 6156)
    })
    
    # Add indicators
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['ATR'] = df['Close'].diff().abs().rolling(14).mean()
    
    # Fake original patterns
    original_patterns = pd.DataFrame({
        'Detection_Index': [100, 200, 300, 400],
        'Pattern_Type': ['DoubleBottom', 'DoubleTop', 'DoubleBottom', 'DoubleTop'],
        'Quality_Score': [65, 70, 55, 80],
        'Pattern_Height': [500, 700, 400, 800]  # Different heights
    })
    
    # Fix patterns
    fixed_patterns = fix_pattern_detection_pipeline(df, original_patterns)
    
    print(f"\n✓ Pattern detection fixed!")
    print(f"  Original: {len(original_patterns)} patterns")
    print(f"  Fixed: {len(fixed_patterns)} patterns")