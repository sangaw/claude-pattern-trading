# ============================================================================
# FILE: utils/helpers.py
# High-Performance Helper Utilities for Pattern Trading
# ============================================================================

import pandas as pd
import numpy as np
from typing import Optional

def adaptive_quality_threshold(
    scored_patterns: pd.DataFrame,
    target_trades: int = 30,
    min_acceptable: float = 30.0
) -> float:
    """
    Dynamically adjusts the "Quality Score" bar based on available data.
    If you have 100 patterns, it picks the threshold for the top 30.
    """
    if scored_patterns.empty or 'Quality_Score' not in scored_patterns.columns:
        return min_acceptable
    
    # Sort scores to find the "cut-off" point for the top N trades
    scores = scored_patterns['Quality_Score'].sort_values(ascending=False)
    
    if len(scores) <= target_trades:
        # If we have fewer patterns than our target, take the best ones 
        # but don't go below the absolute minimum floor.
        threshold = max(scores.min(), min_acceptable)
        print(f"fewer trades than planned {target_trades} with threshold {threshold}")
    else:
        # Find the score at the Nth position
        threshold = scores.iloc[target_trades - 1]
        print(f"more trades than planned {target_trades} with threshold {threshold}")
    
    # Ensure we don't return a "trash" threshold just to meet trade count
    threshold = max(threshold, min_acceptable)
    
    print(f"\n 💡 Adaptive Threshold Analysis:")
    print(f"    Target trades: {target_trades} | Available: {len(scores)}")
    print(f"    Selected Threshold: {threshold:.2f}")
    
    return float(threshold)


def filter_patterns_by_trend(patterns_df, df):
    filtered = []
    for _, pattern in patterns_df.iterrows():
        # 1. Use your standardized column name
        idx = int(pattern['Confirmed_At'])
        
        # 2. Safety check for SMA_50
        if 'SMA_50' not in df.columns:
            df['SMA_50'] = df['Close'].rolling(50).mean()
            
        # 3. Look at the trend leading UP TO the pattern
        # We check the SMA 10 bars before confirmation to see the 'entering' trend
        prev_idx = max(0, idx - 20)
        price_before = df['Close'].iloc[prev_idx]
        sma_before = df['SMA_50'].iloc[prev_idx]
        
        # 4. REVERSAL LOGIC
        if pattern['Pattern_Type'] == 'DoubleBottom':
            # Was the market trending down before the bottoms formed?
            if price_before < sma_before:
                filtered.append(pattern)
                
        elif pattern['Pattern_Type'] == 'DoubleTop':
            # Was the market trending up before the tops formed?
            if price_before > sma_before:
                filtered.append(pattern)
    
    return pd.DataFrame(filtered)


def generate_sample_data(bars: int = 1000) -> pd.DataFrame:
    """
    Generates realistic OHLCV data with consistent price relationship.
    Ensures High >= Open/Close >= Low.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=bars, freq='H')
    
    # Generate a Geometric Brownian Motion (more realistic than a linear trend)
    returns = np.random.normal(0.0001, 0.01, bars)
    price_path = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'Close': price_path,
        'Volume': np.random.uniform(1000, 5000, bars)
    })
    
    # Create realistic High/Low/Open based on volatility
    vol = df['Close'] * 0.005
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, vol)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, vol)
    
    # Reorder columns
    return df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]