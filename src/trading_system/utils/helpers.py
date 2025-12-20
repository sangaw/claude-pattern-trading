# ============================================================================
# FILE: utils/helpers.py
# Helper utility functions
# ============================================================================

import pandas as pd
from typing import Optional


def adaptive_quality_threshold(
    scored_patterns: pd.DataFrame,
    target_trades: int = 30
) -> float:
    """
    Calculate adaptive quality threshold
    
    Args:
        scored_patterns: Patterns with Quality_Score
        target_trades: Desired number of trades
    
    Returns:
        Recommended threshold value
    """
    if scored_patterns.empty:
        return 40.0
    
    scores = scored_patterns['Quality_Score'].sort_values(ascending=False)
    
    if len(scores) < target_trades:
        threshold = scores.min() - 1
    else:
        threshold = scores.iloc[target_trades - 1]
    
    threshold = max(threshold, 35)
    
    print(f"\n  💡 Adaptive Threshold Analysis:")
    print(f"     Target trades: {target_trades}")
    print(f"     Patterns available: {len(scores)}")
    print(f"     Recommended threshold: {threshold:.1f}")
    print(f"     This will give ~{(scores >= threshold).sum()} trades")
    
    return threshold


def filter_patterns_by_trend(
    patterns_df: pd.DataFrame,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter patterns to only keep trend-aligned ones
    
    Args:
        patterns_df: Detected patterns
        df: Price data with indicators
    
    Returns:
        Filtered patterns DataFrame
    """
    filtered = []
    
    for _, pattern in patterns_df.iterrows():
        idx = pattern['Detection_Index']
        sma_20 = df['Close'].rolling(20).mean().iloc[idx]
        current_price = df['Close'].iloc[idx]
        
        is_uptrend = current_price > sma_20
        
        # Only keep trend-aligned patterns
        if pattern['Pattern_Type'] == 'DoubleBottom' and is_uptrend:
            filtered.append(pattern)
        elif pattern['Pattern_Type'] == 'DoubleTop' and not is_uptrend:
            filtered.append(pattern)
    
    return pd.DataFrame(filtered)


def generate_sample_data(bars: int = 1000) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing
    
    Args:
        bars: Number of bars to generate
    
    Returns:
        DataFrame with sample data
    """
    import numpy as np
    
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=bars, freq='D')
    
    trend = np.linspace(10000, 11500, bars)
    noise = np.cumsum(np.random.randn(bars) * 40)
    close = trend + noise
    
    high = close + np.random.uniform(20, 100, bars)
    low = close - np.random.uniform(20, 100, bars)
    open_price = close + np.random.uniform(-50, 50, bars)
    volume = np.random.uniform(1000000, 5000000, bars)
    
    return pd.DataFrame({
        'date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })