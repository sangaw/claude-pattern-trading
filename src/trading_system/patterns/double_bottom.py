# ============================================================================
# FILE: patterns/double_bottom.py
# Double Bottom pattern detector
# ============================================================================

from .base import BasePatternDetector
# Standard library imports
import pandas as pd
import numpy as np

from config import TradingConfig


class DoubleBottomDetector(BasePatternDetector):
    """
    Detects Double Bottom (bullish reversal) patterns
    
    Pattern Characteristics:
    - Two troughs at similar price levels
    - Neckline resistance between troughs
    - Target: Pattern height above neckline
    """
    
    def detect_patterns(
        self,
        min_bars: int = None,
        max_bars: int = None,
        tolerance: float = None
    ) -> pd.DataFrame:
        """Detect Double Bottom patterns"""
        min_bars = min_bars or TradingConfig.PATTERN_MIN_BARS
        max_bars = max_bars or TradingConfig.PATTERN_MAX_BARS
        tolerance = tolerance or TradingConfig.PATTERN_TOLERANCE
        
        patterns = []
        trough_indices = self.df[self.df['Trough']].index.tolist()
        
        for i in range(len(trough_indices) - 1):
            for j in range(i + 1, len(trough_indices)):
                pattern = self._check_pattern_pair(
                    trough_indices[i],
                    trough_indices[j],
                    min_bars,
                    max_bars,
                    tolerance
                )
                if pattern:
                    patterns.append(pattern)
        
        return pd.DataFrame(patterns)
    
    def _check_pattern_pair(
        self,
        idx1: int,
        idx2: int,
        min_bars: int,
        max_bars: int,
        tolerance: float
    ) -> dict:
        """Check if two troughs form a valid Double Bottom"""
        bars_between = idx2 - idx1
        
        if bars_between < min_bars or bars_between > max_bars:
            return None
        
        bottom1 = self.df.loc[idx1, 'Low']
        bottom2 = self.df.loc[idx2, 'Low']
        
        if abs(bottom1 - bottom2) / bottom1 > tolerance:
            return None
        
        peak_idx = self.df.loc[idx1:idx2, 'High'].idxmax()
        neckline = self.df.loc[peak_idx, 'High']
        pattern_height = neckline - min(bottom1, bottom2)
        
        return {
            'Pattern_Type': 'DoubleBottom',
            'Detection_Index': idx2,
            'Detection_Date': self.df.loc[idx2, 'date'],
            'Neckline': neckline,
            'Pattern_Height': pattern_height,
            'Stop_Loss': min(bottom1, bottom2) * 0.998,
            'Target': neckline + (pattern_height * 1.5),
            'RSI': self.df.loc[idx2, 'RSI'],
            'ATR': self.df.loc[idx2, 'ATR'],
            'Volatility': self.df.loc[idx2, 'Volatility'],
            'Trend': self.df.loc[idx2, 'Trend'],
        }