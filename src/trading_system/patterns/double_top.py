# ============================================================================
# FILE: patterns/double_top.py
# Double Top pattern detector
# ============================================================================

from .base import BasePatternDetector
# Standard library imports
import pandas as pd
import numpy as np

# Third-party imports
from sklearn.preprocessing import StandardScaler

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradingConfig


class DoubleTopDetector(BasePatternDetector):
    """
    Detects Double Top (bearish reversal) patterns
    
    Pattern Characteristics:
    - Two peaks at similar price levels
    - Neckline support between peaks
    - Target: Pattern height below neckline
    """
    
    def detect_patterns(
        self,
        min_bars: int = None,
        max_bars: int = None,
        tolerance: float = None
    ) -> pd.DataFrame:
        """
        Detect Double Top patterns
        
        Args:
            min_bars: Minimum bars between peaks
            max_bars: Maximum bars between peaks
            tolerance: Price match tolerance (as decimal)
        
        Returns:
            DataFrame with detected patterns
        """
        # Use config defaults if not provided
        min_bars = min_bars or TradingConfig.PATTERN_MIN_BARS
        max_bars = max_bars or TradingConfig.PATTERN_MAX_BARS
        tolerance = tolerance or TradingConfig.PATTERN_TOLERANCE
        
        patterns = []
        peak_indices = self.df[self.df['Peak']].index.tolist()
        
        for i in range(len(peak_indices) - 1):
            for j in range(i + 1, len(peak_indices)):
                pattern = self._check_pattern_pair(
                    peak_indices[i], 
                    peak_indices[j],
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
        """Check if two peaks form a valid Double Top"""
        bars_between = idx2 - idx1
        
        # Check bar spacing
        if bars_between < min_bars or bars_between > max_bars:
            return None
        
        peak1 = self.df.loc[idx1, 'High']
        peak2 = self.df.loc[idx2, 'High']
        
        # Check price similarity
        if abs(peak1 - peak2) / peak1 > tolerance:
            return None
        
        # Find neckline
        trough_idx = self.df.loc[idx1:idx2, 'Low'].idxmin()
        neckline = self.df.loc[trough_idx, 'Low']
        pattern_height = max(peak1, peak2) - neckline
        
        return {
            'Pattern_Type': 'DoubleTop',
            'Detection_Index': idx2,
            'Detection_Date': self.df.loc[idx2, 'date'],
            'Neckline': neckline,
            'Pattern_Height': pattern_height,
            'Stop_Loss': max(peak1, peak2) * 1.002,
            'Target': neckline - (pattern_height * 1.5),
            'RSI': self.df.loc[idx2, 'RSI'],
            'ATR': self.df.loc[idx2, 'ATR'],
            'Volatility': self.df.loc[idx2, 'Volatility'],
            'Trend': self.df.loc[idx2, 'Trend'],
        }
