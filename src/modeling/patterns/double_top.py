# ============================================================================
# FILE: patterns/double_top.py
# Refined Double Top detector with Dynamic ATR-Based Levels
# ============================================================================

from .base import BasePatternDetector
import pandas as pd
import numpy as np
from config import TradingConfig

class DoubleTopDetector(BasePatternDetector):
    """
    Enhanced Detects Double Top (bearish reversal) patterns.
    Uses ATR for dynamic Stop Loss and Profit Targets.
    """
    
    def detect_patterns(
        self,
        min_bars: int = None,
        max_bars: int = None,
        tolerance: float = None
    ) -> pd.DataFrame:
        """Detect confirmed Double Top patterns"""
        min_bars = min_bars or TradingConfig.PATTERN_MIN_BARS
        max_bars = max_bars or TradingConfig.PATTERN_MAX_BARS
        tolerance = tolerance or TradingConfig.PATTERN_TOLERANCE
        
        patterns = []
        # Peak indices identified by the base class
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
    
    def _check_pattern_pair(self, idx1, idx2, min_bars, max_bars, tolerance) -> dict:
        """Logic for high-probability Double Top identification"""
        
        # 1. DISTANCE CHECK
        bars_between = idx2 - idx1
        if not (min_bars <= bars_between <= max_bars):
            return None
        
        # 2. PRICE SIMILARITY (The two peaks)
        peak1 = self.df.loc[idx1, 'High']
        peak2 = self.df.loc[idx2, 'High']
        price_diff = abs(peak1 - peak2) / peak1
        if price_diff > tolerance:
            return None
            
        # 3. NECKLINE/VALLEY CHECK (Must have a significant drop between peaks)
        trough_idx = self.df.loc[idx1:idx2, 'Low'].idxmin()
        neckline = self.df.loc[trough_idx, 'Low']
        valley_depth = (max(peak1, peak2) - neckline) / neckline
        
        if valley_depth < 0.015: # 1.5% drop required
            return None

        # 4. TREND CONTEXT
        # Double Tops are reversals; they should appear after an UPTREND.
        if 'SMA_50' in self.df.columns:
            if self.df.loc[idx1, 'Close'] < self.df.loc[idx1, 'SMA_50']:
                return None

        # 5. RSI DIVERGENCE (Bearish)
        # Price makes equal peaks, but RSI should be LOWER on the second peak.
        if 'RSI' in self.df.columns:
            rsi1 = self.df.loc[idx1, 'RSI']
            rsi2 = self.df.loc[idx2, 'RSI']
            if rsi2 > rsi1: 
                return None

        # 6. CONFIRMATION (The Bearish Breakout)
        # Search forward up to 15 bars for a neckline breach (Close below neckline)
        search_limit = min(idx2 + 15, len(self.df) - 1)
        breakout_slice = self.df.loc[idx2 : search_limit]
        confirmed_idx = breakout_slice[breakout_slice['Close'] < neckline].index.min()
        
        if pd.isna(confirmed_idx):
            return None 

        # 7. DYNAMIC ATR-BASED LEVELS
        # Calculate ATR or fallback to standard deviation
        if 'ATR' in self.df.columns:
            atr = self.df.loc[confirmed_idx, 'ATR']
        else:
            atr = self.df['Close'].rolling(14).std().iloc[confirmed_idx]
            
        entry_price = self.df.loc[confirmed_idx, 'Close']
        
        # Stop loss is above the highest peak plus a half-ATR buffer
        stop_loss = max(peak1, peak2) + (atr * 0.5)
        
        # Target is entry minus (Pattern Height or ATR-based volatility move)
        pattern_height = max(peak1, peak2) - neckline
        dynamic_target = entry_price - max(pattern_height * 1.5, atr * 2.0)

        

        return {
            'Pattern_Type': 'DoubleTop',
            'Confirmed_At': confirmed_idx,
            'Detection_Date': self.df.loc[confirmed_idx, 'date'],
            'Entry_Price': entry_price,
            'Neckline': neckline,
            'Pattern_Height': pattern_height,
            'Stop_Loss': stop_loss,
            'Target': dynamic_target,
            'RSI_at_Bottom': self.df.loc[idx2, 'RSI'] if 'RSI' in self.df.columns else 0,
            'Valley_Depth_Pct': valley_depth * 100,
            'RSI_Divergence': rsi2 - rsi1 if 'RSI' in self.df.columns else 0
        }