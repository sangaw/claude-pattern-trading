# ============================================================================
# FILE: patterns/double_bottom.py
# Refined Double Bottom detector with Dynamic ATR-Based Levels
# ============================================================================

from .base import BasePatternDetector
import pandas as pd
import numpy as np
from config import TradingConfig

class DoubleBottomDetector(BasePatternDetector):
    """
    Enhanced Detects Double Bottom (bullish reversal) patterns.
    Uses ATR for dynamic Stop Loss and Profit Targets to handle low volatility.
    """
    
    def detect_patterns(
        self,
        min_bars: int = None,
        max_bars: int = None,
        tolerance: float = None
    ) -> pd.DataFrame:
        """Detect confirmed Double Bottom patterns"""
        min_bars = min_bars or TradingConfig.PATTERN_MIN_BARS
        max_bars = max_bars or TradingConfig.PATTERN_MAX_BARS
        tolerance = tolerance or TradingConfig.PATTERN_TOLERANCE
        
        patterns = []
        # Trough indices from base class
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
    
    def _check_pattern_pair(self, idx1, idx2, min_bars, max_bars, tolerance) -> dict:
        """Logic for high-probability Double Bottom identification"""
        
        # 1. DISTANCE CHECK
        bars_between = idx2 - idx1
        if not (min_bars <= bars_between <= max_bars):
            return None
        
        # 2. PRICE SIMILARITY (The two bottoms)
        bottom1 = self.df.loc[idx1, 'Low']
        bottom2 = self.df.loc[idx2, 'Low']
        price_diff = abs(bottom1 - bottom2) / bottom1
        if price_diff > tolerance:
            return None
            
        # 3. VALLEY DEPTH (Significance Check)
        peak_idx = self.df.loc[idx1:idx2, 'High'].idxmax()
        neckline = self.df.loc[peak_idx, 'High']
        valley_depth = (neckline - min(bottom1, bottom2)) / min(bottom1, bottom2)
        
        if valley_depth < 0.015: # 1.5% bounce required (adjusted for low vol)
            return None

        # 4. TREND CONTEXT (The 0% Fix)
        # We need the price to be in a downtrend (below SMA_50) to reverse it.
        if 'SMA_50' in self.df.columns:
            if self.df.loc[idx1, 'Close'] > self.df.loc[idx1, 'SMA_50']:
                return None

        # 5. RSI STRENGTH FILTER
        # Only trade if the second bottom shows strength (Divergence)
        if 'RSI' in self.df.columns:
            rsi1 = self.df.loc[idx1, 'RSI']
            rsi2 = self.df.loc[idx2, 'RSI']
            if rsi2 < rsi1: 
                return None

        # 6. CONFIRMATION (The Breakout)
        # Search forward up to 15 bars for a neckline breach
        search_limit = min(idx2 + 15, len(self.df) - 1)
        breakout_slice = self.df.loc[idx2 : search_limit]
        confirmed_idx = breakout_slice[breakout_slice['Close'] > neckline].index.min()
        
        if pd.isna(confirmed_idx):
            return None 

        # 7. DYNAMIC ATR-BASED LEVELS
        # This replaces fixed percentages with volatility-aware math
        atr = self.df.loc[confirmed_idx, 'ATR'] if 'ATR' in self.df.columns else self.df['Close'].rolling(14).std().iloc[confirmed_idx]
        entry_price = self.df.loc[confirmed_idx, 'Close']
        
        # Stop loss is below the lowest bottom minus a half-ATR buffer to avoid noise
        stop_loss = min(bottom1, bottom2) - (atr * 0.5)
        
        # Target is neckline + (Pattern Height adjusted by Volatility)
        # We use a 2.0x ATR multiplier or 1.5x Pattern Height (whichever is more conservative)
        pattern_height = neckline - min(bottom1, bottom2)
        dynamic_target = entry_price + max(pattern_height * 1.5, atr * 2.0)

        

        return {
            'Pattern_Type': 'DoubleBottom',
            'Confirmed_At': confirmed_idx,
            'Detection_Date': self.df.loc[confirmed_idx, 'date'],
            'Entry_Price': entry_price,
            'Neckline': neckline,
            'Pattern_Height': pattern_height,
            'Stop_Loss': stop_loss,
            'Target': dynamic_target,
            'RSI_at_Bottom': self.df.loc[idx2, 'RSI'],
            'Valley_Depth_Pct': valley_depth * 100,
            'RSI_Divergence': rsi2 - rsi1 if 'RSI' in self.df.columns else 0
        }