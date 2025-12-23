from .base import BasePatternDetector
import pandas as pd
import numpy as np
from config import TradingConfig

class BullishFlagDetector(BasePatternDetector):
    """
    Detects Bullish Flag (trend continuation) patterns.
    Designed for trending markets where price 'rests' before the next leg up.
    """
    
    def detect_patterns(
        self,
        pole_period: int = 5,
        flag_period: int = 10,
        max_flag_retrace: float = 0.5
    ) -> pd.DataFrame:
        """
        Detects Bullish Flags based on momentum and tight consolidation.
        """
        patterns = []
        # We start the search after we have enough data for the pole and flag
        search_start = pole_period + flag_period
        
        for idx in range(search_start, len(self.df) - 1):
            pattern = self._check_at_index(idx, pole_period, flag_period, max_flag_retrace)
            if pattern:
                patterns.append(pattern)
        
        return pd.DataFrame(patterns)

    def _check_at_index(self, idx, pole_period, flag_period, max_retrace) -> dict:
        """Logic to identify a flag breakout at a specific index"""
        
        # 1. POLE IDENTIFICATION (The sharp move up)
        # Price change over the pole period
        pole_start_idx = idx - flag_period - pole_period
        pole_end_idx = idx - flag_period
        
        pole_low = self.df.loc[pole_start_idx:pole_end_idx, 'Low'].min()
        pole_high = self.df.loc[pole_start_idx:pole_end_idx, 'High'].max()
        pole_height = pole_high - pole_low
        
        
        # Check if the pole is "strong" enough (e.g., > 2% move)
        if (pole_height / pole_low) < 0.02:
            return None

        # 2. FLAG CONSOLIDATION (The 'resting' phase)
        flag_slice = self.df.loc[pole_end_idx : idx-1]
        flag_max = flag_slice['High'].max()
        flag_min = flag_slice['Low'].min()
        
        # Filter: The flag must not retrace more than 'max_retrace' of the pole
        retrace_amount = pole_high - flag_min
        if retrace_amount > (pole_height * max_retrace):
            return None

        # Filter: Tightness (Standard Deviation check)
        # If the flag is too "messy/volatile", it's not a high-probability flag
        flag_std = flag_slice['Close'].std()
        if flag_std > (pole_height * 0.15): # Flag is too wide
            return None

        # 3. BREAKOUT CONFIRMATION
        # Current bar must close above the flag's highest point
        current_close = self.df.loc[idx, 'Close']
        if current_close <= flag_max:
            return None

        # 4. TREND ALIGNMENT (The 66% Uptrend Filter)
        # Ensure we are above a medium-term trend line
        if 'SMA_50' in self.df.columns:
            if current_close < self.df.loc[idx, 'SMA_50']:
                return None

        # 5. DYNAMIC ATR-BASED LEVELS (Consistent with your DoubleBottom code)
        atr = self.df.loc[idx, 'ATR'] if 'ATR' in self.df.columns else self.df['Close'].rolling(14).std().iloc[idx]
        rsi = self.df.loc[idx, 'rsi_14']
        volatility = self.df.loc[idx, 'volatility_5']
        entry_price = current_close
        
        # Stop loss is below the flag's low with a small buffer
        stop_loss = flag_min - (atr * 0.3)
        
        # Target: Measured move (The height of the pole projected from the breakout)
        # We take 80% of the pole height as a conservative target
        target = entry_price + (pole_height * 0.8)

        return {
            'Pattern_Type': 'BullishFlag',
            'Confirmed_At': idx,
            'Detection_Date': self.df.loc[idx, 'date'],
            'Entry_Price': entry_price,
            'Pole_Height': pole_height,
            'Flag_Tightness': flag_std,
            'Stop_Loss': stop_loss,
            'Target': target,
            'RSI':rsi,
            'ATR':atr,
            'Volatility':volatility,
            'Risk_Reward': (target - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
        }