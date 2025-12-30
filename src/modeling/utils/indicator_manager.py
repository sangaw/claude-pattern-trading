"""
Indicator Manager
=================

Manages technical indicator calculations.
Only computes indicators that are missing from the DataFrame.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class IndicatorManager:
    """
    Manages technical indicator calculations.
    Checks if indicators exist in DataFrame before computing.
    """
    
    def __init__(self, df: pd.DataFrame, config=None):
        """
        Initialize indicator manager
        
        Args:
            df: DataFrame with market data
            config: Configuration object (optional)
        """
        self.df = df.copy()
        self.config = config
        
        # Import config if not provided
        if config is None:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import TradingConfig
            self.config = TradingConfig()
    
    def require(self, indicator_names: List[str]) -> pd.DataFrame:
        """
        Ensure required indicators are present in DataFrame.
        Computes missing indicators only.
        
        Args:
            indicator_names: List of indicator names needed
        
        Returns:
            DataFrame with all required indicators
        """
        for indicator in indicator_names:
            if indicator not in self.df.columns:
                logger.info(f"Computing missing indicator: {indicator}")
                self._compute_indicator(indicator)
            else:
                logger.debug(f"Using existing indicator: {indicator}")
        
        return self.df
    
    def _compute_indicator(self, indicator_name: str):
        """Compute a specific indicator"""
        if indicator_name == 'RSI':
            self._calculate_rsi()
        elif indicator_name == 'MACD':
            self._calculate_macd()
        elif indicator_name == 'MACD_Signal':
            self._calculate_macd()  # MACD_Signal is computed with MACD
        elif indicator_name == 'MACD_Histogram':
            self._calculate_macd()  # MACD_Histogram is computed with MACD
        elif indicator_name == 'SMA_20':
            self._calculate_sma(20)
        elif indicator_name == 'SMA_50':
            self._calculate_sma(50)
        elif indicator_name == 'SMA_200':
            self._calculate_sma(200)
        elif indicator_name == 'ATR':
            self._calculate_atr()
        elif indicator_name == 'Volatility':
            self._calculate_volatility()
        elif indicator_name == 'Returns':
            self._calculate_returns()
        elif indicator_name == 'Volume_Avg':
            self._calculate_volume_avg()
        elif indicator_name == 'Trend':
            self._calculate_trend()
        else:
            logger.warning(f"Unknown indicator: {indicator_name}")
    
    def _calculate_rsi(self):
        """Calculate RSI if not present"""
        if 'RSI' in self.df.columns:
            return
        
        period = getattr(self.config, 'RSI_PERIOD', 14)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
    
    def _calculate_macd(self):
        """Calculate MACD, MACD_Signal, and MACD_Histogram if not present"""
        if 'MACD' in self.df.columns and 'MACD_Signal' in self.df.columns:
            return
        
        fast = getattr(self.config, 'MACD_FAST', 12)
        slow = getattr(self.config, 'MACD_SLOW', 26)
        signal = getattr(self.config, 'MACD_SIGNAL', 9)
        
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['MACD_Signal']
    
    def _calculate_sma(self, period: int):
        """Calculate Simple Moving Average"""
        col_name = f'SMA_{period}'
        if col_name in self.df.columns:
            return
        
        self.df[col_name] = self.df['Close'].rolling(period).mean()
    
    def _calculate_atr(self):
        """Calculate Average True Range"""
        if 'ATR' in self.df.columns:
            return
        
        period = getattr(self.config, 'ATR_PERIOD', 14)
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['ATR'] = true_range.rolling(period).mean()
    
    def _calculate_volatility(self):
        """Calculate price volatility"""
        if 'Volatility' in self.df.columns:
            return
        
        period = getattr(self.config, 'VOLATILITY_PERIOD', 20)
        if 'Returns' not in self.df.columns:
            self._calculate_returns()
        
        self.df['Volatility'] = self.df['Returns'].rolling(period).std()
    
    def _calculate_returns(self):
        """Calculate returns"""
        if 'Returns' in self.df.columns:
            return
        
        self.df['Returns'] = self.df['Close'].pct_change()
    
    def _calculate_volume_avg(self):
        """Calculate average volume"""
        if 'Volume_Avg' in self.df.columns:
            return
        
        if 'volume' not in self.df.columns:
            logger.warning("Volume column not found, cannot calculate Volume_Avg")
            return
        
        self.df['Volume_Avg'] = self.df['volume'].rolling(20).mean()
    
    def _calculate_trend(self):
        """Calculate trend direction"""
        if 'Trend' in self.df.columns:
            return
        
        # Ensure SMAs exist
        if 'SMA_20' not in self.df.columns:
            self._calculate_sma(20)
        if 'SMA_50' not in self.df.columns:
            self._calculate_sma(50)
        
        self.df['Trend'] = np.where(
            self.df['SMA_20'] > self.df['SMA_50'], 1, -1
        )
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get DataFrame with all computed indicators"""
        return self.df

