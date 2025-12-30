"""
patterns/base.py - Base Pattern Detector
========================================

Base class for all pattern detectors with technical indicator calculations.
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradingConfig


class BasePatternDetector:
    """
    Base class for all pattern detectors
    
    Responsibilities:
    - Load and validate OHLCV data
    - Calculate technical indicators (RSI, MACD, ATR, etc.)
    - Identify peaks and troughs
    - Provide common utility methods
    
    Usage:
        detector = BasePatternDetector(df)
        # Indicators automatically calculated
        # Access via detector.df['RSI'], etc.
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[TradingConfig] = None):
        """
        Initialize base pattern detector
        
        Args:
            df: DataFrame with columns [date, Open, High, Low, Close, Volume]
            config: Configuration object (uses defaults if None)
        
        Raises:
            ValueError: If required columns are missing
        """
        self.config = config or TradingConfig()
        self.df = self._prepare_data(df)
        self._calculate_indicators()
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate input data
        
        Args:
            df: Raw price data
        
        Returns:
            Cleaned and validated DataFrame
        
        Raises:
            ValueError: If required columns missing
        """
        df = df.copy()
        
        # Validate required columns
        required_cols = ['date', 'Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert date and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Validate data types
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for missing values in OHLC
        if df[['Open', 'High', 'Low', 'Close']].isnull().any().any():
            print("Warning: Missing values in OHLC data")
        
        return df
    
    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_moving_averages()
        self._calculate_atr()
        self._calculate_volatility()
        self._calculate_trend()
        self._identify_peaks_troughs()
    
    def _calculate_rsi(self):
        """
        Calculate Relative Strength Index (RSI)
        
        Formula:
            RSI = 100 - (100 / (1 + RS))
            where RS = Average Gain / Average Loss
        """
        period = self.config.RSI_PERIOD
        
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
    
    def _calculate_macd(self):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Components:
            - MACD Line: EMA(12) - EMA(26)
            - Signal Line: EMA(9) of MACD
        """
        fast = self.config.MACD_FAST
        slow = self.config.MACD_SLOW
        signal = self.config.MACD_SIGNAL
        
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
    
    def _calculate_moving_averages(self):
        """Calculate Simple Moving Averages"""
        self.df['SMA_20'] = self.df['Close'].rolling(self.config.SMA_SHORT).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(self.config.SMA_LONG).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(self.config.SMA_TREND).mean()
    
    def _calculate_atr(self):
        """
        Calculate Average True Range (ATR)
        
        True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ATR = SMA(True Range)
        """
        period = self.config.ATR_PERIOD
        
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        self.df['ATR'] = true_range.rolling(period).mean()
    
    def _calculate_volatility(self):
        """
        Calculate price volatility (standard deviation of returns)
        """
        period = self.config.VOLATILITY_PERIOD
        
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(period).std()
    
    def _calculate_trend(self):
        """
        Calculate trend direction
        
        Trend = 1 if SMA_20 > SMA_50 (uptrend)
        Trend = -1 if SMA_20 < SMA_50 (downtrend)
        """
        self.df['Trend'] = np.where(
            self.df['SMA_20'] > self.df['SMA_50'], 1, -1
        )
    
    def _identify_peaks_troughs(self):
        """
        Identify local peaks and troughs using scipy
        
        Uses argrelextrema to find local maxima (peaks) and minima (troughs)
        """
        order = self.config.PATTERN_PEAK_ORDER
        
        self.df['Peak'] = False
        self.df['Trough'] = False
        
        # Find peaks (local maxima in High prices)
        peak_idx = argrelextrema(
            self.df['High'].values,
            np.greater,
            order=order
        )[0]
        
        # Find troughs (local minima in Low prices)
        trough_idx = argrelextrema(
            self.df['Low'].values,
            np.less,
            order=order
        )[0]
        
        self.df.loc[peak_idx, 'Peak'] = True
        self.df.loc[trough_idx, 'Trough'] = True
    
    def detect_patterns(self, **kwargs) -> pd.DataFrame:
        """
        Detect patterns - must be implemented by subclasses
        
        Args:
            **kwargs: Pattern-specific parameters
        
        Returns:
            DataFrame with detected patterns
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "Subclasses must implement detect_patterns() method"
        )
    
    def get_indicator_at_index(self, idx: int, indicator: str) -> float:
        """
        Get indicator value at specific index
        
        Args:
            idx: DataFrame index
            indicator: Indicator name (e.g., 'RSI', 'MACD')
        
        Returns:
            Indicator value
        """
        if indicator not in self.df.columns:
            raise ValueError(f"Indicator '{indicator}' not found")
        
        return self.df.loc[idx, indicator]
    
    def get_price_at_index(self, idx: int, price_type: str = 'Close') -> float:
        """
        Get price at specific index
        
        Args:
            idx: DataFrame index
            price_type: 'Open', 'High', 'Low', or 'Close'
        
        Returns:
            Price value
        """
        return self.df.loc[idx, price_type]
    
    def summary(self) -> dict:
        """
        Get summary statistics of the data
        
        Returns:
            Dictionary with summary stats
        """
        return {
            'total_bars': len(self.df),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'price_range': (self.df['Close'].min(), self.df['Close'].max()),
            'peaks_detected': self.df['Peak'].sum(),
            'troughs_detected': self.df['Trough'].sum(),
            'avg_volatility': self.df['Volatility'].mean(),
            'current_trend': 'Uptrend' if self.df['Trend'].iloc[-1] > 0 else 'Downtrend'
        }


if __name__ == "__main__":
    # Test with sample data
    from datetime import datetime, timedelta
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 102 + np.random.randn(100).cumsum(),
        'Low': 98 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Test base detector
    detector = BasePatternDetector(df)
    print("Base Pattern Detector Test")
    print("="*50)
    print("\nSummary:")
    for key, value in detector.summary().items():
        print(f"  {key}: {value}")
    
    print("\n✓ Base detector working correctly")