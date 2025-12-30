"""
Data Loader Module
===================

Loads CSV data and detects which technical indicators are already present.
Maps CSV column names to standard indicator names used by the system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and prepares market data from CSV files.
    Detects existing indicators and standardizes column names.
    """
    
    # Mapping from CSV column names to standard indicator names
    COLUMN_MAPPING = {
        # Price data
        'date': 'date',
        'Date': 'date',
        'timestamp': 'date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'close': 'Close',
        'Volume': 'volume',
        'volume': 'volume',
        'Volume': 'volume',
        'turnover': 'turnover',
        
        # RSI variations
        'rsi_14': 'RSI',
        'RSI': 'RSI',
        'rsi': 'RSI',
        
        # MACD variations
        'macd_12_26': 'MACD',
        'MACD': 'MACD',
        'macd': 'MACD',
        'macd_signal_12_26': 'MACD_Signal',
        'MACD_Signal': 'MACD_Signal',
        'macd_signal': 'MACD_Signal',
        'macd_histogram_12_26': 'MACD_Histogram',
        'MACD_Histogram': 'MACD_Histogram',
        
        # Moving Averages
        'SMA_200': 'SMA_200',
        'sma_200': 'SMA_200',
        'ma_20': 'SMA_20',
        'SMA_20': 'SMA_20',
        'sma_20': 'SMA_20',
        'ma_50': 'SMA_50',
        'SMA_50': 'SMA_50',
        'sma_50': 'SMA_50',
        'ma_5': 'SMA_5',
        'SMA_5': 'SMA_5',
        
        # Volume
        'Volume_Avg': 'Volume_Avg',
        'volume_avg': 'Volume_Avg',
        
        # Volatility
        'volatility_5': 'Volatility',
        'volatility_20': 'Volatility',
        'Volatility': 'Volatility',
        
        # ATR (if present)
        'ATR': 'ATR',
        'atr': 'ATR',
        
        # Returns
        'daily_return': 'Returns',
        'log_return': 'Returns',
        'Returns': 'Returns',
    }
    
    # Required columns for pattern detection
    REQUIRED_COLUMNS = ['date', 'Open', 'High', 'Low', 'Close']
    
    # Indicators that can be computed if missing
    COMPUTABLE_INDICATORS = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'SMA_20', 'SMA_50', 'SMA_200',
        'ATR', 'Volatility', 'Returns',
        'Volume_Avg', 'Trend'
    ]
    
    def __init__(self, csv_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Initialize data loader
        
        Args:
            csv_path: Path to CSV file (if loading from file)
            df: DataFrame (if data already loaded)
        """
        if csv_path:
            self.df = self.load_csv(csv_path)
        elif df is not None:
            self.df = self.prepare_data(df)
        else:
            raise ValueError("Either csv_path or df must be provided")
        
        self.indicators_present = self._detect_indicators()
        self.indicators_missing = self._identify_missing_indicators()
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load CSV file and prepare data"""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        return self.prepare_data(df)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and standardize DataFrame
        
        - Standardize column names
        - Convert date column
        - Sort by date
        - Validate required columns
        """
        df = df.copy()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
        
        # Validate required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure numeric columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Data prepared: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map CSV column names to standard names"""
        rename_map = {}
        for old_name, new_name in self.COLUMN_MAPPING.items():
            if old_name in df.columns and old_name != new_name:
                # Check if new_name already exists (avoid overwriting)
                if new_name not in df.columns:
                    rename_map[old_name] = new_name
                else:
                    logger.warning(f"Column {new_name} already exists, keeping {old_name} as-is")
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Renamed columns: {rename_map}")
        
        return df
    
    def _detect_indicators(self) -> Dict[str, bool]:
        """Detect which indicators are present in the DataFrame"""
        indicators = {}
        
        for indicator in self.COMPUTABLE_INDICATORS:
            indicators[indicator] = indicator in self.df.columns
        
        # Special handling for volume
        if 'volume' in self.df.columns:
            indicators['volume'] = True
        
        return indicators
    
    def _identify_missing_indicators(self) -> List[str]:
        """Identify indicators that need to be computed"""
        return [ind for ind, present in self.indicators_present.items() if not present]
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the prepared DataFrame"""
        return self.df
    
    def get_indicator_status(self) -> Dict[str, bool]:
        """Get status of all indicators"""
        return self.indicators_present.copy()
    
    def get_missing_indicators(self) -> List[str]:
        """Get list of missing indicators"""
        return self.indicators_missing.copy()
    
    def has_indicator(self, indicator_name: str) -> bool:
        """Check if a specific indicator is present"""
        return self.indicators_present.get(indicator_name, False)
    
    def summary(self) -> Dict:
        """Get summary of loaded data"""
        return {
            'total_rows': len(self.df),
            'date_range': (self.df['date'].min(), self.df['date'].max()) if 'date' in self.df.columns else None,
            'indicators_present': sum(self.indicators_present.values()),
            'indicators_missing': len(self.indicators_missing),
            'missing_list': self.indicators_missing,
            'present_list': [k for k, v in self.indicators_present.items() if v]
        }


if __name__ == "__main__":
    # Test the data loader
    import sys
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with featured.csv
    csv_path = Path(__file__).parent.parent.parent / "data" / "input" / "featured.csv"
    
    if csv_path.exists():
        loader = DataLoader(csv_path=str(csv_path))
        print("\n" + "="*80)
        print("DATA LOADER TEST")
        print("="*80)
        print(f"\nLoaded {len(loader.df)} rows")
        print(f"\nIndicators Present:")
        for ind, present in loader.indicators_present.items():
            status = "✓" if present else "✗"
            print(f"  {status} {ind}")
        
        print(f"\nMissing Indicators: {loader.indicators_missing}")
        print(f"\nSummary: {loader.summary()}")
    else:
        print(f"Test file not found: {csv_path}")

