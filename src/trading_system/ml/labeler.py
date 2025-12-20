# ============================================================================
# FILE: ml/labeler.py
# Pattern labeling for ML training
# ============================================================================

import pandas as pd
from config import TradingConfig


class PatternLabeler:
    """
    Labels patterns with success/failure based on forward price action
    
    Success Criteria:
    - Pattern reaches target before stop-loss
    - Within specified forward bars
    """
    
    def __init__(self, df: pd.DataFrame, patterns_df: pd.DataFrame):
        """
        Initialize labeler
        
        Args:
            df: Price data
            patterns_df: Detected patterns
        """
        self.df = df
        self.patterns_df = patterns_df
    
    def label_patterns(self, forward_bars: int = None) -> pd.DataFrame:
        """
        Label all patterns with success/failure
        
        Args:
            forward_bars: Number of bars to look ahead
        
        Returns:
            DataFrame with labeled patterns
        """
        forward_bars = forward_bars or TradingConfig.FORWARD_BARS
        
        labeled = []
        for _, pattern in self.patterns_df.iterrows():
            result = self._evaluate_pattern(pattern, forward_bars)
            labeled.append({**pattern.to_dict(), **result})
        
        return pd.DataFrame(labeled)
    
    def _evaluate_pattern(self, pattern: dict, forward_bars: int) -> dict:
        """
        Evaluate single pattern
        
        Returns:
            dict with 'Success' (0/1) and 'Return' (-1/0/1)
        """
        start_idx = pattern['Detection_Index']
        stop_loss = pattern['Stop_Loss']
        target = pattern['Target']
        is_bearish = pattern['Pattern_Type'] == 'DoubleTop'
        
        if start_idx >= len(self.df) - 1:
            return {'Success': 0, 'Return': 0}
        
        # Check each bar forward
        for i in range(start_idx + 1, min(start_idx + forward_bars, len(self.df))):
            high = self.df.loc[i, 'High']
            low = self.df.loc[i, 'Low']
            
            if is_bearish:
                if high >= stop_loss:
                    return {'Success': 0, 'Return': -1}
                if low <= target:
                    return {'Success': 1, 'Return': 1}
            else:
                if low <= stop_loss:
                    return {'Success': 0, 'Return': -1}
                if high >= target:
                    return {'Success': 1, 'Return': 1}
        
        return {'Success': 0, 'Return': 0}