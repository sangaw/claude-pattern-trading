"""
rl/environment.py - Pattern-Aware Trading Environment (FIXED)
=============================================================

Gymnasium-compatible trading environment with pattern signals and proper error handling.
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Optional RL imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Note: gymnasium not installed. RL features disabled.")

# Optional deep RL algorithms
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Note: stable-baselines3 not installed. RL training disabled.")

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from stable_baselines3.common.callbacks import BaseCallback

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradingConfig

try:
    import gymnasium as gym
    from gymnasium import spaces
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("gymnasium not installed. RL features disabled.")


if RL_AVAILABLE:
    class PatternAwareTradingEnv(gym.Env):
        """
        Trading environment with pattern awareness
        
        Observation Space (7 features):
        - Price change (%)
        - RSI (normalized 0-1)
        - MACD (normalized)
        - MACD Signal (normalized)
        - ATR (normalized)
        - Position (0 or 1)
        - Pattern Signal (-1 to 1)
        
        Action Space (3 discrete):
        - 0: Hold (do nothing)
        - 1: Go/Keep Long (enter or maintain)
        - 2: Flatten (close position)
        """
        
        def __init__(
            self,
            df: pd.DataFrame,
            scored_patterns: pd.DataFrame = None,
            config: TradingConfig = None,
            initial_capital: float = 100000,
            max_steps: int = 300
        ):
            """
            Initialize environment
            
            Args:
                df: Price data with indicators (MUST have RSI, MACD, ATR, etc.)
                scored_patterns: Patterns with quality scores
                config: Configuration object
                initial_capital: Starting capital
                max_steps: Maximum steps per episode
            """
            super().__init__()
            
            self.config = config or TradingConfig()
            self.df = df.reset_index(drop=True)
            self.scored_patterns = scored_patterns if scored_patterns is not None else pd.DataFrame()
            self.initial_capital = initial_capital
            self.max_steps = max_steps
            
            # Validate DataFrame has required columns
            self._validate_dataframe()
            
            # Action space: Hold(0), Long(1), Flatten(2)
            self.action_space = spaces.Discrete(3)
            
            # Observation space: 7 features
            self.observation_space = spaces.Box(
                low=-10, 
                high=10, 
                shape=(7,), 
                dtype=np.float32
            )
            
            self.reset()
        
        def _validate_dataframe(self):
            """Validate that DataFrame has all required indicators"""
            required = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Volatility']
            missing = [col for col in required if col not in self.df.columns]
            
            if missing:
                raise ValueError(
                    f"DataFrame missing required columns: {missing}\n"
                    f"Available columns: {list(self.df.columns)}\n"
                    f"Tip: Pass the enriched DataFrame from BasePatternDetector, not raw data"
                )
            
            # Check for NaN values
            if self.df[required].isnull().any().any():
                print("⚠️ Warning: NaN values in indicators, will be filled with defaults")
        
        def _get_pattern_signal(self) -> float:
            """
            Get pattern signal at current index
            
            Returns:
                float: Signal strength (-1 to 1)
                    Positive = bullish pattern
                    Negative = bearish pattern
                    0 = no pattern
            """
            if self.scored_patterns.empty:
                return 0.0
            
            # Find recent patterns (within last 5 bars)
            recent_patterns = self.scored_patterns[
                (self.scored_patterns['Detection_Index'] >= self.idx - 5) &
                (self.scored_patterns['Detection_Index'] <= self.idx)
            ]
            
            if recent_patterns.empty:
                return 0.0
            
            # Get highest quality pattern
            best = recent_patterns.loc[recent_patterns['Quality_Score'].idxmax()]
            quality_norm = best['Quality_Score'] / 100.0  # Normalize to 0-1
            
            # Return signed signal
            if best['Pattern_Type'] == 'DoubleBottom':
                return quality_norm  # Bullish: 0 to 1
            elif best['Pattern_Type'] == 'DoubleTop':
                return -quality_norm  # Bearish: -1 to 0
            
            return 0.0
        
        def _get_obs(self) -> np.ndarray:
            """
            Get current observation
            
            Returns:
                numpy array of 7 features
            """
            if self.idx >= len(self.df):
                self.idx = len(self.df) - 1
            
            row = self.df.loc[self.idx]
            prev_close = self.df.loc[max(0, self.idx - 1), 'Close']
            
            # Calculate features with safe defaults
            pct_change = (row['Close'] - prev_close) / prev_close if prev_close != 0 else 0
            
            rsi = row.get('RSI', 50.0)
            rsi_norm = rsi / 100.0 if not pd.isna(rsi) else 0.5
            
            macd = row.get('MACD', 0.0)
            macd_norm = macd / 100.0 if not pd.isna(macd) else 0.0
            
            macd_signal = row.get('MACD_Signal', 0.0)
            macd_signal_norm = macd_signal / 100.0 if not pd.isna(macd_signal) else 0.0
            
            atr = row.get('ATR', 0.0)
            close = row['Close']
            atr_norm = atr / close if close != 0 and not pd.isna(atr) else 0.0
            
            pattern_signal = self._get_pattern_signal()
            
            obs = np.array([
                np.clip(pct_change, -1, 1),
                np.clip(rsi_norm, 0, 1),
                np.clip(macd_norm, -1, 1),
                np.clip(macd_signal_norm, -1, 1),
                np.clip(atr_norm, 0, 0.1) * 10,  # Scale ATR
                float(self.position),
                np.clip(pattern_signal, -1, 1)
            ], dtype=np.float32)
            
            # Handle any NaN values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return obs
        
        def reset(self, seed=None, options=None):
            """
            Reset environment to initial state
            
            Returns:
                tuple: (observation, info)
            """
            super().reset(seed=seed)
            
            # Start after indicators are filled (60 bars for safety)
            self.idx = min(60, len(self.df) - self.max_steps - 1)
            self.steps = 0
            self.position = 0  # 0=flat, 1=long
            self.entry_price = 0
            self.entry_step = 0
            self.equity = self.initial_capital
            self.peak_equity = self.initial_capital
            self.trades = []
            
            obs = self._get_obs()
            info = {}
            
            return obs, info
        
        def step(self, action):
            """
            Execute action and return next state
            
            Args:
                action: 0=Hold, 1=Long, 2=Flatten
            
            Returns:
                tuple: (observation, reward, terminated, truncated, info)
            """
            self.steps += 1
            
            # Check if episode should end
            terminated = False
            truncated = (
                self.steps >= self.max_steps or 
                self.idx >= len(self.df) - 2
            )
            
            # Get prices safely
            curr_idx = min(self.idx, len(self.df) - 1)
            prev_idx = max(0, curr_idx - 1)
            
            prev_close = self.df.loc[prev_idx, 'Close']
            curr_close = self.df.loc[curr_idx, 'Close']
            volatility = self.df.loc[curr_idx, 'Volatility']
            volatility = volatility if not pd.isna(volatility) else 0.01
            
            reward = 0.0
            pattern_signal = self._get_pattern_signal()
            
            # Execute action
            if action == 1:  # Go/Keep Long
                if self.position == 0:
                    # Enter long position
                    self.position = 1
                    self.entry_price = curr_close
                    self.entry_step = self.steps
                    
                    # Transaction cost
                    reward -= self.config.RL_TRANSACTION_COST
                    
                    # Bonus for entering on strong bullish pattern
                    if pattern_signal > 0.5:
                        reward += self.config.RL_PATTERN_BONUS
                
            elif action == 2:  # Flatten
                if self.position == 1:
                    # Exit long position
                    trade_return = (curr_close - self.entry_price) / self.entry_price
                    
                    # Cap returns to prevent exploitation
                    capped_return = np.clip(
                        trade_return,
                        -self.config.RL_MAX_TRADE_RETURN,
                        self.config.RL_MAX_TRADE_RETURN
                    )
                    
                    reward += capped_return * self.config.RL_REWARD_SCALE
                    
                    # Record trade
                    self.trades.append({
                        'entry': self.entry_price,
                        'exit': curr_close,
                        'return': trade_return,
                        'holding_period': self.steps - self.entry_step
                    })
                    
                    # Transaction cost
                    reward -= self.config.RL_TRANSACTION_COST
                    
                    # Reset position
                    self.position = 0
                    self.entry_price = 0
                else:
                    # Penalty for flattening when not in position
                    reward -= 0.001
            
            elif action == 0:  # Hold
                # Small penalty for excessive holding
                if self.steps % 50 == 0:
                    reward -= 0.0001
            
            # Mark-to-market for open positions
            if self.position == 1:
                price_change = (curr_close - prev_close) / prev_close if prev_close != 0 else 0
                
                # Risk-adjusted return
                if volatility > 0:
                    risk_adjusted_return = price_change / (volatility + 0.01)
                else:
                    risk_adjusted_return = price_change
                
                reward += risk_adjusted_return * 0.5
                
                # Penalty for holding too long
                holding_period = self.steps - self.entry_step
                if holding_period > self.config.RL_HOLDING_PENALTY_STEPS:
                    reward -= self.config.RL_HOLDING_PENALTY
                
                # Drawdown penalty
                unrealized_pnl_pct = (curr_close - self.entry_price) / self.entry_price if self.entry_price != 0 else 0
                if unrealized_pnl_pct < -0.05:
                    reward -= 0.05
                elif unrealized_pnl_pct < -0.02:
                    reward -= 0.01
                
                self.equity = self.initial_capital * (1 + unrealized_pnl_pct)
            else:
                self.equity = self.initial_capital
            
            self.peak_equity = max(self.peak_equity, self.equity)
            
            # Advance to next bar
            self.idx += 1
            obs = self._get_obs()
            
            # info = {}
            
            # 1. Determine the "Condition" (The 'Why')
            action_names = {0: "Hold", 1: "Long", 2: "Flatten"}
            condition = f"Action: {action_names.get(action)}"
            
            # Add context to the reason if a pattern was present
            if pattern_signal > 0.5:
                condition += f" | Bullish Signal: {pattern_signal:.2f}"
            
            # 2. Populate the info dictionary
            info = {
                'portfolio_value': float(self.equity),
                'total_trades': len(self.trades),
                'reason': condition,
                'trade_taken': action != 0  # True if it's Long or Flatten
            }
            
            return obs, float(reward), terminated, truncated, info
        
        def get_trade_stats(self):
            """
            Get statistics about trades made during episode
            
            Returns:
                dict: Trade statistics
            """
            if not self.trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'total_return': 0
                }
            
            returns = [t['return'] for t in self.trades]
            winning_trades = [r for r in returns if r > 0]
            
            return {
                'total_trades': len(returns),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(returns) if returns else 0,
                'avg_return': np.mean(returns),
                'total_return': sum(returns)
            }

else:
    # Dummy class when gymnasium not available
    class PatternAwareTradingEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gymnasium not installed. "
                "Install with: pip install gymnasium"
            )


if __name__ == "__main__":
    if RL_AVAILABLE:
        print("Testing PatternAwareTradingEnv...")
        
        # Create sample data with indicators
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'date': dates,
            'Close': 100 + np.random.randn(200).cumsum(),
            'RSI': np.random.uniform(30, 70, 200),
            'MACD': np.random.randn(200),
            'MACD_Signal': np.random.randn(200),
            'ATR': np.random.uniform(1, 5, 200),
            'Volatility': np.random.uniform(0.01, 0.03, 200)
        })
        
        # Create environment
        env = PatternAwareTradingEnv(df, max_steps=50)
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        
        # Test step
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print(f"✓ Environment test complete")
        print(f"  Steps taken: {env.steps}")
        print(f"  Trades made: {len(env.trades)}")
        
        stats = env.get_trade_stats()
        print(f"  Trade stats: {stats}")
    else:
        print("⚠️ gymnasium not installed, skipping test")      