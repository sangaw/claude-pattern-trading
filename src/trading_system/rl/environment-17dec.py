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

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class PatternAwareTradingEnv(gym.Env):
        """Enhanced RL environment with pattern signals"""
        
        def __init__(self, df, scored_patterns=None, initial_capital=100000, max_steps=300):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.scored_patterns = scored_patterns if scored_patterns is not None else pd.DataFrame()
            self.initial_capital = initial_capital
            self.max_steps = max_steps
            
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
            
            self.reset()
        
        def _get_pattern_signal(self):
            """Get pattern signal at current index"""
            if self.scored_patterns.empty:
                return 0.0
            
            recent_patterns = self.scored_patterns[
                (self.scored_patterns['Detection_Index'] >= self.idx - 5) &
                (self.scored_patterns['Detection_Index'] <= self.idx)
            ]
            
            if recent_patterns.empty:
                return 0.0
            
            best = recent_patterns.loc[recent_patterns['Quality_Score'].idxmax()]
            quality_norm = best['Quality_Score'] / 100.0
            
            if best['Pattern_Type'] == 'DoubleBottom':
                return quality_norm
            elif best['Pattern_Type'] == 'DoubleTop':
                return -quality_norm
            
            return 0.0
        
        def _get_obs(self):
            """Get current state observation"""
            row = self.df.loc[self.idx]
            prev_close = self.df.loc[self.idx - 1, 'Close']
            
            pct_change = (row['Close'] - prev_close) / prev_close if prev_close != 0 else 0
            atr_norm = row['ATR'] / row['Close'] if row['Close'] != 0 else 0
            pattern_signal = self._get_pattern_signal()
            
            obs = np.array([
                pct_change,
                row.get('RSI', 50) / 100,
                row.get('MACD', 0) / 100,
                row.get('MACD_Signal', 0) / 100,
                atr_norm,
                float(self.position),
                pattern_signal
            ], dtype=np.float32)
            
            return obs
        
        def reset(self, seed=None, options=None):
            """Reset environment"""
            super().reset(seed=seed)
            
            self.idx = 60
            self.steps = 0
            self.position = 0
            self.entry_price = 0
            self.equity = self.initial_capital
            self.peak_equity = self.initial_capital
            self.trades = []
            
            obs = self._get_obs()
            return obs, {}
        
        def step(self, action):
            """Execute action and return next state"""
            self.steps += 1
            
            terminated = False
            truncated = self.steps >= self.max_steps or self.idx >= len(self.df) - 2
            
            prev_close = self.df.loc[self.idx - 1, 'Close']
            curr_close = self.df.loc[self.idx, 'Close']
            volatility = self.df.loc[self.idx, 'Volatility']
            
            reward = 0.0
            
            if action == 1:  # Go/Keep Long
                if self.position == 0:
                    self.position = 1
                    self.entry_price = curr_close
                    reward -= 0.0002
                    
            elif action == 2:  # Flatten
                if self.position == 1:
                    trade_return = (curr_close - self.entry_price) / self.entry_price
                    reward += trade_return
                    
                    self.trades.append({
                        'entry': self.entry_price,
                        'exit': curr_close,
                        'return': trade_return
                    })
                    
                    reward -= 0.0002
                    self.position = 0
                    self.entry_price = 0
            
            if self.position == 1:
                price_change = (curr_close - prev_close) / prev_close
                
                if volatility > 0:
                    risk_adjusted_return = price_change / (volatility + 0.01)
                else:
                    risk_adjusted_return = price_change
                
                reward += risk_adjusted_return
                
                unrealized_pnl_pct = (curr_close - self.entry_price) / self.entry_price
                if unrealized_pnl_pct < -0.05:
                    reward -= 0.1
                elif unrealized_pnl_pct < -0.02:
                    reward -= 0.02
                
                self.equity = self.initial_capital * (1 + unrealized_pnl_pct)
            else:
                self.equity = self.initial_capital
            
            self.peak_equity = max(self.peak_equity, self.equity)
            self.idx += 1
            obs = self._get_obs()
            
            return obs, float(reward), terminated, truncated, {}
        
        def get_trade_stats(self):
            """Get statistics about trades made during episode"""
            if not self.trades:
                return {}
            
            returns = [t['return'] for t in self.trades]
            winning_trades = [r for r in returns if r > 0]
            
            return {
                'total_trades': len(returns),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(returns) if returns else 0,
                'avg_return': np.mean(returns),
                'total_return': sum(returns)
            }