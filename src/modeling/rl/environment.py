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
    from collections import deque
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
            portfolio,  # Explicitly required for Unified Logging
            scored_patterns: pd.DataFrame = None,
            config: TradingConfig = None,
            initial_capital: float = 100000,
            max_steps: int = 300,
            window_size=50
        ):
            """
            Modified Environment to support unified Portfolio Tracking.
            """
            super().__init__()
            
            # 1. CORE COMPONENTS & INJECTION
            self.config = config or TradingConfig()
            self.portfolio = portfolio  # The shared PortfolioRiskManager instance
            self.df = df.reset_index(drop=True)
            self.scored_patterns = scored_patterns if scored_patterns is not None else pd.DataFrame()
            
            # 2. STATE VARIABLES
            self.initial_capital = initial_capital
            self.max_steps = max_steps
            self.window_size = window_size
            
            # 3. RL TRACKING
            # Sliding window for Sharpe Ratio (History of step returns)
            self.returns_history = deque(maxlen=window_size)
            
            # Track ID of the current active trade in the portfolio
            self.active_trade_id = None 
            
            # 4. ACTION & OBSERVATION SPACES
            # Action space: Hold(0), Long(1), Flatten(2)
            self.action_space = spaces.Discrete(3)
            
            # Observation space: 7 features (e.g., RSI, Volatility, Pattern Signal, etc.)
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(7,), 
                dtype=np.float32
            )

            # 5. INITIALIZE
            self._validate_dataframe()
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
                (self.scored_patterns['Confirmed_At'] >= self.idx - 5) &
                (self.scored_patterns['Confirmed_At'] <= self.idx)
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
            """
            super().reset(seed=seed)
            
            # Start after indicators are filled (60 bars for safety)
            self.idx = min(60, len(self.df) - self.max_steps - 1)
            self.steps = 0
            self.position = 0  # 0=flat, 1=long
            self.entry_price = 0
            self.entry_step = 0
            
            # --- MISSING ATTRIBUTES FIX ---
            self.equity = float(self.initial_capital)
            self.peak_equity = float(self.initial_capital)  # This fixes the AttributeError
            self.trades = []
            # ------------------------------
            
            obs = self._get_obs()
            info = {}
            
            return obs, info

        def _get_info(self, reward, pattern_info):
            """
            Diagnostic info for Stable Baselines and Logging.
            Ensures we pull from the shared portfolio and local environment state.
            """
            try:
                # 1. Get metrics from our Unified Portfolio Manager
                # We use silent=True to prevent the infinite printing loop
                metrics = self.portfolio.get_metrics(engine_name="RL-Agent", silent=True)
                
                # 2. Bundle the info for the RL Agent and the Dashboard
                return {
                    'portfolio_value': float(self.portfolio.current_capital),
                    'total_trades': int(metrics.get('total_trades', 0)),
                    'win_rate': float(metrics.get('win_rate', 0)),
                    'reward': float(reward),      # Passes the step-reward to the logger
                    'step': int(self.steps),      # Uses the correct variable (self.steps)
                    'market_idx': int(self.idx),  # Use the correct variable (self.idx)
                    'pattern': str(pattern_info.get('Pattern_Type', 'None'))
                }
            except Exception as e:
                # Fallback to prevent the whole training from crashing if a metric is missing
                return {
                    'portfolio_value': self.initial_capital,
                    'reward': reward,
                    'step': self.steps,
                    'error': str(e)
                }
        
        def _calculate_sharpe_reward(self, current_return):
            """
            Calculates a risk-adjusted reward based on the last N steps.
            """
            self.returns_history.append(current_return)
            
            if len(self.returns_history) < 10:
                return current_return # Not enough data for volatility yet
                
            returns_arr = np.array(self.returns_history)
            avg_return = np.mean(returns_arr)
            std_return = np.std(returns_arr)
            
            # Risk-Free Rate assumed 0 for simplicity
            # We add a small epsilon (1e-6) to avoid division by zero
            sharpe = avg_return / (std_return + 1e-6)
            
            # Scale the reward so it doesn't vanish (Sharpe is usually small)
            return sharpe * 0.1
        
        def step(self, action):
            """
            Execute action and return next state with UNIFIED LOGGING
            """
            self.steps += 1
            
            # 1. Setup indices and terminal conditions
            terminated = bool(self.idx >= len(self.df) - 1) 
            truncated = bool(self.steps >= self.max_steps)
            
            curr_idx = min(self.idx, len(self.df) - 1)
            curr_close = self.df.loc[curr_idx, 'Close']
            reward = 0.0
            
            # 2. GET ACTUAL PATTERN DATA (Using your provided logic)
            # Find recent patterns (within last 5 bars)
            recent_patterns = self.scored_patterns[
                (self.scored_patterns['Confirmed_At'] >= self.idx - 5) &
                (self.scored_patterns['Confirmed_At'] <= self.idx)
            ]
            
            best_pattern = None
            pattern_signal = 0.0
            
            if not recent_patterns.empty:
                # Use your logic to get the highest quality pattern
                best_pattern_row = recent_patterns.loc[recent_patterns['Quality_Score'].idxmax()]
                best_pattern = best_pattern_row.to_dict() # Convert to dict for Portfolio Manager
                
                # Calculate signal for reward logic
                quality_norm = best_pattern['Quality_Score'] / 100.0
                pattern_signal = quality_norm if best_pattern['Pattern_Type'] == 'DoubleBottom' else -quality_norm

            # 3. ACTION EXECUTION
            if action == 1:  # GO LONG
                if self.position == 0:
                    # FIX: Create the 'evaluation' dict with 'shares' to avoid KeyError
                    # We pass the actual 'best_pattern' found above instead of a mock
                    evaluation = {
                        'approved': True, 
                        'shares': 1.0,  # Matches what risk_manager.py:71 expects
                        'reason': 'RL Agent Signal'
                    }
                    
                    # 2. Ensure we have valid Pattern Data for Risk Management
                    if best_pattern is not None:
                        # Use the actual detected pattern
                        trade_pattern = best_pattern
                    else:
                        # Create a "Synthetic Pattern" so RiskManager doesn't crash
                        # Calculate a simple ATR-based Stop Loss
                        atr_val = self.df.loc[curr_idx, 'ATR'] if 'ATR' in self.df.columns else curr_close * 0.02
                        trade_pattern = {
                            'Pattern_Type': 'RL_Trend',
                            'Confirmed_At': curr_idx,
                            'Quality_Score': 50.0,
                            'Stop_Loss': curr_close - (atr_val * 2), # Required by RiskManager
                            'Target': curr_close + (atr_val * 4)     # Often required too
                        }

                    # 3. Call Portfolio with complete data
                    self.active_trade_id = self.portfolio.open_position(
                        trade_pattern, 
                        evaluation, 
                        curr_close, 
                        engine_type="RL-Agent"
                    )
                    
                    self.position = 1
                    self.entry_price = curr_close
                    self.entry_step = self.steps
                    reward -= self.config.RL_TRANSACTION_COST
                    
                    if pattern_signal > 0.5:
                        reward += self.config.RL_PATTERN_BONUS
                        
            elif action == 2:  # FLATTEN
                if self.position == 1:
                    # Sync with Portfolio Manager
                    exit_date = self.df.loc[curr_idx].get('Date', str(datetime.now()))
                    if self.active_trade_id:
                        self.portfolio.close_position(self.active_trade_id, curr_close, exit_date)
                    
                    trade_return = (curr_close - self.entry_price) / self.entry_price
                    reward += np.clip(trade_return, -0.1, 0.1) * self.config.RL_REWARD_SCALE
                    reward -= self.config.RL_TRANSACTION_COST
                    
                    self.position = 0
                    self.entry_price = 0
                    self.active_trade_id = None

            # 4. Mark-to-Market & Equity Updates
            if self.position == 1:
                # Use current portfolio capital for accurate tracking
                self.equity = float(self.portfolio.current_capital)
            else:
                self.equity = float(self.portfolio.current_capital)
                
            self.peak_equity = max(self.peak_equity, self.equity)
            
            # 5. Advance Market
            self.idx += 1
            obs = self._get_obs()
            
            # 6. Build Info (Ensuring keys match what your dashboard expects)
            info = self._get_info(reward, best_pattern if best_pattern else {})
            
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