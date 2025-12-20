"""
COMPLETE MULTI-PATTERN TRADING SYSTEM WITH ENHANCED RL
Single file - No imports needed - Just run it!

This file contains everything: Pattern Detection, ML, Portfolio Management, RL, and Dashboard
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
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

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ============================================================================
# PART 1: BASE PATTERN DETECTOR
# ============================================================================

class BasePatternDetector:
    """Base class for all pattern detectors"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self._calculate_base_indicators()
    
    def _calculate_base_indicators(self):
        """Calculate common technical indicators"""
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Moving Averages
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        
        # ATR
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['ATR'] = true_range.rolling(14).mean()
        
        # Volatility
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(20).std()
        
        # Volume
        if 'Volume' in self.df.columns:
            self.df['Volume_MA'] = self.df['Volume'].rolling(20).mean()
        
        # Trend
        self.df['Trend'] = np.where(self.df['SMA_20'] > self.df['SMA_50'], 1, -1)
        
        # Peaks and Troughs
        self.df['Peak'] = False
        self.df['Trough'] = False
        peak_idx = argrelextrema(self.df['High'].values, np.greater, order=5)[0]
        trough_idx = argrelextrema(self.df['Low'].values, np.less, order=5)[0]
        self.df.loc[peak_idx, 'Peak'] = True
        self.df.loc[trough_idx, 'Trough'] = True


class DoubleTopDetector(BasePatternDetector):
    """Detects Double Top patterns"""
    
    def detect_patterns(self, min_bars=10, max_bars=50, tolerance=0.02):
        patterns = []
        peak_indices = self.df[self.df['Peak'] == True].index.tolist()
        
        for i in range(len(peak_indices) - 1):
            for j in range(i + 1, len(peak_indices)):
                idx1, idx2 = peak_indices[i], peak_indices[j]
                bars_between = idx2 - idx1
                
                if bars_between < min_bars or bars_between > max_bars:
                    continue
                
                peak1 = self.df.loc[idx1, 'High']
                peak2 = self.df.loc[idx2, 'High']
                
                if abs(peak1 - peak2) / peak1 > tolerance:
                    continue
                
                trough_idx = self.df.loc[idx1:idx2, 'Low'].idxmin()
                neckline = self.df.loc[trough_idx, 'Low']
                pattern_height = max(peak1, peak2) - neckline
                
                patterns.append({
                    'Pattern_Type': 'DoubleTop',
                    'Detection_Index': idx2,
                    'Detection_Date': self.df.loc[idx2, 'date'],
                    'Neckline': neckline,
                    'Pattern_Height': pattern_height,
                    'Stop_Loss': max(peak1, peak2) * 1.002,
                    'Target': neckline - pattern_height,
                    'RSI': self.df.loc[idx2, 'RSI'],
                    'ATR': self.df.loc[idx2, 'ATR'],
                    'Volatility': self.df.loc[idx2, 'Volatility'],
                    'Trend': self.df.loc[idx2, 'Trend'],
                })
        
        return pd.DataFrame(patterns)


class DoubleBottomDetector(BasePatternDetector):
    """Detects Double Bottom patterns"""
    
    def detect_patterns(self, min_bars=10, max_bars=50, tolerance=0.02):
        patterns = []
        trough_indices = self.df[self.df['Trough'] == True].index.tolist()
        
        for i in range(len(trough_indices) - 1):
            for j in range(i + 1, len(trough_indices)):
                idx1, idx2 = trough_indices[i], trough_indices[j]
                bars_between = idx2 - idx1
                
                if bars_between < min_bars or bars_between > max_bars:
                    continue
                
                bottom1 = self.df.loc[idx1, 'Low']
                bottom2 = self.df.loc[idx2, 'Low']
                
                if abs(bottom1 - bottom2) / bottom1 > tolerance:
                    continue
                
                peak_idx = self.df.loc[idx1:idx2, 'High'].idxmax()
                neckline = self.df.loc[peak_idx, 'High']
                pattern_height = neckline - min(bottom1, bottom2)
                
                patterns.append({
                    'Pattern_Type': 'DoubleBottom',
                    'Detection_Index': idx2,
                    'Detection_Date': self.df.loc[idx2, 'date'],
                    'Neckline': neckline,
                    'Pattern_Height': pattern_height,
                    'Stop_Loss': min(bottom1, bottom2) * 0.998,
                    'Target': neckline + pattern_height,
                    'RSI': self.df.loc[idx2, 'RSI'],
                    'ATR': self.df.loc[idx2, 'ATR'],
                    'Volatility': self.df.loc[idx2, 'Volatility'],
                    'Trend': self.df.loc[idx2, 'Trend'],
                })
        
        return pd.DataFrame(patterns)


# ============================================================================
# PART 2: PATTERN LABELING & ML TRAINING
# ============================================================================

class PatternLabeler:
    """Labels patterns with success/failure"""
    
    def __init__(self, df, patterns_df):
        self.df = df
        self.patterns_df = patterns_df
    
    def label_patterns(self, forward_bars=30):
        labeled = []
        
        for _, pattern in self.patterns_df.iterrows():
            result = self._evaluate_pattern(pattern, forward_bars)
            labeled.append({**pattern.to_dict(), **result})
        
        return pd.DataFrame(labeled)
    
    def _evaluate_pattern(self, pattern, forward_bars):
        start_idx = pattern['Detection_Index']
        stop_loss = pattern['Stop_Loss']
        target = pattern['Target']
        is_bearish = pattern['Pattern_Type'] in ['DoubleTop']
        
        if start_idx >= len(self.df) - 1:
            return {'Success': 0, 'Return': 0}
        
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


class PatternQualityScorer:
    """ML model to score pattern quality"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
    
    def train(self, labeled_patterns_df):
        pattern_types = labeled_patterns_df['Pattern_Type'].unique()
        results = {}
        
        for ptype in pattern_types:
            pattern_data = labeled_patterns_df[labeled_patterns_df['Pattern_Type'] == ptype].copy()
            
            if len(pattern_data) < 30:
                print(f"  Skipping {ptype} - insufficient data ({len(pattern_data)} samples)")
                continue
            
            feature_cols = ['Pattern_Height', 'RSI', 'ATR', 'Volatility', 'Trend']
            feature_cols = [c for c in feature_cols if c in pattern_data.columns]
            
            X = pattern_data[feature_cols].fillna(0)
            y = pattern_data['Success']
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            accuracy = (y_pred == y_test).mean()
            print(f"  {ptype}: Accuracy = {accuracy:.2%}")
            
            self.models[ptype] = model
            self.scalers[ptype] = scaler
            self.feature_columns[ptype] = feature_cols
            results[ptype] = accuracy
        
        return results
    
    def predict_quality(self, patterns_df):
        if patterns_df.empty:
            return patterns_df
        
        patterns_with_scores = patterns_df.copy()
        patterns_with_scores['Quality_Score'] = 50.0
        
        for ptype in patterns_df['Pattern_Type'].unique():
            if ptype not in self.models:
                continue
            
            mask = patterns_df['Pattern_Type'] == ptype
            pattern_data = patterns_df[mask].copy()
            
            X = pattern_data[self.feature_columns[ptype]].fillna(0)
            X_scaled = self.scalers[ptype].transform(X)
            
            probabilities = self.models[ptype].predict_proba(X_scaled)[:, 1]
            patterns_with_scores.loc[mask, 'Quality_Score'] = probabilities * 100
        
        return patterns_with_scores


# ============================================================================
# PART 3: PORTFOLIO RISK MANAGEMENT
# ============================================================================

class PortfolioRiskManager:
    """Portfolio risk management system"""
    
    def __init__(self, initial_capital=100000, max_positions=5, risk_per_trade=0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        self.active_positions = []
        self.closed_positions = []
        self.daily_equity = []
        self.peak_equity = initial_capital
    
    def evaluate_new_trade(self, pattern):
        """Evaluate if trade should be taken"""
        if len(self.active_positions) >= self.max_positions:
            return {'approved': False, 'reason': 'MAX_POSITIONS'}
        
        entry_price = pattern.get('Neckline', pattern.get('Close', 0))
        stop_loss = pattern['Stop_Loss']
        
        risk_amount = self.current_capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return {'approved': False, 'reason': 'INVALID_STOP'}
        
        shares = int(risk_amount / stop_distance)
        position_value = shares * entry_price
        
        if position_value > self.current_capital * 0.3:
            shares = int(self.current_capital * 0.3 / entry_price)
            position_value = shares * entry_price
        
        return {
            'approved': True,
            'shares': shares,
            'position_value': position_value,
            'risk_amount': shares * stop_distance
        }
    
    def open_position(self, pattern, evaluation, entry_price):
        """Open new position"""
        position = {
            'pattern_type': pattern['Pattern_Type'],
            'entry_date': pattern['Detection_Date'],
            'entry_price': entry_price,
            'shares': evaluation['shares'],
            'position_value': evaluation['position_value'],
            'stop_loss': pattern['Stop_Loss'],
            'target': pattern['Target'],
            'unrealized_pnl': 0
        }
        self.active_positions.append(position)
        return position
    
    def close_position(self, position_idx, exit_price, exit_date):
        """Close position"""
        if position_idx >= len(self.active_positions):
            return None
        
        position = self.active_positions.pop(position_idx)
        pnl = (exit_price - position['entry_price']) * position['shares']
        
        if position['pattern_type'] == 'DoubleTop':
            pnl = -pnl
        
        self.current_capital += pnl
        
        closed = {**position, 'exit_price': exit_price, 'exit_date': exit_date, 'pnl': pnl}
        self.closed_positions.append(closed)
        return closed
    
    def update_positions(self, current_date, current_price):
        """Update all positions"""
        for pos in self.active_positions:
            unrealized = (current_price - pos['entry_price']) * pos['shares']
            if pos['pattern_type'] == 'DoubleTop':
                unrealized = -unrealized
            pos['unrealized_pnl'] = unrealized
        
        total_unrealized = sum(p['unrealized_pnl'] for p in self.active_positions)
        current_equity = self.current_capital + total_unrealized
        
        self.daily_equity.append({'date': current_date, 'equity': current_equity})
        self.peak_equity = max(self.peak_equity, current_equity)
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if not self.closed_positions:
            return {'total_trades': 0}
        
        df = pd.DataFrame(self.closed_positions)
        wins = len(df[df['pnl'] > 0])
        total = len(df)
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if total > wins else 0,
            'final_capital': self.current_capital
        }


# ============================================================================
# PART 4: ENHANCED RL TRADING ENVIRONMENT WITH PATTERN INTEGRATION
# ============================================================================

if RL_AVAILABLE:
    class PatternAwareTradingEnv(gym.Env):
        """
        Enhanced RL environment with pattern signals and improved reward function
        
        Observation: [pct_change, RSI, MACD, MACD_Signal, ATR_norm, position_flag, pattern_signal]
        Actions: 0=hold, 1=go/keep long, 2=flatten
        Reward: Risk-adjusted returns with transaction costs and drawdown penalties
        """
        
        def __init__(self, df, scored_patterns=None, initial_capital=100000, max_steps=300):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.scored_patterns = scored_patterns if scored_patterns is not None else pd.DataFrame()
            self.initial_capital = initial_capital
            self.max_steps = max_steps
            
            # Action space: 0=hold, 1=go/keep long, 2=flatten
            self.action_space = spaces.Discrete(3)
            
            # Observation space: 7 features
            self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
            
            self.reset()
        
        def _get_pattern_signal(self):
            """Get pattern signal at current index"""
            if self.scored_patterns.empty:
                return 0.0
            
            # Find patterns detected at or near current index (within 5 bars)
            recent_patterns = self.scored_patterns[
                (self.scored_patterns['Detection_Index'] >= self.idx - 5) &
                (self.scored_patterns['Detection_Index'] <= self.idx)
            ]
            
            if recent_patterns.empty:
                return 0.0
            
            # Get highest quality pattern
            best = recent_patterns.loc[recent_patterns['Quality_Score'].idxmax()]
            quality_norm = best['Quality_Score'] / 100.0  # Normalize to 0-1
            
            # Return signed signal based on pattern type
            if best['Pattern_Type'] == 'DoubleBottom':
                return quality_norm  # Bullish: 0 to 1
            elif best['Pattern_Type'] == 'DoubleTop':
                return -quality_norm  # Bearish: -1 to 0
            
            return 0.0
        
        def _get_obs(self):
            """Get current state observation"""
            row = self.df.loc[self.idx]
            prev_close = self.df.loc[self.idx - 1, 'Close']
            
            # Price change
            pct_change = (row['Close'] - prev_close) / prev_close if prev_close != 0 else 0
            
            # Normalized ATR
            atr_norm = row['ATR'] / row['Close'] if row['Close'] != 0 else 0
            
            # Pattern signal
            pattern_signal = self._get_pattern_signal()
            
            obs = np.array([
                pct_change,
                row.get('RSI', 50) / 100,  # Normalize RSI to 0-1
                row.get('MACD', 0) / 100,  # Normalize MACD
                row.get('MACD_Signal', 0) / 100,
                atr_norm,
                float(self.position),  # 0 or 1
                pattern_signal  # -1 to 1
            ], dtype=np.float32)
            
            return obs
        
        def reset(self, seed=None, options=None):
            """Reset environment"""
            super().reset(seed=seed)
            
            # Start after indicators are filled
            self.idx = 60
            self.steps = 0
            self.position = 0  # 0=flat, 1=long
            self.entry_price = 0
            self.equity = self.initial_capital
            self.peak_equity = self.initial_capital
            self.trades = []
            
            obs = self._get_obs()
            return obs, {}
        
        def step(self, action):
            """Execute action and return next state"""
            self.steps += 1
            
            # Check if episode should end
            terminated = False
            truncated = self.steps >= self.max_steps or self.idx >= len(self.df) - 2
            
            # Get prices
            prev_close = self.df.loc[self.idx - 1, 'Close']
            curr_close = self.df.loc[self.idx, 'Close']
            volatility = self.df.loc[self.idx, 'Volatility']
            
            reward = 0.0
            
            # Execute action
            if action == 1:  # Go/Keep Long
                if self.position == 0:
                    # Enter long position
                    self.position = 1
                    self.entry_price = curr_close
                    # Transaction cost
                    reward -= 0.0002  # 0.02% cost
                    
            elif action == 2:  # Flatten
                if self.position == 1:
                    # Exit long position
                    trade_return = (curr_close - self.entry_price) / self.entry_price
                    reward += trade_return
                    
                    # Track trade
                    self.trades.append({
                        'entry': self.entry_price,
                        'exit': curr_close,
                        'return': trade_return
                    })
                    
                    # Transaction cost
                    reward -= 0.0002
                    
                    # Reset position
                    self.position = 0
                    self.entry_price = 0
            
            # Mark-to-market reward for open positions
            if self.position == 1:
                # Price change reward
                price_change = (curr_close - prev_close) / prev_close
                
                # Risk-adjusted reward (Sharpe-like)
                if volatility > 0:
                    risk_adjusted_return = price_change / (volatility + 0.01)
                else:
                    risk_adjusted_return = price_change
                
                reward += risk_adjusted_return
                
                # Drawdown penalty
                unrealized_pnl_pct = (curr_close - self.entry_price) / self.entry_price
                if unrealized_pnl_pct < -0.05:  # More than 5% loss
                    reward -= 0.1  # Heavy penalty for large drawdowns
                elif unrealized_pnl_pct < -0.02:  # More than 2% loss
                    reward -= 0.02  # Small penalty
                
                # Update equity
                self.equity = self.initial_capital * (1 + unrealized_pnl_pct)
            else:
                self.equity = self.initial_capital
            
            # Update peak equity
            self.peak_equity = max(self.peak_equity, self.equity)
            
            # Advance to next bar
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

else:
    class PatternAwareTradingEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium not installed. Please install with: pip install gymnasium")


# ============================================================================
# PART 5: RL TRAINING AND EVALUATION
# ============================================================================

def train_rl_agent(df, scored_patterns, algorithm='PPO', timesteps=50000):
    """
    Train RL agent with pattern signals
    
    Parameters:
    -----------
    df : DataFrame with OHLC and indicators
    scored_patterns : DataFrame with scored patterns
    algorithm : str, 'PPO' or 'A2C'
    timesteps : int, number of training timesteps
    
    Returns:
    --------
    trained model
    """
    if not RL_AVAILABLE or not SB3_AVAILABLE:
        print("RL training skipped: gymnasium/stable-baselines3 not installed")
        return None
    
    print(f"\n{'='*80}")
    print(f"TRAINING RL AGENT ({algorithm})")
    print(f"{'='*80}")
    
    # Split data: 80% train, 20% test
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].reset_index(drop=True)
    test_df = df[train_size:].reset_index(drop=True)
    
    # Create training environment
    train_env = PatternAwareTradingEnv(train_df, scored_patterns)
    
    # Train model
    print(f"\nTraining on {len(train_df)} bars...")
    
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=0,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95
        )
    elif algorithm == 'A2C':
        model = A2C(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    model.learn(total_timesteps=timesteps)
    print(f"✓ Training completed: {timesteps} timesteps")
    
    # Evaluate on training data
    print("\nEvaluating on training data...")
    train_env_eval = PatternAwareTradingEnv(train_df, scored_patterns, max_steps=len(train_df)-100)
    train_reward = evaluate_agent(model, train_env_eval)
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_env = PatternAwareTradingEnv(test_df, scored_patterns, max_steps=len(test_df)-100)
    test_reward = evaluate_agent(model, test_env)
    
    print(f"\n{'='*80}")
    print(f"RL AGENT TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Algorithm              : {algorithm}")
    print(f"Training Timesteps     : {timesteps:,}")
    print(f"Train Cumulative Reward: {train_reward:.4f}")
    print(f"Test Cumulative Reward : {test_reward:.4f}")
    print(f"{'='*80}")
    
    return model


def evaluate_agent(model, env, num_episodes=1):
    """Evaluate RL agent and return cumulative reward"""
    total_reward = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def compare_strategies(df, scored_patterns, rl_model, portfolio_results):
    """
    Compare RL agent vs rule-based portfolio strategy
    
    Parameters:
    -----------
    df : DataFrame with OHLC data
    scored_patterns : DataFrame with scored patterns
    rl_model : trained RL model
    portfolio_results : dict with portfolio backtest results
    
    Returns:
    --------
    dict with comparison metrics
    """
    if not RL_AVAILABLE or rl_model is None:
        print("Strategy comparison skipped: RL not available")
        return None
    
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON: RL vs RULE-BASED")
    print(f"{'='*80}")
    
    # Test RL agent on full dataset
    test_env = PatternAwareTradingEnv(df, scored_patterns, max_steps=len(df)-100)
    obs, _ = test_env.reset()
    
    rl_rewards = []
    done = False
    
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        rl_rewards.append(reward)
        done = terminated or truncated
    
    # Get RL statistics
    rl_stats = test_env.get_trade_stats()
    
    # FIXED: Calculate realistic return from actual trades
    if rl_stats.get('total_trades', 0) > 0:
        rl_total_return = rl_stats['total_return'] * 100  # Use actual trade returns
    else:
        rl_total_return = 0
    
    # Calculate Sharpe from reward distribution (for comparison)
    rl_sharpe = np.mean(rl_rewards) / (np.std(rl_rewards) + 1e-8) * np.sqrt(252)
    
    # Rule-based statistics
    rule_return = portfolio_results.get('total_return', 0)
    rule_win_rate = portfolio_results.get('win_rate', 0)
    rule_trades = portfolio_results.get('total_trades', 0)
    
    # Print comparison with SANITY CHECKS
    print(f"\n{'Strategy':<20} {'Return %':<15} {'Win Rate %':<15} {'Trades':<10} {'Sharpe':<10}", flush=True)
    print(f"{'-'*70}", flush=True)
    print(f"{'RL Agent':<20} {rl_total_return:<15.2f} {rl_stats.get('win_rate', 0)*100:<15.1f} {rl_stats.get('total_trades', 0):<10} {rl_sharpe:<10.2f}", flush=True)
    print(f"{'Rule-Based':<20} {rule_return:<15.2f} {rule_win_rate:<15.1f} {rule_trades:<10} {'N/A':<10}", flush=True)
    print(f"{'-'*70}", flush=True)
    # sys.stdout.flush()
    
    # SANITY CHECKS
    if rl_total_return > 1000:
        print(f"\n⚠️  WARNING: RL return suspiciously high ({rl_total_return:.1f}%)", flush=True)
        print("   Possible reward hacking detected. Consider:", flush=True)
        print("   1. Reducing transaction costs", flush=True)
        print("   2. Increasing position holding penalties", flush=True)
        print("   3. Capping per-trade rewards", flush=True)
        # sys.stdout.flush()
    
    if rl_stats.get('total_trades', 0) > rule_trades * 20:
        print(f"\n⚠️  WARNING: RL trading too frequently ({rl_stats.get('total_trades', 0)} trades)", flush=True)
        print("   Likely overtrading. Rule-based: {rule_trades} trades", flush=True)
        # sys.stdout.flush()
    
    # Determine winner
    winner = "RL Agent" if rl_total_return > rule_return else "Rule-Based"
    improvement = abs(rl_total_return - rule_return)
    
    print(f"\n✓ Winner: {winner} (by {improvement:.2f}%)", flush=True)
    # sys.stdout.flush()
    
    comparison = {
        'rl_return': rl_total_return,
        'rl_win_rate': rl_stats.get('win_rate', 0) * 100,
        'rl_trades': rl_stats.get('total_trades', 0),
        'rl_sharpe': rl_sharpe,
        'rule_return': rule_return,
        'rule_win_rate': rule_win_rate,
        'rule_trades': rule_trades,
        'winner': winner,
        'improvement': improvement
    }
    
    return comparison


# ============================================================================
# PART 6: ENHANCED DASHBOARD WITH RL COMPARISON
# ============================================================================

class TradingDashboard:
    """Enhanced dashboard with RL comparison"""
    
    def __init__(self, portfolio, rl_comparison=None):
        self.portfolio = portfolio
        self.rl_comparison = rl_comparison
    
    def create_dashboard(self, save_path='trading_dashboard.png'):
        """Generate comprehensive dashboard"""
        if self.rl_comparison:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        fig.suptitle('TRADING SYSTEM DASHBOARD', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        ax1 = axes[0, 0] if self.rl_comparison else axes[0, 0]
        if self.portfolio.daily_equity:
            equity_df = pd.DataFrame(self.portfolio.daily_equity)
            ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2, label='Portfolio')
            ax1.axhline(self.portfolio.initial_capital, color='gray', linestyle='--', label='Initial')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Trade Distribution
        ax2 = axes[0, 1] if self.rl_comparison else axes[0, 1]
        if self.portfolio.closed_positions:
            closed_df = pd.DataFrame(self.portfolio.closed_positions)
            returns = closed_df['pnl'].values
            colors = ['green' if x > 0 else 'red' for x in returns]
            ax2.bar(range(len(returns)), returns, color=colors, alpha=0.6)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_title('Trade P&L Distribution')
            ax2.set_ylabel('P&L ($)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance Metrics
        ax3 = axes[1, 0] if self.rl_comparison else axes[1, 0]
        metrics = self.portfolio.get_metrics()
        ax3.axis('off')
        data = [
            ['Total Trades', f"{metrics.get('total_trades', 0)}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1f}%"],
            ['Total P&L', f"${metrics.get('total_pnl', 0):,.0f}"],
            ['Total Return', f"{metrics.get('total_return', 0):.2f}%"],
            ['Avg Win', f"${metrics.get('avg_win', 0):,.0f}"],
            ['Avg Loss', f"${metrics.get('avg_loss', 0):,.0f}"],
        ]
        table = ax3.table(cellText=data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax3.set_title('Performance Metrics', fontweight='bold')
        
        # 4. Pattern Performance
        ax4 = axes[1, 1] if self.rl_comparison else axes[1, 1]
        if self.portfolio.closed_positions:
            closed_df = pd.DataFrame(self.portfolio.closed_positions)
            pattern_stats = closed_df.groupby('pattern_type').agg({
                'pnl': ['count', lambda x: (x > 0).sum()]
            }).reset_index()
            pattern_stats.columns = ['Pattern', 'Total', 'Wins']
            pattern_stats['WinRate'] = pattern_stats['Wins'] / pattern_stats['Total'] * 100
            
            colors = ['green' if x > 50 else 'red' for x in pattern_stats['WinRate']]
            ax4.barh(pattern_stats['Pattern'], pattern_stats['WinRate'], color=colors, alpha=0.6)
            ax4.axvline(50, color='gray', linestyle='--')
            ax4.set_xlabel('Win Rate (%)')
            ax4.set_title('Win Rate by Pattern')
            ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. RL vs Rule-Based Comparison (if available)
        if self.rl_comparison:
            ax5 = axes[0, 2]
            strategies = ['RL Agent', 'Rule-Based']
            returns = [self.rl_comparison['rl_return'], self.rl_comparison['rule_return']]
            colors = ['blue' if returns[0] > returns[1] else 'gray', 
                     'orange' if returns[1] > returns[0] else 'gray']
            ax5.bar(strategies, returns, color=colors, alpha=0.7)
            ax5.set_ylabel('Return (%)')
            ax5.set_title('Strategy Comparison: Returns')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 6. Win Rate Comparison
            ax6 = axes[1, 2]
            win_rates = [self.rl_comparison['rl_win_rate'], self.rl_comparison['rule_win_rate']]
            ax6.bar(strategies, win_rates, color=['blue', 'orange'], alpha=0.7)
            ax6.axhline(50, color='gray', linestyle='--')
            ax6.set_ylabel('Win Rate (%)')
            ax6.set_title('Strategy Comparison: Win Rate')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Dashboard saved as '{save_path}'", flush=True)
        # sys.stdout.flush()
        plt.close()
        return fig


# ============================================================================
# PART 7: MAIN EXECUTION WITH ALL ENHANCEMENTS
# ============================================================================

def generate_sample_data():
    """Generate sample data for testing"""
    print("Generating sample data...", flush=True)
    # sys.stdout.flush()
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    
    trend = np.linspace(10000, 11500, 1000)
    noise = np.cumsum(np.random.randn(1000) * 40)
    close = trend + noise
    
    high = close + np.random.uniform(20, 100, 1000)
    low = close - np.random.uniform(20, 100, 1000)
    open_price = close + np.random.uniform(-50, 50, 1000)
    volume = np.random.uniform(1000000, 5000000, 1000)
    
    df = pd.DataFrame({
        'date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    return df


def interpret_rl_model(rl_model, df, scored_patterns):
    """
    Simple interpretability analysis for RL model
    
    Analyzes:
    1. Action distribution
    2. Feature importance (via surrogate model)
    3. Decision patterns
    """
    print("\n" + "="*80, flush=True)
    print("RL MODEL INTERPRETABILITY ANALYSIS", flush=True)
    print("="*80, flush=True)
    # sys.stdout.flush()
    
    # Create environment for analysis
    env = PatternAwareTradingEnv(df, scored_patterns, max_steps=200)
    
    # Collect episodes
    print("\n[1/3] Collecting action data...", flush=True)
    # sys.stdout.flush()
    
    action_log = []
    state_log = []
    
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            action_log.append(action)
            state_log.append(obs.copy())
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
    action_log = np.array(action_log)
    state_log = np.array(state_log)
    
    print(f"  ✓ Collected {len(action_log)} state-action pairs", flush=True)
    # sys.stdout.flush()
    
    # Analyze action distribution
    print("\n[2/3] Analyzing action distribution...", flush=True)
    # sys.stdout.flush()
    
    action_names = ['Hold', 'Long', 'Flatten']
    action_counts = np.bincount(action_log, minlength=3)
    action_pcts = action_counts / action_counts.sum() * 100
    
    print("\nAction Distribution:", flush=True)
    for name, count, pct in zip(action_names, action_counts, action_pcts):
        print(f"  {name:<10}: {count:>4} ({pct:>5.1f}%)", flush=True)
    # sys.stdout.flush()
    
    # Feature importance via decision tree
    print("\n[3/3] Analyzing feature importance...", flush=True)
    # sys.stdout.flush()
    
    try:
        from sklearn.tree import DecisionTreeClassifier
        
        # Train surrogate model
        tree = DecisionTreeClassifier(max_depth=4, random_state=42)
        tree.fit(state_log, action_log)
        
        accuracy = tree.score(state_log, action_log)
        print(f"  ✓ Surrogate model accuracy: {accuracy:.1%}", flush=True)
        
        # Feature importance
        feature_names = ['Price_Change', 'RSI', 'MACD', 'MACD_Signal', 
                        'ATR', 'Position', 'Pattern_Signal']
        importances = tree.feature_importances_
        
        print("\nFeature Importance:", flush=True)
        importance_pairs = sorted(zip(feature_names, importances), 
                                 key=lambda x: x[1], reverse=True)
        
        for name, importance in importance_pairs[:5]:  # Top 5
            print(f"  {name:<15}: {importance:.3f}", flush=True)
        # sys.stdout.flush()
        
        # Visualize action distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        colors = ['gray', 'green', 'red']
        ax1.bar(action_names, action_counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Frequency')
        ax1.set_title('RL Agent Action Distribution')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Feature importance
        top_features = [name for name, _ in importance_pairs[:5]]
        top_importances = [imp for _, imp in importance_pairs[:5]]
        ax2.barh(top_features, top_importances, alpha=0.7, color='steelblue')
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 5 Feature Importance')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('rl_interpretability.png', dpi=150, bbox_inches='tight')
        print("\n✓ Interpretability visualization saved as 'rl_interpretability.png'", flush=True)
        plt.close()
        
    except Exception as e:
        print(f"  ⚠ Could not compute feature importance: {e}", flush=True)
    
    # Key insights
    print("\n" + "="*80, flush=True)
    print("KEY INSIGHTS", flush=True)
    print("="*80, flush=True)
    
    # Check for overtrading
    if action_pcts[0] < 30:  # Less than 30% hold
        print("⚠ Agent may be overtrading (low hold %))", flush=True)
    elif action_pcts[0] > 80:  # More than 80% hold
        print("⚠ Agent may be too passive (high hold %)", flush=True)
    else:
        print("✓ Action distribution looks reasonable", flush=True)
    
    # Check feature usage
    if 'Pattern_Signal' in [name for name, _ in importance_pairs[:3]]:
        print("✓ Agent is using pattern signals (good!)", flush=True)
    else:
        print("⚠ Agent may be ignoring pattern signals", flush=True)
    
    # sys.stdout.flush()
    print("="*80, flush=True)
    # sys.stdout.flush()


def run_complete_system(df=None, use_sample_data=True, enable_rl=True, rl_timesteps=50000):
    """Run the complete trading system with all enhancements"""
    
    print("="*80, flush=True)
    print("ENHANCED MULTI-PATTERN TRADING SYSTEM WITH RL", flush=True)
    print("="*80, flush=True)
    # sys.stdout.flush()
    
    # Step 1: Load Data
    if df is None:
        if use_sample_data:
            df = generate_sample_data()
        else:
            csv_file = input("Enter path to your CSV file: ")
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n✓ Loaded {len(df)} bars of data", flush=True)
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}", flush=True)
    # sys.stdout.flush()
    
    # Step 2: Detect Patterns
    print("\n[1/6] Detecting patterns...", flush=True)
    # sys.stdout.flush()
    
    detector_dt = DoubleTopDetector(df)
    patterns_dt = detector_dt.detect_patterns()
    
    detector_db = DoubleBottomDetector(df)
    patterns_db = detector_db.detect_patterns()
    
    all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)
    print(f"  ✓ Double Top: {len(patterns_dt)}", flush=True)
    print(f"  ✓ Double Bottom: {len(patterns_db)}", flush=True)
    print(f"  ✓ Total: {len(all_patterns)}", flush=True)
    # sys.stdout.flush()
    
    if all_patterns.empty:
        print("\n⚠ No patterns detected. Try different data or adjust parameters.", flush=True)
        # sys.stdout.flush()
        return None
    
    # Step 3: Label and Train ML
    print("\n[2/6] Training ML models...", flush=True)
    # sys.stdout.flush()
    
    labeler = PatternLabeler(df, all_patterns)
    labeled_patterns = labeler.label_patterns()
    
    print(f"  Success rate by pattern:", flush=True)
    for ptype in labeled_patterns['Pattern_Type'].unique():
        success_rate = labeled_patterns[labeled_patterns['Pattern_Type']==ptype]['Success'].mean()
        print(f"    {ptype}: {success_rate:.1%}", flush=True)
    # sys.stdout.flush()
    
    scorer = PatternQualityScorer()
    scorer.train(labeled_patterns)
    
    # Step 4: Score Patterns
    print("\n[3/6] Scoring pattern quality...", flush=True)
    # sys.stdout.flush()
    
    scored_patterns = scorer.predict_quality(all_patterns)
    high_quality = scored_patterns[scored_patterns['Quality_Score'] >= 60]
    print(f"  ✓ High quality patterns (score >= 60): {len(high_quality)}", flush=True)
    # sys.stdout.flush()
    
    # Step 5: Portfolio Backtest (Rule-Based)
    print("\n[4/6] Running rule-based portfolio backtest...", flush=True)
    # sys.stdout.flush()
    
    portfolio = PortfolioRiskManager(initial_capital=100000, max_positions=5, risk_per_trade=0.02)
    
    trades_executed = 0
    for _, pattern in high_quality.iterrows():
        eval_result = portfolio.evaluate_new_trade(pattern)
        
        if eval_result['approved']:
            entry_price = pattern['Neckline']
            portfolio.open_position(pattern, eval_result, entry_price)
            trades_executed += 1
            
            # Simulate exit
            entry_idx = int(pattern['Detection_Index'])
            is_bearish = pattern['Pattern_Type'] == 'DoubleTop'
            exit_price = None
            exit_date = None
            
            for i in range(entry_idx + 1, min(entry_idx + 30, len(df))):
                high = df.loc[i, 'High']
                low = df.loc[i, 'Low']
                
                if is_bearish:
                    if high >= pattern['Stop_Loss']:
                        exit_price = pattern['Stop_Loss']
                        exit_date = df.loc[i, 'date']
                        break
                    if low <= pattern['Target']:
                        exit_price = pattern['Target']
                        exit_date = df.loc[i, 'date']
                        break
                else:
                    if low <= pattern['Stop_Loss']:
                        exit_price = pattern['Stop_Loss']
                        exit_date = df.loc[i, 'date']
                        break
                    if high >= pattern['Target']:
                        exit_price = pattern['Target']
                        exit_date = df.loc[i, 'date']
                        break
            
            if exit_price is None:
                exit_idx = min(entry_idx + 29, len(df) - 1)
                exit_price = df.loc[exit_idx, 'Close']
                exit_date = df.loc[exit_idx, 'date']
            
            portfolio.close_position(0, exit_price, exit_date)
            portfolio.update_positions(pattern['Detection_Date'], df.loc[pattern['Detection_Index'], 'Close'])
            
            if trades_executed >= 30:
                break
    
    print(f"  ✓ Executed {trades_executed} trades", flush=True)
    # sys.stdout.flush()
    
    # Get portfolio metrics
    metrics = portfolio.get_metrics()
    portfolio_results = {
        'total_trades': metrics['total_trades'],
        'win_rate': metrics['win_rate'],
        'total_return': metrics['total_return'],
        'final_capital': metrics['final_capital']
    }
    
    print("\n[RULE-BASED RESULTS]:", flush=True)
    print(f"  Total Trades   : {metrics['total_trades']}", flush=True)
    print(f"  Win Rate       : {metrics['win_rate']:.1f}%", flush=True)
    print(f"  Total Return   : {metrics['total_return']:.2f}%", flush=True)
    print(f"  Final Capital  : ${metrics['final_capital']:,.2f}", flush=True)
    # sys.stdout.flush()
    
    # Step 6: RL Training and Comparison
    rl_comparison = None
    rl_model = None
    
    if enable_rl and RL_AVAILABLE and SB3_AVAILABLE:
        print("\n[5/6] Training RL agent...", flush=True)
        # sys.stdout.flush()
        
        # Prepare data with indicators
        enriched_df = BasePatternDetector(df).df
        
        # Train RL agent
        rl_model = train_rl_agent(enriched_df, scored_patterns, algorithm='PPO', timesteps=rl_timesteps)
        
        if rl_model:
            # Compare strategies
            rl_comparison = compare_strategies(enriched_df, scored_patterns, rl_model, portfolio_results)
            
            # NEW: Add interpretability analysis
            print("\n[5.5/6] Analyzing RL model interpretability...", flush=True)
            # sys.stdout.flush()
            
            try:
                # Import interpretability module (if available)
                # For now, we'll do a simple analysis
                interpret_rl_model(rl_model, enriched_df, scored_patterns)
            except Exception as e:
                print(f"  ⚠ Interpretability analysis skipped: {e}", flush=True)
                # sys.stdout.flush()
    elif enable_rl:
        print("\n[5/6] RL training skipped: gymnasium/stable-baselines3 not installed", flush=True)
        # sys.stdout.flush()
    
    # Step 7: Create Enhanced Dashboard
    print("\n[6/6] Generating enhanced dashboard...", flush=True)
    # sys.stdout.flush()
    
    dashboard = TradingDashboard(portfolio, rl_comparison)
    dashboard.create_dashboard()
    
    print("\n" + "="*80, flush=True)
    print("SYSTEM EXECUTION COMPLETE", flush=True)
    print("="*80, flush=True)
    # sys.stdout.flush()
    
    return {
        'portfolio': portfolio,
        'scored_patterns': scored_patterns,
        'rl_model': rl_model,
        'rl_comparison': rl_comparison
    }


# ============================================================================
# MAIN ENTRY POINT - EXECUTE ON RUN
# ============================================================================

if __name__ == "__main__":
    # Display main menu and get user choice
    print("\n" + "="*80, flush=True)
    print("ENHANCED TRADING SYSTEM OPTIONS:", flush=True)
    print("="*80, flush=True)
    print("1. Quick Demo (sample data, with RL - ~5 min)", flush=True)
    print("2. Quick Demo (sample data, no RL - ~30 sec)", flush=True)
    print("3. Use my own CSV file (with RL)", flush=True)
    print("4. Use my own CSV file (no RL)", flush=True)
    print("5. Exit", flush=True)
    # sys.stdout.flush()
    
    choice = input("\nEnter choice (1-5): ").strip()
    print(f"\nYou selected: {choice}", flush=True)
    # sys.stdout.flush()
    
    if choice == '1':
        print("\n" + "="*80, flush=True)
        print("Starting Quick Demo with RL...", flush=True)
        print("This will take approximately 5 minutes", flush=True)
        print("="*80, flush=True)
        # sys.stdout.flush()
        result = run_complete_system(use_sample_data=True, enable_rl=True, rl_timesteps=10000)
        
    elif choice == '2':
        print("\n" + "="*80, flush=True)
        print("Starting Quick Demo without RL...", flush=True)
        print("="*80, flush=True)
        # sys.stdout.flush()
        result = run_complete_system(use_sample_data=True, enable_rl=False)
        
    elif choice == '3':
        csv_file = input("\nEnter CSV file path: ").strip()
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} rows from {csv_file}", flush=True)
            # sys.stdout.flush()
            
            timesteps = input("\nRL training timesteps (default 50000): ").strip()
            timesteps = int(timesteps) if timesteps else 50000
            
            print("\n" + "="*80, flush=True)
            print(f"Running with your data and RL training ({timesteps} steps)...", flush=True)
            print("="*80, flush=True)
            # sys.stdout.flush()
            result = run_complete_system(df=df, use_sample_data=False, enable_rl=True, rl_timesteps=timesteps)
                    
        except Exception as e:
            print(f"\n✗ Error: {e}", flush=True)
            print("Make sure CSV has columns: date, Open, High, Low, Close, Volume", flush=True)
            # sys.stdout.flush()
    
    elif choice == '4':
        csv_file = input("\nEnter CSV file path: ").strip()
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} rows from {csv_file}", flush=True)
            # sys.stdout.flush()
            
            print("\n" + "="*80, flush=True)
            print("Running with your data (no RL)...", flush=True)
            print("="*80, flush=True)
            # sys.stdout.flush()
            result = run_complete_system(df=df, use_sample_data=False, enable_rl=False)
                    
        except Exception as e:
            print(f"\n✗ Error: {e}", flush=True)
            print("Make sure CSV has columns: date, Open, High, Low, Close, Volume", flush=True)
            # sys.stdout.flush()
    
    elif choice == '5':
        print("\nExiting... Goodbye!", flush=True)
        # sys.stdout.flush()
    
    else:
        print("\n✗ Invalid choice. Please run again and select 1-5.", flush=True)
        # sys.stdout.flush()
        
    print("\n" + "="*80, flush=True)
    print("Program finished.", flush=True)
    print("="*80, flush=True)
    # sys.stdout.flush()