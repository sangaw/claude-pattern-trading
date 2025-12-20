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

from .environment import PatternAwareTradingEnv


def train_rl_agent(df, scored_patterns, algorithm='PPO', timesteps=50000):
    """Train RL agent with pattern signals"""
    if not RL_AVAILABLE or not SB3_AVAILABLE:
        print("RL training skipped: gymnasium/stable-baselines3 not installed")
        return None
    
    print(f"\n{'='*80}")
    print(f"TRAINING RL AGENT ({algorithm})")
    print(f"{'='*80}")
    
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].reset_index(drop=True)
    test_df = df[train_size:].reset_index(drop=True)
    
    train_env = PatternAwareTradingEnv(train_df, scored_patterns)
    
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
    
    model.learn(total_timesteps=timesteps)
    print(f"✓ Training completed: {timesteps} timesteps")
    
    return model


def compare_strategies(df, scored_patterns, rl_model, portfolio_results):
    """Compare RL agent vs rule-based portfolio strategy"""
    if not RL_AVAILABLE or rl_model is None:
        return None
    
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON: RL vs RULE-BASED")
    print(f"{'='*80}")
    
    test_env = PatternAwareTradingEnv(df, scored_patterns, max_steps=len(df)-100)
    obs, _ = test_env.reset()
    
    rl_rewards = []
    done = False
    
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        rl_rewards.append(reward)
        done = terminated or truncated
    
    rl_stats = test_env.get_trade_stats()
    
    if rl_stats.get('total_trades', 0) > 0:
        rl_total_return = rl_stats['total_return'] * 100
    else:
        rl_total_return = 0
    
    rl_sharpe = np.mean(rl_rewards) / (np.std(rl_rewards) + 1e-8) * np.sqrt(252)
    
    rule_return = portfolio_results.get('total_return', 0)
    rule_win_rate = portfolio_results.get('win_rate', 0)
    rule_trades = portfolio_results.get('total_trades', 0)
    
    print(f"\n{'Strategy':<20} {'Return %':<15} {'Win Rate %':<15} {'Trades':<10} {'Sharpe':<10}")
    print(f"{'-'*70}")
    print(f"{'RL Agent':<20} {rl_total_return:<15.2f} {rl_stats.get('win_rate', 0)*100:<15.1f} {rl_stats.get('total_trades', 0):<10} {rl_sharpe:<10.2f}")
    print(f"{'Rule-Based':<20} {rule_return:<15.2f} {rule_win_rate:<15.1f} {rule_trades:<10} {'N/A':<10}")
    print(f"{'-'*70}")
    
    winner = "RL Agent" if rl_total_return > rule_return else "Rule-Based"
    improvement = abs(rl_total_return - rule_return)
    
    print(f"\n✓ Winner: {winner} (by {improvement:.2f}%)")
    
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