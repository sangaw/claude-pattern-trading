import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Optional RL imports
RL_AVAILABLE = False
SB3_AVAILABLE = False

from .interpretability import RLInterpretabilityReport
from .rl_visualization import RLVisualizer
from .trigger_dashboard_callback import TradingDashboardCallback


try:
    import gymnasium as gym
    from gymnasium import spaces
    RL_AVAILABLE = True
except ImportError:
    pass  # Silent fail, we'll check later

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    pass  # Silent fail, we'll check later

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

# Only import environment if RL is available
if RL_AVAILABLE and SB3_AVAILABLE:
    from .environment import PatternAwareTradingEnv

def train_rl_agent(df, scored_patterns, portfolio_manager, config):
    """Train RL agent with pattern signals"""

    # Check if RL libraries are available
    if not RL_AVAILABLE:
        print(" RL training skipped: gymnasium not installed")
        print("  Install with: pip install gymnasium")
        return None
    
    if not SB3_AVAILABLE:
        print(" RL training skipped: stable-baselines3 not installed")
        print("  Install with: pip install stable-baselines3")
        return None
    

    algorithm = config.RL_ALGORITHM
    timesteps= config.RL_TIMESTEPS 


    print(f"\n{'='*80}")
    print(f"TRAINING RL AGENT ({algorithm})")
    print(f"{'='*80}")
    
    # Rename columns to match environment expectations
    df = df.copy()
    column_mapping = {
        'rsi_14': 'RSI',
        'macd_12_26': 'MACD',
        'macd_signal_12_26': 'MACD_Signal',
        'volatility_20': 'Volatility'
    }
    
    # Apply rename
    df = df.rename(columns=column_mapping)
    
    # Add ATR if missing (use volatility as proxy)
    if 'ATR' not in df.columns:
        df['ATR'] = df['Volatility'] * df['Close']
    
    # Validate required columns exist
    required_cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Volatility']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n ERROR: DataFrame still missing indicators: {missing}")
        print("Available columns:", list(df.columns))
        return None
    
    print(f"✓ DataFrame prepared with {len(df)} bars")
    
    # 1. Define your log directory (Windows safe)
    import os
    from stable_baselines3.common.monitor import Monitor

    log_dir = "./monitoring/logs/tboard_rita_rl_logs/"
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)

    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].reset_index(drop=True)
    test_df = df[train_size:].reset_index(drop=True)

    # Double-DQN specific policy kwargs
    # ddqn_policy_kwargs = dict(net_arch=[256, 256],)

        
    
    print(f"\nTraining {algorithm} on {len(train_df)} bars...")
    
    try:
        # Initialize once
        visualizer = RLVisualizer(
            strategy_name="RITA_RL_Model",
            use_tensorboard=True,
            use_live_plot=True
        )
        # 2. Wrap your custom environment with Monitor to capture reward data
        train_env = PatternAwareTradingEnv(
            df=df,
            portfolio=portfolio_manager,  # <--- INJECTION POINT
            scored_patterns=scored_patterns,
            config=config
        )
        train_env = Monitor(train_env, log_dir)

    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return None
    
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=0,
            tensorboard_log=log_dir,
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
            tensorboard_log=log_dir,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99
        )
    elif algorithm == 'DQN' or algorithm == 'DDQN':
        # Changed 'ddqn_model' to 'model' so the rest of your script works
        model = DQN(
            "MlpPolicy",
            train_env, 
            learning_rate=2.5e-4,
            buffer_size=200000,
            learning_starts=100,
            batch_size=128,
            gamma=0.99,
            tau=1.0,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            verbose=0, # Set to 1 to see progress in console
            device='cpu',
            tensorboard_log=log_dir 
        )
    else:
        print(f"✗ Unknown algorithm: {algorithm}")
        visualizer.close()
        return None
    
    try:
        
        # 3. Give each run a unique name inside the logs
        run_name = f"{algorithm}_trading_run"

        print(f"🚀 Starting training: {algorithm} for {timesteps} timesteps")
        print(f"📊 TensorBoard: tensorboard --logdir={log_dir}")

        # Pass the visualizer to the callback
        dashboard = TradingDashboardCallback(visualizer=visualizer)

        model.learn(
            total_timesteps=timesteps, 
            progress_bar=False, 
            callback=dashboard,
            tb_log_name=run_name)

        print(f"✓ Training completed: {timesteps} timesteps")

        # Create the directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)

        # NOW save the model
        model.save("reports/RITA_swing_model")
        print("✅ Model saved successfully to reports/RITA_swing_model.zip")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 6. Always close visualizer to generate reports
        visualizer.close()
        print("📊 Visualizer closed and report generated.")
    return model


import numpy as np
import pandas as pd

def compare_strategies(df, scored_patterns, rl_model, portfolio_manager, portfolio_results, config):
    """
    Compare RL agent vs rule-based portfolio strategy.
    
    Args:
        df: Historical market data
        scored_patterns: Detected patterns with ML scores
        rl_model: The trained SB3 model
        portfolio_manager: The Unified Portfolio Manager object (for testing)
        portfolio_results: The dictionary of metrics from the earlier Backtest
        config: The system configuration object
    """
    # 1. Validation Checks
    if rl_model is None:
        print("⚠ Strategy comparison skipped: No trained RL model")
        return None
    
    # --- ADD THIS COLUMN MAPPING BLOCK ---
    df = df.copy()
    column_mapping = {
        'rsi_14': 'RSI',
        'macd_12_26': 'MACD',
        'macd_signal_12_26': 'MACD_Signal',
        'volatility_20': 'Volatility'
    }
    
    # Rename only columns that exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Add ATR if missing (Environment usually expects it for position sizing)
    if 'ATR' not in df.columns:
        # Use a simple range-based ATR proxy if the real one isn't there
        df['ATR'] = df['Volatility'] * df['Close'] 

    # --- END MAPPING BLOCK ---


    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON: RL vs RULE-BASED")
    print(f"{'='*80}")
    
    # 2. Environment Preparation
    # We use a copy of the manager to avoid corrupting the original backtest logs
    from modeling.rl.environment import PatternAwareTradingEnv
    
    try:
        # Create a fresh environment for the "final exam" of the RL agent
        test_env = PatternAwareTradingEnv(
            df=df, 
            scored_patterns=scored_patterns, 
            portfolio=portfolio_manager, # Pass the ACTUAL object
            config=config
        )
    except Exception as e:
        print(f"✗ Failed to create test environment: {e}")
        return None
    
    # 3. Running the Evaluation Loop
    obs, _ = test_env.reset()
    rl_rewards = []
    done = False
    
    try:
        while not done:
            # rl_model.predict returns (action, state)
            # action is often a numpy array [1]
            action, _ = rl_model.predict(obs, deterministic=True)
            
            # Convert to standard Python integer to prevent indexing errors
            action_int = int(action.item()) if hasattr(action, 'item') else int(action)
            
            # Step the environment
            obs, reward, terminated, truncated, _ = test_env.step(action_int)

            rl_rewards.append(reward)
            done = terminated or truncated

        # Force close any open position at the end of history
        if test_env.position == 1:
            test_env.step(2) # Action 2 = Flatten
                        
    except Exception as e:
        print(f"✗ RL evaluation failed: {e}")
        return None
    
    # 4. Gather Comparative Metrics
    # Get metrics specifically for the RL engine from our unified manager
    rl_metrics = portfolio_manager.get_metrics(engine_name="RL-Agent", silent=True)
    
    rl_total_return = rl_metrics.get('total_return', 0)
    rl_win_rate = rl_metrics.get('win_rate', 0)
    rl_trades = rl_metrics.get('total_trades', 0)
    rl_sharpe = np.mean(rl_rewards) / (np.std(rl_rewards) + 1e-8) * np.sqrt(252)
    
    # These come from the portfolio_results dict passed from Step 6
    rule_return = portfolio_results.get('total_return', 0)
    rule_win_rate = portfolio_results.get('win_rate', 0)
    rule_trades = portfolio_results.get('total_trades', 0)
    
    # 5. Print Comparison Table
    print(f"\n{'Strategy':<20} {'Return %':<15} {'Win Rate %':<15} {'Trades':<10} {'Sharpe':<10}")
    print(f"{'-'*70}")
    print(f"{'RL Agent':<20} {rl_total_return:<15.2f} {rl_win_rate:<15.1f} {rl_trades:<10} {rl_sharpe:<10.2f}")
    print(f"{'Rule-Based':<20} {rule_return:<15.2f} {rule_win_rate:<15.1f} {rule_trades:<10} {'N/A':<10}")
    print(f"{'-'*70}")
    
    winner = "RL Agent" if rl_total_return > rule_return else "Rule-Based"
    improvement = abs(rl_total_return - rule_return)
    print(f"\n✓ Winner: {winner} (by {improvement:.2f}%)")
    
    # 6. Return Data for Dashboard
    comparison = {
        'rl_return': rl_total_return,
        'rl_win_rate': rl_win_rate,
        'rl_trades': rl_trades,
        'rl_sharpe': rl_sharpe,
        'rule_return': rule_return,
        'rule_win_rate': rule_win_rate,
        'rule_trades': rule_trades,
        'winner': winner,
        'improvement': improvement
    }
    
    # 7. Enhanced Interpretability (Optional SHAP analysis)
    print("\n[Analysis] Generating interpretability report...")
    try:
        from modeling.rl.interpretability import RLInterpretabilityReport
        feature_names = ['RSI', 'MACD_Hist', 'ATR', 'Vol_Ratio', 'Price_Chg', 'Pattern_Score', 'Port_Ret']
        
        # Simple wrapper to handle action types for the interpreter
        class HashableModelWrapper:
            def __init__(self, model):
                self.model = model
            def predict(self, obs, deterministic=True):
                a, s = self.model.predict(obs, deterministic=deterministic)
                return (int(a.item()) if hasattr(a, 'item') else int(a)), s

        interpreter = RLInterpretabilityReport(HashableModelWrapper(rl_model), test_env, feature_names)
        interpreter.generate_full_report()
    except Exception as e:
        print(f"  Interpretability skipped or failed: {e}")
    
    return comparison


# Status check for when module loads
if __name__ != "__main__":
    if RL_AVAILABLE and SB3_AVAILABLE:
        print("✓ RL training available")
    elif not RL_AVAILABLE:
        print("⚠ RL libraries not available (gymnasium missing)")
    elif not SB3_AVAILABLE:
        print("⚠ RL libraries not available (stable-baselines3 missing)")