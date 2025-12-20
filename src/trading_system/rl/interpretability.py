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

class RLInterpretabilityReport:
    """Comprehensive interpretability analysis for RL trading agents"""
    
    def __init__(self, rl_model, env, feature_names):
        self.rl_model = rl_model
        self.env = env
        self.feature_names = feature_names
        self.action_names = ['Hold', 'Long', 'Flatten']
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_returns = []
    
    def collect_episode_data(self, num_episodes=20, deterministic=True):
        """Collect state-action-reward data"""
        print(f"\n[1/6] Collecting data from {num_episodes} episodes...", flush=True)
        sys.stdout.flush()
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=deterministic)
                action_scalar = action.item()

                self.states.append(obs.copy())
                self.actions.append(action_scalar)
                
                obs, reward, terminated, truncated, _ = self.env.step(action_scalar)
                self.rewards.append(reward)
                episode_reward += reward
                done = terminated or truncated
            
            self.episode_returns.append(episode_reward)
        
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        
        print(f"  ✓ Collected {len(self.states)} state-action pairs", flush=True)
        sys.stdout.flush()
    
    def analyze_action_distribution(self):
        """Analyze action frequency"""
        print("\n[2/6] Analyzing action distribution...", flush=True)
        sys.stdout.flush()
        
        action_counts = np.bincount(self.actions, minlength=3)
        action_pcts = action_counts / action_counts.sum() * 100
        
        print("\n  Action Distribution:", flush=True)
        for name, count, pct in zip(self.action_names, action_counts, action_pcts):
            bar = '█' * int(pct / 2)
            print(f"    {name:<10}: {count:>5} ({pct:>5.1f}%) {bar}", flush=True)
        sys.stdout.flush()
        
        return {'counts': action_counts, 'percentages': action_pcts}
    
    def analyze_feature_importance(self):
        """Analyze feature importance using surrogate models"""
        print("\n[3/6] Analyzing feature importance...", flush=True)
        sys.stdout.flush()
        
        try:
            # Decision Tree
            tree = DecisionTreeClassifier(max_depth=5, random_state=42)
            tree.fit(self.states, self.actions)
            tree_acc = tree.score(self.states, self.actions)
            tree_imp = tree.feature_importances_
            
            print(f"\n  Decision Tree Surrogate Accuracy: {tree_acc:.1%}", flush=True)
            print("  Top 5 Features:", flush=True)
            
            for name, imp in sorted(zip(self.feature_names, tree_imp), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                bar = '█' * int(imp * 40)
                print(f"    {name:<20}: {imp:.3f} {bar}", flush=True)
            sys.stdout.flush()
            
            # Random Forest
            forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            forest.fit(self.states, self.actions)
            forest_imp = forest.feature_importances_
            
            return {'tree': tree_imp, 'forest': forest_imp}
            
        except Exception as e:
            print(f"  ⚠ Could not compute feature importance: {e}", flush=True)
            sys.stdout.flush()
            return None
    
    def analyze_state_action_relationships(self):
        """Analyze state-action patterns"""
        print("\n[4/6] Analyzing state-action relationships...", flush=True)
        sys.stdout.flush()
        
        df = pd.DataFrame(self.states, columns=self.feature_names)
        df['Action'] = [self.action_names[a] for a in self.actions]
        df['Reward'] = self.rewards
        
        print("\n  Average State Values by Action:", flush=True)
        summary_cols = [c for c in ['RSI', 'Pattern_Signal', 'MACD'] if c in df.columns]
        if summary_cols:
            print(df.groupby('Action')[summary_cols].mean().round(3), flush=True)
        sys.stdout.flush()
        
        return df
    
    def visualize_interpretability(self):
        """Create visualization dashboard"""
        print("\n[5/6] Generating visualizations...", flush=True)
        sys.stdout.flush()
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('RL AGENT INTERPRETABILITY DASHBOARD', fontsize=14, fontweight='bold')
            
            # 1. Action Distribution
            ax = axes[0, 0]
            action_counts = np.bincount(self.actions, minlength=3)
            colors = ['gray', 'green', 'red']
            ax.bar(self.action_names, action_counts, color=colors, alpha=0.7)
            ax.set_ylabel('Frequency')
            ax.set_title('Action Distribution')
            ax.grid(True, alpha=0.3)
            
            # 2. Feature Importance
            ax = axes[0, 1]
            tree = DecisionTreeClassifier(max_depth=5, random_state=42)
            tree.fit(self.states, self.actions)
            importance = tree.feature_importances_
            sorted_idx = np.argsort(importance)[::-1][:6]
            ax.barh([self.feature_names[i] for i in sorted_idx], 
                   importance[sorted_idx], alpha=0.7, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3)
            
            # 3. Reward by Action
            ax = axes[0, 2]
            for i, name in enumerate(self.action_names):
                mask = self.actions == i
                if mask.sum() > 0:
                    ax.hist(self.rewards[mask], bins=20, alpha=0.6, label=name)
            ax.set_xlabel('Reward')
            ax.set_title('Reward Distribution by Action')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. RSI Distribution
            ax = axes[1, 0]
            if 'RSI' in self.feature_names:
                rsi_idx = self.feature_names.index('RSI')
                for i, name in enumerate(self.action_names):
                    mask = self.actions == i
                    if mask.sum() > 0:
                        ax.hist(self.states[mask, rsi_idx], bins=20, alpha=0.6, label=name)
                ax.set_xlabel('RSI')
                ax.set_title('RSI Distribution by Action')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 5. Pattern Signal
            ax = axes[1, 1]
            if 'Pattern_Signal' in self.feature_names:
                pattern_idx = self.feature_names.index('Pattern_Signal')
                for i, name in enumerate(self.action_names):
                    mask = self.actions == i
                    if mask.sum() > 0:
                        ax.scatter(self.states[mask, pattern_idx], 
                                 self.rewards[mask], alpha=0.3, label=name, s=10)
                ax.set_xlabel('Pattern Signal')
                ax.set_ylabel('Reward')
                ax.set_title('Pattern Signal vs Reward')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 6. Episode Returns
            ax = axes[1, 2]
            ax.hist(self.episode_returns, bins=15, alpha=0.7, color='purple')
            ax.axvline(np.mean(self.episode_returns), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(self.episode_returns):.3f}')
            ax.set_xlabel('Episode Return')
            ax.set_title('Episode Returns Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('rl_interpretability_enhanced.png', dpi=150, bbox_inches='tight')
            print("  ✓ Saved as 'rl_interpretability_enhanced.png'", flush=True)
            plt.close()
            
        except Exception as e:
            print(f"  ⚠ Visualization failed: {e}", flush=True)
        
        sys.stdout.flush()
    
    def generate_insights(self, action_stats, feature_importance):
        """Generate key insights"""
        print("\n[6/6] Generating insights...", flush=True)
        sys.stdout.flush()
        
        print("\n" + "="*80, flush=True)
        print("KEY INSIGHTS", flush=True)
        print("="*80, flush=True)
        
        action_pcts = action_stats['percentages']
        
        # Trading behavior
        print("\n[Trading Behavior]", flush=True)
        if action_pcts[0] > 70:
            print("  ⚠ Agent is very passive (>70% hold)", flush=True)
        elif action_pcts[0] < 30:
            print("  ⚠ Agent may be overtrading (<30% hold)", flush=True)
        else:
            print("  ✓ Trading frequency appears reasonable", flush=True)
        
        # Feature usage
        if feature_importance and 'tree' in feature_importance:
            print("\n[Feature Usage]", flush=True)
            top_3 = sorted(zip(self.feature_names, feature_importance['tree']), 
                          key=lambda x: x[1], reverse=True)[:3]
            print("  Top 3 Features:", flush=True)
            for name, imp in top_3:
                print(f"    {name}: {imp:.3f}", flush=True)
            
            if 'Pattern_Signal' in [name for name, _ in top_3]:
                print("  ✓ Agent is using pattern signals", flush=True)
            else:
                print("  ⚠ Agent may be ignoring pattern signals", flush=True)
        
        # Performance
        print("\n[Performance]", flush=True)
        mean_return = np.mean(self.episode_returns)
        if mean_return > 0:
            print(f"  ✓ Positive average return: {mean_return:.4f}", flush=True)
        else:
            print(f"  ⚠ Negative average return: {mean_return:.4f}", flush=True)
        
        sys.stdout.flush()
        print("="*80, flush=True)
        sys.stdout.flush()
    
    def generate_full_report(self):
        """Run complete analysis"""
        self.collect_episode_data(num_episodes=20)
        action_stats = self.analyze_action_distribution()
        feature_importance = self.analyze_feature_importance()
        state_action_df = self.analyze_state_action_relationships()
        self.visualize_interpretability()
        self.generate_insights(action_stats, feature_importance)
        
        return {
            'action_stats': action_stats,
            'feature_importance': feature_importance,
            'state_action_df': state_action_df
        }