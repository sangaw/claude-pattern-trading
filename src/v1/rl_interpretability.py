"""
ENHANCED RL MODEL INTERPRETABILITY MODULE
Comprehensive analysis tools for understanding RL trading agent behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class RLInterpretabilityReport:
    """
    Comprehensive interpretability analysis for RL trading agents
    
    Features:
    - Action distribution analysis
    - Feature importance (multiple methods)
    - Decision boundary visualization
    - State-action relationship analysis
    - Episode trajectory analysis
    - Performance attribution
    """
    
    def __init__(self, rl_model, env, feature_names):
        """
        Parameters:
        -----------
        rl_model : trained RL model (PPO, A2C, etc.)
        env : gym environment used for training
        feature_names : list of feature names matching observation space
        """
        self.rl_model = rl_model
        self.env = env
        self.feature_names = feature_names
        self.action_names = ['Hold', 'Long', 'Flatten']
        
        # Storage for collected data
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_returns = []
        
    def collect_episode_data(self, num_episodes=20, deterministic=True):
        """Collect state-action-reward data from multiple episodes"""
        print(f"\nCollecting data from {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=deterministic)
                
                self.states.append(obs.copy())
                self.actions.append(action)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                self.rewards.append(reward)
                episode_reward += reward
                
                done = terminated or truncated
            
            self.episode_returns.append(episode_reward)
        
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        
        print(f"✓ Collected {len(self.states)} state-action pairs")
        print(f"  Mean episode return: {np.mean(self.episode_returns):.4f}")
        print(f"  Std episode return: {np.std(self.episode_returns):.4f}")
        
    def analyze_action_distribution(self):
        """Analyze how frequently each action is taken"""
        print("\n" + "="*80)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("="*80)
        
        action_counts = np.bincount(self.actions, minlength=3)
        action_pcts = action_counts / action_counts.sum() * 100
        
        print("\nOverall Action Distribution:")
        for name, count, pct in zip(self.action_names, action_counts, action_pcts):
            bar = '█' * int(pct / 2)
            print(f"  {name:<10}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        # Analyze action patterns
        print("\nAction Pattern Insights:")
        
        # Check for overtrading
        hold_pct = action_pcts[0]
        if hold_pct < 30:
            print("  ⚠ Low hold percentage - agent may be overtrading")
        elif hold_pct > 80:
            print("  ⚠ High hold percentage - agent may be too passive")
        else:
            print("  ✓ Action distribution appears balanced")
        
        # Action transitions
        transitions = self._calculate_action_transitions()
        print("\nMost Common Action Transitions:")
        for (from_act, to_act), freq in transitions[:5]:
            print(f"  {self.action_names[from_act]} → {self.action_names[to_act]}: {freq:.1%}")
        
        return {
            'counts': action_counts,
            'percentages': action_pcts,
            'transitions': transitions
        }
    
    def _calculate_action_transitions(self):
        """Calculate action transition probabilities"""
        transitions = {}
        for i in range(len(self.actions) - 1):
            key = (self.actions[i], self.actions[i+1])
            transitions[key] = transitions.get(key, 0) + 1
        
        total = sum(transitions.values())
        transitions = {k: v/total for k, v in transitions.items()}
        return sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_feature_importance(self, method='all'):
        """
        Analyze feature importance using multiple methods
        
        Parameters:
        -----------
        method : str, 'tree', 'forest', or 'all'
        """
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        results = {}
        
        # Method 1: Decision Tree Surrogate
        if method in ['tree', 'all']:
            print("\n[1] Decision Tree Surrogate Model:")
            tree = DecisionTreeClassifier(max_depth=5, random_state=42)
            tree.fit(self.states, self.actions)
            
            accuracy = tree.score(self.states, self.actions)
            print(f"  Surrogate model accuracy: {accuracy:.1%}")
            
            tree_importance = tree.feature_importances_
            results['tree'] = tree_importance
            
            print("\n  Top Features (Decision Tree):")
            for name, imp in sorted(zip(self.feature_names, tree_importance), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                bar = '█' * int(imp * 50)
                print(f"    {name:<20}: {imp:.3f} {bar}")
        
        # Method 2: Random Forest Surrogate
        if method in ['forest', 'all']:
            print("\n[2] Random Forest Surrogate Model:")
            forest = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            forest.fit(self.states, self.actions)
            
            accuracy = forest.score(self.states, self.actions)
            print(f"  Surrogate model accuracy: {accuracy:.1%}")
            
            forest_importance = forest.feature_importances_
            results['forest'] = forest_importance
            
            print("\n  Top Features (Random Forest):")
            for name, imp in sorted(zip(self.feature_names, forest_importance), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                bar = '█' * int(imp * 50)
                print(f"    {name:<20}: {imp:.3f} {bar}")
        
        # Method 3: Permutation Importance
        if method == 'all':
            print("\n[3] Permutation Importance (approximate):")
            perm_importance = self._calculate_permutation_importance()
            results['permutation'] = perm_importance
            
            print("\n  Top Features (Permutation):")
            for name, imp in sorted(zip(self.feature_names, perm_importance), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                bar = '█' * int(imp * 10)
                print(f"    {name:<20}: {imp:.3f} {bar}")
        
        return results
    
    def _calculate_permutation_importance(self, n_repeats=5):
        """Calculate permutation importance"""
        baseline_accuracy = self._get_policy_consistency()
        importances = []
        
        for feat_idx in range(self.states.shape[1]):
            scores = []
            for _ in range(n_repeats):
                # Permute feature
                states_perm = self.states.copy()
                np.random.shuffle(states_perm[:, feat_idx])
                
                # Get predictions on permuted data
                actions_perm = []
                for state in states_perm:
                    action, _ = self.rl_model.predict(state, deterministic=True)
                    actions_perm.append(action)
                
                # Calculate accuracy drop
                accuracy = (np.array(actions_perm) == self.actions).mean()
                scores.append(baseline_accuracy - accuracy)
            
            importances.append(np.mean(scores))
        
        return np.array(importances)
    
    def _get_policy_consistency(self):
        """Get baseline policy consistency"""
        actions_pred = []
        for state in self.states:
            action, _ = self.rl_model.predict(state, deterministic=True)
            actions_pred.append(action)
        return (np.array(actions_pred) == self.actions).mean()
    
    def analyze_state_action_relationships(self):
        """Analyze relationships between states and actions"""
        print("\n" + "="*80)
        print("STATE-ACTION RELATIONSHIP ANALYSIS")
        print("="*80)
        
        df = pd.DataFrame(self.states, columns=self.feature_names)
        df['Action'] = [self.action_names[a] for a in self.actions]
        df['Reward'] = self.rewards
        
        print("\nAverage State Values by Action:")
        print(df.groupby('Action')[self.feature_names].mean().round(3))
        
        print("\nAverage Reward by Action:")
        print(df.groupby('Action')['Reward'].mean())
        
        # Find critical thresholds
        print("\nCritical Decision Thresholds:")
        self._find_decision_thresholds(df)
        
        return df
    
    def _find_decision_thresholds(self, df):
        """Find important thresholds in features that trigger actions"""
        for feat in ['RSI', 'Pattern_Signal', 'MACD']:
            if feat in df.columns:
                long_mask = df['Action'] == 'Long'
                if long_mask.sum() > 0:
                    median_val = df[long_mask][feat].median()
                    print(f"  {feat:<20}: Median when going Long = {median_val:.3f}")
    
    def analyze_episode_trajectories(self, num_episodes=3):
        """Analyze full episode trajectories"""
        print("\n" + "="*80)
        print(f"EPISODE TRAJECTORY ANALYSIS (showing {num_episodes} episodes)")
        print("="*80)
        
        trajectories = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'cumulative_reward': 0
            }
            done = False
            
            while not done:
                action, _ = self.rl_model.predict(obs, deterministic=True)
                
                trajectory['states'].append(obs.copy())
                trajectory['actions'].append(action)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                trajectory['rewards'].append(reward)
                trajectory['cumulative_reward'] += reward
                
                done = terminated or truncated
            
            trajectories.append(trajectory)
            
            print(f"\nEpisode {ep+1}:")
            print(f"  Total Steps: {len(trajectory['actions'])}")
            print(f"  Cumulative Reward: {trajectory['cumulative_reward']:.4f}")
            print(f"  Action Counts: ", end="")
            action_counts = np.bincount(trajectory['actions'], minlength=3)
            for name, count in zip(self.action_names, action_counts):
                print(f"{name}={count} ", end="")
            print()
        
        return trajectories
    
    def visualize_interpretability(self, save_path='rl_interpretability_full.png'):
        """Create comprehensive visualization dashboard"""
        print("\nGenerating interpretability visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Action Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        action_counts = np.bincount(self.actions, minlength=3)
        colors = ['gray', 'green', 'red']
        ax1.bar(self.action_names, action_counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Frequency')
        ax1.set_title('Action Distribution')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Feature Importance (Tree)
        ax2 = fig.add_subplot(gs[0, 1])
        tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree.fit(self.states, self.actions)
        importance = tree.feature_importances_
        sorted_idx = np.argsort(importance)[::-1][:7]
        ax2.barh([self.feature_names[i] for i in sorted_idx], 
                importance[sorted_idx], alpha=0.7, color='steelblue')
        ax2.set_xlabel('Importance')
        ax2.set_title('Feature Importance (Decision Tree)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Reward Distribution by Action
        ax3 = fig.add_subplot(gs[0, 2])
        for i, name in enumerate(self.action_names):
            mask = self.actions == i
            if mask.sum() > 0:
                ax3.hist(self.rewards[mask], bins=30, alpha=0.6, label=name)
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution by Action')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. State Feature Distributions by Action
        ax4 = fig.add_subplot(gs[1, 0])
        feature_idx = self.feature_names.index('RSI') if 'RSI' in self.feature_names else 0
        for i, name in enumerate(self.action_names):
            mask = self.actions == i
            if mask.sum() > 0:
                ax4.hist(self.states[mask, feature_idx], bins=20, alpha=0.6, label=name)
        ax4.set_xlabel(self.feature_names[feature_idx])
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'{self.feature_names[feature_idx]} Distribution by Action')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Pattern Signal vs Action
        ax5 = fig.add_subplot(gs[1, 1])
        if 'Pattern_Signal' in self.feature_names:
            pattern_idx = self.feature_names.index('Pattern_Signal')
            for i, name in enumerate(self.action_names):
                mask = self.actions == i
                if mask.sum() > 0:
                    ax5.scatter(self.states[mask, pattern_idx], 
                              self.rewards[mask], alpha=0.3, label=name, s=10)
            ax5.set_xlabel('Pattern Signal')
            ax5.set_ylabel('Reward')
            ax5.set_title('Pattern Signal vs Reward (by Action)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Decision Tree Visualization
        ax6 = fig.add_subplot(gs[1, 2])
        simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        simple_tree.fit(self.states, self.actions)
        plot_tree(simple_tree, ax=ax6, feature_names=self.feature_names, 
                 class_names=self.action_names, filled=True, fontsize=6)
        ax6.set_title('Decision Tree (depth=3)')
        
        # 7. Action Transition Matrix
        ax7 = fig.add_subplot(gs[2, 0])
        transition_matrix = np.zeros((3, 3))
        for i in range(len(self.actions) - 1):
            transition_matrix[self.actions[i], self.actions[i+1]] += 1
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.action_names, yticklabels=self.action_names, ax=ax7)
        ax7.set_title('Action Transition Probabilities')
        ax7.set_ylabel('From Action')
        ax7.set_xlabel('To Action')
        
        # 8. Episode Returns Distribution
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(self.episode_returns, bins=20, alpha=0.7, color='purple')
        ax8.axvline(np.mean(self.episode_returns), color='red', 
                   linestyle='--', label=f'Mean: {np.mean(self.episode_returns):.3f}')
        ax8.set_xlabel('Episode Return')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Episode Returns Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Key Metrics Summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        action_counts = np.bincount(self.actions, minlength=3)
        action_pcts = action_counts / action_counts.sum() * 100
        
        metrics_text = [
            ['Metric', 'Value'],
            ['─'*20, '─'*15],
            ['Total Samples', f'{len(self.actions)}'],
            ['Episodes', f'{len(self.episode_returns)}'],
            ['Mean Episode Return', f'{np.mean(self.episode_returns):.4f}'],
            ['Std Episode Return', f'{np.std(self.episode_returns):.4f}'],
            ['Hold %', f'{action_pcts[0]:.1f}%'],
            ['Long %', f'{action_pcts[1]:.1f}%'],
            ['Flatten %', f'{action_pcts[2]:.1f}%'],
            ['Policy Consistency', f'{self._get_policy_consistency():.1%}'],
        ]
        
        table = ax9.table(cellText=metrics_text, cellLoc='left', 
                         loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax9.set_title('Summary Metrics', fontweight='bold', pad=20)
        
        plt.suptitle('RL AGENT INTERPRETABILITY DASHBOARD', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Full interpretability dashboard saved as '{save_path}'")
        plt.close()
    
    def generate_full_report(self):
        """Generate complete interpretability report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RL INTERPRETABILITY ANALYSIS")
        print("="*80)
        
        # Step 1: Collect data
        self.collect_episode_data(num_episodes=20)
        
        # Step 2: Action analysis
        action_stats = self.analyze_action_distribution()
        
        # Step 3: Feature importance
        feature_importance = self.analyze_feature_importance(method='all')
        
        # Step 4: State-action relationships
        state_action_df = self.analyze_state_action_relationships()
        
        # Step 5: Episode trajectories
        trajectories = self.analyze_episode_trajectories(num_episodes=3)
        
        # Step 6: Visualizations
        self.visualize_interpretability()
        
        # Step 7: Key insights
        print("\n" + "="*80)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        
        action_pcts = action_stats['percentages']
        
        # Trading behavior
        print("\n[Trading Behavior]")
        if action_pcts[0] > 70:
            print("  ⚠ Agent is very passive (>70% hold)")
            print("    → Consider reducing transaction costs or increasing reward signals")
        elif action_pcts[0] < 30:
            print("  ⚠ Agent may be overtrading (<30% hold)")
            print("    → Consider increasing transaction costs or position holding incentives")
        else:
            print("  ✓ Trading frequency appears reasonable")
        
        # Feature usage
        print("\n[Feature Usage]")
        if 'tree' in feature_importance:
            top_features = sorted(zip(self.feature_names, feature_importance['tree']), 
                                key=lambda x: x[1], reverse=True)[:3]
            print("  Top 3 most important features:")
            for name, imp in top_features:
                print(f"    - {name}: {imp:.3f}")
            
            pattern_importance = dict(top_features).get('Pattern_Signal', 0)
            if pattern_importance < 0.05:
                print("  ⚠ Agent may be ignoring pattern signals")
                print("    → Consider increasing pattern signal weight or quality")
        
        # Performance
        print("\n[Performance]")
        mean_return = np.mean(self.episode_returns)
        if mean_return > 0:
            print(f"  ✓ Positive average episode return: {mean_return:.4f}")
        else:
            print(f"  ⚠ Negative average episode return: {mean_return:.4f}")
            print("    → Model may need more training or reward function adjustment")
        
        print("\n" + "="*80)
        print("INTERPRETABILITY ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'action_stats': action_stats,
            'feature_importance': feature_importance,
            'state_action_df': state_action_df,
            'trajectories': trajectories,
            'episode_returns': self.episode_returns
        }


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def run_interpretability_analysis(rl_model, env, feature_names):
    """
    Convenience function to run full interpretability analysis
    
    Parameters:
    -----------
    rl_model : trained RL model
    env : trading environment
    feature_names : list of feature names
    
    Example:
    --------
    >>> feature_names = ['Price_Change', 'RSI', 'MACD', 'MACD_Signal', 
    ...                  'ATR', 'Position', 'Pattern_Signal']
    >>> interpreter = RLInterpretabilityReport(rl_model, env, feature_names)
    >>> results = interpreter.generate_full_report()
    """
    interpreter = RLInterpretabilityReport(rl_model, env, feature_names)
    results = interpreter.generate_full_report()
    return interpreter, results


if __name__ == "__main__":
    """
    Standalone demo of interpretability analysis
    This demonstrates how to use the module independently
    """
    print("\n" + "="*80)
    print("RL INTERPRETABILITY MODULE - STANDALONE DEMO")
    print("="*80)
    print("\nThis module analyzes trained RL trading agents.")
    print("\nUsage Options:")
    print("-" * 80)
    print("\n1. WITH YOUR TRAINED MODEL:")
    print("   from rl_interpretability import RLInterpretabilityReport")
    print("   ")
    print("   # After training your RL model in the main system")
    print("   feature_names = ['Price_Change', 'RSI', 'MACD', 'MACD_Signal',")
    print("                    'ATR', 'Position', 'Pattern_Signal']")
    print("   ")
    print("   interpreter = RLInterpretabilityReport(rl_model, env, feature_names)")
    print("   results = interpreter.generate_full_report()")
    print("\n2. AUTOMATIC INTEGRATION:")
    print("   The main trading system automatically calls this module")
    print("   when you run it with RL enabled (Option 1 in the menu)")
    print("\n3. STANDALONE DEMO:")
    print("   To see a demo with synthetic data, this would require")
    print("   importing the full trading system (PatternAwareTradingEnv, etc.)")
    print("\n" + "="*80)
    print("\nFor full functionality, use this with the main trading system:")
    print("  1. Save this file as 'rl_interpretability.py'")
    print("  2. Run your main trading system Python file")
    print("  3. Choose Option 1 or 3 (with RL enabled)")
    print("  4. Interpretability analysis runs automatically after training")
    print("\n" + "="*80)
    
    # Optional: Create a minimal demo if gym/sb3 are available
    try:
        import gymnasium as gym
        print("\n✓ gymnasium detected - interpretability ready for use")
    except ImportError:
        print("\n⚠ gymnasium not installed - install with: pip install gymnasium")
    
    try:
        from stable_baselines3 import PPO
        print("✓ stable-baselines3 detected - RL training ready")
    except ImportError:
        print("⚠ stable-baselines3 not installed - install with: pip install stable-baselines3")
    
    print("\n" + "="*80)
    print("Setup complete. Ready to analyze RL trading agents!")
    print("="*80 + "\n")