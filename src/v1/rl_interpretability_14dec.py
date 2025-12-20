"""
RL MODEL INTERPRETABILITY & EXPLAINABILITY MODULE

Techniques for understanding what the RL agent has learned:
1. Action Distribution Analysis
2. Feature Importance (SHAP-like)
3. State-Action Heatmaps
4. Decision Path Visualization
5. Policy Network Inspection
6. Counterfactual Analysis
7. Attention Mechanisms
8. Rule Extraction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. ACTION DISTRIBUTION ANALYSIS
# ============================================================================

class ActionDistributionAnalyzer:
    """Analyze when and why the agent takes specific actions"""
    
    def __init__(self, rl_model, env):
        self.model = rl_model
        self.env = env
        self.action_log = []
        self.state_log = []
    
    def collect_episodes(self, n_episodes=10):
        """Collect state-action pairs over multiple episodes"""
        print(f"Collecting {n_episodes} episodes for analysis...")
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                self.action_log.append(action)
                self.state_log.append(obs.copy())
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
        
        self.state_log = np.array(self.state_log)
        self.action_log = np.array(self.action_log)
        
        print(f"✓ Collected {len(self.action_log)} state-action pairs")
    
    def analyze_action_distribution(self):
        """Analyze overall action distribution"""
        action_names = ['Hold', 'Long', 'Flatten']
        action_counts = np.bincount(self.action_log, minlength=3)
        action_pcts = action_counts / action_counts.sum() * 100
        
        print("\n" + "="*80)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("="*80)
        
        for i, (name, count, pct) in enumerate(zip(action_names, action_counts, action_pcts)):
            print(f"{name:<15}: {count:>6} times ({pct:>5.1f}%)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        ax1.bar(action_names, action_counts, color=['gray', 'green', 'red'], alpha=0.7)
        ax1.set_ylabel('Frequency')
        ax1.set_title('Action Distribution')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(action_pcts, labels=action_names, autopct='%1.1f%%',
               colors=['gray', 'green', 'red'], startangle=90)
        ax2.set_title('Action Distribution (%)')
        
        plt.tight_layout()
        plt.savefig('rl_action_distribution.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'rl_action_distribution.png'")
        plt.close()
        
        return action_counts, action_pcts
    
    def analyze_action_by_state(self, feature_idx=0, feature_name='Feature_0'):
        """Analyze which actions are taken in different states"""
        
        # Bin the feature into quintiles
        feature_values = self.state_log[:, feature_idx]
        bins = np.percentile(feature_values, [0, 20, 40, 60, 80, 100])
        digitized = np.digitize(feature_values, bins[:-1])
        
        # Count actions per quintile
        action_names = ['Hold', 'Long', 'Flatten']
        quintile_actions = {}
        
        for q in range(1, 6):
            mask = digitized == q
            actions = self.action_log[mask]
            counts = np.bincount(actions, minlength=3)
            quintile_actions[q] = counts
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(5)
        width = 0.25
        
        for i, action_name in enumerate(action_names):
            counts = [quintile_actions[q][i] for q in range(1, 6)]
            ax.bar(x + i*width, counts, width, label=action_name, alpha=0.7)
        
        ax.set_xlabel(f'{feature_name} Quintiles')
        ax.set_ylabel('Action Frequency')
        ax.set_title(f'Actions by {feature_name} State')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'rl_actions_by_{feature_name}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved 'rl_actions_by_{feature_name}.png'")
        plt.close()


# ============================================================================
# 2. FEATURE IMPORTANCE (SHAP-LIKE)
# ============================================================================

class FeatureImportanceAnalyzer:
    """Approximate feature importance using surrogate models"""
    
    def __init__(self, rl_model, env, feature_names):
        self.model = rl_model
        self.env = env
        self.feature_names = feature_names
        self.surrogate_model = None
    
    def train_surrogate_model(self, state_log, action_log):
        """Train a decision tree to mimic the RL policy"""
        print("\nTraining surrogate model (Decision Tree)...")
        
        # Train decision tree to mimic RL policy
        self.surrogate_model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=42
        )
        self.surrogate_model.fit(state_log, action_log)
        
        accuracy = self.surrogate_model.score(state_log, action_log)
        print(f"✓ Surrogate model accuracy: {accuracy:.2%}")
        
        return self.surrogate_model
    
    def analyze_feature_importance(self):
        """Get feature importance from surrogate model"""
        if self.surrogate_model is None:
            print("Error: Train surrogate model first!")
            return
        
        importances = self.surrogate_model.feature_importances_
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(importance_df.to_string(index=False))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        ax.barh(importance_df['Feature'], importance_df['Importance'], 
               color=colors, alpha=0.8)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance in RL Policy')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('rl_feature_importance.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'rl_feature_importance.png'")
        plt.close()
        
        return importance_df
    
    def visualize_decision_tree(self):
        """Visualize the surrogate decision tree"""
        if self.surrogate_model is None:
            print("Error: Train surrogate model first!")
            return
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(self.surrogate_model, 
                 feature_names=self.feature_names,
                 class_names=['Hold', 'Long', 'Flatten'],
                 filled=True, rounded=True, ax=ax)
        
        plt.tight_layout()
        plt.savefig('rl_decision_tree.png', dpi=150, bbox_inches='tight')
        print("\n✓ Decision tree saved as 'rl_decision_tree.png'")
        plt.close()
        
        # Also print text representation
        tree_rules = export_text(self.surrogate_model, 
                                feature_names=self.feature_names)
        
        print("\n" + "="*80)
        print("DECISION TREE RULES (Approximating RL Policy)")
        print("="*80)
        print(tree_rules)


# ============================================================================
# 3. STATE-ACTION VALUE HEATMAP
# ============================================================================

class StateActionHeatmap:
    """Visualize which states lead to which actions"""
    
    def __init__(self, rl_model, env):
        self.model = rl_model
        self.env = env
    
    def create_heatmap(self, feature1_idx, feature2_idx, 
                       feature1_name, feature2_name, grid_size=20):
        """Create 2D heatmap of actions over state space"""
        
        print(f"\nCreating heatmap for {feature1_name} vs {feature2_name}...")
        
        # Get reasonable ranges for features
        obs, _ = self.env.reset()
        
        # Sample state space
        feature1_range = np.linspace(-2, 2, grid_size)
        feature2_range = np.linspace(-2, 2, grid_size)
        
        action_grid = np.zeros((grid_size, grid_size))
        
        for i, f1 in enumerate(feature1_range):
            for j, f2 in enumerate(feature2_range):
                # Create synthetic state
                test_state = obs.copy()
                test_state[feature1_idx] = f1
                test_state[feature2_idx] = f2
                
                # Get action
                action, _ = self.model.predict(test_state, deterministic=True)
                action_grid[i, j] = action
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(action_grid, cmap='RdYlGn', aspect='auto', origin='lower')
        
        ax.set_xlabel(feature2_name)
        ax.set_ylabel(feature1_name)
        ax.set_title(f'RL Policy Heatmap: Actions across State Space')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.set_label('Action')
        cbar.ax.set_yticklabels(['Hold', 'Long', 'Flatten'])
        
        plt.tight_layout()
        plt.savefig(f'rl_heatmap_{feature1_name}_{feature2_name}.png', 
                   dpi=150, bbox_inches='tight')
        print(f"✓ Heatmap saved as 'rl_heatmap_{feature1_name}_{feature2_name}.png'")
        plt.close()


# ============================================================================
# 4. POLICY NETWORK INSPECTION
# ============================================================================

class PolicyNetworkInspector:
    """Inspect the neural network weights and activations"""
    
    def __init__(self, rl_model):
        self.model = rl_model
    
    def extract_policy_network(self):
        """Extract the policy network from PPO/A2C model"""
        try:
            policy_net = self.model.policy
            return policy_net
        except:
            print("Could not extract policy network")
            return None
    
    def analyze_network_structure(self):
        """Analyze the structure of the policy network"""
        print("\n" + "="*80)
        print("POLICY NETWORK STRUCTURE")
        print("="*80)
        
        policy_net = self.extract_policy_network()
        if policy_net is None:
            return
        
        # Print network architecture
        print(policy_net)
        
        # Count parameters
        total_params = sum(p.numel() for p in policy_net.parameters())
        trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


# ============================================================================
# 5. COUNTERFACTUAL ANALYSIS
# ============================================================================

class CounterfactualAnalyzer:
    """What would happen if we changed specific features?"""
    
    def __init__(self, rl_model, env):
        self.model = rl_model
        self.env = env
    
    def analyze_feature_impact(self, base_state, feature_idx, 
                               feature_name, variation_range=(-2, 2)):
        """See how changing one feature affects the action"""
        
        print(f"\nAnalyzing impact of {feature_name}...")
        
        variations = np.linspace(variation_range[0], variation_range[1], 50)
        actions = []
        
        for var in variations:
            test_state = base_state.copy()
            test_state[feature_idx] = var
            action, _ = self.model.predict(test_state, deterministic=True)
            actions.append(action)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['gray' if a == 0 else 'green' if a == 1 else 'red' for a in actions]
        ax.scatter(variations, actions, c=colors, alpha=0.6, s=50)
        
        ax.set_xlabel(f'{feature_name} Value')
        ax.set_ylabel('Action')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Hold', 'Long', 'Flatten'])
        ax.set_title(f'How {feature_name} Affects RL Action')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'rl_counterfactual_{feature_name}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved 'rl_counterfactual_{feature_name}.png'")
        plt.close()
        
        # Analyze thresholds
        action_changes = np.where(np.diff(actions) != 0)[0]
        if len(action_changes) > 0:
            print(f"\n{feature_name} thresholds where action changes:")
            for idx in action_changes:
                print(f"  {variations[idx]:.3f} → {variations[idx+1]:.3f}: "
                     f"Action {actions[idx]} → {actions[idx+1]}")


# ============================================================================
# 6. COMPREHENSIVE INTERPRETABILITY REPORT
# ============================================================================

class RLInterpretabilityReport:
    """Generate comprehensive interpretability report"""
    
    def __init__(self, rl_model, env, feature_names):
        self.model = rl_model
        self.env = env
        self.feature_names = feature_names
        
        # Initialize analyzers
        self.action_analyzer = ActionDistributionAnalyzer(rl_model, env)
        self.importance_analyzer = FeatureImportanceAnalyzer(rl_model, env, feature_names)
        self.heatmap_analyzer = StateActionHeatmap(rl_model, env)
        self.counterfactual_analyzer = CounterfactualAnalyzer(rl_model, env)
        self.network_inspector = PolicyNetworkInspector(rl_model)
    
    def generate_full_report(self):
        """Generate complete interpretability report"""
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE RL INTERPRETABILITY REPORT")
        print("="*80)
        
        # 1. Collect data
        print("\n[1/6] Collecting episode data...")
        self.action_analyzer.collect_episodes(n_episodes=20)
        
        # 2. Action distribution
        print("\n[2/6] Analyzing action distribution...")
        self.action_analyzer.analyze_action_distribution()
        
        # Analyze by key features
        print("\nAnalyzing actions by RSI...")
        self.action_analyzer.analyze_action_by_state(
            feature_idx=1,  # RSI
            feature_name='RSI'
        )
        
        print("\nAnalyzing actions by pattern signal...")
        self.action_analyzer.analyze_action_by_state(
            feature_idx=6,  # Pattern signal
            feature_name='Pattern_Signal'
        )
        
        # 3. Feature importance
        print("\n[3/6] Analyzing feature importance...")
        self.importance_analyzer.train_surrogate_model(
            self.action_analyzer.state_log,
            self.action_analyzer.action_log
        )
        self.importance_analyzer.analyze_feature_importance()
        self.importance_analyzer.visualize_decision_tree()
        
        # 4. State-action heatmaps
        print("\n[4/6] Creating state-action heatmaps...")
        self.heatmap_analyzer.create_heatmap(
            feature1_idx=1, feature2_idx=6,
            feature1_name='RSI', feature2_name='Pattern_Signal'
        )
        
        # 5. Counterfactual analysis
        print("\n[5/6] Running counterfactual analysis...")
        # Get a typical state
        typical_state = np.median(self.action_analyzer.state_log, axis=0)
        
        self.counterfactual_analyzer.analyze_feature_impact(
            typical_state, 
            feature_idx=1,
            feature_name='RSI'
        )
        
        self.counterfactual_analyzer.analyze_feature_impact(
            typical_state,
            feature_idx=6,
            feature_name='Pattern_Signal'
        )
        
        # 6. Network inspection
        print("\n[6/6] Inspecting policy network...")
        self.network_inspector.analyze_network_structure()
        
        print("\n" + "="*80)
        print("INTERPRETABILITY REPORT COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - rl_action_distribution.png")
        print("  - rl_actions_by_RSI.png")
        print("  - rl_actions_by_Pattern_Signal.png")
        print("  - rl_feature_importance.png")
        print("  - rl_decision_tree.png")
        print("  - rl_heatmap_RSI_Pattern_Signal.png")
        print("  - rl_counterfactual_RSI.png")
        print("  - rl_counterfactual_Pattern_Signal.png")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demo_rl_interpretability():
    """Demo function showing how to use interpretability tools"""
    
    print("""
================================================================================
RL INTERPRETABILITY TOOLKIT - USAGE GUIDE
================================================================================

After training your RL model, use these tools to understand what it learned:

1. ACTION DISTRIBUTION ANALYSIS
   - See which actions the agent prefers
   - Understand when different actions are taken
   
2. FEATURE IMPORTANCE
   - Which market features does the agent pay attention to?
   - Train a decision tree to mimic the policy
   
3. STATE-ACTION HEATMAPS
   - Visualize the policy across different market conditions
   - See decision boundaries
   
4. COUNTERFACTUAL ANALYSIS
   - "What if RSI was higher?"
   - See how each feature affects decisions
   
5. POLICY NETWORK INSPECTION
   - Examine the neural network structure
   - Understand the complexity of learned policy

EXAMPLE CODE:
-------------

# After training your RL model:
from rl_interpretability import RLInterpretabilityReport

# Feature names from your environment
feature_names = [
    'Price_Change', 'RSI', 'MACD', 'MACD_Signal',
    'ATR', 'Position', 'Pattern_Signal'
]

# Create report
interpreter = RLInterpretabilityReport(rl_model, env, feature_names)
interpreter.generate_full_report()

# This generates:
# - 8 visualizations
# - Feature importance rankings
# - Decision tree approximation
# - Actionable insights

KEY INSIGHTS YOU'LL GET:
------------------------
✓ Which market signals the agent trusts most
✓ What RSI/MACD levels trigger actions
✓ How pattern signals influence decisions
✓ When the agent enters vs exits positions
✓ Whether the agent learned reasonable strategies

This helps you:
- Trust the RL agent
- Debug poor performance
- Explain to stakeholders
- Improve training
================================================================================
    """)


if __name__ == "__main__":
    demo_rl_interpretability()