import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns



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
        ax1 = axes[0, 0]
        if self.portfolio.daily_equity:
            equity_df = pd.DataFrame(self.portfolio.daily_equity)
            ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2)
            ax1.axhline(self.portfolio.initial_capital, color='gray', linestyle='--')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Trade Distribution
        ax2 = axes[0, 1]
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
        ax3 = axes[1, 0]
        metrics = self.portfolio.get_metrics()
        ax3.axis('off')
        data = [
            ['Total Trades', f"{metrics.get('total_trades', 0)}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1f}%"],
            ['Total P&L', f"${metrics.get('total_pnl', 0):,.0f}"],
            ['Total Return', f"{metrics.get('total_return', 0):.2f}%"],
        ]
        table = ax3.table(cellText=data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax3.set_title('Performance Metrics', fontweight='bold')
        
        # 4. Pattern Performance
        ax4 = axes[1, 1]
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
        
        if self.rl_comparison:
            # 5. Strategy Returns
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
        print(f"\n✓ Dashboard saved as '{save_path}'")
        plt.close()