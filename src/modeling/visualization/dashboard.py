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
    
    def __init__(self, portfolio_manager, rl_metrics=None):
        """
        Unified Dashboard Logic
        """
        self.portfolio_manager = portfolio_manager
        # We set both names to ensure internal methods find what they need
        self.rl_metrics = rl_metrics
        self.rl_comparison = rl_metrics # <--- This fixes the AttributeError
        
        # Ensure we have trade data
        self.trades_df = pd.DataFrame(portfolio_manager.closed_positions)
    
    def create_dashboard(self, save_path='trading_dashboard.png'):
        """Generate comprehensive dashboard with Unified Portfolio data"""
        
        # 1. Attribute Fix: Ensure we use the correct internal name
        # If your __init__ uses self.portfolio_manager, let's reference that
        pm = self.portfolio_manager 
        
        # 2. Setup the Plot Grid
        # Check if rl_comparison exists and has data
        has_rl = hasattr(self, 'rl_comparison') and self.rl_comparison is not None
        
        if has_rl:
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        fig.suptitle('TRADING SYSTEM DASHBOARD', fontsize=16, fontweight='bold')
        
        # --- SECTION 1: Equity Curve ---
        ax1 = axes[0, 0]
        if pm.daily_equity:
            equity_df = pd.DataFrame(pm.daily_equity)
            ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2, color='#2ecc71')
            ax1.axhline(pm.initial_capital, color='gray', linestyle='--')
            ax1.set_title('Equity Curve (Unified)')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)
        
        # --- SECTION 2: Trade Distribution ---
        ax2 = axes[0, 1]
        if pm.closed_positions:
            closed_df = pd.DataFrame(pm.closed_positions)
            returns = closed_df['pnl'].values
            colors = ['#27ae60' if x > 0 else '#e74c3c' for x in returns]
            ax2.bar(range(len(returns)), returns, color=colors, alpha=0.6)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_title('Trade P&L Distribution')
            ax2.set_ylabel('P&L ($)')
            ax2.grid(True, alpha=0.3)
        
        # --- SECTION 3: Performance Metrics ---
        ax3 = axes[1, 0]
        # Use the specific engine metrics or general ones
        metrics = pm.get_metrics(engine_name="Backtest")
        ax3.axis('off')
        data = [
            ['Total Trades', f"{metrics.get('total_trades', 0)}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1f}%"],
            ['Total P&L', f"${metrics.get('total_pnl', 0):,.0f}"],
            ['Total Return', f"{metrics.get('total_return', 0):.2f}%"],
        ]
        table = ax3.table(cellText=data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.set_fontsize(12)
        table.scale(1, 2)
        ax3.set_title('Performance Metrics', fontweight='bold')
        
        # --- SECTION 4: Pattern Performance ---
        ax4 = axes[1, 1]
        if pm.closed_positions:
            closed_df = pd.DataFrame(pm.closed_positions)
            pattern_stats = closed_df.groupby('pattern_type').agg({
                'pnl': ['count', lambda x: (x > 0).sum()]
            }).reset_index()
            pattern_stats.columns = ['Pattern', 'Total', 'Wins']
            pattern_stats['WinRate'] = (pattern_stats['Wins'] / pattern_stats['Total']) * 100
            
            colors = ['#2ecc71' if x >= 50 else '#e67e22' for x in pattern_stats['WinRate']]
            ax4.barh(pattern_stats['Pattern'], pattern_stats['WinRate'], color=colors, alpha=0.6)
            ax4.axvline(50, color='gray', linestyle='--')
            ax4.set_xlabel('Win Rate (%)')
            ax4.set_title('Win Rate by Pattern')
        
        # --- SECTION 5 & 6: RL COMPARISON (Only if data exists) ---
        if has_rl:
            # We assume rl_comparison is the dict returned by get_metrics from RL
            # We compare it against the main portfolio metrics
            ax5 = axes[0, 2]
            strategies = ['RL Agent', 'Backtest']
            # Safely extract returns
            rl_ret = self.rl_comparison.get('total_return', 0)
            base_ret = metrics.get('total_return', 0)
            
            ax5.bar(strategies, [rl_ret, base_ret], color=['#3498db', '#f39c12'], alpha=0.7)
            ax5.set_ylabel('Return (%)')
            ax5.set_title('Strategy Comparison: Returns')
            
            ax6 = axes[1, 2]
            rl_win = self.rl_comparison.get('win_rate', 0)
            base_win = metrics.get('win_rate', 0)
            
            ax6.bar(strategies, [rl_win, base_win], color=['#3498db', '#f39c12'], alpha=0.7)
            ax6.axhline(50, color='gray', linestyle='--')
            ax6.set_ylabel('Win Rate (%)')
            ax6.set_title('Strategy Comparison: Win Rate')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Unified Dashboard saved as '{save_path}'")
        plt.close()