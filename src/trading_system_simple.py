#!/usr/bin/env python3
"""
Streamlined Multi-Pattern Trading System
Guaranteed to show output and work step by step
"""

# Immediate output so you know it's running
print("\n" + "="*80)
print("TRADING SYSTEM STARTING...")
print("="*80)
print("Loading libraries... (this may take 10-30 seconds)")

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("✓ Core libraries loaded")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb

print("✓ ML libraries loaded")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Visualization libraries loaded")

# Check for optional RL
try:
    import gymnasium as gym
    from gymnasium import spaces
    RL_AVAILABLE = True
    print("✓ RL libraries available (gymnasium)")
except ImportError:
    RL_AVAILABLE = False
    print("⚠ RL libraries not available (optional)")

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
    print("✓ RL training available (stable-baselines3)")
except ImportError:
    SB3_AVAILABLE = False
    print("⚠ RL training not available (optional)")

print("\n" + "="*80)
print("ALL LIBRARIES LOADED SUCCESSFULLY")
print("="*80)

# ============================================================================
# SIMPLIFIED CLASSES
# ============================================================================

class SimplePatternDetector:
    """Simplified pattern detector"""
    
    def __init__(self, df):
        print("  Initializing pattern detector...")
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self._calculate_indicators()
        print("  ✓ Indicators calculated")
    
    def _calculate_indicators(self):
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Simple moving averages
        self.df['SMA_20'] = self.df['Close'].rolling(20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(50).mean()
        
        # ATR
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['ATR'] = true_range.rolling(14).mean()
        
        # Find peaks
        peak_idx = argrelextrema(self.df['High'].values, np.greater, order=5)[0]
        trough_idx = argrelextrema(self.df['Low'].values, np.less, order=5)[0]
        self.df['Peak'] = False
        self.df['Trough'] = False
        self.df.loc[peak_idx, 'Peak'] = True
        self.df.loc[trough_idx, 'Trough'] = True
    
    def detect_double_tops(self):
        print("  Scanning for Double Top patterns...")
        patterns = []
        peak_indices = self.df[self.df['Peak'] == True].index.tolist()
        
        for i in range(len(peak_indices) - 1):
            for j in range(i + 1, len(peak_indices)):
                idx1, idx2 = peak_indices[i], peak_indices[j]
                
                if idx2 - idx1 < 10 or idx2 - idx1 > 50:
                    continue
                
                peak1 = self.df.loc[idx1, 'High']
                peak2 = self.df.loc[idx2, 'High']
                
                if abs(peak1 - peak2) / peak1 > 0.02:
                    continue
                
                trough_idx = self.df.loc[idx1:idx2, 'Low'].idxmin()
                neckline = self.df.loc[trough_idx, 'Low']
                pattern_height = max(peak1, peak2) - neckline
                
                patterns.append({
                    'Type': 'DoubleTop',
                    'Index': idx2,
                    'Date': self.df.loc[idx2, 'date'],
                    'Neckline': neckline,
                    'Height': pattern_height,
                    'StopLoss': max(peak1, peak2) * 1.002,
                    'Target': neckline - pattern_height,
                    'RSI': self.df.loc[idx2, 'RSI'],
                })
        
        return pd.DataFrame(patterns)
    
    def detect_double_bottoms(self):
        print("  Scanning for Double Bottom patterns...")
        patterns = []
        trough_indices = self.df[self.df['Trough'] == True].index.tolist()
        
        for i in range(len(trough_indices) - 1):
            for j in range(i + 1, len(trough_indices)):
                idx1, idx2 = trough_indices[i], trough_indices[j]
                
                if idx2 - idx1 < 10 or idx2 - idx1 > 50:
                    continue
                
                bottom1 = self.df.loc[idx1, 'Low']
                bottom2 = self.df.loc[idx2, 'Low']
                
                if abs(bottom1 - bottom2) / bottom1 > 0.02:
                    continue
                
                peak_idx = self.df.loc[idx1:idx2, 'High'].idxmax()
                neckline = self.df.loc[peak_idx, 'High']
                pattern_height = neckline - min(bottom1, bottom2)
                
                patterns.append({
                    'Type': 'DoubleBottom',
                    'Index': idx2,
                    'Date': self.df.loc[idx2, 'date'],
                    'Neckline': neckline,
                    'Height': pattern_height,
                    'StopLoss': min(bottom1, bottom2) * 0.998,
                    'Target': neckline + pattern_height,
                    'RSI': self.df.loc[idx2, 'RSI'],
                })
        
        return pd.DataFrame(patterns)


class SimpleBacktester:
    """Simplified backtester"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
    
    def backtest(self, df, patterns):
        print("\n  Running backtest...")
        
        if patterns.empty:
            print("  ⚠ No patterns to backtest")
            return
        
        for _, pattern in patterns.iterrows():
            entry_idx = pattern['Index']
            if entry_idx >= len(df) - 30:
                continue
            
            entry_price = pattern['Neckline']
            stop_loss = pattern['StopLoss']
            target = pattern['Target']
            is_bearish = pattern['Type'] == 'DoubleTop'
            
            # Find exit
            for i in range(entry_idx + 1, min(entry_idx + 30, len(df))):
                if is_bearish:
                    if df.loc[i, 'High'] >= stop_loss:
                        pnl = -100  # Lost
                        self.trades.append({'PnL': pnl, 'Result': 'Loss'})
                        break
                    if df.loc[i, 'Low'] <= target:
                        pnl = 200  # Won
                        self.trades.append({'PnL': pnl, 'Result': 'Win'})
                        break
                else:
                    if df.loc[i, 'Low'] <= stop_loss:
                        pnl = -100
                        self.trades.append({'PnL': pnl, 'Result': 'Loss'})
                        break
                    if df.loc[i, 'High'] >= target:
                        pnl = 200
                        self.trades.append({'PnL': pnl, 'Result': 'Win'})
                        break
            
            if len(self.trades) >= 20:  # Limit for demo
                break
        
        self.capital = self.initial_capital + sum(t['PnL'] for t in self.trades)
        print(f"  ✓ Completed {len(self.trades)} trades")
    
    def get_results(self):
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        wins = len(trades_df[trades_df['Result'] == 'Win'])
        total = len(trades_df)
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'total_pnl': sum(t['PnL'] for t in self.trades),
            'final_capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100
        }


def generate_sample_data():
    """Generate sample data"""
    print("\nGenerating sample data...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    trend = np.linspace(10000, 11000, 500)
    noise = np.cumsum(np.random.randn(500) * 30)
    close = trend + noise
    
    high = close + np.random.uniform(10, 50, 500)
    low = close - np.random.uniform(10, 50, 500)
    open_price = close + np.random.uniform(-20, 20, 500)
    
    df = pd.DataFrame({
        'date': dates,
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close
    })
    
    print(f"✓ Generated {len(df)} bars of data")
    return df


def create_simple_dashboard(results):
    """Create simple dashboard"""
    print("\nCreating dashboard...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Trading System Results', fontsize=14, fontweight='bold')
    
    # Chart 1: Metrics
    ax1 = axes[0]
    ax1.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS
    
    Total Trades:     {results['total_trades']}
    Winning Trades:   {results['wins']}
    Losing Trades:    {results['losses']}
    Win Rate:         {results['win_rate']:.1f}%
    
    Total P&L:        ${results['total_pnl']:,.0f}
    Final Capital:    ${results['final_capital']:,.0f}
    Return:           {results['return_pct']:.2f}%
    """
    
    ax1.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    # Chart 2: Win/Loss distribution
    ax2 = axes[1]
    categories = ['Wins', 'Losses']
    values = [results['wins'], results['losses']]
    colors = ['green', 'red']
    
    ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Trades')
    ax2.set_title('Win/Loss Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('trading_results.png', dpi=150, bbox_inches='tight')
    print("✓ Dashboard saved as 'trading_results.png'")
    plt.close()


def run_quick_demo():
    """Run quick demo"""
    
    print("\n" + "="*80)
    print("RUNNING QUICK DEMO")
    print("="*80)
    
    # Step 1: Generate data
    df = generate_sample_data()
    
    # Step 2: Detect patterns
    print("\n[1/3] DETECTING PATTERNS")
    print("="*80)
    detector = SimplePatternDetector(df)
    
    patterns_dt = detector.detect_double_tops()
    print(f"  ✓ Found {len(patterns_dt)} Double Top patterns")
    
    patterns_db = detector.detect_double_bottoms()
    print(f"  ✓ Found {len(patterns_db)} Double Bottom patterns")
    
    all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)
    print(f"  ✓ Total: {len(all_patterns)} patterns")
    
    # Step 3: Backtest
    print("\n[2/3] BACKTESTING")
    print("="*80)
    backtester = SimpleBacktester(initial_capital=100000)
    backtester.backtest(df, all_patterns)
    
    results = backtester.get_results()
    
    # Step 4: Show results
    print("\n[3/3] RESULTS")
    print("="*80)
    print(f"Total Trades      : {results['total_trades']}")
    print(f"Winning Trades    : {results['wins']}")
    print(f"Losing Trades     : {results['losses']}")
    print(f"Win Rate          : {results['win_rate']:.1f}%")
    print(f"Total P&L         : ${results['total_pnl']:,.0f}")
    print(f"Final Capital     : ${results['final_capital']:,.0f}")
    print(f"Return            : {results['return_pct']:.2f}%")
    print("="*80)
    
    # Create dashboard
    create_simple_dashboard(results)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nCheck your directory for 'trading_results.png'")


def run_with_csv(csv_file):
    """Run with user's CSV file"""
    
    print("\n" + "="*80)
    print(f"RUNNING WITH YOUR DATA: {csv_file}")
    print("="*80)
    
    try:
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} rows")
        
        # Detect patterns
        print("\n[1/3] DETECTING PATTERNS")
        print("="*80)
        detector = SimplePatternDetector(df)
        
        patterns_dt = detector.detect_double_tops()
        print(f"  ✓ Found {len(patterns_dt)} Double Top patterns")
        
        patterns_db = detector.detect_double_bottoms()
        print(f"  ✓ Found {len(patterns_db)} Double Bottom patterns")
        
        all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)
        print(f"  ✓ Total: {len(all_patterns)} patterns")
        
        # Backtest
        print("\n[2/3] BACKTESTING")
        print("="*80)
        backtester = SimpleBacktester(initial_capital=100000)
        backtester.backtest(df, all_patterns)
        
        results = backtester.get_results()
        
        # Show results
        print("\n[3/3] RESULTS")
        print("="*80)
        print(f"Total Trades      : {results['total_trades']}")
        print(f"Win Rate          : {results['win_rate']:.1f}%")
        print(f"Final Capital     : ${results['final_capital']:,.0f}")
        print(f"Return            : {results['return_pct']:.2f}%")
        print("="*80)
        
        # Create dashboard
        create_simple_dashboard(results)
        
        print("\n✓ Analysis complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Make sure your CSV has columns: date, Open, High, Low, Close")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main function with menu"""
    
    print("\n" + "="*80)
    print("TRADING SYSTEM - MAIN MENU")
    print("="*80)
    print("1. Quick Demo (sample data)")
    print("2. Use my own CSV file")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        run_quick_demo()
        
    elif choice == '2':
        csv_file = input("\nEnter CSV file path: ").strip()
        run_with_csv(csv_file)
        
    elif choice == '3':
        print("\nExiting... Goodbye!")
        
    else:
        print("\n✗ Invalid choice. Please run again.")
    
    print("\n" + "="*80)
    print("Program finished.")
    print("="*80)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()