import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

class PortfolioManager:
    """Manages portfolio with risk constraints"""
    def __init__(self, initial_capital: float = 100000, max_drawdown: float = 0.10, target_sharpe: float = 1.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_drawdown = max_drawdown
        self.target_sharpe = target_sharpe
        self.position = 0  # Number of shares
        self.trades = []
        self.portfolio_values = []
        self.peak_value = initial_capital
        
    def calculate_position_size(self, price: float) -> int:
        """Calculate position size based on available capital"""
        return int(self.capital / price)
    
    def buy(self, price: float, date: str, shares: int = None):
        """Execute buy order"""
        if shares is None:
            shares = self.calculate_position_size(price)
        
        cost = shares * price
        if cost <= self.capital:
            self.position += shares
            self.capital -= cost
            self.trades.append({'date': date, 'action': 'BUY', 'price': price, 'shares': shares, 'value': cost})
            return True
        return False
    
    def sell(self, price: float, date: str, shares: int = None):
        """Execute sell order"""
        if shares is None:
            shares = self.position
        
        if shares <= self.position:
            revenue = shares * price
            self.position -= shares
            self.capital += revenue
            self.trades.append({'date': date, 'action': 'SELL', 'price': price, 'shares': shares, 'value': revenue})
            return True
        return False
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        return self.capital + (self.position * current_price)
    
    def update_portfolio_value(self, current_price: float, date: str):
        """Update portfolio value and check drawdown"""
        portfolio_value = self.get_portfolio_value(current_price)
        self.portfolio_values.append({'date': date, 'value': portfolio_value})
        
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # Exit all positions if drawdown exceeds limit
        if current_drawdown > self.max_drawdown and self.position > 0:
            self.sell(current_price, date)
            return False  # Signal to stop trading
        
        return True  # Continue trading
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'num_trades': 0, 'final_value': self.initial_capital, 'win_rate': 0}
        
        values = pd.DataFrame(self.portfolio_values)
        values['returns'] = values['value'].pct_change()
        
        total_return = (values['value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (annualized, assuming daily data)
        mean_return = values['returns'].mean()
        std_return = values['returns'].std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0
        
        # Maximum drawdown
        peak = values['value'].expanding(min_periods=1).max()
        drawdown = (values['value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = 0
        if len(self.trades) >= 2:
            for i in range(1, len(self.trades), 2):
                if i < len(self.trades) and self.trades[i]['action'] == 'SELL':
                    buy_price = self.trades[i-1]['price']
                    sell_price = self.trades[i]['price']
                    if sell_price > buy_price:
                        winning_trades += 1
        
        total_trade_pairs = len(self.trades) // 2
        win_rate = (winning_trades / total_trade_pairs * 100) if total_trade_pairs > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'num_trades': len(self.trades),
            'final_value': values['value'].iloc[-1],
            'win_rate': win_rate
        }


class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, name: str):
        self.name = name
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> str:
        """Generate trading signal: 'BUY', 'SELL', or 'HOLD'"""
        raise NotImplementedError


class BuyAndHold(TradingStrategy):
    def __init__(self):
        super().__init__("Buy and Hold")
        self.bought = False
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> str:
        if not self.bought:
            self.bought = True
            return 'BUY'
        return 'HOLD'


class ValueInvesting(TradingStrategy):
    def __init__(self):
        super().__init__("Value Investing")
        self.holding = False
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> str:
        if idx < 20:
            return 'HOLD'
        
        ma_20 = data['close'].iloc[idx-20:idx].mean()
        current_price = data['close'].iloc[idx]
        
        if not self.holding and current_price < ma_20 * 0.95:
            self.holding = True
            return 'BUY'
        
        if self.holding and current_price > ma_20 * 1.05:
            self.holding = False
            return 'SELL'
        
        return 'HOLD'


class MomentumInvesting(TradingStrategy):
    def __init__(self):
        super().__init__("Momentum Investing")
        self.holding = False
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> str:
        if idx < 60:
            return 'HOLD'
        
        price_60_days_ago = data['close'].iloc[idx-60]
        current_price = data['close'].iloc[idx]
        momentum = (current_price - price_60_days_ago) / price_60_days_ago
        
        if not self.holding and momentum > 0.05:
            self.holding = True
            return 'BUY'
        
        if self.holding and momentum < -0.03:
            self.holding = False
            return 'SELL'
        
        return 'HOLD'


class SwingTrading(TradingStrategy):
    def __init__(self):
        super().__init__("Swing Trading")
        self.holding = False
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> str:
        if idx < 20:
            return 'HOLD'
        
        prices = data['close'].iloc[max(0, idx-20):idx+1]
        rsi = self.calculate_rsi(prices)
        
        if not self.holding and rsi < 30:
            self.holding = True
            return 'BUY'
        
        if self.holding and rsi > 70:
            self.holding = False
            return 'SELL'
        
        return 'HOLD'


class SupportResistance(TradingStrategy):
    def __init__(self):
        super().__init__("Support-Resistance-52-Week-High-Low")
        self.holding = False
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> str:
        if idx < 252:
            return 'HOLD'
        
        lookback_period = min(252, idx)
        week_52_high = data['high'].iloc[idx-lookback_period:idx].max()
        week_52_low = data['low'].iloc[idx-lookback_period:idx].min()
        current_price = data['close'].iloc[idx]
        
        support = data['low'].iloc[idx-20:idx].min()
        resistance = data['high'].iloc[idx-20:idx].max()
        
        if not self.holding and (current_price <= week_52_low * 1.02 or current_price <= support * 1.02):
            self.holding = True
            return 'BUY'
        
        if self.holding and (current_price >= week_52_high * 0.98 or current_price >= resistance * 0.98):
            self.holding = False
            return 'SELL'
        
        return 'HOLD'


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load and validate CSV data"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    df.columns = df.columns.str.lower().str.strip()
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nAvailable columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df[required_cols].copy()
    
    if not pd.api.types.is_string_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    df['date_obj'] = pd.to_datetime(df['date'])
    df = df.sort_values('date_obj').drop('date_obj', axis=1).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} rows of data")
    print(f"✓ Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"✓ Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}\n")
    
    return df


def backtest_strategy(strategy: TradingStrategy, data: pd.DataFrame, 
                      trade_year: int = 2025, initial_capital: float = 100000) -> Tuple[Dict, PortfolioManager]:
    """Backtest a trading strategy and return metrics and portfolio manager"""
    trade_data = data[data['date'].str.startswith(str(trade_year))].reset_index(drop=True)
    
    if len(trade_data) == 0:
        return None, None
    
    portfolio = PortfolioManager(initial_capital=initial_capital)
    
    full_data = data[data['date'] < f'{trade_year + 1}-01-01'].reset_index(drop=True)
    
    for i in range(len(full_data)):
        if not full_data['date'].iloc[i].startswith(str(trade_year)):
            continue
        
        current_price = full_data['close'].iloc[i]
        current_date = full_data['date'].iloc[i]
        
        signal = strategy.generate_signal(full_data, i)
        
        if signal == 'BUY' and portfolio.position == 0:
            portfolio.buy(current_price, current_date)
        elif signal == 'SELL' and portfolio.position > 0:
            portfolio.sell(current_price, current_date)
        
        can_continue = portfolio.update_portfolio_value(current_price, current_date)
        if not can_continue:
            break
    
    if portfolio.position > 0:
        last_price = full_data['close'].iloc[-1]
        last_date = full_data['date'].iloc[-1]
        portfolio.sell(last_price, last_date)
    
    return portfolio.calculate_metrics(), portfolio


def create_visualizations(results_df: pd.DataFrame, portfolios: Dict, trade_year: int, output_dir: str = "reports"):
    """Create comprehensive visualization charts"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Create main comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Portfolio Growth Over Time (Large chart)
    ax1 = fig.add_subplot(gs[0, :])
    for i, (strategy_name, portfolio) in enumerate(portfolios.items()):
        if portfolio.portfolio_values:
            df = pd.DataFrame(portfolio.portfolio_values)
            df['date'] = pd.to_datetime(df['date'])
            ax1.plot(df['date'], df['value'], label=strategy_name, linewidth=2, color=colors[i % len(colors)])
    
    ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title(f'Portfolio Growth Comparison - {trade_year}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
    
    # 2. Total Returns Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.barh(results_df['Strategy'], results_df['Total_Return_%'], color=colors[:len(results_df)])
    ax2.set_xlabel('Total Return (%)', fontsize=11)
    ax2.set_title('Total Returns', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # 3. Sharpe Ratio Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    bars = ax3.barh(results_df['Strategy'], results_df['Sharpe_Ratio'], color=colors[:len(results_df)])
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Target (1.0)')
    ax3.set_xlabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Risk-Adjusted Returns', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.legend(fontsize=9)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # 4. Max Drawdown
    ax4 = fig.add_subplot(gs[1, 2])
    bars = ax4.barh(results_df['Strategy'], results_df['Max_Drawdown_%'], color=colors[:len(results_df)])
    ax4.axvline(x=10.0, color='red', linestyle='--', alpha=0.7, label='Limit (10%)')
    ax4.set_xlabel('Maximum Drawdown (%)', fontsize=11)
    ax4.set_title('Maximum Drawdown', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.legend(fontsize=9)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    # 5. Number of Trades
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.bar(range(len(results_df)), results_df['Num_Trades'], color=colors[:len(results_df)])
    ax5.set_xticks(range(len(results_df)))
    ax5.set_xticklabels(results_df['Strategy'], rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Number of Trades', fontsize=11)
    ax5.set_title('Trading Frequency', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df['Num_Trades']):
        ax5.text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontsize=9)
    
    # 6. Win Rate
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.bar(range(len(results_df)), results_df['Win_Rate_%'], color=colors[:len(results_df)])
    ax6.set_xticks(range(len(results_df)))
    ax6.set_xticklabels(results_df['Strategy'], rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Win Rate (%)', fontsize=11)
    ax6.set_title('Trading Accuracy', fontsize=13, fontweight='bold')
    ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df['Win_Rate_%']):
        ax6.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 7. Final Portfolio Value
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.bar(range(len(results_df)), results_df['Final_Value'], color=colors[:len(results_df)])
    ax7.set_xticks(range(len(results_df)))
    ax7.set_xticklabels(results_df['Strategy'], rotation=45, ha='right', fontsize=9)
    ax7.set_ylabel('Final Value (₹)', fontsize=11)
    ax7.set_title('Final Portfolio Value', fontsize=13, fontweight='bold')
    ax7.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax7.legend(fontsize=9)
    ax7.grid(axis='y', alpha=0.3)
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
    
    plt.suptitle(f'Trading Strategies Performance Analysis - {trade_year}', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Save figure
    output_file = os.path.join(output_dir, f'strategy_comparison_{trade_year}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved: {output_file}")
    
    # Create individual strategy charts
    create_individual_charts(portfolios, trade_year, output_dir, colors)


def create_individual_charts(portfolios: Dict, trade_year: int, output_dir: str, colors: List):
    """Create individual detailed charts for each strategy"""
    for i, (strategy_name, portfolio) in enumerate(portfolios.items()):
        if not portfolio.portfolio_values:
            continue
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Portfolio value over time
        df = pd.DataFrame(portfolio.portfolio_values)
        df['date'] = pd.to_datetime(df['date'])
        
        ax1.plot(df['date'], df['value'], linewidth=2, color=colors[i % len(colors)])
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(df['date'], 100000, df['value'], 
                         where=(df['value'] >= 100000), alpha=0.3, color='green', label='Gain')
        ax1.fill_between(df['date'], 100000, df['value'], 
                         where=(df['value'] < 100000), alpha=0.3, color='red', label='Loss')
        
        # Mark trades
        trades_df = pd.DataFrame(portfolio.trades)
        if not trades_df.empty:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'] == 'SELL']
            
            for _, trade in buys.iterrows():
                ax1.axvline(x=trade['date'], color='green', alpha=0.3, linestyle=':')
            for _, trade in sells.iterrows():
                ax1.axvline(x=trade['date'], color='red', alpha=0.3, linestyle=':')
        
        ax1.set_title(f'{strategy_name} - Portfolio Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
        
        # Daily returns
        df['returns'] = df['value'].pct_change() * 100
        ax2.bar(df['date'], df['returns'], color=['green' if x > 0 else 'red' for x in df['returns']], alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Daily Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Daily Return (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{strategy_name.replace(" ", "_")}_{trade_year}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  → {strategy_name} chart saved: {filename}")


def print_comparison_table(results: List[Dict]):
    """Print a formatted comparison table"""
    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON TABLE")
    print("=" * 120)
    
    headers = ["Strategy", "Return %", "Sharpe", "Max DD %", "Trades", "Win Rate %", "Final Value", "Risk Check"]
    
    print(f"{headers[0]:<35} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>8} {headers[5]:>10} {headers[6]:>15} {headers[7]:>15}")
    print("-" * 120)
    
    for r in results:
        meets_sharpe = r['Sharpe_Ratio'] >= 1.0
        meets_dd = r['Max_Drawdown_%'] <= 10.0
        risk_check = "✓ PASS" if (meets_sharpe and meets_dd) else "✗ FAIL"
        
        print(f"{r['Strategy']:<35} {r['Total_Return_%']:>10.2f} {r['Sharpe_Ratio']:>10.3f} "
              f"{r['Max_Drawdown_%']:>10.2f} {r['Num_Trades']:>8} {r['Win_Rate_%']:>10.1f} "
              f"₹{r['Final_Value']:>13,.0f} {risk_check:>15}")
    
    print("=" * 120)


def save_detailed_report(strategy_name: str, portfolio: PortfolioManager, output_dir: str = "reports"):
    """Save detailed trade and portfolio reports to CSV files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if portfolio.trades:
        trades_df = pd.DataFrame(portfolio.trades)
        trades_file = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_trades.csv")
        trades_df.to_csv(trades_file, index=False)
    
    if portfolio.portfolio_values:
        portfolio_df = pd.DataFrame(portfolio.portfolio_values)
        portfolio_df['return_%'] = portfolio_df['value'].pct_change() * 100
        portfolio_file = os.path.join(output_dir, f"{strategy_name.replace(' ', '_')}_portfolio.csv")
        portfolio_df.to_csv(portfolio_file, index=False)


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("\n" + "=" * 80)
        print("NIFTY 50 TRADING STRATEGIES BACKTESTER")
        print("=" * 80)
        print("\nUsage: python nifty_backtest.py <csv_file_path> [trade_year]")
        print("\nExample: python nifty_backtest.py nifty50_data.csv 2025")
        print("\nCSV Requirements:")
        print("  - Columns: Date, Open, High, Low, Close, Volume (case-insensitive)")
        print("  - Date format: YYYY-MM-DD or DD/MM/YYYY")
        print("  - At least 252 trading days of historical data recommended")
        print("\n" + "=" * 80 + "\n")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    trade_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    
    print("\n" + "=" * 80)
    print("NIFTY 50 TRADING STRATEGIES BACKTESTER")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"  • Initial Capital: ₹100,000")
    print(f"  • Trading Period: {trade_year}")
    print(f"  • Risk Constraints: Sharpe Ratio > 1.0, Max Drawdown < 10%")
    print(f"  • CSV File: {csv_file}\n")
    
    try:
        nifty_data = load_csv_data(csv_file)
    except Exception as e:
        print(f"\n❌ Error loading CSV file: {e}\n")
        sys.exit(1)
    
    strategies = [
        BuyAndHold(),
        ValueInvesting(),
        MomentumInvesting(),
        SwingTrading(),
        SupportResistance()
    ]
    
    print("=" * 80)
    print("RUNNING BACKTESTS...")
    print("=" * 80)
    
    results = []
    portfolios = {}
    
    for strategy in strategies:
        print(f"\n⚙️  Testing: {strategy.name}")
        metrics, portfolio = backtest_strategy(strategy, nifty_data, trade_year=trade_year)
        
        if metrics and portfolio:
            results.append({
                'Strategy': strategy.name,
                'Total_Return_%': round(metrics['total_return']*100, 2),
                'Sharpe_Ratio': round(metrics['sharpe_ratio'], 3),
                'Max_Drawdown_%': round(metrics['max_drawdown']*100, 2),
                'Num_Trades': metrics['num_trades'],
                'Win_Rate_%': round(metrics['win_rate'], 1),
                'Final_Value': round(metrics['final_value'], 2)
            })
            
            portfolios[strategy.name] = portfolio
            
            print(f"  ✓ Completed - Return: {metrics['total_return']*100:.2f}%, "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f}, Trades: {metrics['num_trades']}")
            
            save_detailed_report(strategy.name, portfolio)
    
    if not results:
        print(f"\n❌ No results generated. Check if data exists for year {trade_year}.\n")
        sys.exit(1)
    
    print_comparison_table(results)
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['Total_Return_%'].idxmax()
    worst_idx = results_df['Total_Return_%'].idxmin()
    
    print(f"\n🏆 BEST PERFORMER: {results_df.iloc[best_idx]['Strategy']}")
    print(f"   Return: {results_df.iloc[best_idx]['Total_Return_%']:.2f}% | "
          f"Sharpe: {results_df.iloc[best_idx]['Sharpe_Ratio']:.3f} | "
          f"Trades: {results_df.iloc[best_idx]['Num_Trades']}")
    
    print(f"\n📉 WORST PERFORMER: {results_df.iloc[worst_idx]['Strategy']}")
    print(f"   Return: {results_df.iloc[worst_idx]['Total_Return_%']:.2f}% | "
          f"Sharpe: {results_df.iloc[worst_idx]['Sharpe_Ratio']:.3f} | "
          f"Trades: {results_df.iloc[worst_idx]['Num_Trades']}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    create_visualizations(results_df, portfolios, trade_year)
    
    print("\n" + "=" * 80)
    print(f"✓ All reports and charts saved to 'reports/' directory")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()