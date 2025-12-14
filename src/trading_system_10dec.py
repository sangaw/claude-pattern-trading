"""
COMPLETE MULTI-PATTERN TRADING SYSTEM
Single file - No imports needed - Just run it!

This file contains everything: Pattern Detection, ML, Portfolio Management, and Dashboard
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

# Optional deep RL algorithms (only used if installed)
try:
    from stable_baselines3 import PPO, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

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
# PART 4: DASHBOARD
# ============================================================================

class TradingDashboard:
    """Create visual dashboard"""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
    
    def create_dashboard(self, save_path='trading_dashboard.png'):
        """Generate dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('TRADING SYSTEM DASHBOARD', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        if self.portfolio.daily_equity:
            equity_df = pd.DataFrame(self.portfolio.daily_equity)
            axes[0, 0].plot(equity_df['date'], equity_df['equity'], linewidth=2)
            axes[0, 0].axhline(self.portfolio.initial_capital, color='gray', linestyle='--')
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Equity ($)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Trade Distribution
        if self.portfolio.closed_positions:
            closed_df = pd.DataFrame(self.portfolio.closed_positions)
            returns = closed_df['pnl'].values
            colors = ['green' if x > 0 else 'red' for x in returns]
            axes[0, 1].bar(range(len(returns)), returns, color=colors, alpha=0.6)
            axes[0, 1].axhline(0, color='black', linewidth=0.5)
            axes[0, 1].set_title('Trade P&L Distribution')
            axes[0, 1].set_ylabel('P&L ($)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance Metrics
        metrics = self.portfolio.get_metrics()
        axes[1, 0].axis('off')
        data = [
            ['Total Trades', f"{metrics.get('total_trades', 0)}"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1f}%"],
            ['Total P&L', f"${metrics.get('total_pnl', 0):,.0f}"],
            ['Total Return', f"{metrics.get('total_return', 0):.2f}%"],
            ['Avg Win', f"${metrics.get('avg_win', 0):,.0f}"],
            ['Avg Loss', f"${metrics.get('avg_loss', 0):,.0f}"],
        ]
        table = axes[1, 0].table(cellText=data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 0].set_title('Performance Metrics', fontweight='bold')
        
        # 4. Pattern Performance
        if self.portfolio.closed_positions:
            closed_df = pd.DataFrame(self.portfolio.closed_positions)
            pattern_stats = closed_df.groupby('pattern_type').agg({
                'pnl': ['count', lambda x: (x > 0).sum()]
            }).reset_index()
            pattern_stats.columns = ['Pattern', 'Total', 'Wins']
            pattern_stats['WinRate'] = pattern_stats['Wins'] / pattern_stats['Total'] * 100
            
            colors = ['green' if x > 50 else 'red' for x in pattern_stats['WinRate']]
            axes[1, 1].barh(pattern_stats['Pattern'], pattern_stats['WinRate'], color=colors, alpha=0.6)
            axes[1, 1].axvline(50, color='gray', linestyle='--')
            axes[1, 1].set_xlabel('Win Rate (%)')
            axes[1, 1].set_title('Win Rate by Pattern')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Dashboard saved as '{save_path}'")
        plt.show()
        return fig


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def generate_sample_data():
    """Generate sample data for testing"""
    print("Generating sample data...")
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


def run_complete_system(df=None, use_sample_data=True):
    """Run the complete trading system"""
    
    print("="*80)
    print("MULTI-PATTERN TRADING SYSTEM")
    print("="*80)
    
    # Step 1: Load Data
    if df is None:
        if use_sample_data:
            df = generate_sample_data()
        else:
            csv_file = input("Enter path to your CSV file: ")
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n✓ Loaded {len(df)} bars of data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: Detect Patterns
    print("\n[1/5] Detecting patterns...")
    detector_dt = DoubleTopDetector(df)
    patterns_dt = detector_dt.detect_patterns()
    
    detector_db = DoubleBottomDetector(df)
    patterns_db = detector_db.detect_patterns()
    
    all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)
    print(f"  ✓ Double Top: {len(patterns_dt)}")
    print(f"  ✓ Double Bottom: {len(patterns_db)}")
    print(f"  ✓ Total: {len(all_patterns)}")
    
    if all_patterns.empty:
        print("\n⚠ No patterns detected. Try different data or adjust parameters.")
        return None
    
    # Step 3: Label and Train ML
    print("\n[2/5] Training ML models...")
    labeler = PatternLabeler(df, all_patterns)
    labeled_patterns = labeler.label_patterns()
    
    print(f"  Success rate by pattern:")
    for ptype in labeled_patterns['Pattern_Type'].unique():
        success_rate = labeled_patterns[labeled_patterns['Pattern_Type']==ptype]['Success'].mean()
        print(f"    {ptype}: {success_rate:.1%}")
    
    scorer = PatternQualityScorer()
    scorer.train(labeled_patterns)
    
    # Step 4: Score Patterns
    print("\n[3/5] Scoring pattern quality...")
    scored_patterns = scorer.predict_quality(all_patterns)
    high_quality = scored_patterns[scored_patterns['Quality_Score'] >= 60]
    print(f"  ✓ High quality patterns (score >= 60): {len(high_quality)}")

    # Step 4b: RL gating (DDQN/DQN) to accept/reject patterns
    rl_accepted = high_quality.copy()
    if SB3_AVAILABLE and RL_AVAILABLE and not high_quality.empty:
        print("\n[RL] Training DDQN (DQN with Double-Q) for trade gating...")
        try:
            rl_df = BasePatternDetector(df).df  # ensure ATR/MACD/RSI exist
            env = PatternTradeEnv(rl_df, high_quality)
            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                buffer_size=5000,
                learning_rate=1e-3,
                batch_size=64,
                train_freq=32,
                learning_starts=100,
                target_update_interval=200,
                exploration_fraction=0.2,
                exploration_final_eps=0.05,
            )  # SB3 DQN uses Double DQN by default
            model.learn(total_timesteps=min(5000, 200 * len(high_quality)))

            from rl_interpretability import RLInterpretabilityReport

            feature_names = [
                'Price_Change', 'RSI', 'MACD', 'MACD_Signal',
                'ATR', 'Position', 'Pattern_Signal'
            ]

            # Generate comprehensive report
            interpreter = RLInterpretabilityReport(model, env, feature_names)
            interpreter.generate_full_report()
            
            # Use the trained policy to filter trades
            accepted_rows = []
            for _, p in high_quality.iterrows():
                obs = env.pattern_to_obs(p)
                action, _ = model.predict(obs, deterministic=True)
                if int(action) == 1:
                    accepted_rows.append(p)
            rl_accepted = pd.DataFrame(accepted_rows)
            print(f"[RL] Accepted {len(rl_accepted)} of {len(high_quality)} high-quality patterns.")
            if rl_accepted.empty:
                rl_accepted = high_quality
                print("[RL] No patterns accepted; falling back to XGBoost filter only.")
        except Exception as e:
            print(f"[RL] Skipped due to error: {e}")
            rl_accepted = high_quality
    else:
        if high_quality.empty:
            print("[RL] No patterns to train on; skipping RL gate.")
        else:
            print("[RL] Skipped (missing gymnasium/stable-baselines3).")
    
    # Step 5: Portfolio Backtest (filtered by RL if available)
    print("\n[4/5] Running portfolio backtest...")
    portfolio = PortfolioRiskManager(initial_capital=100000, max_positions=5, risk_per_trade=0.02)
    
    trades_executed = 0
    for _, pattern in rl_accepted.iterrows():
        eval_result = portfolio.evaluate_new_trade(pattern)
        
        if eval_result['approved']:
            entry_price = pattern['Neckline']
            portfolio.open_position(pattern, eval_result, entry_price)
            trades_executed += 1
            
            # Simulate exit using actual OHLC path up to 30 bars
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
    
    print(f"  ✓ Executed {trades_executed} trades")
    
    # Step 6: Show Results
    metrics = portfolio.get_metrics()
    
    print("\n[5/5] RESULTS:")
    print("="*80)
    print(f"Total Trades        : {metrics['total_trades']}")
    print(f"Winning Trades      : {metrics['winning_trades']}")
    print(f"Win Rate            : {metrics['win_rate']:.1f}%")
    print(f"Total P&L           : ${metrics['total_pnl']:,.2f}")
    print(f"Total Return        : {metrics['total_return']:.2f}%")
    print(f"Final Capital       : ${metrics['final_capital']:,.2f}")
    print("="*80)
    
    # Step 7: Create Dashboard
    print("\n[6/6] Generating dashboard...")
    dashboard = TradingDashboard(portfolio)
    dashboard.create_dashboard()
    
    return portfolio, scored_patterns


# ============================================================================
# PART 6: OPTIONAL RL TRADING DEMO
# ============================================================================

class SimpleTradingEnv(gym.Env if RL_AVAILABLE else object):
    """
    Minimal long/flat trading environment for RL demos.
    Observation: [pct_change, RSI, MACD, MACD_Signal, ATR_norm, position_flag]
    Actions: 0=hold, 1=go/keep long, 2=flatten.
    Reward: change in equity driven by price moves while positioned.
    """
    def __init__(self, df, max_steps=300):
        if not RL_AVAILABLE:
            raise RuntimeError("gymnasium not installed; cannot build RL environment.")
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self.reset()

    def _get_obs(self):
        row = self.df.loc[self.idx]
        prev_close = self.df.loc[self.idx - 1, 'Close']
        pct_change = (row['Close'] - prev_close) / prev_close
        atr_norm = row['ATR'] / row['Close'] if row['Close'] != 0 else 0
        return np.array([
            pct_change,
            row.get('RSI', 50) / 100,
            row.get('MACD', 0),
            row.get('MACD_Signal', 0),
            atr_norm,
            float(self.position)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start away from the very beginning to ensure indicators are filled
        self.idx = 60
        self.steps = 0
        self.position = 0  # 0=flat, 1=long
        self.entry_price = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps or self.idx >= len(self.df) - 2

        # Price movement
        prev_close = self.df.loc[self.idx - 1, 'Close']
        curr_close = self.df.loc[self.idx, 'Close']
        reward = 0.0

        # Execute action
        if action == 1:  # go/keep long
            if self.position == 0:
                self.position = 1
                self.entry_price = curr_close
        elif action == 2:  # flatten
            if self.position == 1:
                reward = (curr_close - self.entry_price) / self.entry_price
            self.position = 0
            self.entry_price = 0

        # Mark-to-market reward while in position
        if self.position == 1:
            reward += (curr_close - prev_close) / prev_close

        # Advance index
        self.idx += 1
        obs = self._get_obs()

        return obs, float(reward), terminated, truncated, {}


def run_rl_demo(df, timesteps=2000):
    """
    Train a tiny PPO agent on the simple trading environment.
    Safe to skip if RL deps are missing.
    """
    if not RL_AVAILABLE:
        print("RL demo skipped: gymnasium not installed.")
        return None
    if not SB3_AVAILABLE:
        print("RL demo skipped: stable-baselines3 not installed.")
        return None

    print("\n[RL] Training PPO on SimpleTradingEnv...")
    env = SimpleTradingEnv(df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    obs, _ = env.reset()
    cumulative_reward = 0.0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward
        if terminated or truncated:
            break

    print(f"[RL] Demo cumulative reward over 200 steps: {cumulative_reward:.4f}")
    return model


class PatternTradeEnv(gym.Env if RL_AVAILABLE else object):
    """
    Environment where the agent decides to accept/reject patterns.
    Obs: normalized pattern features; Action: 0=reject, 1=accept.
    Reward: realized P&L from deterministic stop/target simulation (scaled).
    """
    FEATURE_COLS = ['Quality_Score', 'Pattern_Height', 'RSI', 'ATR', 'Volatility', 'Trend']

    def __init__(self, df, patterns_df):
        if not RL_AVAILABLE:
            raise RuntimeError("gymnasium not installed; cannot build RL environment.")
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.patterns = patterns_df.reset_index(drop=True)
        self.ptr = 0
        self.max_steps = len(self.patterns)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(len(self.FEATURE_COLS),), dtype=np.float32)

    def pattern_to_obs(self, pattern_row):
        vals = []
        for col in self.FEATURE_COLS:
            v = pattern_row.get(col, 0)
            if col == 'Quality_Score':
                v = v / 100.0
            vals.append(float(v) if pd.notna(v) else 0.0)
        return np.array(vals, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = 0
        if self.max_steps == 0:
            return np.zeros(len(self.FEATURE_COLS), dtype=np.float32), {}
        obs = self.pattern_to_obs(self.patterns.loc[self.ptr])
        return obs, {}

    def step(self, action):
        if self.ptr >= self.max_steps:
            return np.zeros(len(self.FEATURE_COLS), dtype=np.float32), 0.0, True, True, {}

        pattern = self.patterns.loc[self.ptr]
        reward = 0.0

        if int(action) == 1:
            # Simulate deterministic exit
            pnl = self._simulate_pnl(pattern)
            reward = pnl / max(1.0, pattern.get('Neckline', 1.0))  # scale

        self.ptr += 1
        terminated = self.ptr >= self.max_steps
        truncated = terminated
        if terminated:
            obs = np.zeros(len(self.FEATURE_COLS), dtype=np.float32)
        else:
            obs = self.pattern_to_obs(self.patterns.loc[self.ptr])

        return obs, float(reward), terminated, truncated, {}

    def _simulate_pnl(self, pattern):
        idx = int(pattern['Detection_Index'])
        if idx >= len(self.df):
            return 0.0
        is_bearish = pattern['Pattern_Type'] == 'DoubleTop'
        stop_loss = pattern.get('Stop_Loss')
        target = pattern.get('Target')
        entry = pattern.get('Neckline', self.df.loc[idx, 'Close'])

        exit_price = entry
        for i in range(idx + 1, min(idx + 30, len(self.df))):
            high = self.df.loc[i, 'High']
            low = self.df.loc[i, 'Low']
            if is_bearish:
                if high >= stop_loss:
                    exit_price = stop_loss
                    break
                if low <= target:
                    exit_price = target
                    break
            else:
                if low <= stop_loss:
                    exit_price = stop_loss
                    break
                if high >= target:
                    exit_price = target
                    break
        # Fallback to expiry
        if exit_price == entry:
            exit_price = self.df.loc[min(idx + 29, len(self.df) - 1), 'Close']

        pnl = (exit_price - entry)
        if is_bearish:
            pnl = -pnl
        return pnl


# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHOOSE AN OPTION:")
    print("="*80)
    print("1. Quick Demo (sample data)")
    print("2. Use my own CSV file")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\nStarting Quick Demo...")
        result = run_complete_system(use_sample_data=True)
        
    elif choice == '2':
        csv_file = input("\nEnter CSV file path: ").strip()
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} rows from {csv_file}")
            result = run_complete_system(df=df, use_sample_data=False)
        except Exception as e:
            print(f"\n✗ Error loading file: {e}")
            print("Make sure CSV has columns: date, Open, High, Low, Close, Volume")
    
    elif choice == '3':
        print("\nExiting...")
    
    else:
        print("\nInvalid choice. Please run again.")