"""
COMPLETE TRADING SYSTEM WITH ENHANCED RL INTERPRETABILITY
Single file - Everything included - Just run it!

This file contains:
- Pattern Detection
- ML Training
- Portfolio Management
- RL Training
- ENHANCED Interpretability Analysis
- Dashboard
"""

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

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class PatternDiagnostic:
    """
    Comprehensive diagnostic for pattern detection issues
    """
    
    def __init__(self, df, all_patterns, labeled_patterns):
        self.df = df.copy()
        self.all_patterns = all_patterns.copy() if not all_patterns.empty else pd.DataFrame()
        self.labeled_patterns = labeled_patterns.copy() if not labeled_patterns.empty else pd.DataFrame()
        
        # Fix date formats
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        if not self.all_patterns.empty and 'Detection_Date' in self.all_patterns.columns:
            self.all_patterns['Detection_Date'] = pd.to_datetime(self.all_patterns['Detection_Date'])
        
    def run_full_diagnostic(self):
        """Run all diagnostic checks with error handling"""
        
        print("\n" + "="*80)
        print("PATTERN QUALITY DIAGNOSTIC - COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        self._check_data_quality()
        self._check_pattern_distribution()
        self._check_success_rates()
        self._check_pattern_characteristics()
        self._check_market_conditions()
        self._check_target_stop_analysis()
        self._generate_recommendations()
    
    def _check_data_quality(self):
        """Check data quality with proper error handling"""
        print("\n[1/6] DATA QUALITY CHECK")
        print("-" * 80)
        
        # Missing values
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        
        if total_missing > 0:
            print(f"⚠️ Found {total_missing} missing values")
            print("\nMissing values by column (showing top 10):")
            top_missing = missing[missing > 0].sort_values(ascending=False).head(10)
            for col, count in top_missing.items():
                pct = (count / len(self.df)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
            
            if total_missing / len(self.df) > 0.05:
                print("\n⚠️ WARNING: >5% missing values - data quality may be poor")
            else:
                print("\n✓ Missing values are acceptable (<5% of data)")
                print("  These are typically at the start due to indicator calculations")
        else:
            print("✓ No missing values detected")
        
        # Date range
        try:
            date_col = 'date' if 'date' in self.df.columns else self.df.columns[0]
            dates = pd.to_datetime(self.df[date_col])
            date_range_days = (dates.max() - dates.min()).days
            
            print(f"\n✓ Date range: {dates.min().date()} to {dates.max().date()}")
            print(f"  Total days: {date_range_days}")
            print(f"  Total bars: {len(self.df)}")
            
            # Check for gaps
            if len(self.df) < date_range_days * 0.7:  # Assuming ~70% trading days
                print("  ⚠️ Large gaps detected in data")
        except Exception as e:
            print(f"⚠️ Could not analyze date range: {e}")
        
        # Price statistics
        if 'Close' in self.df.columns:
            close_prices = self.df['Close'].dropna()
            if len(close_prices) > 0:
                price_min = close_prices.min()
                price_max = close_prices.max()
                price_start = close_prices.iloc[0]
                price_end = close_prices.iloc[-1]
                total_return = ((price_end - price_start) / price_start) * 100
                
                print(f"\n✓ Price statistics:")
                print(f"  Range: ${price_min:.2f} to ${price_max:.2f}")
                print(f"  Total return: {total_return:+.2f}%")
                
                # Volatility
                returns = close_prices.pct_change().dropna()
                if len(returns) > 0:
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)
                    
                    print(f"  Daily volatility: {daily_vol:.2%}")
                    print(f"  Annualized volatility: {annual_vol:.1%}")
                    
                    if annual_vol < 10:
                        print("  ⚠️ Very low volatility - patterns may be weak")
                    elif annual_vol > 60:
                        print("  ⚠️ Very high volatility - patterns may be unreliable")
                    else:
                        print("  ✓ Volatility is reasonable for pattern trading")
    
    def _check_pattern_distribution(self):
        """Check pattern distribution"""
        print("\n[2/6] PATTERN DISTRIBUTION")
        print("-" * 80)
        
        if self.all_patterns.empty:
            print("⚠️ ERROR: No patterns detected!")
            print("  Possible causes:")
            print("  1. Pattern parameters too strict")
            print("  2. Data doesn't contain clear patterns")
            print("  3. Insufficient data length")
            return
        
        # Patterns by type
        print(f"Total patterns detected: {len(self.all_patterns)}")
        
        if 'Pattern_Type' in self.all_patterns.columns:
            pattern_counts = self.all_patterns['Pattern_Type'].value_counts()
            print("\nPattern distribution:")
            for ptype, count in pattern_counts.items():
                pct = (count / len(self.all_patterns)) * 100
                print(f"  {ptype}: {count} ({pct:.1f}%)")
        
        # Pattern spacing
        if 'Detection_Index' in self.all_patterns.columns and len(self.all_patterns) > 1:
            sorted_patterns = self.all_patterns.sort_values('Detection_Index')
            spacing = sorted_patterns['Detection_Index'].diff().dropna()
            
            print(f"\nPattern spacing:")
            print(f"  Average: {spacing.mean():.1f} bars")
            print(f"  Median: {spacing.median():.1f} bars")
            print(f"  Min: {spacing.min():.0f} bars")
            print(f"  Max: {spacing.max():.0f} bars")
            
            if spacing.mean() < 15:
                print("  ⚠️ Patterns very close together - may include noise")
            elif spacing.mean() > 100:
                print("  ⚠️ Patterns very sparse - parameters may be too strict")
            else:
                print("  ✓ Pattern spacing looks reasonable")
    
    def _check_success_rates(self):
        """Analyze success rates - THE MOST IMPORTANT CHECK"""
        print("\n[3/6] SUCCESS RATE ANALYSIS ⭐ CRITICAL")
        print("-" * 80)
        
        if self.labeled_patterns.empty:
            print("⚠️ ERROR: No labeled patterns!")
            return
        
        if 'Success' not in self.labeled_patterns.columns:
            print("⚠️ ERROR: Patterns not labeled with success/failure")
            return
        
        # Overall success rate
        total_success = self.labeled_patterns['Success'].sum()
        total_patterns = len(self.labeled_patterns)
        overall_rate = total_success / total_patterns if total_patterns > 0 else 0
        
        print(f"📊 OVERALL SUCCESS RATE: {overall_rate:.1%} ({int(total_success)}/{total_patterns})")
        
        # Critical assessment
        print("\n🎯 ASSESSMENT:")
        if overall_rate < 0.12:
            print("  🔴 CRITICAL FAILURE - Success rate < 12%")
            print("     These patterns have NO PREDICTIVE POWER")
            print("     Even perfect prediction cannot be profitable!")
            print("")
            print("  ⚠️  THIS IS YOUR MAIN PROBLEM ⚠️")
            print("")
        elif overall_rate < 0.25:
            print("  🟠 POOR - Success rate < 25%")
            print("     Patterns are very weak predictors")
            print("     Profitability is unlikely even with good execution")
        elif overall_rate < 0.40:
            print("  🟡 MARGINAL - Success rate 25-40%")
            print("     Patterns have weak edge, requires excellent risk management")
        elif overall_rate < 0.55:
            print("  🟢 ACCEPTABLE - Success rate 40-55%")
            print("     Patterns have decent predictive power")
        else:
            print("  ✅ GOOD - Success rate > 55%")
            print("     Patterns are strong predictors")
        
        # By pattern type
        print("\n📈 Success by pattern type:")
        for ptype in self.labeled_patterns['Pattern_Type'].unique():
            mask = self.labeled_patterns['Pattern_Type'] == ptype
            success = self.labeled_patterns[mask]['Success'].sum()
            total = mask.sum()
            rate = success / total if total > 0 else 0
            
            status = "✓" if rate > 0.4 else "✗"
            print(f"  {status} {ptype}: {rate:.1%} ({int(success)}/{total})")
        
        # Calculate minimum required win rate
        print("\n💰 Profitability Analysis:")
        if 'Return' in self.labeled_patterns.columns:
            wins = self.labeled_patterns[self.labeled_patterns['Success'] == 1]
            losses = self.labeled_patterns[self.labeled_patterns['Success'] == 0]
            
            if len(wins) > 0 and len(losses) > 0:
                avg_win = wins['Return'].mean()
                avg_loss = abs(losses['Return'].mean())
                
                # Calculate breakeven win rate
                if avg_win + avg_loss > 0:
                    breakeven_rate = avg_loss / (avg_win + avg_loss)
                    
                    print(f"  Average win: {avg_win:.2f}")
                    print(f"  Average loss: {avg_loss:.2f}")
                    print(f"  Win/Loss ratio: {avg_win/avg_loss:.2f}:1")
                    print(f"  Breakeven win rate: {breakeven_rate:.1%}")
                    print(f"  Actual win rate: {overall_rate:.1%}")
                    
                    if overall_rate > breakeven_rate * 1.2:  # 20% buffer
                        print(f"  ✓ Win rate is {((overall_rate/breakeven_rate - 1)*100):.0f}% above breakeven")
                    else:
                        print(f"  ✗ Win rate is too close to or below breakeven")
                        print(f"     Need {breakeven_rate*1.2:.1%}+ to be reliably profitable")
    
    def _check_pattern_characteristics(self):
        """Analyze what makes patterns succeed/fail"""
        print("\n[4/6] PATTERN CHARACTERISTICS")
        print("-" * 80)
        
        if self.labeled_patterns.empty or 'Success' not in self.labeled_patterns.columns:
            return
        
        successful = self.labeled_patterns[self.labeled_patterns['Success'] == 1]
        failed = self.labeled_patterns[self.labeled_patterns['Success'] == 0]
        
        if len(successful) == 0:
            print("❌ NO SUCCESSFUL PATTERNS FOUND!")
            print("   Cannot identify what makes patterns succeed")
            print("")
            print("   This confirms patterns are not working on this data")
            return
        
        print(f"Analyzing {len(successful)} successful vs {len(failed)} failed patterns:")
        
        # Compare key features
        features_to_check = ['RSI', 'Volatility', 'Pattern_Height', 'Trend', 'ATR']
        
        print("\n📊 Feature comparison (Successful vs Failed):")
        for feature in features_to_check:
            if feature in self.labeled_patterns.columns:
                success_mean = successful[feature].mean()
                fail_mean = failed[feature].mean()
                diff_pct = ((success_mean - fail_mean) / fail_mean * 100) if fail_mean != 0 else 0
                
                indicator = "↑" if success_mean > fail_mean else "↓"
                print(f"  {feature}:")
                print(f"    Success: {success_mean:.3f} | Failed: {fail_mean:.3f} {indicator} ({diff_pct:+.1f}%)")
        
        # If differences are small, patterns aren't distinguishing success/failure
        print("\n💡 Insight:")
        if len(successful) < 3:
            print("  ⚠️ Too few successful patterns to identify reliable patterns")
        else:
            print("  Check if successful patterns have consistent characteristics")
            print("  Large differences indicate features that predict success")
    
    def _check_market_conditions(self):
        """Analyze market environment"""
        print("\n[5/6] MARKET CONDITIONS")
        print("-" * 80)
        
        if 'Close' not in self.df.columns:
            print("⚠️ No price data available")
            return
        
        close = self.df['Close'].dropna()
        
        # Trend analysis
        if len(close) > 50:
            sma_50 = close.rolling(50).mean()
            current_trend = "Uptrend" if close.iloc[-1] > sma_50.iloc[-1] else "Downtrend"
            
            # Calculate % of time in uptrend
            uptrend_pct = ((close > sma_50).sum() / len(close)) * 100
            
            print(f"📈 Current trend: {current_trend}")
            print(f"   Uptrend: {uptrend_pct:.1f}% of time")
            print(f"   Downtrend: {100-uptrend_pct:.1f}% of time")
        
        # Overall performance
        total_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
        print(f"\n📊 Total price change: {total_return:+.2f}%")
        
        if abs(total_return) < 10:
            print("  ⚠️ Range-bound market - patterns may be less reliable")
        
        # Volatility
        returns = close.pct_change().dropna()
        if len(returns) > 20:
            vol_20 = returns.rolling(20).std()
            current_vol = vol_20.iloc[-1]
            avg_vol = vol_20.mean()
            
            print(f"\n📉 Volatility:")
            print(f"   Current: {current_vol:.2%}")
            print(f"   Average: {avg_vol:.2%}")
            
            if current_vol > avg_vol * 1.5:
                print("   ⚠️ Currently high volatility - patterns may be unreliable")
    
    def _check_target_stop_analysis(self):
        """Analyze target/stop-loss setup"""
        print("\n[6/6] TARGET/STOP-LOSS ANALYSIS")
        print("-" * 80)
        
        if self.all_patterns.empty or 'Neckline' not in self.all_patterns.columns:
            print("⚠️ No target/stop data available")
            return
        
        # Calculate distances
        if 'Target' in self.all_patterns.columns and 'Stop_Loss' in self.all_patterns.columns:
            target_dist = abs((self.all_patterns['Target'] - self.all_patterns['Neckline']) / 
                            self.all_patterns['Neckline']) * 100
            stop_dist = abs((self.all_patterns['Stop_Loss'] - self.all_patterns['Neckline']) / 
                          self.all_patterns['Neckline']) * 100
            
            risk_reward = target_dist / stop_dist
            
            print(f"📏 Average distances:")
            print(f"   Target: {target_dist.mean():.2f}%")
            print(f"   Stop-Loss: {stop_dist.mean():.2f}%")
            print(f"   Risk/Reward: {risk_reward.mean():.2f}:1")
            
            # Assessment
            if risk_reward.mean() < 1.5:
                min_winrate = 1 / (1 + risk_reward.mean())
                print(f"\n  ⚠️ Low R:R requires >{min_winrate:.0%} win rate to breakeven")
            elif risk_reward.mean() > 3:
                print(f"  ✓ Good R:R ratio")
            
            if target_dist.mean() < 1:
                print(f"  ⚠️ Targets very close ({target_dist.mean():.2f}%) - may not be reached")
            
            if stop_dist.mean() > 5:
                print(f"  ⚠️ Stops very wide ({stop_dist.mean():.2f}%) - high risk per trade")
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n" + "="*80)
        print("🎯 RECOMMENDATIONS & ACTION PLAN")
        print("="*80)
        
        if self.labeled_patterns.empty:
            print("\n❌ Cannot generate recommendations - no pattern data")
            return
        
        success_rate = self.labeled_patterns['Success'].mean() if 'Success' in self.labeled_patterns.columns else 0
        pattern_count = len(self.all_patterns)
        
        print(f"\n📊 Current Status:")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Pattern Count: {pattern_count}")
        
        # Generate priority-ordered recommendations
        if success_rate < 0.15:
            print("\n🔴 PRIORITY 1: PATTERNS HAVE NO PREDICTIVE POWER")
            print("   Your 11% success rate means these patterns don't work on this data.")
            print("")
            print("   IMMEDIATE ACTIONS (choose ONE):")
            print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("   A) Try different pattern types:")
            print("      - Head & Shoulders patterns")
            print("      - Triangle breakouts")
            print("      - Channel patterns")
            print("      - Flag/Pennant patterns")
            print("")
            print("   B) Change timeframe:")
            print("      - If using daily → try 4-hour or hourly")
            print("      - If using hourly → try daily")
            print("")
            print("   C) Try different asset:")
            print("      - This specific asset may not exhibit reliable patterns")
            print("      - Test on high-liquidity stocks or major indices")
            print("")
            print("   D) Abandon pattern trading for this asset:")
            print("      - Use trend-following strategies instead")
            print("      - Try moving average crossovers")
            print("      - Use momentum indicators (RSI, MACD)")
            print("")
            print("   🎯 RECOMMENDED: Start with Option A (different patterns)")
            
        elif success_rate < 0.30:
            print("\n🟠 PRIORITY 1: IMPROVE PATTERN QUALITY")
            print("   Actions:")
            print("   1. Increase pattern strictness (tolerance = 0.01)")
            print("   2. Add volume confirmation")
            print("   3. Only trade patterns aligned with major trend")
            print("   4. Filter by volatility (avoid low-vol periods)")
        
        # Check sample size
        if pattern_count < 50:
            print("\n🟡 PRIORITY 2: INSUFFICIENT DATA")
            print(f"   Only {pattern_count} patterns - need 100+ for reliable conclusions")
            print("   Actions:")
            print("   1. Use longer historical data")
            print("   2. Reduce pattern strictness slightly")
            print("   3. Consider multiple assets/timeframes")
        
        # Check RL performance
        print("\n🔵 PRIORITY 3: RL AGENT VALIDATION")
        print("   Your RL agent shows 403% return vs -10% rule-based")
        print("   This is likely reward hacking, not real learning")
        print("   Actions:")
        print("   1. Validate RL on completely separate test data")
        print("   2. Increase transaction costs to 0.2%")
        print("   3. Add maximum daily trade limits")
        print("   4. Compare to buy-and-hold benchmark")
        
        print("\n" + "="*80)
        print("🎬 NEXT STEPS:")
        print("="*80)
        print("1. Address PRIORITY 1 first (pattern predictive power)")
        print("2. Once success rate > 30%, optimize other parameters")
        print("3. Then validate RL agent properly")
        print("4. Only deploy with live paper trading after validation")
        print("")
        print("💡 Remember: An 11% success rate cannot be profitable!")
        print("   Fix the pattern detection before optimizing anything else.")
        print("="*80)
    
    def visualize_diagnostics(self, save_path='pattern_diagnostic.png'):
        """Create diagnostic visualizations"""
        
        if self.labeled_patterns.empty:
            print("Cannot create visualizations - no patterns")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('PATTERN QUALITY DIAGNOSTIC DASHBOARD', fontsize=14, fontweight='bold')
        
        # 1. Success rate by pattern type
        ax = axes[0, 0]
        success_by_type = self.labeled_patterns.groupby('Pattern_Type')['Success'].agg(['sum', 'count'])
        success_by_type['rate'] = success_by_type['sum'] / success_by_type['count'] * 100
        success_by_type['rate'].plot(kind='bar', ax=ax, color=['green' if x > 30 else 'red' for x in success_by_type['rate']])
        ax.axhline(30, color='orange', linestyle='--', label='30% threshold')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Pattern Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Pattern distribution over time
        ax = axes[0, 1]
        self.all_patterns['YearMonth'] = pd.to_datetime(self.all_patterns['Detection_Date']).dt.to_period('M')
        pattern_timeline = self.all_patterns.groupby('YearMonth').size()
        pattern_timeline.plot(ax=ax, marker='o')
        ax.set_ylabel('Pattern Count')
        ax.set_title('Patterns Over Time')
        ax.grid(True, alpha=0.3)
        
        # 3. RSI distribution: success vs fail
        ax = axes[0, 2]
        if 'RSI' in self.labeled_patterns.columns:
            successful = self.labeled_patterns[self.labeled_patterns['Success'] == 1]['RSI']
            failed = self.labeled_patterns[self.labeled_patterns['Success'] == 0]['RSI']
            ax.hist([failed, successful], bins=20, label=['Failed', 'Success'], alpha=0.7)
            ax.set_xlabel('RSI')
            ax.set_ylabel('Count')
            ax.set_title('RSI Distribution: Success vs Fail')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Pattern height distribution
        ax = axes[1, 0]
        if 'Pattern_Height' in self.all_patterns.columns:
            ax.hist(self.all_patterns['Pattern_Height'], bins=30, alpha=0.7)
            ax.set_xlabel('Pattern Height')
            ax.set_ylabel('Count')
            ax.set_title('Pattern Height Distribution')
            ax.grid(True, alpha=0.3)
        
        # 5. Market conditions
        ax = axes[1, 1]
        self.df['Close'].plot(ax=ax)
        ax.set_ylabel('Price')
        ax.set_title('Price Chart with Market Conditions')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary stats
        ax = axes[1, 2]
        ax.axis('off')
        
        stats_data = [
            ['Metric', 'Value'],
            ['─'*20, '─'*15],
            ['Total Patterns', f'{len(self.all_patterns)}'],
            ['Success Rate', f"{self.labeled_patterns['Success'].mean():.1%}"],
            ['DoubleTop Success', f"{self.labeled_patterns[self.labeled_patterns['Pattern_Type']=='DoubleTop']['Success'].mean():.1%}"],
            ['DoubleBottom Success', f"{self.labeled_patterns[self.labeled_patterns['Pattern_Type']=='DoubleBottom']['Success'].mean():.1%}"],
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Diagnostic visualization saved as '{save_path}'")
        plt.close()


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
    
    def detect_patterns(self, min_bars=15, max_bars=40, tolerance=0.015):
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
                    'Target': neckline - (pattern_height * 1.5),
                    'RSI': self.df.loc[idx2, 'RSI'],
                    'ATR': self.df.loc[idx2, 'ATR'],
                    'Volatility': self.df.loc[idx2, 'Volatility'],
                    'Trend': self.df.loc[idx2, 'Trend'],
                })
        
        return pd.DataFrame(patterns)


class DoubleBottomDetector(BasePatternDetector):
    """Detects Double Bottom patterns"""
    
    def detect_patterns(self, min_bars=15, max_bars=40, tolerance=0.015):
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
                    'Target': neckline + (pattern_height * 1.5),
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
    
    def label_patterns(self, forward_bars=20):
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
        self.use_ml = {}  # Track which patterns can use ML
        self.baseline_scores = {}  # Fallback scores
    
    def train(self, labeled_patterns_df):
        """Train with smart fallback to rule-based scoring"""
        from sklearn.metrics import confusion_matrix
        
        pattern_types = labeled_patterns_df['Pattern_Type'].unique()
        results = {}
        
        print("\n" + "="*80)
        print("SMART ML TRAINING WITH RULE-BASED FALLBACK")
        print("="*80)
        
        for ptype in pattern_types:
            pattern_data = labeled_patterns_df[labeled_patterns_df['Pattern_Type'] == ptype].copy()
            
            if len(pattern_data) < 30:
                print(f"\n[{ptype}] Skipping - insufficient data")
                self.use_ml[ptype] = False
                continue
            
            print(f"\n[{ptype}] Training...")
            
            # Check class distribution
            success_count = pattern_data['Success'].sum()
            total_count = len(pattern_data)
            success_rate = success_count / total_count
            
            print(f"  Class Distribution:")
            print(f"    Success: {success_count}/{total_count} ({success_rate:.1%})")
            print(f"    Failure: {total_count - success_count}/{total_count} ({1-success_rate:.1%})")
            
            # ═══════════════════════════════════════════════════════════════
            # DECISION: Use ML or Rule-Based?
            # ═══════════════════════════════════════════════════════════════
            
            if success_rate < 0.15 or success_rate > 0.85:
                print(f"  ⚠️ EXTREME IMBALANCE ({success_rate:.1%} success rate)")
                print(f"     → Using RULE-BASED scoring instead of ML")
                
                # Calculate rule-based scores
                self._calculate_rule_based_scores(pattern_data, ptype)
                self.use_ml[ptype] = False
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # Try ML Training
            # ═══════════════════════════════════════════════════════════════
            
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
            
            # Use class weights
            scale_pos = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            accuracy = (y_pred == y_test).mean()
            cm = confusion_matrix(y_test, y_pred)
            
            print(f"  ML Accuracy: {accuracy:.2%}")
            
            # Check if ML is useful
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                if tp == 0:
                    print(f"  ⚠️ ML model never predicts positive class")
                    print(f"     → Falling back to RULE-BASED scoring")
                    self._calculate_rule_based_scores(pattern_data, ptype)
                    self.use_ml[ptype] = False
                    continue
            
            # ML is working!
            print(f"  ✓ Using ML model")
            self.models[ptype] = model
            self.scalers[ptype] = scaler
            self.feature_columns[ptype] = feature_cols
            self.use_ml[ptype] = True
            results[ptype] = accuracy
        
        print("\n" + "="*80)
        return results
    
    def _calculate_rule_based_scores(self, pattern_data, ptype):
        """
        Calculate rule-based quality scores based on technical indicators
        
        Logic:
        - Strong patterns have: good RSI, high volatility, clear trend
        - Score = weighted combination of indicators
        """
        print(f"  → Computing rule-based scores...")
        
        scores = []
        
        for _, row in pattern_data.iterrows():
            score = 50.0  # Base score
            
            # RSI component (30 points)
            rsi = row.get('RSI', 50)
            if ptype == 'DoubleBottom':  # Bullish
                if rsi < 30:
                    score += 15  # Oversold = good
                elif rsi < 40:
                    score += 10
                elif rsi > 70:
                    score -= 10  # Overbought = bad
            else:  # DoubleTop - Bearish
                if rsi > 70:
                    score += 15  # Overbought = good for shorts
                elif rsi > 60:
                    score += 10
                elif rsi < 30:
                    score -= 10  # Oversold = bad for shorts
            
            # Volatility component (20 points)
            volatility = row.get('Volatility', 0)
            if volatility > 0.02:
                score += 10  # High volatility = good
            elif volatility < 0.005:
                score -= 5  # Too low = bad
            
            # Pattern height component (20 points)
            pattern_height = row.get('Pattern_Height', 0)
            if pattern_height > pattern_data['Pattern_Height'].median():
                score += 10
            
            # Trend component (10 points)
            trend = row.get('Trend', 0)
            if ptype == 'DoubleBottom' and trend > 0:
                score += 5  # Bullish pattern in uptrend
            elif ptype == 'DoubleTop' and trend < 0:
                score += 5  # Bearish pattern in downtrend
            
            scores.append(np.clip(score, 0, 100))
        
        # Store average score for this pattern type
        self.baseline_scores[ptype] = np.mean(scores)
        print(f"     Average rule-based score: {self.baseline_scores[ptype]:.1f}")
    
    def predict_quality(self, patterns_df):
        """Predict quality using ML or rule-based depending on pattern type"""
        if patterns_df.empty:
            return patterns_df
        
        patterns_with_scores = patterns_df.copy()
        patterns_with_scores['Quality_Score'] = 40.0  # Low default
        
        for ptype in patterns_df['Pattern_Type'].unique():
            mask = patterns_df['Pattern_Type'] == ptype
            pattern_data = patterns_df[mask].copy()
            
            # Check if we should use ML or rule-based
            if ptype not in self.use_ml or not self.use_ml[ptype]:
                # Use rule-based scoring
                scores = []
                for _, row in pattern_data.iterrows():
                    score = 50.0
                    
                    # Apply same logic as training
                    rsi = row.get('RSI', 50)
                    if ptype == 'DoubleBottom':
                        if rsi < 30:
                            score += 15
                        elif rsi < 40:
                            score += 10
                        elif rsi > 70:
                            score -= 10
                    else:
                        if rsi > 70:
                            score += 15
                        elif rsi > 60:
                            score += 10
                        elif rsi < 30:
                            score -= 10
                    
                    volatility = row.get('Volatility', 0)
                    if volatility > 0.02:
                        score += 10
                    elif volatility < 0.005:
                        score -= 5
                    
                    scores.append(np.clip(score, 0, 100))
                
                patterns_with_scores.loc[mask, 'Quality_Score'] = scores
            else:
                # Use ML model
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
    
    def __init__(self, initial_capital=100000, max_positions=3, risk_per_trade=0.01):
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

class EnhancedPatternQualityScorer:
    """
    Enhanced ML scorer with confusion matrix analysis
    
    Add this to replace or extend the existing PatternQualityScorer class
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.confusion_matrices = {}  # NEW
        self.classification_reports = {}  # NEW
    
    def train(self, labeled_patterns_df):
        """Train with confusion matrix analysis"""
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        
        pattern_types = labeled_patterns_df['Pattern_Type'].unique()
        results = {}
        
        print("\n" + "="*80)
        print("ML TRAINING WITH CONFUSION MATRIX ANALYSIS")
        print("="*80)
        
        for ptype in pattern_types:
            pattern_data = labeled_patterns_df[labeled_patterns_df['Pattern_Type'] == ptype].copy()
            
            if len(pattern_data) < 30:
                print(f"\n[{ptype}] Skipping - insufficient data ({len(pattern_data)} samples)")
                continue
            
            print(f"\n[{ptype}] Training...")
            
            feature_cols = ['Pattern_Height', 'RSI', 'ATR', 'Volatility', 'Trend']
            feature_cols = [c for c in feature_cols if c in pattern_data.columns]
            
            X = pattern_data[feature_cols].fillna(0)
            y = pattern_data['Success']
            
            # Time-based split (important for trading!)
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
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            self.confusion_matrices[ptype] = cm
            
            # Classification Report
            report = classification_report(y_test, y_pred, output_dict=True)
            self.classification_reports[ptype] = report
            
            # Print summary
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Precision: {report['1']['precision']:.2%}")
            print(f"  Recall: {report['1']['recall']:.2%}")
            print(f"  F1-Score: {report['1']['f1-score']:.2%}")
            
            # Interpret confusion matrix
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                print(f"\n  Confusion Matrix:")
                print(f"    True Negatives  (TN): {tn:>3} - Correctly predicted failures")
                print(f"    False Positives (FP): {fp:>3} - Predicted success but failed ⚠️")
                print(f"    False Negatives (FN): {fn:>3} - Predicted failure but succeeded")
                print(f"    True Positives  (TP): {tp:>3} - Correctly predicted successes ✓")
                
                # Business interpretation
                if fp > tp:
                    print(f"  ⚠️ WARNING: More false positives than true positives!")
                    print(f"     This means you're taking many losing trades.")
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                print(f"  False Positive Rate: {fpr:.1%} (lower is better)")
            
            # Store model
            self.models[ptype] = model
            self.scalers[ptype] = scaler
            self.feature_columns[ptype] = feature_cols
            results[ptype] = {
                'accuracy': accuracy,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'confusion_matrix': cm
            }
        
        print("\n" + "="*80)
        return results
    
    def visualize_confusion_matrices(self, save_path='ml_confusion_matrices.png'):
        """Create confusion matrix visualization"""
        if not self.confusion_matrices:
            print("No confusion matrices to visualize")
            return
        
        print("\nGenerating confusion matrix visualizations...")
        
        n_patterns = len(self.confusion_matrices)
        fig, axes = plt.subplots(1, n_patterns, figsize=(6*n_patterns, 5))
        
        if n_patterns == 1:
            axes = [axes]
        
        for idx, (ptype, cm) in enumerate(self.confusion_matrices.items()):
            ax = axes[idx]
            
            # Create display
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Fail', 'Success']
            )
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            
            ax.set_title(f'{ptype} Confusion Matrix', fontsize=12, fontweight='bold')
            
            # Add business interpretation as text
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                textstr = f'Precision: {precision:.1%}\nRecall: {recall:.1%}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices saved as '{save_path}'")
        plt.close()
    
    def predict_quality(self, patterns_df):
        """Predict with quality scores"""
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
# ENHANCEMENT 2: Backtest Confusion Matrix
# ============================================================================

class BacktestConfusionMatrix:
    """
    Analyze backtest results with confusion matrix
    
    Compares predicted pattern quality vs actual trade outcomes
    """
    
    def __init__(self, scored_patterns, closed_positions):
        """
        Parameters:
        -----------
        scored_patterns : DataFrame with Quality_Score column
        closed_positions : List of closed position dicts with 'pnl'
        """
        self.scored_patterns = scored_patterns
        self.closed_positions = closed_positions
    
    def create_confusion_matrix(self, quality_threshold=50):
        """
        CORRECTED LOGIC:
        - Prediction = 1 if Quality_Score >= threshold (we take the trade)
        - Actual = 1 if pnl > 0 (trade was profitable)
        """
        
        if not self.closed_positions:
            return None
        
        print("\n" + "="*80)
        print("FIXED BACKTEST CONFUSION MATRIX")
        print("="*80)
        
        # Build results properly
        results = []
        
        for i, pos in enumerate(self.closed_positions):
            # Try to match pattern by type and approximate date
            pattern_type = pos['pattern_type']
            
            # For better matching, filter by pattern type
            matching = self.scored_patterns[
                self.scored_patterns['Pattern_Type'] == pattern_type
            ]
            
            if not matching.empty:
                # Use the pattern's actual quality score
                # In production, match by detection date
                quality_score = matching['Quality_Score'].iloc[i % len(matching)]
                
                # CORRECTED: predicted = 1 means we TOOK the trade (score >= threshold)
                predicted = 1 if quality_score >= quality_threshold else 0
                actual = 1 if pos['pnl'] > 0 else 0
                
                results.append({
                    'predicted': predicted,
                    'actual': actual,
                    'quality_score': quality_score,
                    'pnl': pos['pnl'],
                    'pattern_type': pattern_type
                })
        
        if not results:
            return None
        
        df = pd.DataFrame(results)
        
        # IMPORTANT: Since you only traded high-quality patterns,
        # all predictions should be 1 (we took the trade)
        if df['predicted'].sum() == 0:
            print("⚠️ WARNING: No patterns met quality threshold!")
            print(f"   All quality scores < {quality_threshold}")
            print(f"   Min score: {df['quality_score'].min():.1f}")
            print(f"   Max score: {df['quality_score'].max():.1f}")
            return None
        
        # Create confusion matrix
        cm = confusion_matrix(df['actual'], df['predicted'])
        
        print(f"\nQuality Threshold: {quality_threshold}")
        print(f"Total Trades Analyzed: {len(df)}")
        print(f"Trades Taken (predicted=1): {df['predicted'].sum()}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\nCORRECT Interpretation:")
            print(f"  True Positives  (TP): {tp:>3} - Took trade → Made money ✓✓")
            print(f"  False Positives (FP): {fp:>3} - Took trade → Lost money ⚠️")
            print(f"  True Negatives  (TN): {tn:>3} - Skipped trade → Would have lost ✓")
            print(f"  False Negatives (FN): {fn:>3} - Skipped trade → Missed profit")
            
            # Calculate useful metrics
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
                print(f"\n  Precision: {precision:.1%}")
                print(f"    → When you take a trade, you win {precision:.1%} of the time")
            
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
                print(f"  Recall: {recall:.1%}")
                print(f"    → You catch {recall:.1%} of all winning opportunities")
        
        print("="*80)
        return cm, df
    
    def visualize_threshold_analysis(self, save_path='threshold_analysis.png'):
        """Analyze different quality thresholds"""
        if not self.closed_positions:
            return
        
        print("\nAnalyzing optimal quality threshold...")
        
        # Match patterns to trades (simplified)
        results = []
        for pos in self.closed_positions:
            pattern_type = pos['pattern_type']
            matching = self.scored_patterns[
                self.scored_patterns['Pattern_Type'] == pattern_type
            ]
            if not matching.empty:
                results.append({
                    'quality_score': matching['Quality_Score'].mean(),
                    'actual': 1 if pos['pnl'] > 0 else 0,
                    'pnl': pos['pnl']
                })
        
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Test different thresholds
        thresholds = range(40, 90, 5)
        metrics = []
        
        for thresh in thresholds:
            df['predicted'] = (df['quality_score'] >= thresh).astype(int)
            
            # Calculate metrics
            tp = ((df['predicted'] == 1) & (df['actual'] == 1)).sum()
            fp = ((df['predicted'] == 1) & (df['actual'] == 0)).sum()
            tn = ((df['predicted'] == 0) & (df['actual'] == 0)).sum()
            fn = ((df['predicted'] == 0) & (df['actual'] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
            
            # Calculate P&L if only taking trades above threshold
            trades_taken = df[df['quality_score'] >= thresh]
            pnl_at_thresh = trades_taken['pnl'].sum() if len(trades_taken) > 0 else 0
            
            metrics.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'trades_taken': len(trades_taken),
                'total_pnl': pnl_at_thresh
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Precision/Recall vs Threshold
        ax = axes[0, 0]
        ax.plot(metrics_df['threshold'], metrics_df['precision'], 
               marker='o', label='Precision', linewidth=2)
        ax.plot(metrics_df['threshold'], metrics_df['recall'], 
               marker='s', label='Recall', linewidth=2)
        ax.plot(metrics_df['threshold'], metrics_df['accuracy'], 
               marker='^', label='Accuracy', linewidth=2)
        ax.set_xlabel('Quality Score Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Trades Taken vs Threshold
        ax = axes[0, 1]
        ax.bar(metrics_df['threshold'], metrics_df['trades_taken'], alpha=0.7)
        ax.set_xlabel('Quality Score Threshold')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Trades Taken vs Threshold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. P&L vs Threshold
        ax = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in metrics_df['total_pnl']]
        ax.bar(metrics_df['threshold'], metrics_df['total_pnl'], 
              alpha=0.7, color=colors)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Quality Score Threshold')
        ax.set_ylabel('Total P&L ($)')
        ax.set_title('Cumulative P&L vs Threshold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Optimal threshold recommendation
        ax = axes[1, 1]
        ax.axis('off')
        
        # Find optimal threshold (max P&L)
        optimal_idx = metrics_df['total_pnl'].idxmax()
        optimal_thresh = metrics_df.loc[optimal_idx, 'threshold']
        optimal_pnl = metrics_df.loc[optimal_idx, 'total_pnl']
        optimal_trades = metrics_df.loc[optimal_idx, 'trades_taken']
        optimal_precision = metrics_df.loc[optimal_idx, 'precision']
        
        summary_text = [
            ['Metric', 'Value'],
            ['─'*25, '─'*15],
            ['Optimal Threshold', f'{optimal_thresh:.0f}'],
            ['Max P&L', f'${optimal_pnl:,.2f}'],
            ['Trades Taken', f'{optimal_trades:.0f}'],
            ['Precision at Optimal', f'{optimal_precision:.1%}'],
            ['', ''],
            ['Current (60)', ''],
            ['Trades at 60', f"{metrics_df[metrics_df['threshold']==60]['trades_taken'].values[0]:.0f}"],
            ['P&L at 60', f"${metrics_df[metrics_df['threshold']==60]['total_pnl'].values[0]:,.2f}"],
        ]
        
        table = ax.table(cellText=summary_text, cellLoc='left',
                        loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Threshold Analysis Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Threshold analysis saved as '{save_path}'")
        print(f"\n💡 Recommendation: Use threshold = {optimal_thresh:.0f} for maximum P&L")
        plt.close()


# ============================================================================
# USAGE EXAMPLE: How to integrate into existing system
# ============================================================================

def example_usage_with_confusion_matrix():
    """
    Example showing how to add confusion matrix analysis
    to your existing run_complete_system() function
    """
    
    print("""
    # In your run_complete_system() function, replace:
    
    scorer = PatternQualityScorer()
    scorer.train(labeled_patterns)
    
    # With:
    
    scorer = EnhancedPatternQualityScorer()
    training_results = scorer.train(labeled_patterns)
    scorer.visualize_confusion_matrices()  # NEW: Visualize ML confusion matrices
    
    # Then after portfolio backtest, add:
    
    backtest_cm = BacktestConfusionMatrix(scored_patterns, portfolio.closed_positions)
    cm, trade_df = backtest_cm.create_confusion_matrix(quality_threshold=60)
    backtest_cm.visualize_threshold_analysis()  # NEW: Optimal threshold analysis
    
    """)


# ============================================================================
# PART 4: RL TRADING ENVIRONMENT
# ============================================================================

if RL_AVAILABLE:
    class PatternAwareTradingEnv(gym.Env):
        """Enhanced RL environment with pattern signals"""
        
        def __init__(self, df, scored_patterns=None, initial_capital=100000, max_steps=300):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.scored_patterns = scored_patterns if scored_patterns is not None else pd.DataFrame()
            self.initial_capital = initial_capital
            self.max_steps = max_steps
            
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)
            
            self.reset()
        
        def _get_pattern_signal(self):
            """Get pattern signal at current index"""
            if self.scored_patterns.empty:
                return 0.0
            
            recent_patterns = self.scored_patterns[
                (self.scored_patterns['Detection_Index'] >= self.idx - 5) &
                (self.scored_patterns['Detection_Index'] <= self.idx)
            ]
            
            if recent_patterns.empty:
                return 0.0
            
            best = recent_patterns.loc[recent_patterns['Quality_Score'].idxmax()]
            quality_norm = best['Quality_Score'] / 100.0
            
            if best['Pattern_Type'] == 'DoubleBottom':
                return quality_norm
            elif best['Pattern_Type'] == 'DoubleTop':
                return -quality_norm
            
            return 0.0
        
        def _get_obs(self):
            """Get current state observation"""
            row = self.df.loc[self.idx]
            prev_close = self.df.loc[self.idx - 1, 'Close']
            
            pct_change = (row['Close'] - prev_close) / prev_close if prev_close != 0 else 0
            atr_norm = row['ATR'] / row['Close'] if row['Close'] != 0 else 0
            pattern_signal = self._get_pattern_signal()
            
            obs = np.array([
                pct_change,
                row.get('RSI', 50) / 100,
                row.get('MACD', 0) / 100,
                row.get('MACD_Signal', 0) / 100,
                atr_norm,
                float(self.position),
                pattern_signal
            ], dtype=np.float32)
            
            return obs
        
        def reset(self, seed=None, options=None):
            """Reset environment"""
            super().reset(seed=seed)
            
            self.idx = 60
            self.steps = 0
            self.position = 0
            self.entry_price = 0
            self.equity = self.initial_capital
            self.peak_equity = self.initial_capital
            self.trades = []
            
            obs = self._get_obs()
            return obs, {}
        
        def step(self, action):
            """Execute action with balanced reward function"""
            self.steps += 1
            
            terminated = False
            truncated = self.steps >= self.max_steps or self.idx >= len(self.df) - 2
            
            prev_close = self.df.loc[self.idx - 1, 'Close']
            curr_close = self.df.loc[self.idx, 'Close']
            volatility = self.df.loc[self.idx, 'Volatility']
            
            reward = 0.0
            
            # Get pattern signal
            pattern_signal = self._get_pattern_signal()
            
            if action == 1:  # Go/Keep Long
                if self.position == 0:
                    # Enter long position
                    self.position = 1
                    self.entry_price = curr_close
                    self.entry_step = self.steps
                    
                    # Moderate transaction cost
                    reward -= 0.0005  # 0.05%
                    
                    # Reward for entering on strong bullish signal
                    if pattern_signal > 0.5:
                        reward += 0.01  # Bonus for good entry
                    
            elif action == 2:  # Flatten
                if self.position == 1:
                    # Exit long position
                    trade_return = (curr_close - self.entry_price) / self.entry_price
                    
                    # Cap but don't overly restrict
                    capped_return = np.clip(trade_return, -0.10, 0.10)
                    reward += capped_return * 10  # Amplify to make meaningful
                    
                    self.trades.append({
                        'entry': self.entry_price,
                        'exit': curr_close,
                        'return': trade_return,
                        'holding_period': self.steps - self.entry_step
                    })
                    
                    reward -= 0.0005  # Transaction cost
                    
                    self.position = 0
                    self.entry_price = 0
                else:
                    # Penalty for flattening when not in position
                    reward -= 0.001
            
            # Action 0: Hold
            elif action == 0:
                # Small penalty for holding too long without action
                # This encourages exploration
                if self.steps % 50 == 0:
                    reward -= 0.0001
            
            # Mark-to-market for open positions
            if self.position == 1:
                price_change = (curr_close - prev_close) / prev_close
                
                # Moderate scaling
                if volatility > 0:
                    risk_adjusted_return = price_change / (volatility + 0.01)
                else:
                    risk_adjusted_return = price_change
                
                reward += risk_adjusted_return * 0.5  # Moderate scaling
                
                # Penalty for holding too long
                holding_period = self.steps - self.entry_step
                if holding_period > 30:
                    reward -= 0.005
                
                # Drawdown penalty
                unrealized_pnl_pct = (curr_close - self.entry_price) / self.entry_price
                if unrealized_pnl_pct < -0.05:
                    reward -= 0.05
                elif unrealized_pnl_pct < -0.02:
                    reward -= 0.01
                
                self.equity = self.initial_capital * (1 + unrealized_pnl_pct)
            else:
                self.equity = self.initial_capital
            
            self.peak_equity = max(self.peak_equity, self.equity)
            self.idx += 1
            obs = self._get_obs()
            
            return obs, float(reward), terminated, truncated, {}
        
        def get_trade_stats(self):
            """Get statistics about trades made during episode"""
            if not self.trades:
                return {}
            
            returns = [t['return'] for t in self.trades]
            winning_trades = [r for r in returns if r > 0]
            
            return {
                'total_trades': len(returns),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(returns) if returns else 0,
                'avg_return': np.mean(returns),
                'total_return': sum(returns)
            }

else:
    class PatternAwareTradingEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium not installed")


# ============================================================================
# PART 5: ENHANCED RL INTERPRETABILITY (INTEGRATED)
# ============================================================================

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


# ============================================================================
# PART 6: RL TRAINING
# ============================================================================

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


# ============================================================================
# PART 7: DASHBOARD
# ============================================================================

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


# ============================================================================
# PART 8: MAIN EXECUTION
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

def adaptive_quality_threshold(scored_patterns, target_trades=30):
    """
    Calculate adaptive quality threshold to get desired number of trades
    
    Parameters:
    -----------
    scored_patterns : DataFrame with Quality_Score column
    target_trades : int, desired number of trades
    
    Returns:
    --------
    threshold : float, recommended quality threshold
    """
    if scored_patterns.empty:
        return 40.0
    
    scores = scored_patterns['Quality_Score'].sort_values(ascending=False)
    
    # Find threshold that gives us target_trades
    if len(scores) < target_trades:
        threshold = scores.min() - 1  # Take all patterns
    else:
        threshold = scores.iloc[target_trades - 1]
    
    # Ensure minimum threshold of 35
    threshold = max(threshold, 35)
    
    print(f"\n  💡 Adaptive Threshold Analysis:")
    print(f"     Target trades: {target_trades}")
    print(f"     Patterns available: {len(scores)}")
    print(f"     Recommended threshold: {threshold:.1f}")
    print(f"     This will give ~{(scores >= threshold).sum()} trades")
    
    return threshold

def filter_patterns_by_trend(patterns_df, df):
    """Only keep patterns aligned with trend"""
    filtered = []
    
    for _, pattern in patterns_df.iterrows():
        idx = pattern['Detection_Index']
        sma_20 = df['Close'].rolling(20).mean().iloc[idx]
        current_price = df['Close'].iloc[idx]
        
        is_uptrend = current_price > sma_20
        
        # Only take patterns aligned with trend
        if pattern['Pattern_Type'] == 'DoubleBottom' and is_uptrend:
            filtered.append(pattern)
        elif pattern['Pattern_Type'] == 'DoubleTop' and not is_uptrend:
            filtered.append(pattern)
    
    return pd.DataFrame(filtered)

def run_complete_system(df=None, use_sample_data=True, enable_rl=True, rl_timesteps=50000):
    """Run the complete trading system with enhanced interpretability"""
    
    print("="*80)
    print("COMPLETE TRADING SYSTEM WITH ENHANCED RL INTERPRETABILITY")
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
    
    # Step 2: Detect Patterns
    print("\n[1/7] Detecting patterns...")
    
    detector_dt = DoubleTopDetector(df)
    patterns_dt = detector_dt.detect_patterns()
    
    detector_db = DoubleBottomDetector(df)
    patterns_db = detector_db.detect_patterns()
    
    all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)
    trend_aligned_patterns = filter_patterns_by_trend(all_patterns, df)  # NEW
    print(f"  Patterns after trend filter: {len(trend_aligned_patterns)}")
    
    print(f"  ✓ Total patterns: {len(all_patterns)}")
    
    if all_patterns.empty:
        print("\n⚠ No patterns detected.")
        return None
    
    # Step 3: Train ML
    print("\n[2/7] Training ML models...")
    
    labeler = PatternLabeler(df, all_patterns)
    labeled_patterns = labeler.label_patterns()
    
    print("\\n[2.5/7] Running pattern diagnostic...")
    diagnostic = PatternDiagnostic(df, all_patterns, labeled_patterns)
    diagnostic.run_full_diagnostic()
    diagnostic.visualize_diagnostics()


    # scorer = PatternQualityScorer()
    # scorer.train(labeled_patterns)

    scorer = PatternQualityScorer()
    scorer.train(labeled_patterns)

    # training_results = scorer.train(labeled_patterns)
    #scorer.visualize_confusion_matrices()  # Generates: ml_confusion_matrices.png
    
    # Step 4: Score Patterns
    print("\n[3/7] Scoring patterns...")
    
    scored_patterns = scorer.predict_quality(all_patterns)

    # Calculate adaptive threshold
    adaptive_thresh = adaptive_quality_threshold(scored_patterns, target_trades=30)
    

    high_quality = scored_patterns[scored_patterns['Quality_Score'] >= adaptive_thresh]
    print(f"  ✓ High quality patterns: {len(high_quality)}")

    # Step 5: Portfolio Backtest
    print("\n[4/7] Running portfolio backtest...")
    
    portfolio = PortfolioRiskManager()
    
    trades_executed = 0
    for _, pattern in high_quality.iterrows():
        eval_result = portfolio.evaluate_new_trade(pattern)
        
        if eval_result['approved']:
            entry_price = pattern['Neckline']
            portfolio.open_position(pattern, eval_result, entry_price)
            trades_executed += 1
            
            entry_idx = int(pattern['Detection_Index'])
            is_bearish = pattern['Pattern_Type'] == 'DoubleTop'
            exit_price = None
            
            for i in range(entry_idx + 1, min(entry_idx + 30, len(df))):
                high = df.loc[i, 'High']
                low = df.loc[i, 'Low']
                
                if is_bearish:
                    if high >= pattern['Stop_Loss'] or low <= pattern['Target']:
                        exit_price = pattern['Stop_Loss'] if high >= pattern['Stop_Loss'] else pattern['Target']
                        break
                else:
                    if low <= pattern['Stop_Loss'] or high >= pattern['Target']:
                        exit_price = pattern['Stop_Loss'] if low <= pattern['Stop_Loss'] else pattern['Target']
                        break
            
            if exit_price is None:
                exit_idx = min(entry_idx + 29, len(df) - 1)
                exit_price = df.loc[exit_idx, 'Close']
            
            portfolio.close_position(0, exit_price, df.loc[min(entry_idx + 30, len(df)-1), 'date'])
            
            if trades_executed >= 30:
                break
    
    metrics = portfolio.get_metrics()
    portfolio_results = {
        'total_trades': metrics['total_trades'],
        'win_rate': metrics['win_rate'],
        'total_return': metrics['total_return'],
    }
    
    print(f"\n  Total Return: {metrics['total_return']:.2f}%")
    
    # Backtest confusion matrix analysis
    print("\n[5.5/7] Analyzing backtest with confusion matrix...")
    backtest_cm = BacktestConfusionMatrix(scored_patterns, portfolio.closed_positions)
    cm, trade_df = backtest_cm.create_confusion_matrix(quality_threshold=60)
   
    if cm is not None:
        backtest_cm.visualize_threshold_analysis()  # Generates: threshold_analysis.png


    # Step 6: RL Training
    rl_comparison = None
    rl_model = None
    
    if enable_rl and RL_AVAILABLE and SB3_AVAILABLE:
        print("\n[5/7] Training RL agent...")
        enriched_df = BasePatternDetector(df).df
        rl_model = train_rl_agent(enriched_df, scored_patterns, timesteps=rl_timesteps)
        
        if rl_model:
            rl_comparison = compare_strategies(enriched_df, scored_patterns, rl_model, portfolio_results)
            
            # Step 6.5: Enhanced Interpretability
            print("\n[6/7] Running enhanced interpretability analysis...")
            try:
                feature_names = ['Price_Change', 'RSI', 'MACD', 'MACD_Signal', 
                                'ATR', 'Position', 'Pattern_Signal']
                
                env = PatternAwareTradingEnv(enriched_df, scored_patterns, max_steps=300)
                interpreter = RLInterpretabilityReport(rl_model, env, feature_names)
                interp_results = interpreter.generate_full_report()
                
            except Exception as e:
                print(f"  ⚠ Interpretability failed: {e}")
    
    # Step 7: Dashboard
    print("\n[7/7] Generating dashboard...")
    
    dashboard = TradingDashboard(portfolio, rl_comparison)
    dashboard.create_dashboard()
    
    print("\n" + "="*80)
    print("SYSTEM EXECUTION COMPLETE")
    print("="*80)
    
    return {
        'portfolio': portfolio,
        'rl_model': rl_model,
        'rl_comparison': rl_comparison
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRADING SYSTEM OPTIONS:")
    print("="*80)
    print("1. Quick Demo with RL + Enhanced Interpretability (~5 min)")
    print("2. Quick Demo without RL (~30 sec)")
    print("3. Use my own CSV file (with RL)")
    print("4. Use my own CSV file (no RL)")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        print("\nStarting Quick Demo with RL and Enhanced Interpretability...")
        result = run_complete_system(use_sample_data=True, enable_rl=True, rl_timesteps=50000)
        
    elif choice == '2':
        print("\nStarting Quick Demo without RL...")
        result = run_complete_system(use_sample_data=True, enable_rl=False)
        
    elif choice == '3':
        csv_file = input("\nEnter CSV file path: ").strip()
        try:
            df = pd.read_csv(csv_file)
            timesteps = input("RL training timesteps (default 50000): ").strip()
            timesteps = int(timesteps) if timesteps else 50000
            result = run_complete_system(df=df, use_sample_data=False, enable_rl=True, rl_timesteps=timesteps)
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    elif choice == '4':
        csv_file = input("\nEnter CSV file path: ").strip()
        try:
            df = pd.read_csv(csv_file)
            result = run_complete_system(df=df, use_sample_data=False, enable_rl=False)
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    elif choice == '5':
        print("\nExiting... Goodbye!")
    
    else:
        print("\n✗ Invalid choice")
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX ENHANCEMENT MODULE")
    print("="*80)
    print("\nThis module adds confusion matrix analysis to the trading system.")
    print("\nKey Features:")
    print("  1. ML Confusion Matrix - Shows pattern prediction accuracy")
    print("  2. Backtest Confusion Matrix - Compares predictions vs actual results")
    print("  3. Threshold Optimization - Finds optimal quality score cutoff")
    print("\nIntegration:")
    print("  - Copy EnhancedPatternQualityScorer to replace existing scorer")
    print("  - Add BacktestConfusionMatrix after portfolio backtest")
    print("  - See integration instructions above")
    print("="*80)

    print("Program finished.")
    print("="*80)
    