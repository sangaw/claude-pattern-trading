"""
PATTERN QUALITY DIAGNOSTIC TOOL

This will help identify WHY your patterns are failing
and what needs to be fixed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        print("\n[3/6] SUCCESS RATE ANALYSIS")
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
        print(self.labeled_patterns.columns)
        
        # Compare key features
        features_to_check = ['RSI', 'Volatility', 'Pole_Height', 'Trend', 'ATR']
        
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
        
        required_cols = ['Entry_Price', 'Target', 'Stop_Loss']
        missing = [col for col in required_cols if col not in self.all_patterns.columns]
        
        if self.all_patterns.empty or missing:
            print(f"⚠️ Missing columns for analysis: {missing}")
            return
        
        # Calculate distances relative to the Entry (Breakout) Price
        # target_dist: How far the Target is above the entry
        # stop_dist: How far the Stop Loss is below the entry
        target_dist = ((self.all_patterns['Target'] - self.all_patterns['Entry_Price']) / 
                        self.all_patterns['Entry_Price']) * 100
        
        stop_dist = ((self.all_patterns['Entry_Price'] - self.all_patterns['Stop_Loss']) / 
                      self.all_patterns['Entry_Price']) * 100
        
        # Filter out invalid/zero stop distances to avoid division by zero
        valid_mask = stop_dist > 0
        if not valid_mask.any():
            print("⚠️ Invalid Stop-Loss data (Stop Distances <= 0)")
            return

        risk_reward = target_dist[valid_mask] / stop_dist[valid_mask]
        
        print(f"📏 Flag Analysis (Averages):")
        print(f"   Target Distance:  {target_dist.mean():.2f}%")
        print(f"   Stop-Loss Dist:   {stop_dist.mean():.2f}%")
        print(f"   Risk/Reward:      {risk_reward.mean():.2f}:1")
        
        # Pole Height context if available
        if 'Pole_Height' in self.all_patterns.columns:
            avg_pole = self.all_patterns['Pole_Height'].mean()
            print(f"   Avg Pole Height:  {avg_pole:.2f}%")

        # --- ASSESSMENT LOGIC ---
        avg_rr = risk_reward.mean()
        
        # 1. Risk/Reward Check
        if avg_rr < 1.5:
            breakeven = 1 / (1 + avg_rr)
            print(f"\n   ⚠️ AGGRESSIVE: Low R:R ({avg_rr:.2f}) requires >{breakeven:.0%} win rate.")
        elif avg_rr >= 2.0:
            print(f"   ✓ HEALTHY: Good R:R ratio for trend following.")

        # 2. Volatility Alignment (Targets too close/far)
        avg_target = target_dist.mean()
        if avg_target < 0.5:
            print(f"   ⚠️ WARNING: Targets are extremely tight ({avg_target:.2f}%). Transaction costs may eat profits.")
        elif avg_target > 10.0:
            print(f"   ⚠️ WARNING: Targets are very ambitious ({avg_target:.2f}%). May not be reached in 30 bars.")

        # 3. Stop Loss Logic
        avg_stop = stop_dist.mean()
        if avg_stop > 5.0:
            print(f"   ⚠️ WARNING: Wide Stops ({avg_stop:.2f}%). Reduce position size to manage risk.")
    
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
            print("   Your f'{success_rate} success rate means these patterns don't work on this data.")
            print("")
            print("   IMMEDIATE ACTIONS (choose ONE):")
            print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("   A) Try different pattern types:")
            print("      - Head & Shoulders patterns")
            print("      - Triangle breakouts")
            print("      - Channel patterns")
            print("      - Use trend-following strategies instead")
            print("      - Try moving average crossovers")
            print("      - Use momentum indicators (RSI, MACD)")
            
            print("")
            print("   B) Change timeframe:")
            print("      - If using daily → try 4-hour or hourly")
            print("      - If using hourly → try daily")
            print("")
            
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
        
        print("\n" + "="*80)
        print("🎬 NEXT STEPS:")
        print("="*80)
        
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