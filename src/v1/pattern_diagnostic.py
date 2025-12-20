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
        self.df = df
        self.all_patterns = all_patterns
        self.labeled_patterns = labeled_patterns
    
    def run_full_diagnostic(self):
        """Run all diagnostic checks"""
        
        print("\n" + "="*80)
        print("PATTERN QUALITY DIAGNOSTIC")
        print("="*80)
        
        # Check 1: Data Quality
        self._check_data_quality()
        
        # Check 2: Pattern Distribution
        self._check_pattern_distribution()
        
        # Check 3: Success Rate Analysis
        self._check_success_rates()
        
        # Check 4: Pattern Characteristics
        self._check_pattern_characteristics()
        
        # Check 5: Market Conditions
        self._check_market_conditions()
        
        # Check 6: Target/Stop-Loss Analysis
        self._check_target_stop_analysis()
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _check_data_quality(self):
        """Check if data has issues"""
        print("\n[1/6] DATA QUALITY CHECK")
        print("-" * 80)
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ WARNING: Missing values detected!")
            print(missing[missing > 0])
        else:
            print("✓ No missing values")
        
        # Check date range
        date_range = (self.df['date'].max() - self.df['date'].min()).days
        print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Total days: {date_range}")
        print(f"  Total bars: {len(self.df)}")
        
        # Check price range
        price_range = self.df['Close'].max() - self.df['Close'].min()
        price_pct_change = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[0]) / self.df['Close'].iloc[0]
        print(f"✓ Price range: {self.df['Close'].min():.2f} to {self.df['Close'].max():.2f}")
        print(f"  Total change: {price_pct_change:.1%}")
        
        # Check volatility
        daily_returns = self.df['Close'].pct_change()
        avg_volatility = daily_returns.std()
        print(f"✓ Average daily volatility: {avg_volatility:.2%}")
        
        if avg_volatility < 0.005:
            print("⚠️ WARNING: Very low volatility - patterns may not be meaningful")
        elif avg_volatility > 0.05:
            print("⚠️ WARNING: Very high volatility - patterns may be unreliable")
    
    def _check_pattern_distribution(self):
        """Check pattern distribution over time"""
        print("\n[2/6] PATTERN DISTRIBUTION")
        print("-" * 80)
        
        if self.all_patterns.empty:
            print("⚠️ ERROR: No patterns detected!")
            return
        
        # Patterns by type
        pattern_counts = self.all_patterns['Pattern_Type'].value_counts()
        print("Pattern counts:")
        for ptype, count in pattern_counts.items():
            print(f"  {ptype}: {count}")
        
        # Patterns over time
        self.all_patterns['Year'] = pd.to_datetime(self.all_patterns['Detection_Date']).dt.year
        yearly = self.all_patterns.groupby('Year').size()
        print("\nPatterns by year:")
        for year, count in yearly.items():
            print(f"  {year}: {count}")
        
        # Pattern spacing
        if len(self.all_patterns) > 1:
            sorted_patterns = self.all_patterns.sort_values('Detection_Index')
            spacing = sorted_patterns['Detection_Index'].diff().dropna()
            print(f"\nPattern spacing:")
            print(f"  Average bars between patterns: {spacing.mean():.1f}")
            print(f"  Min: {spacing.min():.0f}, Max: {spacing.max():.0f}")
            
            if spacing.mean() < 20:
                print("⚠️ WARNING: Patterns too close together - may be noise")
    
    def _check_success_rates(self):
        """Analyze pattern success rates"""
        print("\n[3/6] SUCCESS RATE ANALYSIS")
        print("-" * 80)
        
        if self.labeled_patterns.empty:
            print("⚠️ ERROR: No labeled patterns!")
            return
        
        # Overall success rate
        total_success = self.labeled_patterns['Success'].sum()
        total_patterns = len(self.labeled_patterns)
        overall_rate = total_success / total_patterns
        
        print(f"Overall success rate: {overall_rate:.1%} ({total_success}/{total_patterns})")
        
        # By pattern type
        print("\nSuccess rate by pattern type:")
        for ptype in self.labeled_patterns['Pattern_Type'].unique():
            mask = self.labeled_patterns['Pattern_Type'] == ptype
            success = self.labeled_patterns[mask]['Success'].sum()
            total = mask.sum()
            rate = success / total if total > 0 else 0
            print(f"  {ptype}: {rate:.1%} ({success}/{total})")
        
        # Critical thresholds
        print("\nCritical Analysis:")
        if overall_rate < 0.15:
            print("🔴 CRITICAL: Success rate < 15% - Patterns are NOT predictive!")
            print("   → These patterns CANNOT be profitable even with perfect prediction")
        elif overall_rate < 0.30:
            print("🟡 WARNING: Success rate < 30% - Patterns are weak")
            print("   → Profitability will be difficult")
        elif overall_rate < 0.50:
            print("🟢 ACCEPTABLE: Success rate 30-50% - Patterns have some edge")
        else:
            print("✓ GOOD: Success rate > 50% - Patterns are predictive")
        
        # Win/Loss distribution
        if 'Return' in self.labeled_patterns.columns:
            wins = self.labeled_patterns[self.labeled_patterns['Success'] == 1]['Return']
            losses = self.labeled_patterns[self.labeled_patterns['Success'] == 0]['Return']
            
            if len(wins) > 0 and len(losses) > 0:
                print(f"\nWin/Loss characteristics:")
                print(f"  Average win: {wins.mean():.2f}")
                print(f"  Average loss: {losses.mean():.2f}")
                print(f"  Win/Loss ratio: {abs(wins.mean() / losses.mean()):.2f}")
    
    def _check_pattern_characteristics(self):
        """Analyze what makes patterns succeed/fail"""
        print("\n[4/6] PATTERN CHARACTERISTICS")
        print("-" * 80)
        
        if self.labeled_patterns.empty or 'Success' not in self.labeled_patterns.columns:
            return
        
        successful = self.labeled_patterns[self.labeled_patterns['Success'] == 1]
        failed = self.labeled_patterns[self.labeled_patterns['Success'] == 0]
        
        if len(successful) == 0:
            print("⚠️ WARNING: NO successful patterns to analyze!")
            print("   Cannot identify success factors")
            return
        
        print("Comparing successful vs failed patterns:")
        
        # RSI
        if 'RSI' in self.labeled_patterns.columns:
            success_rsi = successful['RSI'].mean()
            fail_rsi = failed['RSI'].mean()
            print(f"  RSI: Success={success_rsi:.1f}, Fail={fail_rsi:.1f}")
        
        # Volatility
        if 'Volatility' in self.labeled_patterns.columns:
            success_vol = successful['Volatility'].mean()
            fail_vol = failed['Volatility'].mean()
            print(f"  Volatility: Success={success_vol:.4f}, Fail={fail_vol:.4f}")
        
        # Pattern Height
        if 'Pattern_Height' in self.labeled_patterns.columns:
            success_height = successful['Pattern_Height'].mean()
            fail_height = failed['Pattern_Height'].mean()
            print(f"  Pattern Height: Success={success_height:.2f}, Fail={fail_height:.2f}")
        
        # Trend
        if 'Trend' in self.labeled_patterns.columns:
            success_trend = successful['Trend'].mean()
            fail_trend = failed['Trend'].mean()
            print(f"  Trend: Success={success_trend:.2f}, Fail={fail_trend:.2f}")
    
    def _check_market_conditions(self):
        """Analyze overall market conditions"""
        print("\n[5/6] MARKET CONDITIONS")
        print("-" * 80)
        
        # Trend analysis
        sma_20 = self.df['Close'].rolling(20).mean()
        sma_50 = self.df['Close'].rolling(50).mean()
        
        uptrend_pct = ((sma_20 > sma_50).sum() / len(self.df)) * 100
        downtrend_pct = 100 - uptrend_pct
        
        print(f"Market trend distribution:")
        print(f"  Uptrend: {uptrend_pct:.1f}%")
        print(f"  Downtrend: {downtrend_pct:.1f}%")
        
        # Volatility regimes
        returns = self.df['Close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        
        high_vol_pct = ((rolling_vol > rolling_vol.quantile(0.75)).sum() / len(self.df)) * 100
        low_vol_pct = ((rolling_vol < rolling_vol.quantile(0.25)).sum() / len(self.df)) * 100
        
        print(f"\nVolatility regimes:")
        print(f"  High volatility: {high_vol_pct:.1f}%")
        print(f"  Low volatility: {low_vol_pct:.1f}%")
        
        # Price movement
        total_return = (self.df['Close'].iloc[-1] / self.df['Close'].iloc[0] - 1) * 100
        print(f"\nOverall price movement: {total_return:+.2f}%")
        
        if abs(total_return) < 5:
            print("⚠️ Range-bound market - patterns may be less effective")
        elif total_return > 50:
            print("⚠️ Strong uptrend - bearish patterns may fail")
        elif total_return < -50:
            print("⚠️ Strong downtrend - bullish patterns may fail")
    
    def _check_target_stop_analysis(self):
        """Analyze if targets/stops are reasonable"""
        print("\n[6/6] TARGET/STOP-LOSS ANALYSIS")
        print("-" * 80)
        
        if self.all_patterns.empty:
            return
        
        # Calculate target/stop distances as % of price
        if 'Neckline' in self.all_patterns.columns:
            self.all_patterns['Target_Distance_Pct'] = abs(
                (self.all_patterns['Target'] - self.all_patterns['Neckline']) / 
                self.all_patterns['Neckline']
            ) * 100
            
            self.all_patterns['Stop_Distance_Pct'] = abs(
                (self.all_patterns['Stop_Loss'] - self.all_patterns['Neckline']) / 
                self.all_patterns['Neckline']
            ) * 100
            
            avg_target = self.all_patterns['Target_Distance_Pct'].mean()
            avg_stop = self.all_patterns['Stop_Distance_Pct'].mean()
            risk_reward = avg_target / avg_stop if avg_stop > 0 else 0
            
            print(f"Average distances from entry:")
            print(f"  Target: {avg_target:.2f}%")
            print(f"  Stop-Loss: {avg_stop:.2f}%")
            print(f"  Risk/Reward ratio: {risk_reward:.2f}:1")
            
            if avg_target < 1:
                print("⚠️ WARNING: Targets too close - may not reach before noise")
            if avg_stop > 5:
                print("⚠️ WARNING: Stops too far - excessive risk per trade")
            if risk_reward < 1.5:
                print("⚠️ WARNING: Risk/Reward < 1.5:1 - need >40% win rate to profit")
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        success_rate = self.labeled_patterns['Success'].mean() if not self.labeled_patterns.empty else 0
        
        recommendations = []
        
        # Based on success rate
        if success_rate < 0.15:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': 'Patterns have no predictive power (11% success)',
                'solutions': [
                    '1. Try different pattern types (Head & Shoulders, Triangles, Flags)',
                    '2. Use different timeframe (hourly if using daily, or vice versa)',
                    '3. Try different asset class (this asset may not exhibit patterns)',
                    '4. Increase pattern strictness (lower tolerance to 0.01)',
                    '5. Consider this asset is not suitable for pattern trading'
                ]
            })
        elif success_rate < 0.30:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Patterns are weak predictors',
                'solutions': [
                    '1. Add more filters (volume confirmation, trend alignment)',
                    '2. Reduce pattern tolerance for clearer signals',
                    '3. Combine with other indicators (momentum, volume)',
                    '4. Focus on patterns in specific market conditions'
                ]
            })
        
        # Based on RL vs Rule-based disparity
        if hasattr(self, 'rl_return') and hasattr(self, 'rule_return'):
            if abs(self.rl_return - self.rule_return) > 100:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': f'RL agent outperforms by {abs(self.rl_return - self.rule_return):.1f}% - reward hacking',
                    'solutions': [
                        '1. Increase transaction costs to 0.1%',
                        '2. Cap daily/per-trade returns',
                        '3. Add position holding time penalties',
                        '4. Validate RL on out-of-sample data'
                    ]
                })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{rec['priority']} PRIORITY #{i}:")
            print(f"Issue: {rec['issue']}")
            print("Solutions:")
            for sol in rec['solutions']:
                print(f"  {sol}")
        
        if not recommendations:
            print("\n✓ No critical issues detected")
        
        print("\n" + "="*80)
    
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
# USAGE
# ============================================================================

def run_diagnostic(df, all_patterns, labeled_patterns):
    """
    Run complete diagnostic on your pattern detection results
    
    Usage:
    ------
    # After pattern detection and labeling in your system:
    diagnostic = PatternDiagnostic(df, all_patterns, labeled_patterns)
    diagnostic.run_full_diagnostic()
    diagnostic.visualize_diagnostics()
    """
    
    diagnostic = PatternDiagnostic(df, all_patterns, labeled_patterns)
    diagnostic.run_full_diagnostic()
    diagnostic.visualize_diagnostics()
    
    return diagnostic


if __name__ == "__main__":
    print("""
================================================================================
PATTERN QUALITY DIAGNOSTIC TOOL
================================================================================

This tool helps identify WHY your patterns are failing.

INTEGRATION:
Add this after pattern labeling in your run_complete_system():

    # After Step 2 (labeling)
    print("\\n[2.5/7] Running pattern diagnostic...")
    diagnostic = PatternDiagnostic(df, all_patterns, labeled_patterns)
    diagnostic.run_full_diagnostic()
    diagnostic.visualize_diagnostics()

WHAT IT CHECKS:
1. Data quality (missing values, volatility)
2. Pattern distribution (spacing, frequency)
3. Success rates (overall and by type)
4. Pattern characteristics (what makes them succeed)
5. Market conditions (trend, volatility regimes)
6. Target/Stop-loss reasonableness

OUTPUT:
- Console: Detailed analysis and recommendations
- File: pattern_diagnostic.png with 6 diagnostic charts

This will tell you EXACTLY what's wrong and how to fix it!
================================================================================
    """)