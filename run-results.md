

================================================================================
MULTI-PATTERN TRADING SYSTEM
================================================================================

✓ Loaded 991 bars of data
  Date range: 2021-01-01 00:00:00+05:30 to 2024-12-31 00:00:00+05:30

[1/5] Detecting patterns...
  ✓ Double Top: 43
  ✓ Double Bottom: 49
  ✓ Total: 92

[2/5] Training ML models...
  Success rate by pattern:
    DoubleTop: 4.7%
    DoubleBottom: 14.3%
  DoubleTop: Accuracy = 100.00%
  DoubleBottom: Accuracy = 100.00%

[3/5] Scoring pattern quality...
  ✓ High quality patterns (score >= 60): 5

[4/5] Running portfolio backtest...
  ✓ Executed 5 trades

[5/5] RESULTS:
================================================================================
Total Trades        : 5
Winning Trades      : 5
Win Rate            : 100.0%
Total P&L           : $2,454.55
Total Return        : 2.45%
Final Capital       : $102,454.55
================================================================================

[6/6] Generating dashboard...





Enter CSV file path: C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\featured.csv
✓ Loaded 6156 rows from C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\featured.csv
================================================================================
MULTI-PATTERN TRADING SYSTEM
================================================================================

✓ Loaded 6156 bars of data
  Date range: 2001-01-01 00:00:00+05:30 to 2025-10-03 00:00:00+05:30

[1/5] Detecting patterns...
  ✓ Double Top: 224
  ✓ Double Bottom: 216
  ✓ Total: 440

[2/5] Training ML models...
  Success rate by pattern:
    DoubleTop: 19.6%
    DoubleBottom: 19.9%
  DoubleTop: Accuracy = 97.78%
  DoubleBottom: Accuracy = 84.09%

[3/5] Scoring pattern quality...
  ✓ High quality patterns (score >= 60): 62

[RL] Training RL agent on enriched data...

[RL] Training PPO on SimpleTradingEnv...
[RL] Demo cumulative reward over 200 steps: -0.0055

[4/5] Running portfolio backtest...
  ✓ Executed 30 trades

[5/5] RESULTS:
================================================================================
Total Trades        : 30
Winning Trades      : 30
Win Rate            : 100.0%
Total P&L           : $43,778.45
Total Return        : 43.78%
Final Capital       : $143,778.45
================================================================================

[6/6] Generating dashboard...


############ Second Run

Enter CSV file path: C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\featured.csv
✓ Loaded 6156 rows from C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\featured.csv

RL training timesteps (default 50000): 50000

================================================================================
Running with your data and RL training (50000 steps)...
================================================================================
================================================================================
ENHANCED MULTI-PATTERN TRADING SYSTEM WITH RL
================================================================================

✓ Loaded 6156 bars of data
  Date range: 2001-01-01 00:00:00+05:30 to 2025-10-03 00:00:00+05:30

[1/6] Detecting patterns...
  ✓ Double Top: 224
  ✓ Double Bottom: 216
  ✓ Total: 440

[2/6] Training ML models...
  Success rate by pattern:
    DoubleTop: 19.6%
    DoubleBottom: 19.9%
  DoubleTop: Accuracy = 97.78%
  DoubleBottom: Accuracy = 84.09%

[3/6] Scoring pattern quality...
  ✓ High quality patterns (score >= 60): 62

[4/6] Running rule-based portfolio backtest...
  ✓ Executed 30 trades

[RULE-BASED RESULTS]:
  Total Trades   : 30
  Win Rate       : 100.0%
  Total Return   : 43.78%
  Final Capital  : $143,778.45

[5/6] Training RL agent...

================================================================================
TRAINING RL AGENT (PPO)
================================================================================

Training on 4924 bars...
✓ Training completed: 50000 timesteps

Evaluating on training data...

Evaluating on test data...

================================================================================
RL AGENT TRAINING SUMMARY
================================================================================
Algorithm              : PPO
Training Timesteps     : 50,000
Train Cumulative Reward: 826.3520
Test Cumulative Reward : 57.3361
================================================================================

================================================================================
STRATEGY COMPARISON: RL vs RULE-BASED
================================================================================

Strategy             Return %        Win Rate %      Trades     Sharpe
----------------------------------------------------------------------
RL Agent             454.72          53.7            1010       7.06
Rule-Based           43.78           100.0           30         N/A
----------------------------------------------------------------------

⚠️  WARNING: RL trading too frequently (1010 trades)
   Likely overtrading. Rule-based: {rule_trades} trades

✓ Winner: RL Agent (by 410.95%)

[5.5/6] Analyzing RL model interpretability...

================================================================================
RL MODEL INTERPRETABILITY ANALYSIS
================================================================================

[1/3] Collecting action data...
  ✓ Collected 2000 state-action pairs

[2/3] Analyzing action distribution...

Action Distribution:
  Hold      :    0 (  0.0%)
  Long      : 1060 ( 53.0%)
  Flatten   :  940 ( 47.0%)

[3/3] Analyzing feature importance...
  ✓ Surrogate model accuracy: 98.5%

Feature Importance:
  Price_Change   : 0.847
  RSI            : 0.121
  Pattern_Signal : 0.032
  MACD           : 0.000
  MACD_Signal    : 0.000

✓ Interpretability visualization saved as 'rl_interpretability.png'

================================================================================
KEY INSIGHTS
================================================================================
⚠ Agent may be overtrading (low hold %))
✓ Agent is using pattern signals (good!)
================================================================================

[6/6] Generating enhanced dashboard...

✓ Dashboard saved as 'trading_dashboard.png'


############# Run Next

RL training timesteps (default 50000): 50000

================================================================================
Running with your data and RL training (50000 steps)...
================================================================================
================================================================================
ENHANCED MULTI-PATTERN TRADING SYSTEM WITH RL
================================================================================

✓ Loaded 6156 bars of data
  Date range: 2001-01-01 00:00:00+05:30 to 2025-10-03 00:00:00+05:30

[1/6] Detecting patterns...
  ✓ Double Top: 224
  ✓ Double Bottom: 216
  ✓ Total: 440

[2/6] Training ML models...
  Success rate by pattern:
    DoubleTop: 19.6%
    DoubleBottom: 19.9%
  DoubleTop: Accuracy = 97.78%
  DoubleBottom: Accuracy = 84.09%

[3/6] Scoring pattern quality...
  ✓ High quality patterns (score >= 60): 62

[4/6] Running rule-based portfolio backtest...
  ✓ Executed 30 trades

[RULE-BASED RESULTS]:
  Total Trades   : 30
  Win Rate       : 100.0%
  Total Return   : 43.78%
  Final Capital  : $143,778.45

[5/6] Training RL agent...

================================================================================
TRAINING RL AGENT (PPO)
================================================================================

Training on 4924 bars...
✓ Training completed: 50000 timesteps

Evaluating on training data...

Evaluating on test data...

================================================================================
RL AGENT TRAINING SUMMARY
================================================================================
Algorithm              : PPO
Training Timesteps     : 50,000
Train Cumulative Reward: 732.0793
Test Cumulative Reward : 48.1599
================================================================================

================================================================================
STRATEGY COMPARISON: RL vs RULE-BASED
================================================================================

Strategy             Return %        Win Rate %      Trades     Sharpe
----------------------------------------------------------------------
RL Agent             405.52          56.2            857        6.72
Rule-Based           43.78           100.0           30         N/A
----------------------------------------------------------------------

⚠️  WARNING: RL trading too frequently (857 trades)
   Likely overtrading. Rule-based: {rule_trades} trades

✓ Winner: RL Agent (by 361.74%)

[5.5/6] Analyzing RL model interpretability...

================================================================================
RL MODEL INTERPRETABILITY ANALYSIS
================================================================================

[1/3] Collecting action data...
  ✓ Collected 2000 state-action pairs

[2/3] Analyzing action distribution...

Action Distribution:
  Hold      :    0 (  0.0%)
  Long      :  960 ( 48.0%)
  Flatten   : 1040 ( 52.0%)

[3/3] Analyzing feature importance...
  ✓ Surrogate model accuracy: 96.5%

Feature Importance:
  Price_Change   : 0.831
  Pattern_Signal : 0.083
  RSI            : 0.047
  MACD_Signal    : 0.028
  MACD           : 0.011

✓ Interpretability visualization saved as 'rl_interpretability.png'

================================================================================
KEY INSIGHTS
================================================================================
⚠ Agent may be overtrading (low hold %))
✓ Agent is using pattern signals (good!)
================================================================================

[6/6] Generating enhanced dashboard...

✓ Dashboard saved as 'trading_dashboard.png'




RL training timesteps (default 50000): 5000
================================================================================
COMPLETE TRADING SYSTEM WITH ENHANCED RL INTERPRETABILITY
================================================================================

✓ Loaded 6156 bars of data

[1/7] Detecting patterns...
  ✓ Total patterns: 440

[2/7] Training ML models...

================================================================================
ML TRAINING WITH CONFUSION MATRIX ANALYSIS
================================================================================

[DoubleTop] Training...
  Accuracy: 97.78%
  Precision: 0.00%
  Recall: 0.00%
  F1-Score: 0.00%

  Confusion Matrix:
    True Negatives  (TN):  44 - Correctly predicted failures
    False Positives (FP):   0 - Predicted success but failed ⚠️
    False Negatives (FN):   1 - Predicted failure but succeeded
    True Positives  (TP):   0 - Correctly predicted successes ✓
  False Positive Rate: 0.0% (lower is better)

[DoubleBottom] Training...
  Accuracy: 84.09%
  Precision: 25.00%
  Recall: 66.67%
  F1-Score: 36.36%

  Confusion Matrix:
    True Negatives  (TN):  35 - Correctly predicted failures
    False Positives (FP):   6 - Predicted success but failed ⚠️
    False Negatives (FN):   1 - Predicted failure but succeeded
    True Positives  (TP):   2 - Correctly predicted successes ✓
  ⚠️ WARNING: More false positives than true positives!
     This means you're taking many losing trades.
  False Positive Rate: 14.6% (lower is better)

================================================================================

Generating confusion matrix visualizations...
✓ Confusion matrices saved as 'ml_confusion_matrices.png'

[3/7] Scoring patterns...
  ✓ High quality patterns: 62

[4/7] Running portfolio backtest...

  Total Return: 43.78%

[5.5/7] Analyzing backtest with confusion matrix...

================================================================================
BACKTEST CONFUSION MATRIX ANALYSIS
================================================================================

Quality Threshold: 60
Total Trades: 30

Confusion Matrix:
[[ 0  0]
 [30  0]]

Interpretation:
  True Negatives  (TN):   0 - Low quality → Lost money ✓
  False Positives (FP):   0 - High quality → Lost money ⚠️ (BAD)
  False Negatives (FN):  30 - Low quality → Made money (Missed opportunity)
  True Positives  (TP):   0 - High quality → Made money ✓✓ (GOOD)

Metrics:
  Accuracy : 0.0% - Overall correctness
  Precision: 0.0% - When we predict success, we're right 0.0% of time
  Recall   : 0.0% - We catch 0.0% of all successful patterns

Business Impact:
  Total P&L: $43,778.45
  Winning Trades P&L: $43,778.45
  Losing Trades P&L: $0.00
================================================================================

Analyzing optimal quality threshold...
✓ Threshold analysis saved as 'threshold_analysis.png'

💡 Recommendation: Use threshold = 40 for maximum P&L

[5/7] Training RL agent...

================================================================================
TRAINING RL AGENT (PPO)
================================================================================

Training on 4924 bars...
✓ Training completed: 5000 timesteps

================================================================================
STRATEGY COMPARISON: RL vs RULE-BASED
================================================================================

Strategy             Return %        Win Rate %      Trades     Sharpe
----------------------------------------------------------------------
RL Agent             241.59          59.9            1569       3.22
Rule-Based           43.78           100.0           30         N/A
----------------------------------------------------------------------

✓ Winner: RL Agent (by 197.81%)

[6/7] Running enhanced interpretability analysis...

[1/6] Collecting data from 20 episodes...
  ✓ Collected 6000 state-action pairs

[2/6] Analyzing action distribution...

  Action Distribution:
    Hold      :     0 (  0.0%)
    Long      :   960 ( 16.0%) ████████
    Flatten   :  5040 ( 84.0%) ██████████████████████████████████████████

[3/6] Analyzing feature importance...

  Decision Tree Surrogate Accuracy: 99.7%
  Top 5 Features:
    Position            : 0.393 ███████████████
    RSI                 : 0.367 ██████████████
    Pattern_Signal      : 0.137 █████
    Price_Change        : 0.063 ██
    MACD                : 0.022

[4/6] Analyzing state-action relationships...

  Average State Values by Action:
           RSI  Pattern_Signal   MACD
Action
Flatten  0.447           0.001 -0.073
Long     0.689           0.023  0.098

[5/6] Generating visualizations...
  ✓ Saved as 'rl_interpretability_enhanced.png'

[6/6] Generating insights...

================================================================================
KEY INSIGHTS
================================================================================

[Trading Behavior]
  ⚠ Agent may be overtrading (<30% hold)

[Feature Usage]
  Top 3 Features:
    Position: 0.393
    RSI: 0.367
    Pattern_Signal: 0.137
  ✓ Agent is using pattern signals

[Performance]
  ✓ Positive average return: 15.0828
================================================================================

[7/7] Generating dashboard...

✓ Dashboard saved as 'trading_dashboard.png'

================================================================================
SYSTEM EXECUTION COMPLETE
================================================================================

================================================================================
CONFUSION MATRIX ENHANCEMENT MODULE
================================================================================

This module adds confusion matrix analysis to the trading system.

Key Features:
  1. ML Confusion Matrix - Shows pattern prediction accuracy
  2. Backtest Confusion Matrix - Compares predictions vs actual results
  3. Threshold Optimization - Finds optimal quality score cutoff

Integration:
  - Copy EnhancedPatternQualityScorer to replace existing scorer
  - Add BacktestConfusionMatrix after portfolio backtest
  - See integration instructions above
================================================================================
Program finished.
================================================================================

================================================================================
TRADING SYSTEM OPTIONS:
================================================================================
1. Quick Demo with RL + Enhanced Interpretability (~5 min)
2. Quick Demo without RL (~30 sec)
3. Use my own CSV file (with RL)
4. Use my own CSV file (no RL)
5. Exit

Enter choice (1-5): 1

Starting Quick Demo with RL and Enhanced Interpretability...
================================================================================
COMPLETE TRADING SYSTEM WITH ENHANCED RL INTERPRETABILITY
================================================================================
Generating sample data...

✓ Loaded 1000 bars of data

[1/7] Detecting patterns...
  ✓ Total patterns: 115

[2/7] Training ML models...
\n[2.5/7] Running pattern diagnostic...

================================================================================
PATTERN QUALITY DIAGNOSTIC
================================================================================

[1/6] DATA QUALITY CHECK
--------------------------------------------------------------------------------
✓ No missing values
checking date range
✓ Date range: 2020-01-01 00:00:00 to 2022-09-26 00:00:00
  Total days: 999
  Total bars: 1000
✓ Price range: 9597.61 to 12555.45
  Total change: 22.5%
✓ Average daily volatility: 0.37%
⚠️ WARNING: Very low volatility - patterns may not be meaningful

[2/6] PATTERN DISTRIBUTION
--------------------------------------------------------------------------------
Pattern counts:
  DoubleBottom: 58
  DoubleTop: 57

Patterns by year:
  2020: 42
  2021: 52
  2022: 21

Pattern spacing:
  Average bars between patterns: 8.2
  Min: 0, Max: 67
⚠️ WARNING: Patterns too close together - may be noise

[3/6] SUCCESS RATE ANALYSIS
--------------------------------------------------------------------------------
Overall success rate: 3.5% (4/115)

Success rate by pattern type:
  DoubleTop: 1.8% (1/57)
  DoubleBottom: 5.2% (3/58)

Critical Analysis:
🔴 CRITICAL: Success rate < 15% - Patterns are NOT predictive!
   → These patterns CANNOT be profitable even with perfect prediction

Win/Loss characteristics:
  Average win: 1.00
  Average loss: -0.30
  Win/Loss ratio: 3.36

[4/6] PATTERN CHARACTERISTICS
--------------------------------------------------------------------------------
Comparing successful vs failed patterns:
  RSI: Success=40.9, Fail=49.4
  Volatility: Success=0.0034, Fail=0.0037
  Pattern Height: Success=265.48, Fail=368.34
  Trend: Success=0.50, Fail=0.23

[5/6] MARKET CONDITIONS
--------------------------------------------------------------------------------
Market trend distribution:
  Uptrend: 60.7%
  Downtrend: 39.3%

Volatility regimes:
  High volatility: 24.5%
  Low volatility: 24.5%

Overall price movement: +22.49%

[6/6] TARGET/STOP-LOSS ANALYSIS
--------------------------------------------------------------------------------
Average distances from entry:
  Target: 3.44%
  Stop-Loss: 3.64%
  Risk/Reward ratio: 0.94:1
⚠️ WARNING: Risk/Reward < 1.5:1 - need >40% win rate to profit

================================================================================
RECOMMENDATIONS
================================================================================

CRITICAL PRIORITY #1:
Issue: Patterns have no predictive power (11% success)
Solutions:
  1. Try different pattern types (Head & Shoulders, Triangles, Flags)
  2. Use different timeframe (hourly if using daily, or vice versa)
  3. Try different asset class (this asset may not exhibit patterns)
  4. Increase pattern strictness (lower tolerance to 0.01)
  5. Consider this asset is not suitable for pattern trading

================================================================================

✓ Diagnostic visualization saved as 'pattern_diagnostic.png'

================================================================================
SMART ML TRAINING WITH RULE-BASED FALLBACK
================================================================================

[DoubleTop] Training...
  Class Distribution:
    Success: 1/57 (1.8%)
    Failure: 56/57 (98.2%)
  ⚠️ EXTREME IMBALANCE (1.8% success rate)
     → Using RULE-BASED scoring instead of ML
  → Computing rule-based scores...
     Average rule-based score: 56.2

[DoubleBottom] Training...
  Class Distribution:
    Success: 3/58 (5.2%)
    Failure: 55/58 (94.8%)
  ⚠️ EXTREME IMBALANCE (5.2% success rate)
     → Using RULE-BASED scoring instead of ML
  → Computing rule-based scores...
     Average rule-based score: 58.2

================================================================================

[3/7] Scoring patterns...

  💡 Adaptive Threshold Analysis:
     Target trades: 30
     Patterns available: 115
     Recommended threshold: 55.0
     This will give ~48 trades
  ✓ High quality patterns: 48

[4/7] Running portfolio backtest...

  Total Return: -13.15%

[5.5/7] Analyzing backtest with confusion matrix...

================================================================================
FIXED BACKTEST CONFUSION MATRIX
================================================================================

Quality Threshold: 60
Total Trades Analyzed: 30
Trades Taken (predicted=1): 2

Confusion Matrix:
[[26  2]
 [ 2  0]]

CORRECT Interpretation:
  True Positives  (TP):   0 - Took trade → Made money ✓✓
  False Positives (FP):   2 - Took trade → Lost money ⚠️
  True Negatives  (TN):  26 - Skipped trade → Would have lost ✓
  False Negatives (FN):   2 - Skipped trade → Missed profit

  Precision: 0.0%
    → When you take a trade, you win 0.0% of the time
  Recall: 0.0%
    → You catch 0.0% of all winning opportunities
================================================================================

Analyzing optimal quality threshold...
✓ Threshold analysis saved as 'threshold_analysis.png'

💡 Recommendation: Use threshold = 55 for maximum P&L

[5/7] Training RL agent...

================================================================================
TRAINING RL AGENT (PPO)
================================================================================

Training on 800 bars...
✓ Training completed: 50000 timesteps

================================================================================
STRATEGY COMPARISON: RL vs RULE-BASED
================================================================================

Strategy             Return %        Win Rate %      Trades     Sharpe
----------------------------------------------------------------------
RL Agent             31.42           49.1            165        5.94
Rule-Based           -13.15          6.7             30         N/A
----------------------------------------------------------------------

✓ Winner: RL Agent (by 44.57%)

[6/7] Running enhanced interpretability analysis...

[1/6] Collecting data from 20 episodes...
  ✓ Collected 6000 state-action pairs

[2/6] Analyzing action distribution...

  Action Distribution:
    Hold      :    40 (  0.7%)
    Long      :  3560 ( 59.3%) █████████████████████████████
    Flatten   :  2400 ( 40.0%) ████████████████████

[3/6] Analyzing feature importance...

  Decision Tree Surrogate Accuracy: 92.0%
  Top 5 Features:
    Pattern_Signal      : 0.261 ██████████
    RSI                 : 0.237 █████████
    Price_Change        : 0.220 ████████
    MACD_Signal         : 0.175 ██████
    MACD                : 0.084 ███

[4/6] Analyzing state-action relationships...

  Average State Values by Action:
           RSI  Pattern_Signal   MACD
Action
Flatten  0.487          -0.161  0.230
Hold     0.185           0.600 -0.277
Long     0.592           0.146  0.197

[5/6] Generating visualizations...
  ✓ Saved as 'rl_interpretability_enhanced.png'

[6/6] Generating insights...

================================================================================
KEY INSIGHTS
================================================================================

[Trading Behavior]
  ⚠ Agent may be overtrading (<30% hold)

[Feature Usage]
  Top 3 Features:
    Pattern_Signal: 0.261
    RSI: 0.237
    Price_Change: 0.220
  ✓ Agent is using pattern signals

[Performance]
  ✓ Positive average return: 10.7059
================================================================================

[7/7] Generating dashboard...

✓ Dashboard saved as 'trading_dashboard.png'

================================================================================
SYSTEM EXECUTION COMPLETE
================================================================================

================================================================================
CONFUSION MATRIX ENHANCEMENT MODULE
================================================================================

This module adds confusion matrix analysis to the trading system.

Key Features:
  1. ML Confusion Matrix - Shows pattern prediction accuracy
  2. Backtest Confusion Matrix - Compares predictions vs actual results
  3. Threshold Optimization - Finds optimal quality score cutoff

Integration:
  - Copy EnhancedPatternQualityScorer to replace existing scorer
  - Add BacktestConfusionMatrix after portfolio backtest
  - See integration instructions above
================================================================================
Program finished.
================================================================================



✓ Loaded 6156 bars of data

[1/7] Detecting patterns...
  ✓ Total patterns: 211

[2/7] Training ML models...
\n[2.5/7] Running pattern diagnostic...

================================================================================
PATTERN QUALITY DIAGNOSTIC - COMPREHENSIVE ANALYSIS
================================================================================

[1/6] DATA QUALITY CHECK
--------------------------------------------------------------------------------
⚠️ Found 232 missing values

Missing values by column (showing top 10):
  macd_signal_12_26: 33 (0.5%)
  macd_histogram_12_26: 33 (0.5%)
  macd_signal_strength: 33 (0.5%)
  macd_12_26: 25 (0.4%)
  volatility_20: 20 (0.3%)
  ma_20: 19 (0.3%)
  stoch_smoothd: 17 (0.3%)
  stoch_smoothk: 15 (0.2%)
  rsi_14: 13 (0.2%)
  stoch_14: 13 (0.2%)

✓ Missing values are acceptable (<5% of data)
  These are typically at the start due to indicator calculations

✓ Date range: 2001-01-01 to 2025-10-03
  Total days: 9041
  Total bars: 6156
  ⚠️ Large gaps detected in data

✓ Price statistics:
  Range: $854.20 to $26216.05
  Total return: +1884.71%
  Daily volatility: 1.34%
  Annualized volatility: 21.3%
  ⚠️ Very low volatility - patterns may be weak

[2/6] PATTERN DISTRIBUTION
--------------------------------------------------------------------------------
Total patterns detected: 211

Pattern distribution:
  DoubleTop: 106 (50.2%)
  DoubleBottom: 105 (49.8%)

Pattern spacing:
  Average: 29.1 bars
  Median: 11.0 bars
  Min: 0 bars
  Max: 403 bars
  ✓ Pattern spacing looks reasonable

[3/6] SUCCESS RATE ANALYSIS ⭐ CRITICAL
--------------------------------------------------------------------------------
📊 OVERALL SUCCESS RATE: 11.4% (24/211)

🎯 ASSESSMENT:
  🔴 CRITICAL FAILURE - Success rate < 12%
     These patterns have NO PREDICTIVE POWER
     Even perfect prediction cannot be profitable!

  ⚠️  THIS IS YOUR MAIN PROBLEM ⚠️


📈 Success by pattern type:
  ✗ DoubleTop: 11.3% (12/106)
  ✗ DoubleBottom: 11.4% (12/105)

💰 Profitability Analysis:
  Average win: 1.00
  Average loss: 0.49
  Win/Loss ratio: 2.03:1
  Breakeven win rate: 33.0%
  Actual win rate: 11.4%
  ✗ Win rate is too close to or below breakeven
     Need 39.6%+ to be reliably profitable

[4/6] PATTERN CHARACTERISTICS
--------------------------------------------------------------------------------
Analyzing 24 successful vs 187 failed patterns:

📊 Feature comparison (Successful vs Failed):
  RSI:
    Success: 46.535 | Failed: 51.935 ↓ (-10.4%)
  Volatility:
    Success: 0.008 | Failed: 0.009 ↓ (-10.9%)
  Pattern_Height:
    Success: 329.494 | Failed: 630.426 ↓ (-47.7%)
  Trend:
    Success: 0.250 | Failed: 0.230 ↑ (+8.7%)
  ATR:
    Success: 83.402 | Failed: 131.929 ↓ (-36.8%)

💡 Insight:
  Check if successful patterns have consistent characteristics
  Large differences indicate features that predict success

[5/6] MARKET CONDITIONS
--------------------------------------------------------------------------------
📈 Current trend: Uptrend
   Uptrend: 62.5% of time
   Downtrend: 37.5% of time

📊 Total price change: +1884.71%

📉 Volatility:
   Current: 0.47%
   Average: 1.15%

[6/6] TARGET/STOP-LOSS ANALYSIS
--------------------------------------------------------------------------------
📏 Average distances:
   Target: 6.28%
   Stop-Loss: 6.48%
   Risk/Reward: 0.96:1

  ⚠️ Low R:R requires >51% win rate to breakeven
  ⚠️ Stops very wide (6.48%) - high risk per trade


  Apply trend filter (5 minutes) - copy the code above
Increase targets by 1.5x (2 minutes) - change one line
Re-run diagnostic to confirm improvement
If still < 30% success → Switch to trend-following strategy entirely

