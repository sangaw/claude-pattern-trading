

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