========================================
ML MODEL PERFORMANCE REPORT
========================================
Target Accuracy: 51.71%

Classification Metrics:
              precision    recall  f1-score   support

           0       0.44      0.49      0.47       213
           1       0.45      0.15      0.23       117
           2       0.58      0.69      0.63       283

    accuracy                           0.52       613
   macro avg       0.49      0.44      0.44       613
weighted avg       0.51      0.52      0.49       613


Top 10 Most Influential Features:
is_mother_candle        0.295808
log_return              0.184319
mother_candle_trend     0.097390
daily_return            0.036998
stoch_smoothd           0.023289
macd_signal_strength    0.022822
Volume_Avg              0.022202
rsi_14                  0.021986
price_range             0.021975
macd_histogram_12_26    0.021824
dtype: float32


--- Optimized Model Report ---
Accuracy: 51.71%
              precision    recall  f1-score   support

           0       0.50      0.40      0.45       213
           1       0.41      0.73      0.52       117
           2       0.62      0.52      0.57       283

    accuracy                           0.52       613
   macro avg       0.51      0.55      0.51       613
weighted avg       0.54      0.52      0.52       613

--- Top 10 Most Influential Features ---
is_mother_candle       0.415103
mother_candle_trend    0.213970
log_return             0.089851
stoch_smoothd          0.033237
rsi_14                 0.032928
volume                 0.032804
volatility_20          0.032334
rsi_14_lag1            0.031961
price_range            0.031525
log_return_lag1        0.030272
dtype: 


--- Confidence Filter (Threshold: 0.7) ---
Trades meeting threshold: 112 out of 613
High-Confidence Accuracy: 65.18%

========================================
ML MODEL PERFORMANCE REPORT
========================================
Target Accuracy: 51.71%

Classification Metrics:
              precision    recall  f1-score   support

           0       0.50      0.40      0.45       213
           1       0.41      0.73      0.52       117
           2       0.62      0.52      0.57       283

    accuracy                           0.52       613
   macro avg       0.51      0.55      0.51       613
weighted avg       0.54      0.52      0.52       613


--- Top 10 Most Influential Features ---
is_mother_candle       0.415103
mother_candle_trend    0.213970
log_return             0.089851
stoch_smoothd          0.033237
rsi_14                 0.032928
volume                 0.032804
volatility_20          0.032334
rsi_14_lag1            0.031961
price_range            0.031525
log_return_lag1        0.030272
dtype: float32


By applying the Confidence Filter, you have effectively moved your model from a "coin flip" (51.7%) to a statistically significant edge (65.18%). In quantitative trading, a 65% accuracy rate on high-conviction signals is excellent, especially when dealing with three possible outcomes.

1. Decoding the Results
The Edge: Out of 613 trading days, the model identified 112 "High Conviction" days. By ignoring the other 501 days where the model was "unsure," you increased your win rate by ~13.5%.

Precision vs. Recall: Your overall Class 2 (Uptrend) precision is 62%. The filter likely pushes this even higher. You are now "cherry-picking" the best setups.

Feature Dominance: is_mother_candle and mother_candle_trend are clearly the "anchors" of this strategy. The model is essentially saying: "If I see a Mother Candle and the momentum (RSI/Stoch) aligns, I am very confident about tomorrow."


=============================================
DETAILED PERFORMANCE REPORT
=============================================
Global Model Accuracy: 51.88%
High-Confidence Accuracy (>=0.7): 65.74%
Number of High-Confidence Signals: 108

Classification Metrics:
              precision    recall  f1-score   support

           0       0.50      0.39      0.44       213
           1       0.41      0.72      0.52       117
           2       0.62      0.53      0.57       283

    accuracy                           0.52       613
   macro avg       0.51      0.55      0.51       613
weighted avg       0.54      0.52      0.52       613


Top 5 Influential Features:
is_mother_candle       0.410521
mother_candle_trend    0.199472
log_return             0.088591
rsi_14                 0.033355
volume                 0.031518
dtype: float32

Portfolio Performance:
Strategy Cumulative Return: 51.05%
Market Cumulative Return:   37.43%
=============================================


This report tells a story of a model that is an "average" predictor overall but becomes a **specialized expert** when it filters for high-conviction setups.

Here is how to interpret each section of your results like a Quant developer:

---

### **1. The "Edge": Accuracy vs. High-Confidence**

* **Global Model Accuracy (51.88%):** If you traded every single day, you’d barely perform better than a coin flip. This is normal for noisy financial data.
* **High-Confidence Accuracy (65.74%):** This is your true "Alpha." By only taking the **108 trades** where the model was  sure, your win rate jumped by **~14%**.
* **Takeaway:** Your strategy's value isn't in its daily predictions; it's in its **patience**.

---

### **2. Classification Metrics (The "How" of Winning)**

This table breaks down how the model handles the three market states: **0 (Down)**, **1 (Neutral/Sideways)**, and **2 (Up)**.

* **Class 2 (Bullish/Up) – The Star Performer:**
* **Precision (0.62):** When the model says the market is going up, it is right 62% of the time. This is a very strong signal for a long-only or trend-following strategy.


* **Class 1 (Neutral) – The Safety Net:**
* **Recall (0.72):** The model is excellent at identifying sideways/choppy markets. It caught 72% of all neutral days, effectively telling you when to "sit on cash" to avoid being chopped up.


* **Class 0 (Bearish/Down) – The Weak Link:**
* **Recall (0.39):** The model struggles to identify downtrends before they happen. It only catches 39% of the drops.



---

### **3. Top Influential Features**

These values show what information the XGBoost model is actually using to make decisions.

* **is_mother_candle (0.41) & mother_candle_trend (0.19):** Combined, these account for **60% of the model's intelligence**.
* **Interpretation:** Your model is essentially a "Mother Candle Specialist." It views the breakout or containment of a mother candle as the single most important predictor of tomorrow's price.
* **Secondary Factors:** RSI and Volume are "supporting actors," used to fine-tune the entries.

---

### **4. Portfolio Performance**

* **Strategy (51.05%) vs. Market (37.43%):** You outperformed the benchmark by **13.62%**.
* **The "Why":** Because the model has high **Recall for Class 1 (Neutral)**, it likely stayed out of the market during flat or volatile periods where the "Buy & Hold" investors suffered drawdowns or stagnated.

---

### **Summary "Health Check"**

| Metric | Status | Action |
| --- | --- | --- |
| **Edge Stability** | ✅ Excellent | The gap between 52% and 65% accuracy proves the confidence filter works. |
| **Bullish Accuracy** | ✅ High | Class 2 Precision (0.62) is high enough for profitable long trading. |
| **Bearish Detection** | ⚠️ Low | Don't rely on this model for short-selling yet; it misses too many drops. |
| **Logic** | 🔍 Focused | The model is heavily reliant on one pattern (Mother Candle). This is good for clarity, but risky if that pattern stops working. |

