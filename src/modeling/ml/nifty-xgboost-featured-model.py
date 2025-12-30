import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils import class_weight

# --- 1. DATA PREPARATION ---
def prepare_market_data(df):
    # Normalize column names to lowercase to prevent KeyErrors
    df.columns = [col.lower() for col in df.columns]
    
    if 'close' not in df.columns:
        if 'price' in df.columns: df = df.rename(columns={'price': 'close'})
        elif 'adj close' in df.columns: df = df.rename(columns={'adj close': 'close'})

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    # Momentum Lags
    if 'rsi_14' in df.columns: df['rsi_14_lag1'] = df['rsi_14'].shift(1)
    if 'log_return' in df.columns: df['log_return_lag1'] = df['log_return'].shift(1)

    # Market Regime Clustering
    cluster_features = ['rsi_14', 'volatility_20', 'volume', 'daily_return']
    cluster_features = [f for f in cluster_features if f in df.columns]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features].ffill().fillna(0))
    df['cluster_id'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled_data)

    # Casting for XGBoost compatibility
    if 'is_mother_candle' in df.columns: 
        df['is_mother_candle'] = df['is_mother_candle'].astype(int)
    
    for col in df.columns:
        if df[col].dtype == 'object': 
            df[col] = df[col].astype('category')
    
    df['cluster_id'] = df['cluster_id'].astype('category')
    if 'mother_candle_trend' in df.columns: 
        df['mother_candle_trend'] = df['mother_candle_trend'].astype('category')

    # Shift trend code to create the target (predicting tomorrow)
    df['target_trend'] = df['trend_code'].shift(-1)
    df.dropna(subset=['target_trend'], inplace=True)
    return df

# --- 2. MODEL TRAINING ---
def train_predictive_model(df):
    features = [
        'volume', 'log_return', 'price_range', 'volatility_20', 'rsi_14', 
        'macd_12_26', 'macd_signal_strength', 'stoch_smoothd',
        'is_mother_candle', 'mother_candle_trend', 'cluster_id',
        'rsi_14_lag1', 'log_return_lag1'
    ]
    features = [f for f in features if f in df.columns]
    X, y = df[features], df['target_trend'].astype(int)

    # Balanced class weights for better recall on smaller classes
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
    weights = np.array(weights, dtype=np.float32).flatten()
    
    # Time-series split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.02,
        objective='multi:softprob', tree_method='hist',
        enable_categorical=True, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    
    return model, y_test, model.predict(X_test), model.predict_proba(X_test), features

# --- 3. TRADE LOG & PERFORMANCE ---
def generate_trade_report(df, y_test, y_pred, y_proba, threshold=0.70, initial_capital=100000):
    test_df = df.loc[y_test.index].copy().reset_index(drop=True)
    test_df['confidence'] = np.max(y_proba, axis=1)
    test_df['predicted_signal'] = y_pred
    
    trades = []
    equity = initial_capital
    trade_id = 1
    
    # Identify high-confidence indices
    signal_indices = test_df[test_df['confidence'] >= threshold].index
    
    for i in signal_indices:
        if i + 1 >= len(test_df): continue
            
        row, next_row = test_df.iloc[i], test_df.iloc[i+1]
        entry_price, exit_price = row['close'], next_row['close']
        direction = "LONG" if row['predicted_signal'] == 2 else "SHORT" if row['predicted_signal'] == 0 else None
        
        if not direction: continue

        # Logic for Stop Loss, Target, and Shares (Risk 2% per trade)
        stop_loss = entry_price * 0.97 if direction == "LONG" else entry_price * 1.03
        target_val = entry_price * 1.06 if direction == "LONG" else entry_price * 0.94
        shares = (equity * 0.02) / abs(entry_price - stop_loss)
        
        pnl = (exit_price - entry_price) * shares if direction == "LONG" else (entry_price - exit_price) * shares
        equity += pnl
        
        trades.append({
            'id': trade_id, 'engine_type': 'ML-Model', 
            'pattern_type': row.get('mother_candle_trend', 'ML-Regime'),
            'entry_date': row['date'], 'entry_price': entry_price, 'shares': shares,
            'stop_loss': stop_loss, 'target': target_val, 'unrealized_pnl': 0,
            'exit_price': exit_price, 'exit_date': next_row['date'], 'pnl': pnl,
            'return_pct': (pnl / (entry_price * shares)), 'equity_curve': equity
        })
        trade_id += 1

    # Daily strategy return for portfolio comparison
    test_df['strat_ret'] = 0.0
    conf_mask = (test_df['confidence'].values >= threshold)
    test_df.loc[conf_mask & (test_df['predicted_signal'] == 2), 'strat_ret'] = test_df['daily_return']
    test_df.loc[conf_mask & (test_df['predicted_signal'] == 0), 'strat_ret'] = -test_df['daily_return']
    
    test_df['cum_strat'] = (1 + test_df['strat_ret']).cumprod()
    test_df['cum_mkt'] = (1 + test_df['daily_return']).cumprod()

    trade_df = pd.DataFrame(trades)
    os.makedirs('../data/output/', exist_ok=True)
    trade_df.to_csv('../data/output/trades_report.csv', index=False)
    
    return trade_df, test_df

# --- 4. MAIN ---
def main():
    try:
        # Load Data
        df = pd.read_csv('../data/input/full-featured.csv')
        processed_df = prepare_market_data(df)
        
        # Train & Predict
        model, y_test, y_pred, y_proba, features = train_predictive_model(processed_df)
        
        # Backtest
        trade_log, daily_results = generate_trade_report(processed_df, y_test, y_pred, y_proba)
        
        # --- THE DETAILED PERFORMANCE REPORT ---
        high_conf = daily_results[daily_results['confidence'] >= 0.70]
        
        print("\n" + "="*45)
        print("DETAILED PERFORMANCE REPORT")
        print("="*45)
        print(f"Global Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print(f"High-Confidence Accuracy (>=0.7): {accuracy_score(high_conf['target_trend'], high_conf['predicted_signal']):.2%}")
        print(f"Number of High-Confidence Signals: {len(high_conf)}")
        
        print("\nClassification Metrics:")
        print(classification_report(y_test, y_pred))

        print("\nTop 5 Influential Features:")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print(importances.head(5))

        print("\nPortfolio Performance:")
        strat_final = (daily_results['cum_strat'].iloc[-1] - 1) * 100
        mkt_final = (daily_results['cum_mkt'].iloc[-1] - 1) * 100
        print(f"Strategy Cumulative Return: {strat_final:.2f}%")
        print(f"Market Cumulative Return:   {mkt_final:.2f}%")
        print("="*45)
        
        # Visualize Results
        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Equity Curve
        ax1.plot(daily_results['date'], daily_results['cum_mkt'], label='Market (Index)', color='gray', alpha=0.5)
        ax1.plot(daily_results['date'], daily_results['cum_strat'], label='ML Strategy', color='blue', linewidth=2)
        ax1.set_title('Cumulative Returns')
        ax1.legend()

        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['Down (0)', 'Neutral (1)', 'Up (2)'],
                    yticklabels=['Down (0)', 'Neutral (1)', 'Up (2)'])
        ax2.set_title('Model Confusion Matrix (Leakage Detection)')
        ax2.set_ylabel('Actual Market State')
        ax2.set_xlabel('Model Prediction')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()