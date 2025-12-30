from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradingConfig

class PatternQualityScorer:
    """
    Smart ML scorer with rule-based fallback.
    Supports DoubleTop, DoubleBottom, and BullishFlag patterns.
    """

    def __init__(self, config=None):
        self.config = config or TradingConfig()
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.use_ml = {}
        self.baseline_scores = {}

    def train(self, labeled_patterns_df):
        if labeled_patterns_df.empty:
            print("⚠️ No patterns to train on")
            return {}
        
        pattern_types = labeled_patterns_df['Pattern_Type'].unique()
        results = {}
        
        print("\n" + "="*80)
        print("SMART ML TRAINING WITH RULE-BASED FALLBACK")
        print("="*80)
        
        for ptype in pattern_types:
            pattern_data = labeled_patterns_df[labeled_patterns_df['Pattern_Type'] == ptype].copy()
            
            if len(pattern_data) < self.config.ML_MIN_SAMPLES:
                print(f"\n[{ptype}] Skipping ML - insufficient data ({len(pattern_data)} samples)")
                self._calculate_rule_based_scores(pattern_data, ptype)
                self.use_ml[ptype] = False
                continue
            
            print(f"\n[{ptype}] Training...")
            success_count = pattern_data['Success'].sum()
            total_count = len(pattern_data)
            success_rate = success_count / total_count
            
            print(f"   Class Distribution: Success {success_count}/{total_count} ({success_rate:.1%})")
            
            # Decision: Use ML or Rule-Based?
            if success_rate < self.config.ML_IMBALANCE_THRESHOLD or success_rate > (1 - self.config.ML_IMBALANCE_THRESHOLD):
                print(f"   ⚠️ EXTREME IMBALANCE: Using RULE-BASED fallback")
                self._calculate_rule_based_scores(pattern_data, ptype)
                self.use_ml[ptype] = False
                continue
            
            try:
                ml_result = self._train_ml_model(pattern_data, ptype)
                if ml_result['success']:
                    results[ptype] = ml_result
                else:
                    self._calculate_rule_based_scores(pattern_data, ptype)
                    self.use_ml[ptype] = False
            except Exception as e:
                print(f"   ⚠️ ML training failed: {e}")
                self._calculate_rule_based_scores(pattern_data, ptype)
                self.use_ml[ptype] = False
        
        print("\n" + "="*80)
        return results

    def _train_ml_model(self, pattern_data, ptype):
        # 1. Feature Engineering & Selection
        feature_cols = ['Pattern_Height', 'RSI', 'ATR', 'Volatility', 'Trend', 'Pole_Height', 'Flag_Tightness']
        print(pattern_data.columns)
        
        # Ensure column alignment for Bullish Flags (Pole_Height maps to Pattern_Height)
        if ptype == 'BullishFlag' and 'Pole_Height' in pattern_data.columns:
            if 'Pattern_Height' not in pattern_data.columns or pattern_data['Pattern_Height'].isna().all():
                pattern_data['Pattern_Height'] = pattern_data['Pole_Height']

        available_cols = [c for c in feature_cols if c in pattern_data.columns]
        
        # If critical indicators are missing, ML will be weak. 
        # We should log a warning but allow it to try.
        if len(available_cols) < 3:
            print(f"  Very few features available for ML: {available_cols}")

        # 2. Strict Casting
        X = pattern_data[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pattern_data['Success'].astype(int)

        # --- THE FIX FOR THE SPLIT ERROR ---
        if len(pattern_data) < 50:
            # For tiny datasets, don't use a time-split for training validation
            # Use the whole set just to see if a model CAN be fit (overfitting risk, but stops the crash)
            X_train, X_test = X, X
            y_train, y_test = y, y
            print("   Small dataset: Using full-set training (skipping split)")
        else:
            split_idx = int(len(X) * self.config.ML_TRAIN_SPLIT)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if y_train.sum() == 0 or y_test.sum() == 0:
            print("   ⚠️ Insufficient class variety in split")
            return {'success': False}
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=self.config.ML_N_ESTIMATORS,
            max_depth=self.config.ML_MAX_DEPTH,
            learning_rate=self.config.ML_LEARNING_RATE,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            base_score=0.5
        )
        
        model.fit(X_train_scaled, y_train, verbose=False)
        
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2, 2) and (cm[1,1] + cm[0,1]) > 0:
            print(f"   ✓ ML Accuracy: {(y_pred == y_test).mean():.2%}")
            self.models[ptype] = model
            self.scalers[ptype] = scaler
            self.feature_columns[ptype] = available_cols
            self.use_ml[ptype] = True
            return {'success': True, 'confusion_matrix': cm}
        
        print("   ⚠️ ML model failed to find predictive edge")
        return {'success': False}

    def _calculate_rule_based_scores(self, pattern_data, ptype):
        """Standardizes rule-based scores across all pattern types."""
        print(f"   → Computing rule-based scores for {ptype}...")
        
        # Determine the median height for the 'Small Pattern' insight
        height_col = 'Pole_Height' if ptype == 'BullishFlag' else 'Pattern_Height'
        median_height = pattern_data[height_col].median() if height_col in pattern_data.columns else 0

        scores = []
        for _, row in pattern_data.iterrows():
            score = 55.0  # Base neutral score
            
            # 1. RSI Logic
            rsi = row.get('RSI', 50)
            if ptype in ['DoubleBottom', 'BullishFlag']:
                if rsi < 30: score += 15
                elif rsi > 70: score -= 10
            elif ptype == 'DoubleTop':
                if rsi > 70: score += 15
                elif rsi < 30: score -= 10
            
            # 2. Pattern Height (Your Discovery: Successes are ~22.8% smaller)
            current_height = row.get(height_col, 0)
            if current_height > 0 and current_height < median_height:
                score += 10
            
            # 3. Volatility Check
            vol = row.get('Volatility', 0)
            if vol > 0.015: score += 10
            
            # 4. Bullish Flag Specific Logic
            if ptype == 'BullishFlag':
                tightness = row.get('Flag_Tightness', 1.0)
                pole = row.get('Pole_Height', 1.0)
                if pole > 0 and (tightness / pole) < 0.12:
                    score += 15  # Tight flags are much higher quality

            # 5. Trend Alignment
            trend = row.get('Trend', 0)
            if ptype in ['DoubleBottom', 'BullishFlag'] and trend > 0: score += 5
            elif ptype == 'DoubleTop' and trend < 0: score += 5
            
            scores.append(np.clip(score, 0, 100))
        
        self.baseline_scores[ptype] = np.mean(scores) if scores else 50.0
        return scores

    def predict_quality(self, patterns_df):
        if patterns_df.empty:
            return patterns_df
        
        output_df = patterns_df.copy()
        output_df['Quality_Score'] = 50.0
        
        for ptype in patterns_df['Pattern_Type'].unique():
            mask = output_df['Pattern_Type'] == ptype
            subset = output_df[mask]
            
            if self.use_ml.get(ptype, False):
                try:
                    # Prepare ML features strictly
                    X = subset[self.feature_columns[ptype]].apply(pd.to_numeric, errors='coerce').fillna(0)
                    X_scaled = self.scalers[ptype].transform(X)
                    probs = self.models[ptype].predict_proba(X_scaled)[:, 1]
                    output_df.loc[mask, 'Quality_Score'] = probs * 100
                except Exception:
                    output_df.loc[mask, 'Quality_Score'] = self._calculate_rule_based_scores(subset, ptype)
            else:
                output_df.loc[mask, 'Quality_Score'] = self._calculate_rule_based_scores(subset, ptype)
                
        return output_df

    def get_model_info(self):
        return {ptype: {'ml': self.use_ml.get(ptype, False), 'base': self.baseline_scores.get(ptype, 50)} 
                for ptype in self.use_ml}