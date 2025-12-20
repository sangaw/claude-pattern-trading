
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradingConfig

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc



class PatternQualityScorer:
    """
    Smart ML scorer with rule-based fallback

    Features:
    - Handles extreme class imbalance
    - Automatic fallback to rule-based scoring
    - Proper XGBoost configuration
    - Per-pattern-type models
    """

    def __init__(self, config=None):
        """
        Initialize scorer
        
        Args:
            config: TradingConfig object
        """
        self.config = config or TradingConfig()
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.use_ml = {}
        self.baseline_scores = {}

    def train(self, labeled_patterns_df):
        """
        Train ML models or use rule-based fallback
        
        Args:
            labeled_patterns_df: DataFrame with patterns and Success labels
        
        Returns:
            dict: Training results per pattern type
        """
        if labeled_patterns_df.empty:
            print("⚠️ No patterns to train on")
            return {}
        
        pattern_types = labeled_patterns_df['Pattern_Type'].unique()
        results = {}
        
        print("\n" + "="*80)
        print("SMART ML TRAINING WITH RULE-BASED FALLBACK")
        print("="*80)
        
        for ptype in pattern_types:
            pattern_data = labeled_patterns_df[
                labeled_patterns_df['Pattern_Type'] == ptype
            ].copy()
            
            if len(pattern_data) < self.config.ML_MIN_SAMPLES:
                print(f"\n[{ptype}] Skipping - insufficient data ({len(pattern_data)} samples)")
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
            
            # DECISION: Use ML or Rule-Based?
            if success_rate < self.config.ML_IMBALANCE_THRESHOLD or success_rate > (1 - self.config.ML_IMBALANCE_THRESHOLD):
                print(f"  ⚠️ EXTREME IMBALANCE ({success_rate:.1%} success rate)")
                print(f"     → Using RULE-BASED scoring instead of ML")
                self._calculate_rule_based_scores(pattern_data, ptype)
                self.use_ml[ptype] = False
                continue
            
            # Try ML Training
            try:
                ml_result = self._train_ml_model(pattern_data, ptype)
                if ml_result['success']:
                    results[ptype] = ml_result
                else:
                    # ML failed, use rule-based
                    self._calculate_rule_based_scores(pattern_data, ptype)
                    self.use_ml[ptype] = False
            except Exception as e:
                print(f"  ⚠️ ML training failed: {e}")
                print(f"     → Falling back to RULE-BASED scoring")
                self._calculate_rule_based_scores(pattern_data, ptype)
                self.use_ml[ptype] = False
        
        print("\n" + "="*80)
        return results

    def _train_ml_model(self, pattern_data, ptype):
        """
        Train XGBoost model with proper configuration
        
        Args:
            pattern_data: DataFrame with pattern features
            ptype: Pattern type name
        
        Returns:
            dict: Training results
        """
        # Feature selection
        feature_cols = ['Pattern_Height', 'RSI', 'ATR', 'Volatility', 'Trend']
        feature_cols = [c for c in feature_cols if c in pattern_data.columns]
        
        X = pattern_data[feature_cols].fillna(0)
        y = pattern_data['Success']
        
        # Time-based split (important for trading!)
        split_idx = int(len(X) * self.config.ML_TRAIN_SPLIT)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Check if we have any positive samples in train set
        if y_train.sum() == 0:
            print(f"  ⚠️ No positive samples in training set")
            return {'success': False}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale_pos_weight properly
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        
        if pos_count == 0:
            print(f"  ⚠️ No positive samples for weight calculation")
            return {'success': False}
        
        scale_pos_weight = float(neg_count) / float(pos_count)
        
        # FIXED: XGBoost configuration for binary classification
        model = xgb.XGBClassifier(
            n_estimators=self.config.ML_N_ESTIMATORS,
            max_depth=self.config.ML_MAX_DEPTH,
            learning_rate=self.config.ML_LEARNING_RATE,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',  # FIXED: Explicit objective
            eval_metric='logloss',        # FIXED: Explicit metric
            use_label_encoder=False,      # FIXED: Disable deprecated encoder
            random_state=42,
            subsample=self.config.ML_SUBSAMPLE,
            colsample_bytree=self.config.ML_COLSAMPLE_BYTREE,
            min_child_weight=1,
            base_score=0.5                # FIXED: Proper base score for logistic
        )
        
        # Train model
        model.fit(
            X_train_scaled, 
            y_train,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        # Check if model is useful
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            if tp == 0 and fp == 0:
                print(f"  ⚠️ ML model never predicts positive class")
                return {'success': False}
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"  ✓ ML Accuracy: {accuracy:.2%}")
            print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}")
        
        # Store model
        self.models[ptype] = model
        self.scalers[ptype] = scaler
        self.feature_columns[ptype] = feature_cols
        self.use_ml[ptype] = True
        
        return {
            'success': True,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }

    def _calculate_rule_based_scores(self, pattern_data, ptype):
        """
        Calculate rule-based quality scores
        
        Uses technical indicators to estimate pattern quality when ML fails.
        
        Args:
            pattern_data: DataFrame with pattern features
            ptype: Pattern type ('DoubleTop' or 'DoubleBottom')
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
            median_height = pattern_data['Pattern_Height'].median()
            if pattern_height > median_height:
                score += 10
            
            # Trend component (10 points)
            trend = row.get('Trend', 0)
            if ptype == 'DoubleBottom' and trend > 0:
                score += 5  # Bullish pattern in uptrend
            elif ptype == 'DoubleTop' and trend < 0:
                score += 5  # Bearish pattern in downtrend
            
            scores.append(np.clip(score, 0, 100))
        
        # Store average score
        self.baseline_scores[ptype] = np.mean(scores)
        print(f"     Average rule-based score: {self.baseline_scores[ptype]:.1f}")

    def predict_quality(self, patterns_df):
        """
        Predict quality scores for patterns
        
        Args:
            patterns_df: DataFrame with patterns
        
        Returns:
            DataFrame with Quality_Score column added
        """
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
                scores = self._score_patterns_rule_based(pattern_data, ptype)
                patterns_with_scores.loc[mask, 'Quality_Score'] = scores
            else:
                # Use ML model
                try:
                    X = pattern_data[self.feature_columns[ptype]].fillna(0)
                    X_scaled = self.scalers[ptype].transform(X)
                    probabilities = self.models[ptype].predict_proba(X_scaled)[:, 1]
                    patterns_with_scores.loc[mask, 'Quality_Score'] = probabilities * 100
                except Exception as e:
                    print(f"⚠️ ML prediction failed for {ptype}: {e}")
                    # Fallback to rule-based
                    scores = self._score_patterns_rule_based(pattern_data, ptype)
                    patterns_with_scores.loc[mask, 'Quality_Score'] = scores
        
        return patterns_with_scores

    def _score_patterns_rule_based(self, pattern_data, ptype):
        """
        Score patterns using rule-based approach
        
        Args:
            pattern_data: DataFrame with patterns
            ptype: Pattern type
        
        Returns:
            List of scores
        """
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
        
        return scores
    
    def get_model_info(self):
        """
        Get information about trained models
        
        Returns:
            dict: Model information
        """
        info = {}
        for ptype in self.use_ml:
            info[ptype] = {
                'using_ml': self.use_ml[ptype],
                'has_model': ptype in self.models,
                'baseline_score': self.baseline_scores.get(ptype, None)
            }
        return info