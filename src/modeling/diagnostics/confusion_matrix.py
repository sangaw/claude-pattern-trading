import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc

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