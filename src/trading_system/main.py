# ============================================================================
# FILE: main.py
# Main execution script
# ============================================================================

"""
Main execution script for the trading system

Usage:
    python main.py
    
Or programmatically:
    from main import TradingSystem
    system = TradingSystem()
    results = system.run(df)
"""

import pandas as pd
from typing import Optional, Dict
from .config import TradingConfig
from .patterns.double_top import DoubleTopDetector
from .patterns.double_bottom import DoubleBottomDetector
from .visualization import TradingDashboard
from diagnostics.confusion_matrix import BacktestConfusionMatrix
from ml.labeler import PatternLabeler
from ml.scorer import PatternQualityScorer
from diagnostics.pattern_diagnostic import PatternDiagnostic
from portfolio.risk_manager import PortfolioRiskManager
from utils.helpers import (
    adaptive_quality_threshold,
    filter_patterns_by_trend,
    generate_sample_data
)


class TradingSystem:
    """
    Main trading system orchestrator
    
    Responsibilities:
    - Coordinate all components
    - Execute trading pipeline
    - Generate reports
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize trading system
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or TradingConfig()
        self.results = {}
    
    def run(
        self,
        df: pd.DataFrame,
        enable_rl: bool = False,
        run_diagnostic: bool = True
    ) -> Dict:
        """
        Run complete trading system
        
        Args:
            df: Price data
            enable_rl: Whether to train RL agent
            run_diagnostic: Whether to run diagnostics
        
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("TRADING SYSTEM EXECUTION")
        print("="*80)
        
        # Step 1: Detect Patterns
        print("\n[1/6] Detecting patterns...")
        all_patterns = self._detect_patterns(df)
        
        if all_patterns.empty:
            print(" No patterns detected")
            return {}
        
        # Step 2: Label Patterns
        print("\n[2/6] Labeling patterns...")
        labeled_patterns = self._label_patterns(df, all_patterns)
        
        # Step 3: Run Diagnostic (if enabled)
        if run_diagnostic:
            print("\n[3/6] Running diagnostic...")
            self._run_diagnostic(df, all_patterns, labeled_patterns)
        
        # Step 4: Train ML & Score
        print(f"\n[{4 if run_diagnostic else 3}/6] Training ML...")
        scored_patterns = self._train_and_score(labeled_patterns, all_patterns)
        
        # Step 5: Portfolio Backtest
        print(f"\n[{5 if run_diagnostic else 4}/6] Running backtest...")
        portfolio_results = self._run_backtest(df, scored_patterns)

        # Step 6: RL Training (if enabled)
        if enable_rl:
            print(f"\n[{6 if run_diagnostic else 5}/6] Training RL...")
            rl_results = self._train_rl(df, scored_patterns, portfolio_results)
            self.results['rl'] = rl_results
        
        # Step 7: Dashboard
        print("\n[7/7] Generating dashboard...")
            
        dashboard = TradingDashboard(portfolio_results["portfolio"], rl_results["comparison"])
        dashboard.create_dashboard()
        
        # Store results
        self.results.update({
            'patterns': all_patterns,
            'labeled_patterns': labeled_patterns,
            'scored_patterns': scored_patterns,
            'portfolio': portfolio_results
        })
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETE")
        print("="*80)
        
        return self.results
    
    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns"""
        detector_dt = DoubleTopDetector(df)
        patterns_dt = detector_dt.detect_patterns()
        
        detector_db = DoubleBottomDetector(df)
        patterns_db = detector_db.detect_patterns()
        
        all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)
        print(f"  ✓ Detected {len(all_patterns)} patterns")
        
        return all_patterns
    
    def _label_patterns(
        self,
        df: pd.DataFrame,
        patterns: pd.DataFrame
    ) -> pd.DataFrame:
        """Label patterns for ML"""
        labeler = PatternLabeler(df, patterns)
        labeled = labeler.label_patterns()
        
        success_rate = labeled['Success'].mean()
        print(f"  ✓ Success rate: {success_rate:.1%}")
        
        return labeled
    
    def _run_diagnostic(
        self,
        df: pd.DataFrame,
        patterns: pd.DataFrame,
        labeled_patterns: pd.DataFrame
    ):
        """Run comprehensive diagnostic"""
        diagnostic = PatternDiagnostic(df, patterns, labeled_patterns)
        diagnostic.run_full_diagnostic()
        diagnostic.visualize_diagnostics()
    
    def _train_and_score(
        self,
        labeled_patterns: pd.DataFrame,
        all_patterns: pd.DataFrame
    ) -> pd.DataFrame:
        """Train ML and score patterns"""
        scorer = PatternQualityScorer()
        scorer.train(labeled_patterns)
        scored = scorer.predict_quality(all_patterns)
        
        return scored
    
    def _run_backtest(
        self,
        df: pd.DataFrame,
        scored_patterns: pd.DataFrame
    ) -> Dict:
        """Run portfolio backtest"""
        # Filter high-quality patterns
        threshold = adaptive_quality_threshold(scored_patterns)
        high_quality = scored_patterns[
            scored_patterns['Quality_Score'] >= threshold
        ]
        
        # Initialize portfolio
        portfolio = PortfolioRiskManager(
            initial_capital=self.config.INITIAL_CAPITAL,
            max_positions=self.config.MAX_POSITIONS,
            risk_per_trade=self.config.RISK_PER_TRADE
        )
        
        # Execute trades (simplified)

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
        
        metrics = portfolio.get_metrics()
        print(f"  ✓ Return: {metrics.get('total_return', 0):.2f}%")
        
        # Backtest confusion matrix analysis
        print("\n[5.5/7] Analyzing backtest with confusion matrix...")
        backtest_cm = BacktestConfusionMatrix(scored_patterns, portfolio.closed_positions)
        cm, trade_df = backtest_cm.create_confusion_matrix(quality_threshold=60)
   
        if cm is not None:
            backtest_cm.visualize_threshold_analysis()  # Generates: threshold_analysis.png
        
        return {'portfolio': portfolio, 'metrics': metrics}
    
    def _train_rl(
        self,
        df: pd.DataFrame,
        scored_patterns: pd.DataFrame,
        portfolio_results: Dict
    ) -> Dict:
        """Train RL agent"""
        # Import RL components
        try:
            from rl.trainer import train_rl_agent, compare_strategies
            
            rl_model = train_rl_agent(
                df,
                scored_patterns,
                self.config.RL_ALGORITHM,
                timesteps=self.config.RL_TIMESTEPS
            )
            
            comparison = compare_strategies(
                df,
                scored_patterns,
                rl_model,
                portfolio_results['metrics']
            )

            return {'model': rl_model, 'comparison': comparison}
        except ImportError:
            print("  ⚠ RL libraries not available")
            return {}


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("TRADING SYSTEM OPTIONS:")
    print("="*80)
    print("1. Quick Demo with RL (~5 min)")
    print("2. Quick Demo without RL (~30 sec)")
    print("3. Use my own CSV file (with RL)")
    print("4. Use my own CSV file (no RL)")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    # Initialize system
    system = TradingSystem()
    
    if choice == '1':
        print("\nRunning demo with RL...")
        df = generate_sample_data()
        results = system.run(df, enable_rl=True)
        
    elif choice == '2':
        print("\nRunning demo without RL...")
        df = generate_sample_data()
        results = system.run(df, enable_rl=False)
        
    elif choice == '3':
        csv_file = input("\nEnter CSV file path: ").strip()
        df = pd.read_csv(csv_file)
        results = system.run(df, enable_rl=True)
        
    elif choice == '4':
        csv_file = input("\nEnter CSV file path: ").strip()
        df = pd.read_csv(csv_file)
        results = system.run(df, enable_rl=False)
        
    elif choice == '5':
        print("\nExiting...")
        return
    
    else:
        print("\n✗ Invalid choice")
        return
    
    print("\n" + "="*80)
    print("Program finished.")
    print("="*80)


if __name__ == "__main__":
    main()