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
from .patterns.bullish_flag import BullishFlagDetector
from .visualization import TradingDashboard
from diagnostics.confusion_matrix import BacktestConfusionMatrix
from diagnostics.data_quality_check import DataQualityCheck
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

        self.results = {}
        rl_results = None
        
        # Step 1: Data Quality Check
        print("\n[1/7] Perform Data Checks...")
        self._data_quality_checks(df)
        df_imputed = self._impute_missing_values(df)
        
        df = df_imputed
        print("ready for data science")
        print(df.columns)


        # Step 2: Detect Patterns
        print("\n[2/7] Detecting patterns...")
        all_patterns = self._detect_patterns(df)
        
        if all_patterns.empty:
            print(" No patterns detected")
            return {}
        
        print(f"DEBUG: Found columns: {all_patterns.columns.tolist()}")

        # Step 3: Label Patterns
        print("\n[3/7] Labeling patterns...")
        labeled_patterns = self._label_patterns(df, all_patterns)
        
        # Step 4: Run Diagnostic (if enabled)
        if run_diagnostic:
            print("\n[4/7] Running diagnostic...")
            self._run_diagnostic(df, all_patterns, labeled_patterns)
        
        # Step 5: Train ML & Score
        print(f"\n[{5/7 if run_diagnostic else 6}/7] Training ML...")
        scored_patterns = self._train_and_score(labeled_patterns, all_patterns)
        
       # --- Step 6: Backtest (Rule-Based / ML) ---
        print(f"\n[6/7] Running Backtest (Mode: {'ML' if self.config.ENABLE_ML else 'Rule-Based'})...")

        # Capture the dictionary: contains {"portfolio": object, "total_return": value, ...}
        portfolio_results = self._run_backtest(
            df, 
            scored_patterns, 
            engine_type="ML-Model" if self.config.ENABLE_ML else "Rule-Based"
        )

        rl_results = None

        # --- Step 7: RL Training ---
        if enable_rl:
            print(f"\n[7/7] Training RL...")
            # PASS THE WHOLE DICTIONARY portfolio_results
            rl_results = self._train_rl(
                df, 
                scored_patterns, 
                portfolio_results  # <--- Changed from portfolio_results["portfolio"]
            )
        
        # Step 8: Dashboard
        print("\n[7/7] Generating dashboard...")
            
        # This will now work perfectly because rl_results is at least None
        rl_metrics = rl_results if (enable_rl and isinstance(rl_results, dict)) else None
            
        # Call with the corrected argument names
        dashboard = TradingDashboard(
            portfolio_manager=portfolio_results["portfolio"], 
            rl_metrics=rl_metrics
        )
        dashboard.create_dashboard()
        
        # 4. Consolidate final results dictionary
        self.results.update({
            'patterns': all_patterns,
            'labeled_patterns': labeled_patterns,
            'scored_patterns': scored_patterns,
            'portfolio_results': portfolio_results,
            'rl_results': rl_results
        })
        
        print("\n" + "="*80)
        print(f"  EXECUTION COMPLETE | Mode: {'ML + RL' if enable_rl else 'ML Only'}")
        print("="*80)
        
        return self.results
    
    def _evaluate_rl(self, df, agent):
        # 1. Use the same unified Portfolio Manager
        rl_portfolio = PortfolioRiskManager(initial_capital=self.config.INITIAL_CAPITAL)
        
        # 2. Create the Env for evaluation
        env = TradingEnv(df=df, portfolio=rl_portfolio)
        obs, _ = env.reset()
        
        done = False
        while not done:
            # Agent picks action
            action, _ = agent.predict(obs) 
            # Env step calls portfolio.open_position(..., engine_type="RL-Agent")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        # 3. CALL THE UNIFIED METRICS
        # This generates 'trade_log_RL-AGENT_datetime.csv'
        return rl_portfolio.get_metrics(engine_name="RL-Agent")


    def _data_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform Data Quality checks"""
        data_quality_check = DataQualityCheck(df)
        data_quality_check.run_data_quality_check()
        print(f"  ✓ Checked Data Quality")

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform Data Quality checks"""
        data_quality_check = DataQualityCheck(df)
        df_imputed = data_quality_check.impute_missing_values()
        print(f"  ✓ Perfomed Impute Values")
        return df_imputed
    

    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns"""
        # detector_dt = DoubleTopDetector(df)
        # patterns_dt = detector_dt.detect_patterns()
        
        # detector_db = DoubleBottomDetector(df)
        # patterns_db = detector_db.detect_patterns()

        detector_bf = BullishFlagDetector(df)
        patterns_bf = detector_bf.detect_patterns()
        
        # all_patterns = pd.concat([patterns_dt, patterns_db], ignore_index=True)

        all_patterns = pd.concat([patterns_bf], ignore_index=True)
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
    
    def _standardize_column_names(self, patterns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mapping layer to ensure compatibility between legacy Detectors 
        and the new Helper/Labeler requirements.
        """
        if patterns_df.empty:
            return patterns_df

        # Define your mapping: {'Existing_Name': 'New_Required_Name'}
        mapping = {
            'Confirmed_At': 'Detection_Index',
            'breakout_idx': 'Detection_Index',
            'index': 'Detection_Index',
            'Date': 'Detection_Date',
            'timestamp': 'Detection_Date'
        }
        
        # Only rename columns that actually exist in the dataframe
        to_rename = {k: v for k, v in mapping.items() if k in patterns_df.columns}
        
        return patterns_df.rename(columns=to_rename)

    def _apply_advanced_filters(self, market_row):
        """
        Returns (True, "Reason") if filters pass, or (False, "Failure Reason").
        """
        # 1. Trend Filter (SMA 200)
        sma_200 = market_row.get('SMA_200', 0)
        if sma_200 > 0 and market_row['Close'] < sma_200:
            return False, "BEARISH_TREND"
        
        # 2. RSI Filter (Momentum check)
        rsi = market_row.get('RSI', 50)
        if rsi > 75: return False, "OVERBOUGHT"
        if rsi < 40: return False, "LOW_MOMENTUM"
        
        # 3. Volume Filter (Relative to 20-day average)
        vol_avg = market_row.get('Volume_Avg', 1)
        if market_row['volume'] < (vol_avg * 1.1):
            return False, "LOW_VOLUME"
            
        return True, "PROCEED"

    def _run_backtest(
        self,
        df: pd.DataFrame,
        scored_patterns: pd.DataFrame,
        engine_type: str = "Rule-Based" # Added parameter for reusability
    ) -> Dict:
        
        """
        Complete Portfolio Backtest
        Handles multiple pattern types (Flags/Double Patterns) and price gaps.
        """
        # Determine the best threshold for the current pattern set
        threshold = adaptive_quality_threshold(scored_patterns)
        high_quality = scored_patterns[scored_patterns['Quality_Score'] >= threshold]

        # print("high_quality")
        # print(high_quality)
        
        portfolio = PortfolioRiskManager(
            initial_capital=self.config.INITIAL_CAPITAL,
            max_positions=self.config.MAX_POSITIONS,
            risk_per_trade=self.config.RISK_PER_TRADE
        )
        
        trades_count = 0
        
        # Sort by quality to ensure we look at the best patterns first
        high_quality = high_quality.sort_values('Quality_Score', ascending=False)

        for idx, pattern in high_quality.iterrows():
            # 1. DATA ALIGNMENT
            entry_idx = int(pattern['Confirmed_At'])
            market_context = df.iloc[entry_idx]

            # --- THE FILTER CALL ---
            passed, reason = self._apply_advanced_filters(market_context)
            if not passed:
                # print(f"Trade {idx} skipped: {reason}")
                continue

            # 2. PORTFOLIO EVALUATION
            eval_result = portfolio.evaluate_new_trade(pattern)
            
            if eval_result.get('approved'):
                # Handle Entry Price Mapping
                entry_price = pattern.get('Entry_Price', pattern.get('Neckline'))
                if pd.isna(entry_price): continue
                
                # OPEN
                # Determine your engine label based on the current system mode
                current_engine = "ML-Engine" if self.config.ENABLE_ML else "Rule-Based"

                # Pass the engine_type to the open_position call
                active_trade = portfolio.open_position(
                    pattern, 
                    eval_result, 
                    entry_price, 
                    engine_type=engine_type
                )
                
                # Retrieve ID (Handles both Dict and Object returns)
                t_id = active_trade.get('id') if isinstance(active_trade, dict) else getattr(active_trade, 'id', 0)

                trades_count += 1
                exit_price = None
                
            # 4. SCAN NEXT 30 BARS FOR EXIT (Consistently using iloc)
            for i in range(entry_idx + 1, min(entry_idx + 31, len(df))):
                future_bar = df.iloc[i]
                
                if future_bar['Low'] <= pattern['Stop_Loss']:
                    exit_price = pattern['Stop_Loss']
                    break
                elif future_bar['High'] >= pattern['Target']:
                    exit_price = pattern['Target']
                    break
            
            # 5. FINAL EXIT & DATE
            if exit_price is None:
                exit_pos = min(entry_idx + 30, len(df) - 1)
                exit_price = df.iloc[exit_pos]['Close']
            
            final_exit_idx = min(entry_idx + 30, len(df) - 1)
            exit_date = df.iloc[final_exit_idx].get('date', df.index[final_exit_idx])
            
            # CLOSE
            portfolio.close_position(t_id, exit_price, exit_date)
            
            if trades_count >= 30: 
                break

        mode_label = "ML-Model" if self.config.ENABLE_ML else "Rule-Based"
        metrics = portfolio.get_metrics(engine_name=mode_label)
        
        # FINAL DEBUG PRINT
        print(f"DEBUG: Active Positions Remaining: {len(portfolio.positions)}")
        print(f"DEBUG: Closed Trades in Ledger: {len(portfolio.trade_history)}")
        
        portfolio_results = {
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'total_return': metrics['total_return'],
        }
        
        print(f"\n  Total Return: {metrics['total_return']:.2f}%")
        
        metrics = portfolio.get_metrics()
        print(f"  ✓ Return: {metrics.get('total_return', 0):.2f}%")
        
        # Backtest confusion matrix analysis
        # [5.5/7] Analyzing backtest confusion matrix
        print("\n[5.5/7] Analyzing backtest with confusion matrix...")
        backtest_cm = BacktestConfusionMatrix(scored_patterns, portfolio.closed_positions)
        
        # FIX: Use the same threshold that was used for the backtest
        # Instead of a hard-coded 60, use the 'threshold' variable calculated earlier
        cm_result = backtest_cm.create_confusion_matrix(quality_threshold=threshold)
   
        if cm_result is not None:
            cm, trade_df = cm_result
            backtest_cm.visualize_threshold_analysis()
        else:
            print(f" ⚠️ Skipping Confusion Matrix: No trades found at threshold {threshold:.2f}")

        return {'portfolio': portfolio, 'metrics': metrics}
    
    def _train_rl(self, df, scored_patterns, portfolio_results_dict):
        """
        Train RL agent and compare against the provided backtest results.
        """
        try:
            from modeling.rl.trainer import train_rl_agent, compare_strategies
            
            # 1. Extract the manager object from the dictionary
            portfolio_manager = portfolio_results_dict["portfolio"]
            
            # 2. Train the RL Agent
            rl_model = train_rl_agent(
                df,
                scored_patterns,
                portfolio_manager, 
                self.config                
            )
            
            # 3. Generate comparison
            # Now portfolio_results exists in this local scope!
            comparison = compare_strategies(
                df=df,
                scored_patterns=scored_patterns,
                rl_model=rl_model,
                portfolio_manager=portfolio_manager,
                portfolio_results=portfolio_results_dict, # Passed successfully
                config=self.config                
            )

            return {'model': rl_model, 'comparison': comparison}
            
        except Exception as e:
            print(f"❌ RL Error: {e}")
            return None


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