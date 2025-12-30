import pandas as pd
import numpy as np

class PortfolioRiskManager:
    """Enhanced Portfolio Risk Management for Pattern Trading"""
    
    def __init__(self, initial_capital=100000, max_positions=5, risk_per_trade=0.01):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # Matches the attributes expected by your backtest loop
        self.active_positions = []  # List of dicts
        self.closed_positions = []
        self.daily_equity = []
        self.peak_equity = initial_capital

    @property
    def positions(self):
        """Helper to avoid AttributeError if loop calls .positions"""
        return self.active_positions

    @property
    def trade_history(self):
        """Helper to avoid AttributeError if loop calls .trade_history"""
        return self.closed_positions

    def evaluate_new_trade(self, pattern):
        """Calculate position sizing and approve trade"""
        if len(self.active_positions) >= self.max_positions:
            return {'approved': False, 'reason': 'MAX_POSITIONS'}
        
        # Priority: Entry_Price (Flags) > Neckline (Reversals) > Close
        entry_price = pattern.get('Entry_Price', pattern.get('Neckline', pattern.get('Close', 0)))
        stop_loss = pattern.get('Stop_Loss', 0)
        
        if entry_price <= 0 or stop_loss <= 0:
            return {'approved': False, 'reason': 'INVALID_PRICES'}

        risk_amount = self.current_capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return {'approved': False, 'reason': 'INVALID_STOP'}
        
        # CRITICAL FIX: Allow fractional shares for Nifty/Index backtesting 
        # to avoid the "0 shares" bug that leads to 0.00% return
        shares = risk_amount / stop_distance
        
        # Max position sizing (don't put more than 30% capital in one trade)
        max_val = self.current_capital * 0.3
        if (shares * entry_price) > max_val:
            shares = max_val / entry_price
            
        return {
            'approved': True,
            'shares': shares,
            'position_value': shares * entry_price,
            'risk_amount': shares * stop_distance
        }
    
    def open_position(self, pattern, evaluation, entry_price, engine_type="Rule-Based"):
        """Open and record new position"""
        position = {
            'id': len(self.closed_positions) + len(self.active_positions) + 1,
            'engine_type': engine_type,  # Whether Rule Based, ML ALgorithm or RL Agent
            'pattern_type': pattern['Pattern_Type'],
            'entry_date': pattern.get('Detection_Date', 'Unknown'),
            'entry_price': entry_price,
            'shares': evaluation['shares'],
            'stop_loss': pattern['Stop_Loss'],
            'target': pattern['Target'],
            'unrealized_pnl': 0
        }
        self.active_positions.append(position)
        return position  # Return the dict so the loop can track the ID

    def close_position(self, trade_ref, exit_price, exit_date):
        """Close position by ID or reference"""
        target_idx = -1
        
        # Determine trade ID
        t_id = trade_ref.get('id') if isinstance(trade_ref, dict) else trade_ref
        
        for i, pos in enumerate(self.active_positions):
            if pos['id'] == t_id:
                target_idx = i
                break
        
        if target_idx == -1 and self.active_positions:
            # Fallback: close the oldest if ID not found
            target_idx = 0
        elif not self.active_positions:
            return None

        position = self.active_positions.pop(target_idx)
        
        # PnL Calculation
        pnl = (exit_price - position['entry_price']) * position['shares']
        
        # Inverse for bearish patterns
        if position['pattern_type'] in ['DoubleTop', 'BearishFlag']:
            pnl = -pnl
        
        self.current_capital += pnl
        
        closed = {
            **position, 
            'exit_price': exit_price, 
            'exit_date': exit_date, 
            'pnl': pnl,
            'return_pct': (pnl / (position['entry_price'] * position['shares'])) * 100 if position['shares'] > 0 else 0
        }
        self.closed_positions.append(closed)
        return closed

    def get_metrics(self, engine_name="Rule-Based", silent=False):
        """
        Calculates performance stats and exports a labeled CSV log.
        """
        import datetime
        import pandas as pd
        
        # Guard clause for no trades
        if not self.closed_positions:
            if not silent:
                print(f"⚠️ No trades recorded for {engine_name}.")
            return {
                'total_trades': 0, 
                'win_rate': 0, 
                'total_return': 0, 
                'max_drawdown': 0
            }
        
        # 1. Prepare Data
        df_results = pd.DataFrame(self.closed_positions)
        
        # 2. Performance Math
        total_trades = len(df_results)
        wins = len(df_results[df_results['pnl'] > 0])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = df_results['pnl'].sum()
        
        # Ensure initial_capital exists (default to 100k if not set)
        initial_cap = getattr(self, 'initial_capital', 100000)
        total_return = (total_pnl / initial_cap) * 100
        
        # 3. Max Drawdown (Closed-Trade Equity)
        df_results['equity_curve'] = initial_cap + df_results['pnl'].cumsum()
        rolling_max = df_results['equity_curve'].cummax()
        drawdown = (df_results['equity_curve'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100 if not drawdown.empty else 0

        # 4. Package Results
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_dd
        }

        # 5. Conditional Output (The "Spam Filter")
        if not silent:
            # Generate the unique filename for logging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"trade_log_{engine_name.upper()}_{timestamp}.csv"
            
            # Save the logs
            df_results.to_csv(filename, index=False)
            
            print(f"\n📊 [{engine_name}] Backtest Results:")
            print(f"   Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Return: {total_return:.2f}%")
            print(f"   Max Drawdown: {max_dd:.2f}%")
            print(f"   📁 Log saved as: {filename}")
            
            # If your class has a separate save_logs method, you can call it here instead:
            # self.save_logs(engine_name)

        return metrics
    
    def _calculate_drawdown(self, df):
        """Helper to calculate max drawdown percentage"""
        df['equity'] = self.initial_capital + df['pnl'].cumsum()
        peak = df['equity'].cummax()
        dd = (df['equity'] - peak) / peak
        return dd.min() * 100    


    def save_logs(self, engine_name="System"):
        """Saves the current trade ledger to a CSV file."""
        import os
        import pandas as pd
        from datetime import datetime
        
        if not hasattr(self, 'closed_positions') or not self.closed_positions:
            return None
            
        os.makedirs("logs", exist_ok=True)
        df_log = pd.DataFrame(self.closed_positions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"logs/trade_log_{engine_name}_{timestamp}.csv"
        df_log.to_csv(filename, index=False)
        return filename   