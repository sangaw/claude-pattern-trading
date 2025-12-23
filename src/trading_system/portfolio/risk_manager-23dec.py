import pandas as pd
import numpy as np

# ============================================================================
# PART 3: PORTFOLIO RISK MANAGEMENT
# ============================================================================

class PortfolioRiskManager:
    """Portfolio risk management system"""
    
    def __init__(self, initial_capital=100000, max_positions=3, risk_per_trade=0.01):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        self.active_positions = []
        self.closed_positions = []
        self.daily_equity = []
        self.peak_equity = initial_capital
    
    def evaluate_new_trade(self, pattern):
        """Evaluate if trade should be taken"""
        if len(self.active_positions) >= self.max_positions:
            return {'approved': False, 'reason': 'MAX_POSITIONS'}
        
        entry_price = pattern.get('Neckline', pattern.get('Close', 0))
        stop_loss = pattern['Stop_Loss']
        
        risk_amount = self.current_capital * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return {'approved': False, 'reason': 'INVALID_STOP'}
        
        shares = int(risk_amount / stop_distance)
        position_value = shares * entry_price
        
        if position_value > self.current_capital * 0.3:
            shares = int(self.current_capital * 0.3 / entry_price)
            position_value = shares * entry_price
        
        return {
            'approved': True,
            'shares': shares,
            'position_value': position_value,
            'risk_amount': shares * stop_distance
        }
    
    def open_position(self, pattern, evaluation, entry_price):
        """Open new position"""
        position = {
            'pattern_type': pattern['Pattern_Type'],
            'entry_date': pattern['Detection_Date'],
            'entry_price': entry_price,
            'shares': evaluation['shares'],
            'position_value': evaluation['position_value'],
            'stop_loss': pattern['Stop_Loss'],
            'target': pattern['Target'],
            'unrealized_pnl': 0
        }
        self.active_positions.append(position)
        return position
    
    def close_position(self, position_idx, exit_price, exit_date):
        """Close position"""
        if position_idx >= len(self.active_positions):
            return None
        
        position = self.active_positions.pop(position_idx)
        pnl = (exit_price - position['entry_price']) * position['shares']
        
        if position['pattern_type'] == 'DoubleTop':
            pnl = -pnl
        
        self.current_capital += pnl
        
        closed = {**position, 'exit_price': exit_price, 'exit_date': exit_date, 'pnl': pnl}
        self.closed_positions.append(closed)
        return closed
    
    def update_positions(self, current_date, current_price):
        """Update all positions"""
        for pos in self.active_positions:
            unrealized = (current_price - pos['entry_price']) * pos['shares']
            if pos['pattern_type'] == 'DoubleTop':
                unrealized = -unrealized
            pos['unrealized_pnl'] = unrealized
        
        total_unrealized = sum(p['unrealized_pnl'] for p in self.active_positions)
        current_equity = self.current_capital + total_unrealized
        
        self.daily_equity.append({'date': current_date, 'equity': current_equity})
        self.peak_equity = max(self.peak_equity, current_equity)
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if not self.closed_positions:
            return {'total_trades': 0}
        
        df = pd.DataFrame(self.closed_positions)
        wins = len(df[df['pnl'] > 0])
        total = len(df)
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if total > wins else 0,
            'final_capital': self.current_capital
        }