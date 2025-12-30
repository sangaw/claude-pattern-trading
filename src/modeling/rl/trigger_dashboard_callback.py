from stable_baselines3.common.callbacks import BaseCallback

class TradingDashboardCallback(BaseCallback):
    def __init__(self, visualizer, verbose=0):
        super().__init__(verbose)
        self.visualizer = visualizer
        self.current_episode = 0
        self.is_first_step = True        

    def _on_step(self) -> bool:
        # Sign of Life check for the first step
        if self.is_first_step:
            print("🟢 Callback Active: Training started and logging initialized.")
            self.is_first_step = False

        # Access info safely
        # Note: self.locals['infos'] is a list of dicts (one for each env)
        info = self.locals['infos'][0]
        
        # 1. Dashboard: Daily Equity (Force to float to avoid numpy hash errors)
        equity = float(info.get('portfolio_value', 10000))
        self.logger.record("dashboard/equity", equity)
        
        # 2. Dashboard: Activity (Force to int)
        trades = int(info.get('total_trades', 0))
        self.logger.record("dashboard/trades_count", trades)
        
        # 3. Log the "Why" to console periodically
        if info.get('trade_taken') and self.num_timesteps % 500 == 0:
            reason = str(info.get('reason', 'N/A'))
            print(f"📊 [Step {self.num_timesteps}] {reason}")

        # Get data from the environment info dict
        info = self.locals['infos'][0]
        
        # SB3 doesn't give us "episode number" directly in _on_step, 
        # so we track it or use total_timesteps
        self.visualizer.log_step(
            episode=self.current_episode,
            step=self.num_timesteps,
            portfolio_value=float(info.get('portfolio_value', 0)),
            reward=float(self.locals['rewards'][0]),
            action=int(self.locals['actions'][0]),
            price=float(info.get('current_price', 0)), # Ensure your Env info has this
            date=info.get('current_date', ""),        # Ensure your Env info has this
            loss=None # Loss isn't easily available mid-step in PPO/DQN
        )

        return True
    
    def _on_rollout_end(self) -> None:
        # Increment episode count when a rollout finishes
        self.current_episode += 1