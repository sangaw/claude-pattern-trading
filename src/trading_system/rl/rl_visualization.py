"""
RL Trading Visualization Framework
Add this to your existing nifty_backtest.py or use as separate module


Usage:
1. Import: from rl_visualization import RLVisualizer
2. Initialize: visualizer = RLVisualizer(strategy_name="DQN_Trading")
3. During training: visualizer.log_step(portfolio_value, reward, action, etc.)
4. View: visualizer.create_dashboard() or check TensorBoard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

class RLVisualizer:
    """
    Comprehensive visualization framework for RL trading models
    Integrates with TensorBoard, W&B, live plotting, and Dash
    """
    
    def __init__(self, strategy_name: str = "RL_Strategy", 
                 use_tensorboard: bool = True,
                 use_live_plot: bool = True,
                 save_video: bool = False):
        """
        Initialize the visualization framework
        
        Args:
            strategy_name: Name of your RL strategy
            use_tensorboard: Enable TensorBoard logging
            use_live_plot: Enable live matplotlib animation
            save_video: Save training as MP4 video
        """
        self.strategy_name = strategy_name
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_live_plot = use_live_plot
        self.save_video = save_video
        
        # Data storage
        self.episodes = []
        self.steps = []
        self.portfolio_values = []
        self.rewards = []
        self.actions = []  # 0=hold, 1=buy, 2=sell
        self.losses = []
        self.prices = []
        self.dates = []
        self.epsilons = []
        
        # Metrics
        self.episode_rewards = []
        self.episode_portfolio_values = []
        self.win_rates = []
        
        # Initialize loggers
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(f'runs/{strategy_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            print(f"✓ TensorBoard initialized. Run: tensorboard --logdir=runs")
        
        # Live plot setup
        if self.use_live_plot:
            self.fig = None
            self.axes = None
            self.lines = {}
            self.setup_live_plot()
        
        # Dash app
        self.dash_app = None
        self.dash_thread = None
        self.data_queue = queue.Queue()
        
    def setup_live_plot(self):
        """Setup live matplotlib plot"""
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Portfolio Value
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Portfolio Value')
        self.ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial')
        self.ax1.set_xlabel('Step')
        self.ax1.set_ylabel('Portfolio Value (₹)')
        self.ax1.set_title('Real-time Portfolio Performance', fontweight='bold')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Rewards
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_xlabel('Step')
        self.ax2.set_ylabel('Reward')
        self.ax2.set_title('Reward per Step', fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        
        # Actions Distribution
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax3.set_xlabel('Action')
        self.ax3.set_ylabel('Count')
        self.ax3.set_title('Action Distribution', fontweight='bold')
        self.ax3.set_xticks([0, 1, 2])
        self.ax3.set_xticklabels(['Hold', 'Buy', 'Sell'])
        
        # Episode Rewards
        self.ax4 = self.fig.add_subplot(gs[2, 0])
        self.line4, = self.ax4.plot([], [], 'r-', linewidth=2)
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('Total Reward')
        self.ax4.set_title('Episode Rewards', fontweight='bold')
        self.ax4.grid(True, alpha=0.3)
        
        # Loss
        self.ax5 = self.fig.add_subplot(gs[2, 1])
        self.line5, = self.ax5.plot([], [], 'orange', linewidth=2)
        self.ax5.set_xlabel('Step')
        self.ax5.set_ylabel('Loss')
        self.ax5.set_title('Training Loss', fontweight='bold')
        self.ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def log_step(self, episode: int, step: int, portfolio_value: float, 
                 reward: float, action: int, price: float, 
                 date: str = None, loss: float = None, epsilon: float = None):
        """
        Log a single training step
        
        Args:
            episode: Current episode number
            step: Current step within episode
            portfolio_value: Current portfolio value
            reward: Reward received
            action: Action taken (0=hold, 1=buy, 2=sell)
            price: Current stock price
            date: Date string (optional)
            loss: Training loss (optional)
            epsilon: Exploration rate (optional)
        """
        # Store data
        self.episodes.append(episode)
        self.steps.append(step)
        self.portfolio_values.append(portfolio_value)
        self.rewards.append(reward)
        self.actions.append(action)
        self.prices.append(price)
        self.dates.append(date or f"Step_{step}")
        if loss is not None:
            self.losses.append(loss)
        if epsilon is not None:
            self.epsilons.append(epsilon)
        
        # TensorBoard logging
        if self.use_tensorboard:
            global_step = episode * 1000 + step  # Unique step counter
            self.tb_writer.add_scalar('Portfolio/Value', portfolio_value, global_step)
            self.tb_writer.add_scalar('Reward/Step', reward, global_step)
            self.tb_writer.add_scalar('Action/Type', action, global_step)
            if loss is not None:
                self.tb_writer.add_scalar('Training/Loss', loss, global_step)
            if epsilon is not None:
                self.tb_writer.add_scalar('Training/Epsilon', epsilon, global_step)
        
        # Update live plot (every N steps to avoid slowdown)
        if self.use_live_plot and len(self.steps) % 10 == 0:
            self.update_live_plot()
        
        # Queue for Dash
        self.data_queue.put({
            'episode': episode,
            'step': step,
            'portfolio_value': portfolio_value,
            'reward': reward,
            'action': action
        })
    
    def log_episode(self, episode: int, total_reward: float, 
                   final_portfolio_value: float, win_rate: float = None):
        """
        Log episode-level metrics
        
        Args:
            episode: Episode number
            total_reward: Total reward for the episode
            final_portfolio_value: Final portfolio value
            win_rate: Win rate percentage (optional)
        """
        self.episode_rewards.append(total_reward)
        self.episode_portfolio_values.append(final_portfolio_value)
        if win_rate is not None:
            self.win_rates.append(win_rate)
        
        if self.use_tensorboard:
            self.tb_writer.add_scalar('Episode/Total_Reward', total_reward, episode)
            self.tb_writer.add_scalar('Episode/Portfolio_Value', final_portfolio_value, episode)
            if win_rate is not None:
                self.tb_writer.add_scalar('Episode/Win_Rate', win_rate, episode)
        
        
    def update_live_plot(self):
        """Update the live matplotlib plot"""
        if not self.use_live_plot or self.fig is None:
            return
        
        try:
            # Portfolio Value
            self.line1.set_data(self.steps, self.portfolio_values)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Rewards
            self.line2.set_data(self.steps, self.rewards)
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            # Actions Distribution
            self.ax3.clear()
            if self.actions:
                action_counts = [self.actions.count(i) for i in range(3)]
                self.ax3.bar([0, 1, 2], action_counts, color=['gray', 'green', 'red'])
                self.ax3.set_xlabel('Action')
                self.ax3.set_ylabel('Count')
                self.ax3.set_title('Action Distribution', fontweight='bold')
                self.ax3.set_xticks([0, 1, 2])
                self.ax3.set_xticklabels(['Hold', 'Buy', 'Sell'])
            
            # Episode Rewards
            if self.episode_rewards:
                episodes_list = list(range(len(self.episode_rewards)))
                self.line4.set_data(episodes_list, self.episode_rewards)
                self.ax4.relim()
                self.ax4.autoscale_view()
            
            # Loss
            if self.losses:
                loss_steps = list(range(len(self.losses)))
                self.line5.set_data(loss_steps, self.losses)
                self.ax5.relim()
                self.ax5.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Warning: Live plot update failed: {e}")
    
    def create_plotly_dashboard(self):
        """Create interactive Plotly charts"""
        if not self.steps:
            print("No data to visualize yet!")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value Over Time', 'Cumulative Reward',
                          'Action Distribution', 'Price vs Portfolio',
                          'Episode Rewards', 'Training Loss'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'type': 'bar'}, {'secondary_y': True}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Portfolio Value
        fig.add_trace(
            go.Scatter(x=self.steps, y=self.portfolio_values, 
                      name='Portfolio Value', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Cumulative Reward
        cumulative_rewards = np.cumsum(self.rewards)
        fig.add_trace(
            go.Scatter(x=self.steps, y=cumulative_rewards, 
                      name='Cumulative Reward', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Action Distribution
        action_counts = [self.actions.count(i) for i in range(3)]
        fig.add_trace(
            go.Bar(x=['Hold', 'Buy', 'Sell'], y=action_counts, 
                  name='Actions', marker_color=['gray', 'green', 'red']),
            row=2, col=1
        )
        
        # Price vs Portfolio
        fig.add_trace(
            go.Scatter(x=self.steps, y=self.prices, 
                      name='Stock Price', line=dict(color='orange', width=2)),
            row=2, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=self.steps, y=self.portfolio_values, 
                      name='Portfolio', line=dict(color='blue', width=2, dash='dash')),
            row=2, col=2, secondary_y=True
        )
        
        # Episode Rewards
        if self.episode_rewards:
            episodes = list(range(len(self.episode_rewards)))
            fig.add_trace(
                go.Scatter(x=episodes, y=self.episode_rewards, 
                          name='Episode Rewards', line=dict(color='red', width=2)),
                row=3, col=1
            )
        
        # Training Loss
        if self.losses:
            loss_steps = list(range(len(self.losses)))
            fig.add_trace(
                go.Scatter(x=loss_steps, y=self.losses, 
                          name='Loss', line=dict(color='purple', width=2)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200, 
            title_text=f"{self.strategy_name} - Training Visualization",
            showlegend=True,
            template='plotly_white'
        )
        
        # Save to HTML
        output_file = f'reports/{self.strategy_name}_interactive.html'
        fig.write_html(output_file)
        print(f"\n✓ Interactive Plotly dashboard saved: {output_file}")
        print(f"  Open in browser to explore!")
        
        return fig
    
    def create_dash_app(self, port: int = 8050):
        """
        Create live Dash dashboard (runs in separate thread)
        Access at http://localhost:8050
        """
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1(f'{self.strategy_name} - Live Training Dashboard', 
                   style={'textAlign': 'center'}),
            
            dcc.Graph(id='live-portfolio', style={'height': '400px'}),
            dcc.Graph(id='live-rewards', style={'height': '400px'}),
            dcc.Graph(id='live-actions', style={'height': '400px'}),
            
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            )
        ])
        
        @app.callback(
            [Output('live-portfolio', 'figure'),
             Output('live-rewards', 'figure'),
             Output('live-actions', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            if not self.steps:
                # Return empty figures
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig
            
            # Portfolio figure
            portfolio_fig = go.Figure()
            portfolio_fig.add_trace(go.Scatter(
                x=self.steps, y=self.portfolio_values,
                mode='lines', name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            portfolio_fig.update_layout(
                title='Portfolio Value',
                xaxis_title='Step',
                yaxis_title='Value (₹)',
                template='plotly_white'
            )
            
            # Rewards figure
            rewards_fig = go.Figure()
            rewards_fig.add_trace(go.Scatter(
                x=self.steps, y=self.rewards,
                mode='lines', name='Reward',
                line=dict(color='green', width=2)
            ))
            rewards_fig.update_layout(
                title='Rewards',
                xaxis_title='Step',
                yaxis_title='Reward',
                template='plotly_white'
            )
            
            # Actions figure
            action_counts = [self.actions.count(i) for i in range(3)]
            actions_fig = go.Figure()
            actions_fig.add_trace(go.Bar(
                x=['Hold', 'Buy', 'Sell'],
                y=action_counts,
                marker_color=['gray', 'green', 'red']
            ))
            actions_fig.update_layout(
                title='Action Distribution',
                xaxis_title='Action',
                yaxis_title='Count',
                template='plotly_white'
            )
            
            return portfolio_fig, rewards_fig, actions_fig
        
        def run_dash():
            app.run_server(debug=False, port=port, use_reloader=False)
        
        self.dash_thread = threading.Thread(target=run_dash, daemon=True)
        self.dash_thread.start()
        
        print(f"\n✓ Dash dashboard started at http://localhost:{port}")
        print(f"  Dashboard updates every 2 seconds with live data!")
    
    def save_training_video(self, filename: str = None, fps: int = 10):
        """Save training process as MP4 video"""
        if not self.steps:
            print("No data to create video!")
            return
        
        if filename is None:
            filename = f'reports/{self.strategy_name}_training.mp4'
        
        print(f"\nCreating training video: {filename}")
        print(f"This may take a few minutes...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        def animate(frame):
            idx = min(frame * 10, len(self.steps) - 1)  # Sample every 10 steps
            
            # Portfolio Value
            axes[0, 0].clear()
            axes[0, 0].plot(self.steps[:idx], self.portfolio_values[:idx], 'b-', linewidth=2)
            axes[0, 0].set_title('Portfolio Value')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Value (₹)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Rewards
            axes[0, 1].clear()
            axes[0, 1].plot(self.steps[:idx], self.rewards[:idx], 'g-', linewidth=2)
            axes[0, 1].set_title('Rewards')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Actions
            axes[1, 0].clear()
            action_counts = [self.actions[:idx].count(i) for i in range(3)]
            axes[1, 0].bar([0, 1, 2], action_counts, color=['gray', 'green', 'red'])
            axes[1, 0].set_title('Action Distribution')
            axes[1, 0].set_xticks([0, 1, 2])
            axes[1, 0].set_xticklabels(['Hold', 'Buy', 'Sell'])
            
            # Episode info
            axes[1, 1].clear()
            axes[1, 1].axis('off')
            if idx < len(self.episodes):
                info_text = f"""
                Episode: {self.episodes[idx]}
                Step: {self.steps[idx]}
                Portfolio: ₹{self.portfolio_values[idx]:,.2f}
                Total Reward: {sum(self.rewards[:idx]):.2f}
                """
                axes[1, 1].text(0.1, 0.5, info_text, fontsize=14, 
                              verticalalignment='center', family='monospace')
            
            plt.tight_layout()
        
        frames = len(self.steps) // 10
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=150)
        plt.close()
        
        print(f"✓ Video saved: {filename}")
    
    def generate_final_report(self):
        """Generate comprehensive final report with all visualizations"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE RL TRAINING REPORT")
        print("=" * 80)
        
        # Save final matplotlib figure
        if self.use_live_plot and self.fig:
            output_file = f'reports/{self.strategy_name}_final.png'
            self.fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n✓ Final training chart saved: {output_file}")
        
        # Create Plotly dashboard
        self.create_plotly_dashboard()
        
        # Save CSV data
        df = pd.DataFrame({
            'episode': self.episodes,
            'step': self.steps,
            'portfolio_value': self.portfolio_values,
            'reward': self.rewards,
            'action': self.actions,
            'price': self.prices,
            'date': self.dates
        })
        csv_file = f'reports/{self.strategy_name}_training_data.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Training data saved: {csv_file}")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total Episodes: {max(self.episodes) if self.episodes else 0}")
        print(f"Total Steps: {len(self.steps)}")
        print(f"Final Portfolio Value: ₹{self.portfolio_values[-1]:,.2f}")
        print(f"Total Return: {(self.portfolio_values[-1] / 100000 - 1) * 100:.2f}%")
        print(f"Total Reward: {sum(self.rewards):.2f}")
        print(f"Actions - Hold: {self.actions.count(0)}, Buy: {self.actions.count(1)}, Sell: {self.actions.count(2)}")
        print("=" * 80)
    
    def close(self):
        """Close all loggers and save final reports"""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_live_plot and self.fig:
            plt.close(self.fig)
        
        self.generate_final_report()
        
        print("\n✓ All visualization resources closed and saved!")


# Example integration with existing PortfolioManager
class RLPortfolioManager:
    """
    Extended PortfolioManager with RL visualization
    Drop-in replacement for existing PortfolioManager
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 visualizer: Optional[RLVisualizer] = None):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.position = 0
        self.visualizer = visualizer
        self.episode = 0
        self.step = 0
        
    def set_episode(self, episode: int):
        """Set current episode"""
        self.episode = episode
        self.step = 0
    
    def execute_action(self, action: int, price: float, date: str, 
                      reward: float = 0, loss: float = None):
        """
        Execute trading action with visualization
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            price: Current stock price
            date: Date string
            reward: Reward received (optional)
            loss: Training loss (optional)
        """
        self.step += 1
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            shares = int(self.capital / price)
            if shares > 0:
                self.capital -= shares * price
                self.position = shares
        
        elif action == 2 and self.position > 0:  # Sell
            self.capital += self.position * price
            self.position = 0
        
        # Calculate portfolio value
        portfolio_value = self.capital + (self.position * price)
        
        # Log to visualizer
        if self.visualizer:
            self.visualizer.log_step(
                episode=self.episode,
                step=self.step,
                portfolio_value=portfolio_value,
                reward=reward,
                action=action,
                price=price,
                date=date,
                loss=loss
            )
        
        return portfolio_value


# Quick start example
if __name__ == "__main__":
    print("""
    RL Trading Visualization Framework
    ===================================
    
    Quick Start:
    -----------
    1. Initialize visualizer:
       visualizer = RLVisualizer(
           strategy_name="DQN_Trading",
           use_tensorboard=True,
           use_live_plot=True
       )
    
    2. During training loop:
       visualizer.log_step(
           episode=episode,
           step=step,
           portfolio_value=portfolio_value,
           reward=reward,
           action=action,
           price=current_price,
           date=current_date,
           loss=loss
       )
    
    3. After episode:
       visualizer.log_episode(
           episode=episode,
           total_reward=total_reward,
           final_portfolio_value=final_value
       )
    
    4. View results:
       - TensorBoard: tensorboard --logdir=runs
       - Live Plot: Auto-updates during training
       - Dash: visualizer.create_dash_app()
       - Final Report: visualizer.close()
    
    See documentation in code for detailed usage.
    """)