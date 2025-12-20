"""
config.py - Central Configuration for Trading System
====================================================

All system parameters in one place for easy tuning.
"""


class TradingConfig:
    """Central configuration for all trading system parameters"""
    
    # ========================================================================
    # PATTERN DETECTION PARAMETERS
    # ========================================================================
    PATTERN_MIN_BARS = 15           # Minimum bars between pattern peaks/troughs
    PATTERN_MAX_BARS = 40           # Maximum bars between pattern peaks/troughs
    PATTERN_TOLERANCE = 0.015       # Price match tolerance (1.5%)
    PATTERN_PEAK_ORDER = 5          # Order for peak/trough detection
    
    # ========================================================================
    # PATTERN LABELING PARAMETERS
    # ========================================================================
    FORWARD_BARS = 20               # Bars to look ahead for pattern evaluation
    
    # ========================================================================
    # PORTFOLIO RISK MANAGEMENT
    # ========================================================================
    INITIAL_CAPITAL = 100000        # Starting capital ($)
    MAX_POSITIONS = 3               # Maximum concurrent positions
    RISK_PER_TRADE = 0.01          # Risk 1% of capital per trade
    MAX_POSITION_SIZE = 0.3        # Max 30% of capital in one position
    
    # ========================================================================
    # MACHINE LEARNING PARAMETERS
    # ========================================================================
    ML_TRAIN_SPLIT = 0.8           # 80% train, 20% test
    ML_MIN_SAMPLES = 30            # Minimum samples needed for ML training
    ML_IMBALANCE_THRESHOLD = 0.15  # If success rate < 15%, use rule-based
    
    # XGBoost Parameters
    ML_N_ESTIMATORS = 100
    ML_MAX_DEPTH = 4
    ML_LEARNING_RATE = 0.05
    ML_SUBSAMPLE = 0.8
    ML_COLSAMPLE_BYTREE = 0.8
    
    # ========================================================================
    # QUALITY SCORING & THRESHOLDS
    # ========================================================================
    QUALITY_THRESHOLD = 50         # Default quality score threshold
    TARGET_TRADES = 30             # Target number of trades for adaptive threshold
    ADAPTIVE_MIN_THRESHOLD = 35    # Minimum adaptive threshold
    
    # ========================================================================
    # REINFORCEMENT LEARNING PARAMETERS
    # ========================================================================
    
    # Training Parameters
    RL_ALGORITHM = 'DQN'           # 'PPO' or 'A2C' or 'DQN'
    RL_TIMESTEPS = 500             # Total training timesteps
    RL_LEARNING_RATE = 0.0003      # Learning rate
    RL_N_STEPS = 2048              # Steps before update
    RL_BATCH_SIZE = 64             # Batch size
    RL_N_EPOCHS = 10               # Training epochs per update
    RL_GAMMA = 0.99                # Discount factor
    RL_GAE_LAMBDA = 0.95           # GAE lambda
    
    # Environment Parameters
    RL_MAX_STEPS = 300             # Max steps per episode
    RL_TRANSACTION_COST = 0.0005   # 0.05% transaction cost
    RL_HOLDING_PENALTY_STEPS = 30  # Penalty after holding this many steps
    RL_HOLDING_PENALTY = 0.005     # Penalty amount
    RL_MAX_TRADE_RETURN = 0.10     # Cap returns at ±10%
    RL_REWARD_SCALE = 10.0         # Scale trade returns
    RL_PATTERN_BONUS = 0.01        # Bonus for entering on strong pattern
    
    # ========================================================================
    # DIAGNOSTIC PARAMETERS
    # ========================================================================
    DIAGNOSTIC_EPISODES = 20       # Episodes to collect for RL interpretability
    
    # ========================================================================
    # VISUALIZATION PARAMETERS
    # ========================================================================
    DASHBOARD_DPI = 150
    DASHBOARD_FIGSIZE = (20, 10)
    DASHBOARD_SAVE_PATH = 'trading_dashboard.png'
    
    DIAGNOSTIC_FIGSIZE = (18, 10)
    DIAGNOSTIC_SAVE_PATH = 'pattern_diagnostic.png'
    
    CONFUSION_MATRIX_SAVE_PATH = 'ml_confusion_matrices.png'
    THRESHOLD_ANALYSIS_SAVE_PATH = 'threshold_analysis.png'
    RL_INTERPRETABILITY_SAVE_PATH = 'rl_interpretability_enhanced.png'
    
    # ========================================================================
    # TECHNICAL INDICATOR PARAMETERS
    # ========================================================================
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    VOLATILITY_PERIOD = 20
    SMA_SHORT = 20
    SMA_LONG = 50
    SMA_TREND = 200
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Load config from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("\n" + "="*80)
        print("TRADING SYSTEM CONFIGURATION")
        print("="*80)
        
        sections = {
            'PATTERN': 'Pattern Detection',
            'ML': 'Machine Learning',
            'RL': 'Reinforcement Learning',
            'INITIAL': 'Portfolio Management',
            'QUALITY': 'Quality Scoring',
        }
        
        for prefix, section_name in sections.items():
            params = {k: v for k, v in cls.to_dict().items() if k.startswith(prefix)}
            if params:
                print(f"\n{section_name}:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
        
        print("="*80)


# Preset configurations for different trading styles
class ConservativeConfig(TradingConfig):
    """Conservative trading configuration"""
    MAX_POSITIONS = 2
    RISK_PER_TRADE = 0.005
    QUALITY_THRESHOLD = 60
    PATTERN_MIN_BARS = 20
    PATTERN_TOLERANCE = 0.01


class AggressiveConfig(TradingConfig):
    """Aggressive trading configuration"""
    MAX_POSITIONS = 5
    RISK_PER_TRADE = 0.02
    QUALITY_THRESHOLD = 40
    PATTERN_MAX_BARS = 50
    PATTERN_TOLERANCE = 0.02


class DayTradingConfig(TradingConfig):
    """Configuration for day trading"""
    PATTERN_MIN_BARS = 5
    PATTERN_MAX_BARS = 20
    FORWARD_BARS = 10
    MAX_POSITIONS = 3
    RISK_PER_TRADE = 0.01
    RL_HOLDING_PENALTY_STEPS = 10


class SwingTradingConfig(TradingConfig):
    """Configuration for swing trading"""
    PATTERN_MIN_BARS = 20
    PATTERN_MAX_BARS = 60
    FORWARD_BARS = 40
    MAX_POSITIONS = 5
    RISK_PER_TRADE = 0.015
    RL_HOLDING_PENALTY_STEPS = 50


if __name__ == "__main__":
    # Print default configuration
    TradingConfig.print_config()
    
    print("\n\nPreset Configurations Available:")
    print("  - ConservativeConfig")
    print("  - AggressiveConfig")
    print("  - DayTradingConfig")
    print("  - SwingTradingConfig")