from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from pathlib import Path

# Initialize MCP
mcp = FastMCP("RITA_Swing_Trader")

# Load your model and data once at startup
script_dir = Path(__file__).parent.resolve()
MODEL_PATH = script_dir / ".." / ".." / "reports" / "RITA_swing_model.zip"
model = DQN.load(MODEL_PATH)

@mcp.tool()
def get_current_trade_audit(
    rsi: float, 
    macd: float, 
    pattern_signal: float,
    atr: float,
    price_change: float,
    position: float,
    macd_signal: float
) -> str:
    """
    Analyzes 7 market indicators using the RL model to return a swing trading verdict.
    """
    # Order matters! Ensure this matches your training dataframe columns exactly.
    obs = np.array([
        rsi, 
        macd, 
        pattern_signal, 
        atr, 
        price_change, 
        position, 
        macd_signal
    ], dtype=np.float32)
    
    # Predict with the model
    action, _ = model.predict(obs, deterministic=True)
    
    mapping = {0: "HOLD (Stay Flat)", 1: "LONG (Enter)", 2: "FLATTEN (Exit)"}
    return f"RL Agent Verdict: {mapping[int(action)]}"

@mcp.resource("trading://portfolio/status")
def get_portfolio_status() -> str:
    """Returns a summary of the trading bot's health and strategy mode."""
    return "Mode: Swing Trading | Goal: High Precision | Risk: Sharpe-Adjusted"

if __name__ == "__main__":
    mcp.run()