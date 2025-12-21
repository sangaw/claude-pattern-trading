from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Trade-Auditor")

@mcp.tool()
def audit_trade_opportunity(rsi: float, pattern_score: float, trend_strength: float):
    """
    Audits an RL signal. If the signal is 'Long' but trend is weak, 
    the AI Agent can advise the user to skip.
    """
    if trend_strength < 0.2 and rsi > 60:
        return "ADVICE: RL Agent wants to go Long, but MACD trend is weak and RSI is high. Likely a Scalp trap. Suggest: SKIP."
    
    return "ADVICE: Trade aligns with Swing criteria. Proceed."

@mcp.resource("trading://metrics/current")
def get_metrics():
    return f"Current Mode: Swing | Strategy: Pattern-Aware | Precision Goal: >40%"

mcp.run()