import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
from datetime import datetime

def generate_dashboard():
    # 1. Setup CLI
    parser = argparse.ArgumentParser(description="RL vs ML Engine Dashboard")
    parser.add_argument("folder", help="Folder containing trade_log_*.csv and price data")
    args = parser.parse_args()

    # 2. Find latest log and price data
    log_files = glob.glob(os.path.join(args.folder, "trade_log_*.csv"))
    if not log_files:
        print(f"❌ No logs found in {args.folder}")
        return
    
    latest_log = max(log_files, key=os.path.getctime)
    df = pd.read_csv(latest_log)
    
    # 3. FIX: Convert dates and handle "Unknown" values
    # errors='coerce' turns "Unknown" or invalid strings into NaT (Not a Time)
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')

    # 4. Filter for Completed vs Active Trades
    closed_trades = df.dropna(subset=['exit_date']).copy()
    active_trades = df[df['exit_date'].isna()].copy()

    # 5. Create Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    
    ax1 = fig.add_subplot(gs[0, :]) # Top: Return Distribution
    ax2 = fig.add_subplot(gs[1, 0]) # Bottom Left: Equity Curve
    ax3 = fig.add_subplot(gs[1, 1]) # Bottom Right: Summary Table

    # --- TOP: Profitability Per Trade ---
    if not closed_trades.empty:
        sns.boxplot(data=closed_trades, x='engine_type', y='return_pct', ax=ax1, palette='viridis')
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f"Trade Performance: {os.path.basename(latest_log)}", fontsize=14)
    else:
        ax1.text(0.5, 0.5, "No closed trades to display yet.", ha='center')

    # --- BOTTOM LEFT: Equity Curve ---
    for engine in closed_trades['engine_type'].unique():
        engine_df = closed_trades[closed_trades['engine_type'] == engine].sort_values('exit_date')
        engine_df['cum_pnl'] = engine_df['pnl'].cumsum()
        ax2.plot(engine_df['exit_date'], engine_df['cum_pnl'], label=f"{engine}", marker='o', markersize=3)
    
    ax2.set_title("Cumulative Net PnL (Closed Trades)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- BOTTOM RIGHT: Statistics Table ---
    ax3.axis('off')
    if not closed_trades.empty:
        stats = closed_trades.groupby('engine_type').agg({
            'pnl': 'sum',
            'return_pct': 'mean',
            'id': 'count'
        }).round(2)
        stats.columns = ['Total PnL', 'Avg Return %', 'Trades']
        
        # Add Active Trades count
        active_counts = active_trades['engine_type'].value_counts()
        stats['Active'] = stats.index.map(active_counts).fillna(0).astype(int)

        table = ax3.table(cellText=stats.values, 
                          colLabels=stats.columns, 
                          rowLabels=stats.index,
                          loc='center', 
                          cellLoc='center')
        table.scale(1, 2)
        ax3.set_title("Performance Metrics", pad=20)

    plt.tight_layout()
    
    # 6. Save and Finish
    output_path = os.path.join(args.folder, f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(output_path, dpi=150)
    print(f"✅ Dashboard saved: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_dashboard()

# python .\modeling\visualization\trade_log_visualization.py "C:\Users\Sandeep\Documents\Work\code\claude-pattern-trading\data\output"    