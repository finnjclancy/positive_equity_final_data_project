# deep_dive_zscore_mr_20ma_1z.py

# This script performs a focused backtest on a single mean-reversion strategy:
# - Moving Average: 20-period
# - Entry: When the Z-score deviates by more than 1.0.
# - Exit: When the Z-score crosses back to 0 (spread crosses the MA).
# It produces a performance table and two charts: one for trades and one for the equity curve.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_spread_and_indicators(df, ma_period):
    """Calculates spread, MA, Std Dev, and Z-Score."""
    print("Step 1: Calculating spread and all technical indicators...")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

    df['FESX_h-l'] = df['FESX_high'] - df['FESX_low']
    df['FDXM_h-l'] = df['FDXM_high'] - df['FDXM_low']
    df['FESX_h-pc'] = abs(df['FESX_high'] - df['FESX_close'].shift(1))
    df['FDXM_h-pc'] = abs(df['FDXM_high'] - df['FDXM_close'].shift(1))
    df['FESX_l-pc'] = abs(df['FESX_low'] - df['FESX_close'].shift(1))
    df['FDXM_l-pc'] = abs(df['FDXM_low'] - df['FDXM_close'].shift(1))
    df['FESX_TR'] = df[['FESX_h-l', 'FESX_h-pc', 'FESX_l-pc']].max(axis=1)
    df['FDXM_TR'] = df[['FDXM_h-l', 'FDXM_h-pc', 'FDXM_l-pc']].max(axis=1)
    df['FESX_ATR'] = df['FESX_TR'].rolling(window=14).mean()
    df['FDXM_ATR'] = df['FDXM_TR'].rolling(window=14).mean()

    price_ratio = df['FDXM_ATR'].mean() / df['FESX_ATR'].mean()
    df['spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)

    df['spread_ma'] = df['spread'].rolling(window=ma_period).mean()
    df['spread_std'] = df['spread'].rolling(window=ma_period).std()
    df['z_score'] = (df['spread'] - df['spread_ma']) / df['spread_std']
    
    return df.dropna()

def run_mean_reversion_backtest(df, z_score_threshold):
    """Simulates the mean reversion trading strategy."""
    print(f"Step 2: Running backtest for Z-Score entry at +/- {z_score_threshold}...")
    
    completed_trades = []
    in_trade = False
    trade_type = None
    entry_price = 0
    entry_time = None

    for i, row in df.iterrows():
        # Exit Logic
        if in_trade:
            if trade_type == 'Long' and row['z_score'] >= 0:
                profit = row['spread'] - entry_price
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': row['timestamp'],
                    'entry_price': entry_price, 'exit_price': row['spread'],
                    'trade_type': 'Long', 'profit': profit
                })
                in_trade = False
            elif trade_type == 'Short' and row['z_score'] <= 0:
                profit = entry_price - row['spread']
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': row['timestamp'],
                    'entry_price': entry_price, 'exit_price': row['spread'],
                    'trade_type': 'Short', 'profit': profit
                })
                in_trade = False

        # Entry Logic
        if not in_trade:
            if row['z_score'] > z_score_threshold:
                in_trade = True
                trade_type = 'Short'
                entry_price = row['spread']
                entry_time = row['timestamp']
            elif row['z_score'] < -z_score_threshold:
                in_trade = True
                trade_type = 'Long'
                entry_price = row['spread']
                entry_time = row['timestamp']
                
    trades_df = pd.DataFrame(completed_trades)
    return df, trades_df

def analyze_and_print_results(trades_df):
    """Calculates and prints performance metrics."""
    print("\nStep 3: Analyzing results...")
    if trades_df.empty:
        print("No trades were executed for this strategy.")
        return

    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']

    total_trades = len(trades_df)
    total_pnl = trades_df['profit'].sum()
    avg_time_in_trade = trades_df['duration'].mean()
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    win_percentage = (len(winners) / total_trades) * 100
    avg_winner = winners['profit'].mean() if not winners.empty else 0
    avg_loser = losers['profit'].mean() if not losers.empty else 0

    print("--- Mean Reversion Performance (MA=20, Z=1.0) ---")
    print("------------------------------------------------------")
    print(f"{'Total PnL:':<25} {total_pnl:>20.2f} points")
    print(f"{'Total Trades:':<25} {total_trades:>20}")
    print(f"{'Win Percentage:':<25} {win_percentage:>19.2f} %")
    print(f"{'Average Winner:':<25} {avg_winner:>20.2f} points")
    print(f"{'Average Loser:':<25} {avg_loser:>20.2f} points")
    print(f"{'Average Time in Trade:':<25} {str(avg_time_in_trade)}")
    print("------------------------------------------------------\n")

def plot_trades_and_equity(strategy_df, trades_df, ma_period, z_score_threshold):
    """Generates and saves the trades chart and the equity curve chart."""
    if trades_df.empty:
        print("Cannot plot charts: No trades were made.")
        return
        
    print("Step 4: Generating charts...")
    strategy_df.reset_index(drop=True, inplace=True)
    
    # --- Trades Chart ---
    fig1, ax1 = plt.subplots(figsize=(18, 9))
    ax1.plot(strategy_df.index, strategy_df['spread'], label='Spread', color='dodgerblue', alpha=0.7)
    ax1.plot(strategy_df.index, strategy_df['spread_ma'], label=f'{ma_period}-Period MA', color='orange', linewidth=2)
    
    # Create Z-score bands for plotting
    ax1.plot(strategy_df.index, strategy_df['spread_ma'] + (strategy_df['spread_std'] * z_score_threshold), color='red', linestyle='--', alpha=0.5, label=f'+/- {z_score_threshold} Z-Score')
    ax1.plot(strategy_df.index, strategy_df['spread_ma'] - (strategy_df['spread_std'] * z_score_threshold), color='red', linestyle='--', alpha=0.5)
    
    entry_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in trades_df['entry_time']]
    exit_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in trades_df['exit_time']]
    
    ax1.scatter(entry_indices, trades_df['entry_price'], label='Entry', marker='^', color='lime', s=100, edgecolor='black', zorder=5)
    ax1.scatter(exit_indices, trades_df['exit_price'], label='Exit (Cross MA)', marker='x', color='black', s=120, zorder=5)
    
    ax1.set_title(f'Mean Reversion Trades (MA={ma_period}, Z-Score={z_score_threshold})', fontsize=16)
    ax1.set_ylabel('Spread Value', fontsize=12)
    ax1.set_xlabel('Sequence Index (Gaps Removed)', fontsize=12)
    ax1.legend()
    fig1.tight_layout()
    trades_filename = f'zscore_mr_trades_{ma_period}ma_{z_score_threshold}z.png'
    plt.savefig(trades_filename, dpi=150)
    plt.close(fig1)
    print(f"Saved trades chart as '{trades_filename}'")
    
    # --- Equity Curve Chart ---
    trades_df.sort_values('exit_time', inplace=True)
    trades_df.reset_index(drop=True, inplace=True)
    trades_df['equity_curve'] = trades_df['profit'].cumsum()
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(trades_df.index, trades_df['equity_curve'], label='Equity Curve', color='navy')
    ax2.set_title(f'Equity Curve (MA={ma_period}, Z-Score={z_score_threshold})', fontsize=16)
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Cumulative PnL (in points)', fontsize=12)
    ax2.grid(True)
    fig2.tight_layout()
    
    equity_filename = f'zscore_mr_equity_{ma_period}ma_{z_score_threshold}z.png'
    plt.savefig(equity_filename, dpi=150)
    plt.close(fig2)
    print(f"Saved equity curve as '{equity_filename}'")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_FILE_PATH = '../../../Data/clean_data/april_may_june_futures_data.csv'
    MA_PERIOD = 20
    Z_SCORE_THRESHOLD = 1.0

    try:
        main_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
        exit()
        
    prepared_df = calculate_spread_and_indicators(main_df, MA_PERIOD)
    
    # Run the single backtest
    strategy_data, trades_data = run_mean_reversion_backtest(prepared_df.copy(), Z_SCORE_THRESHOLD)
    
    # Generate all outputs
    analyze_and_print_results(trades_data)
    plot_trades_and_equity(strategy_data, trades_data, MA_PERIOD, Z_SCORE_THRESHOLD)
    
    print("\n\nDeep-dive analysis complete.")