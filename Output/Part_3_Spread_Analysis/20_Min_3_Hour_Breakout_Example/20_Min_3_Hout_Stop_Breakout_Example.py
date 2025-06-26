# deep_dive_breakout_20ma_3hr_stop.py

# This script performs a focused backtest on a single breakout strategy variation:
# - Strategy: Breakout (Trend-Following)
# - Moving Average: 20-period
# - Entry: When the spread deviates 40 points from the 20 MA.
# - Exit: EITHER when the spread crosses back to the 20 MA OR after 3 hours.
# It produces a performance table, a detailed trades chart with differentiated
# exits, and an equity curve chart.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_spread(df):
    """Calculates the ATR-normalized spread."""
    print("Calculating ATR and Spread...")
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
    df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)
    return df

def run_breakout_with_time_stop(df, ma_period, deviation_threshold, time_limit_periods):
    """Simulates the breakout strategy with a time stop."""
    print(f"Running breakout backtest for MA={ma_period}, Dev={deviation_threshold}, Time Limit={time_limit_periods*10} mins...")
    
    strategy_df = df.copy()
    strategy_df['Spread_MA'] = strategy_df['Spread'].rolling(window=ma_period).mean()
    strategy_df['Upper_Band'] = strategy_df['Spread_MA'] + deviation_threshold
    strategy_df['Lower_Band'] = strategy_df['Spread_MA'] - deviation_threshold
    strategy_df.dropna(inplace=True)
    strategy_df.reset_index(drop=True, inplace=True)

    completed_trades = []
    in_trade = False
    trade_type = None
    entry_price = 0
    entry_time = None
    entry_index = 0

    for i, row in strategy_df.iterrows():
        # Exit Logic
        if in_trade:
            is_ma_cross_exit = False
            if trade_type == 'Long' and row['Spread'] < row['Spread_MA']:
                is_ma_cross_exit = True
            elif trade_type == 'Short' and row['Spread'] > row['Spread_MA']:
                is_ma_cross_exit = True
            
            is_time_out_exit = (i - entry_index) >= time_limit_periods
            
            if is_ma_cross_exit or is_time_out_exit:
                exit_price = row['Spread']
                exit_reason = 'MA Cross' if is_ma_cross_exit else 'Time-Out'
                profit = (exit_price - entry_price) if trade_type == 'Long' else (entry_price - exit_price)
                
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': row['timestamp'],
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': trade_type, 'profit': profit,
                    'exit_reason': exit_reason
                })
                in_trade = False
        
        # Entry Logic
        if not in_trade:
            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Long' # Buy breakout
                entry_price = row['Spread']
                entry_time = row['timestamp']
                entry_index = i
            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Short' # Sell breakdown
                entry_price = row['Spread']
                entry_time = row['timestamp']
                entry_index = i
    
    trades_df = pd.DataFrame(completed_trades)
    return strategy_df, trades_df

def print_performance_table(trades_df):
    """Calculates and prints performance metrics."""
    if trades_df.empty:
        print("\nNo trades were executed for this strategy.")
        return

    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
    
    total_trades = len(trades_df)
    total_pnl = trades_df['profit'].sum()
    avg_time_in_trade = trades_df['duration'].mean()
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    win_percentage = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
    avg_winner = winners['profit'].mean() if not winners.empty else 0
    avg_loser = losers['profit'].mean() if not losers.empty else 0
    time_out_exits = (trades_df['exit_reason'] == 'Time-Out').sum()

    print("\n--- Breakout Strategy Performance Summary ---")
    print("------------------------------------------------------")
    print(f"{'Total PnL:':<25} {total_pnl:>20.2f} points")
    print(f"{'Total Trades:':<25} {total_trades:>20}")
    print(f"{'Win Percentage:':<25} {win_percentage:>19.2f} %")
    print(f"{'Average Winner:':<25} {avg_winner:>20.2f} points")
    print(f"{'Average Loser:':<25} {avg_loser:>20.2f} points")
    print(f"{'Average Time in Trade:':<25} {str(avg_time_in_trade)}")
    print(f"{'Trades Closed by Time-Out:':<25} {time_out_exits} ({time_out_exits/total_trades:.1%})")
    print("------------------------------------------------------\n")

def plot_trades_chart_with_dates(strategy_df, trades_df, num_labels=10):
    """Creates a chart showing all trade entries and both types of exits."""
    if trades_df.empty:
        print("Cannot plot trades chart: No trades were made.")
        return
        
    print("Generating trades chart with custom date labels...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    
    ax.plot(strategy_df.index, strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    ax.plot(strategy_df.index, strategy_df['Spread_MA'], label='20-Period MA', color='orange', linewidth=2)
    ax.plot(strategy_df.index, strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    ax.plot(strategy_df.index, strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label='+/- 40 Bands')
    
    entry_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in trades_df['entry_time']]
    
    ma_cross_exits = trades_df[trades_df['exit_reason'] == 'MA Cross']
    timeout_exits = trades_df[trades_df['exit_reason'] == 'Time-Out']
    
    ma_cross_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in ma_cross_exits['exit_time']]
    timeout_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in timeout_exits['exit_time']]
    
    long_entries = trades_df[trades_df['trade_type'] == 'Long']
    short_entries = trades_df[trades_df['trade_type'] == 'Short']
    
    long_entry_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in long_entries['entry_time']]
    short_entry_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in short_entries['entry_time']]

    ax.scatter(long_entry_indices, long_entries['entry_price'], label='Long Entry', marker='^', color='lime', s=100, edgecolor='black', zorder=5)
    ax.scatter(short_entry_indices, short_entries['entry_price'], label='Short Entry', marker='v', color='red', s=100, edgecolor='black', zorder=5)
    ax.scatter(ma_cross_indices, ma_cross_exits['exit_price'], label='Exit (MA Cross)', marker='x', color='black', s=120, zorder=5)
    ax.scatter(timeout_indices, timeout_exits['exit_price'], label='Exit (Time-Out)', marker='s', color='magenta', s=100, edgecolor='black', zorder=5)
    
    ax.set_title('Breakout Trades (MA=20, Dev=40, Time Stop=3hr)', fontsize=16)
    ax.set_ylabel('Spread Value', fontsize=12)
    ax.legend()
    
    tick_positions = np.linspace(0, len(strategy_df) - 1, num_labels, dtype=int)
    tick_labels_ts = strategy_df['timestamp'].iloc[tick_positions]
    tick_labels_str = [ts.strftime('%b-%d') for ts in tick_labels_ts]
    
    plt.xticks(ticks=tick_positions, labels=tick_labels_str, rotation=45, ha="right")
    ax.set_xlabel('Date', fontsize=12)

    fig.tight_layout()
    plt.savefig('breakout_20ma_3hr_stop_trades.png', dpi=150)
    plt.close(fig)
    print("Saved 'breakout_20ma_3hr_stop_trades.png'")

def plot_equity_curve(trades_df):
    """Creates and saves a chart of the strategy's equity curve."""
    if trades_df.empty:
        print("Cannot plot equity curve: No trades were made.")
        return

    print("Generating equity curve chart...")
    trades_df.sort_values('exit_time', inplace=True)
    trades_df.reset_index(drop=True, inplace=True)
    trades_df['equity_curve'] = trades_df['profit'].cumsum()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(trades_df.index, trades_df['equity_curve'], label='Equity Curve', color='crimson')
    ax.set_title('Breakout Strategy Equity Curve (MA=20, Time Stop=3hr)', fontsize=16)
    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Cumulative PnL (in points)', fontsize=12)
    ax.grid(True)
    
    plt.savefig('breakout_20ma_3hr_stop_equity.png', dpi=150)
    plt.close(fig)
    print("Saved 'breakout_20ma_3hr_stop_equity.png'\n")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_FILE_PATH = '../../../Data/clean_data/april_may_june_futures_data.csv'
    MA_PERIOD = 20
    DEVIATION = 40
    # 3 hours * 6 periods/hour (since data is 10-min) = 18 periods
    TIME_LIMIT_PERIODS = 3 * 6 

    # --- Run Analysis ---
    try:
        main_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
        exit()
        
    df_with_spread = calculate_spread(main_df)
    strategy_data, trades_data = run_breakout_with_time_stop(df_with_spread, MA_PERIOD, DEVIATION, TIME_LIMIT_PERIODS)
    
    # --- Generate All Outputs ---
    print_performance_table(trades_data)
    plot_trades_chart_with_dates(strategy_data, trades_data)
    plot_equity_curve(trades_data)
    
    print("\nDeep-dive analysis complete.")