# 10_breakout_with_fixed_time_hold.py

# This script tests a pure momentum strategy with a fixed holding period.
#
# Strategy Logic:
# - Entry (Long): Spread is 40 points ABOVE the Moving Average.
# - Entry (Short): Spread is 40 points BELOW the Moving Average.
# - Exit: The trade is ALWAYS exited after a fixed time period (e.g., 6 hours),
#         regardless of what the price does.

import pandas as pd
import matplotlib.pyplot as plt

def calculate_spread(df):
    """
    This function takes a DataFrame and calculates the ATR-normalized spread.
    It remains unchanged.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
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

def run_breakout_fixed_time_hold(df, ma_period, deviation_threshold, time_limit_periods, time_limit_label):
    """
    This is the main analysis function. It runs the breakout backtest
    with a fixed-time exit rule.
    """
    # --- Part 1: Setup the Strategy Indicators ---
    print(f"\n{'='*50}")
    print(f"--- Testing Breakout MA={ma_period} | Fixed Hold Time={time_limit_label} ---")
    print(f"{'='*50}")

    strategy_df = df.copy()
    strategy_df['Spread_MA'] = strategy_df['Spread'].rolling(window=ma_period).mean()
    strategy_df['Upper_Band'] = strategy_df['Spread_MA'] + deviation_threshold
    strategy_df['Lower_Band'] = strategy_df['Spread_MA'] - deviation_threshold
    strategy_df.dropna(inplace=True)
    strategy_df.reset_index(drop=True, inplace=True)

    # --- Part 2: Simulate the Trading Logic ---
    completed_trades = []
    in_trade = False
    trade_type = None
    entry_price = 0
    entry_time = None
    exit_time_index = 0

    for i, row in strategy_df.iterrows():
        # --- A: CHECK FOR EXIT ---
        # If we are in a trade and have reached the scheduled exit time, close the trade.
        if in_trade and i >= exit_time_index:
            if exit_time_index < len(strategy_df):
                exit_row = strategy_df.loc[exit_time_index]
                exit_price = exit_row['Spread']
                exit_time = exit_row['timestamp']
                
                if trade_type == 'Short':
                    profit = entry_price - exit_price
                else: # Long trade
                    profit = exit_price - entry_price
                
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': trade_type, 'profit': profit,
                })
            in_trade = False

        # --- B: CHECK FOR ENTRY ---
        # This only runs if we are not currently in a trade.
        if not in_trade:
            # *** THIS IS THE KEY LOGIC CHANGE FROM SCRIPT 06 ***
            # Condition: Spread is 40 points ABOVE the MA.
            # Action: Go LONG (buy the breakout).
            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Long'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                exit_time_index = i + time_limit_periods # Schedule the exit.
                
            # Condition: Spread is 40 points BELOW the MA.
            # Action: Go SHORT (sell the breakdown).
            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Short'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                exit_time_index = i + time_limit_periods # Schedule the exit.
    
    # --- Part 3: Analyze and Print the Results ---
    if not completed_trades:
        print("No trades were completed for this strategy configuration.")
        return

    trades_df = pd.DataFrame(completed_trades)
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']

    total_profit = trades_df['profit'].sum()
    number_of_trades = len(trades_df)
    average_profit_per_trade = trades_df['profit'].mean()
    average_duration = trades_df['duration'].mean()
    win_rate = (trades_df['profit'] > 0).sum() / number_of_trades * 100

    print("--- Breakout Strategy Results (Fixed Hold Time) ---")
    print(f"Total Completed Trades: {number_of_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_profit:.2f} points")
    print(f"Average P/L per Trade: {average_profit_per_trade:.2f} points")
    print(f"Average Time in Trade: {average_duration} (Target: {time_limit_label})")
    print("-" * 25)
    
    # --- Part 4: Plot the Trades ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 9))
    plt.plot(strategy_df['timestamp'], strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    plt.plot(strategy_df['timestamp'], strategy_df['Spread_MA'], label=f'{ma_period}-Period MA', color='orange', linewidth=2)
    plt.plot(strategy_df['timestamp'], strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    plt.plot(strategy_df['timestamp'], strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label=f'+/- {deviation_threshold} Bands')
    
    long_trades = trades_df[trades_df['trade_type'] == 'Long']
    short_trades = trades_df[trades_df['trade_type'] == 'Short']
    
    plt.scatter(long_trades['entry_time'], long_trades['entry_price'], label='Long Entry (Breakout)', marker='^', color='lime', s=120, edgecolor='black', zorder=5)
    plt.scatter(short_trades['entry_time'], short_trades['entry_price'], label='Short Entry (Breakdown)', marker='v', color='red', s=120, edgecolor='black', zorder=5)
    plt.scatter(trades_df['exit_time'], trades_df['exit_price'], label='Exit (Fixed Time)', marker='x', color='black', s=120, zorder=5)
    
    plt.title(f"Breakout Strategy with Fixed Hold Time: MA={ma_period}, Hold={time_limit_label}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Spread Value", fontsize=12)
    plt.legend()
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    DATA_FILE_PATH = '../../data/clean_data/april_may_june_futures_data.csv'
    try:
        main_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        exit()

    main_df_with_spread = calculate_spread(main_df)

    # --- Define all parameters to be tested ---
    ma_periods_to_test = [20, 50, 100]
    DEVIATION = 40
    
    # Define time limits (assuming 10-minute data).
    time_limits_to_test = {
        '6 Hours': 6 * 6,
        '3 Hours': 3 * 6,
        '1 Hour': 1 * 6,
        '30 Mins': 3
    }

    # --- Run the nested loop for all 12 combinations ---
    for ma in ma_periods_to_test:
        for label, periods in time_limits_to_test.items():
            run_breakout_fixed_time_hold(
                df=main_df_with_spread,
                ma_period=ma,
                deviation_threshold=DEVIATION,
                time_limit_periods=periods,
                time_limit_label=label
            )

    print("\n\nAll analyses are complete.")