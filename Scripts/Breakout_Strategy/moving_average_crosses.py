# 08a_breakout_strategy_clarified.py

# This script tests a "breakout" or "trend-following" strategy.
# The logic has been confirmed to match the user request:
#
# - GO LONG when Spread is 40 points ABOVE the Moving Average.
# - GO SHORT when Spread is 40 points BELOW the Moving Average.
# - EXIT any trade when the Spread crosses back over the Moving Average.

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

def run_breakout_strategy(df, ma_period, deviation_threshold):
    """
    This function runs the breakout strategy backtest.
    """
    print(f"\n{'='*50}")
    print(f"--- Testing Breakout Strategy for MA={ma_period} ---")
    print(f"{'='*50}")

    strategy_df = df.copy()
    strategy_df['Spread_MA'] = strategy_df['Spread'].rolling(window=ma_period).mean()
    strategy_df['Upper_Band'] = strategy_df['Spread_MA'] + deviation_threshold
    strategy_df['Lower_Band'] = strategy_df['Spread_MA'] - deviation_threshold
    strategy_df.dropna(inplace=True)
    strategy_df.reset_index(drop=True, inplace=True)

    # --- Trading Logic Simulation ---
    completed_trades = []
    in_trade = False
    trade_type = None
    entry_price = 0
    entry_time = None

    for i, row in strategy_df.iterrows():
        # --- A: CHECK FOR ENTRIES (if we are not already in a trade) ---
        if not in_trade:
            # *** THIS IS THE CORE BREAKOUT ENTRY LOGIC ***
            
            # Condition: Spread is 40 points ABOVE the MA.
            # Action: Go LONG.
            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Long' # We are buying the breakout.
                entry_price = row['Spread']
                entry_time = row['timestamp']
                
            # Condition: Spread is 40 points BELOW the MA.
            # Action: Go SHORT.
            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Short' # We are selling the breakdown.
                entry_price = row['Spread']
                entry_time = row['timestamp']
        
        # --- B: CHECK FOR EXITS (if we are in a trade) ---
        else:
            # *** THIS IS THE CORE EXIT LOGIC ***

            # Condition: We are in a LONG trade and the spread falls below the MA.
            # Action: Exit the trade.
            if trade_type == 'Long' and row['Spread'] < row['Spread_MA']:
                exit_price = row['Spread']
                exit_time = row['timestamp']
                profit = exit_price - entry_price
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': trade_type, 'profit': profit,
                })
                in_trade = False

            # Condition: We are in a SHORT trade and the spread rallies above the MA.
            # Action: Exit the trade.
            elif trade_type == 'Short' and row['Spread'] > row['Spread_MA']:
                exit_price = row['Spread']
                exit_time = row['timestamp']
                profit = entry_price - exit_price
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': trade_type, 'profit': profit,
                })
                in_trade = False
    
    # --- Results Analysis and Plotting (Unchanged) ---
    if not completed_trades:
        print("No trades were completed for this strategy configuration.")
        return

    trades_df = pd.DataFrame(completed_trades)
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']

    total_profit = trades_df['profit'].sum()
    number_of_trades = len(trades_df)
    average_profit = trades_df['profit'].mean()
    average_duration = trades_df['duration'].mean()
    winning_trades = (trades_df['profit'] > 0).sum()
    win_rate = (winning_trades / number_of_trades) * 100

    print("--- Breakout Strategy Results ---")
    print(f"Total Completed Trades: {number_of_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_profit:.2f} points")
    print(f"Average P/L per Trade: {average_profit:.2f} points")
    print(f"Average Time in Trade: {average_duration}")
    print("-" * 25)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 9))
    plt.plot(strategy_df['timestamp'], strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    plt.plot(strategy_df['timestamp'], strategy_df['Spread_MA'], label=f'{ma_period}-Period MA', color='orange', linewidth=2)
    plt.plot(strategy_df['timestamp'], strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    plt.plot(strategy_df['timestamp'], strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label=f'+/- {deviation_threshold} Bands')

    long_trades = trades_df[trades_df['trade_type'] == 'Long']
    short_trades = trades_df[trades_df['trade_type'] == 'Short']
    
    plt.scatter(long_trades['entry_time'], long_trades['entry_price'], label='Long Entry (Buy Breakout)', marker='^', color='lime', s=120, edgecolor='black', zorder=5)
    plt.scatter(short_trades['entry_time'], short_trades['entry_price'], label='Short Entry (Sell Breakdown)', marker='v', color='red', s=120, edgecolor='black', zorder=5)
    plt.scatter(trades_df['exit_time'], trades_df['exit_price'], label='Exit (Crosses MA)', marker='x', color='black', s=120, zorder=5)
    
    plt.title(f"Breakout (Trend-Following) Strategy: MA={ma_period}", fontsize=16)
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

    ma_periods_to_test = [20, 50, 100]
    DEVIATION = 40
    
    for ma in ma_periods_to_test:
        run_breakout_strategy(
            df=main_df_with_spread,
            ma_period=ma,
            deviation_threshold=DEVIATION
        )

    print("\n\nAll analyses are complete.")