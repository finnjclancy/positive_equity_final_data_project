# 06_backtest_fixed_time_hold.py

# This script tests a "fixed time hold" strategy.
# Entry: When the spread deviates 40 points from a moving average.
# Exit: After a fixed period (e.g., 6 hours), regardless of price action.

# Import the necessary libraries.
# pandas is for working with our data tables (DataFrames).
import pandas as pd
# matplotlib.pyplot is for creating charts and plots.
import matplotlib.pyplot as plt

def calculate_spread(df):
    """
    This function takes a DataFrame and calculates the ATR-normalized spread.
    This function is unchanged and ensures our spread calculation is consistent.
    """
    # Convert timestamp column to a proper date format.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Calculate True Range components.
    df['FESX_h-l'] = df['FESX_high'] - df['FESX_low']
    df['FDXM_h-l'] = df['FDXM_high'] - df['FDXM_low']
    df['FESX_h-pc'] = abs(df['FESX_high'] - df['FESX_close'].shift(1))
    df['FDXM_h-pc'] = abs(df['FDXM_high'] - df['FDXM_close'].shift(1))
    df['FESX_l-pc'] = abs(df['FESX_low'] - df['FESX_close'].shift(1))
    df['FDXM_l-pc'] = abs(df['FDXM_low'] - df['FDXM_close'].shift(1))
    # Calculate TR and ATR.
    df['FESX_TR'] = df[['FESX_h-l', 'FESX_h-pc', 'FESX_l-pc']].max(axis=1)
    df['FDXM_TR'] = df[['FDXM_h-l', 'FDXM_h-pc', 'FDXM_l-pc']].max(axis=1)
    df['FESX_ATR'] = df['FESX_TR'].rolling(window=14).mean()
    df['FDXM_ATR'] = df['FDXM_TR'].rolling(window=14).mean()
    # Calculate the Spread using the Price Ratio.
    price_ratio = df['FDXM_ATR'].mean() / df['FESX_ATR'].mean()
    df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)
    return df

def run_fixed_time_hold_strategy(df, ma_period, deviation_threshold, time_limit_periods, time_limit_label):
    """
    This is the main analysis function. It runs the backtest for a specific
    MA period and a fixed holding time.
    """
    # --- Part 1: Setup the Strategy Indicators ---
    print(f"\n{'='*50}")
    print(f"--- Testing MA={ma_period} | Fixed Hold Time={time_limit_label} ---")
    print(f"{'='*50}")

    # Make a copy to avoid changing the original data.
    strategy_df = df.copy()
    # Calculate the moving average and the entry bands.
    strategy_df['Spread_MA'] = strategy_df['Spread'].rolling(window=ma_period).mean()
    strategy_df['Upper_Band'] = strategy_df['Spread_MA'] + deviation_threshold
    strategy_df['Lower_Band'] = strategy_df['Spread_MA'] - deviation_threshold

    # Remove rows with empty values from the start and reset the index.
    strategy_df.dropna(inplace=True)
    strategy_df.reset_index(drop=True, inplace=True)

    # --- Part 2: Simulate the Trading Logic with a Fixed-Time Exit ---
    # This list will store the details of each completed trade.
    completed_trades = []
    
    # These variables track the state of our current trade.
    in_trade = False
    trade_type = None
    entry_price = 0
    entry_time = None
    exit_time_index = 0 # This will store the index where we MUST exit the trade.

    # Loop through each row of the DataFrame, which represents one time step.
    for i, row in strategy_df.iterrows():
        # --- A: CHECK FOR AN EXIT ---
        # If we are in a trade AND the current time index has reached our scheduled exit time,
        # we must close the trade.
        if in_trade and i >= exit_time_index:
            # We can't exit if there's no more data, so we check if the exit index is valid.
            if exit_time_index < len(strategy_df):
                exit_row = strategy_df.loc[exit_time_index]
                exit_price = exit_row['Spread']
                exit_time = exit_row['timestamp']
                
                # Calculate profit based on trade type.
                if trade_type == 'Short':
                    profit = entry_price - exit_price
                else: # Long trade
                    profit = exit_price - entry_price
                
                # Record all details of the completed trade.
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': trade_type, 'profit': profit,
                })

            # Reset our state variables to be ready for a new trade.
            in_trade = False

        # --- B: CHECK FOR AN ENTRY ---
        # This part only runs if we are not currently in a trade.
        # We also make sure not to enter a new trade on the same bar we just exited.
        if not in_trade:
            # Check for a Short entry (spread is far above the mean).
            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Short'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                # *** KEY LOGIC ***: Schedule the exact future index for the exit.
                exit_time_index = i + time_limit_periods
                
            # Check for a Long entry (spread is far below the mean).
            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Long'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                # *** KEY LOGIC ***: Schedule the exact future index for the exit.
                exit_time_index = i + time_limit_periods
    
    # --- Part 3: Analyze and Print the Results ---
    if not completed_trades:
        print("No trades were completed for this strategy configuration.")
        return # Exit if there's nothing to analyze.

    # Convert our list of trades into a DataFrame for easier analysis.
    trades_df = pd.DataFrame(completed_trades)
    # Calculate the time duration of each trade.
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']

    # Calculate key performance metrics.
    total_profit = trades_df['profit'].sum()
    number_of_trades = len(trades_df)
    average_profit_per_trade = trades_df['profit'].mean()
    average_duration = trades_df['duration'].mean() # Should be very close to the fixed time limit.
    winning_trades = (trades_df['profit'] > 0).sum()
    win_rate = (winning_trades / number_of_trades) * 100

    # Print a clear summary of the results.
    print("--- Backtest Results ---")
    print(f"Total Completed Trades: {number_of_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_profit:.2f} points")
    print(f"Average P/L per Trade: {average_profit_per_trade:.2f} points")
    print(f"Average Time in Trade: {average_duration} (Target: {time_limit_label})")
    print("-" * 25)
    
    # --- Part 4: Plot the Trades ---
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 9))

    # Plot the main strategy lines.
    plt.plot(strategy_df['timestamp'], strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    plt.plot(strategy_df['timestamp'], strategy_df['Spread_MA'], label=f'{ma_period}-Period MA', color='orange', linewidth=2)
    plt.plot(strategy_df['timestamp'], strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    plt.plot(strategy_df['timestamp'], strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label=f'+/- {deviation_threshold} Bands')

    # Separate long and short trades for plotting.
    long_trades = trades_df[trades_df['trade_type'] == 'Long']
    short_trades = trades_df[trades_df['trade_type'] == 'Short']

    # Plot the entry markers.
    plt.scatter(long_trades['entry_time'], long_trades['entry_price'], label='Long Entry', marker='^', color='lime', s=120, edgecolor='black', zorder=5)
    plt.scatter(short_trades['entry_time'], short_trades['entry_price'], label='Short Entry', marker='v', color='red', s=120, edgecolor='black', zorder=5)

    # Plot the exit markers. Since all exits are for the same reason, we only need one style.
    plt.scatter(trades_df['exit_time'], trades_df['exit_price'], label='Exit (Fixed Time)', marker='x', color='black', s=120, zorder=5)
    
    # Add title and legend.
    plt.title(f"Fixed Time Hold Strategy: MA={ma_period}, Hold Time={time_limit_label}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Spread Value", fontsize=12)
    plt.legend()
    plt.show()

# --- Main Execution Block ---
# This is where the script starts running.
if __name__ == "__main__":
    # --- Step 1: Load and Prepare Data ---
    DATA_FILE_PATH = 'data/clean_data/april_may_june_futures_data.csv'
    try:
        main_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        exit() # Stop if data file is missing.

    # Calculate the spread just once on the main DataFrame.
    main_df_with_spread = calculate_spread(main_df)

    # --- Step 2: Define Strategy Parameters to Test ---
    ma_periods_to_test = [20, 50, 100]
    DEVIATION = 40
    
    # Define time limits as a dictionary for easy use.
    # (Assuming 10-minute data: 60 mins/hour / 10 mins/period = 6 periods per hour).
    time_limits_to_test = {
        '6 Hours': 6 * 6,
        '3 Hours': 3 * 6,
        '1 Hour': 1 * 6,
        '30 Mins': 3
    }

    # --- Step 3: Run the Nested Loop for All Combinations ---
    # The outer loop iterates through the Moving Average periods.
    for ma in ma_periods_to_test:
        # The inner loop iterates through the time limits.
        for label, periods in time_limits_to_test.items():
            # Run the full analysis for this specific combination.
            run_fixed_time_hold_strategy(
                df=main_df_with_spread,
                ma_period=ma,
                deviation_threshold=DEVIATION,
                time_limit_periods=periods,
                time_limit_label=label
            )

    print("\n\nAll analyses are complete.")