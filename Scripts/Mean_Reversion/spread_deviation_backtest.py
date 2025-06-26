# 04_backtest_with_returns.py

# Import the necessary libraries.
# pandas is for working with data tables (DataFrames).
import pandas as pd
# matplotlib.pyplot is for creating charts and plots.
import matplotlib.pyplot as plt

def calculate_spread(df):
    """
    This function takes a DataFrame and calculates the ATR-normalized spread.
    It's the same function as before, ensuring our starting point is consistent.
    """
    # --- Step 1: Convert timestamp column to a proper date format ---
    # This helps in plotting and time-based calculations.
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Step 2: Calculate the components of True Range (TR) ---
    # The range of the current candle.
    df['FESX_h-l'] = df['FESX_high'] - df['FESX_low']
    df['FDXM_h-l'] = df['FDXM_high'] - df['FDXM_low']
    # The absolute difference between the current high and the previous close.
    # .shift(1) looks at the value from the previous row.
    df['FESX_h-pc'] = abs(df['FESX_high'] - df['FESX_close'].shift(1))
    df['FDXM_h-pc'] = abs(df['FDXM_high'] - df['FDXM_close'].shift(1))
    # The absolute difference between the current low and the previous close.
    df['FESX_l-pc'] = abs(df['FESX_low'] - df['FESX_close'].shift(1))
    df['FDXM_l-pc'] = abs(df['FDXM_low'] - df['FDXM_close'].shift(1))

    # --- Step 3: Calculate the True Range (TR) and Average True Range (ATR) ---
    # TR is the maximum of the three values calculated above.
    df['FESX_TR'] = df[['FESX_h-l', 'FESX_h-pc', 'FESX_l-pc']].max(axis=1)
    df['FDXM_TR'] = df[['FDXM_h-l', 'FDXM_h-pc', 'FDXM_l-pc']].max(axis=1)
    # ATR is a smoothed moving average of the TR, typically over 14 periods.
    df['FESX_ATR'] = df['FESX_TR'].rolling(window=14).mean()
    df['FDXM_ATR'] = df['FDXM_TR'].rolling(window=14).mean()

    # --- Step 4: Calculate the Spread ---
    # Calculate the price ratio using the average ATR over the whole period.
    price_ratio = df['FDXM_ATR'].mean() / df['FESX_ATR'].mean()
    # Calculate the spread for every point in time using the formula.
    df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)

    # Return the DataFrame with all the new calculated columns.
    return df

def run_strategy_analysis(df, ma_period, deviation_threshold):
    """
    This function runs the backtest for a given moving average period and deviation.
    It calculates returns, trade duration, and generates a plot.
    """
    # --- Part 1: Setup the Strategy Indicators ---
    print(f"\n--- Running Analysis for {ma_period}-Period Moving Average ---")

    # Make a copy of the DataFrame to avoid modifying the original one.
    strategy_df = df.copy()

    # Calculate the moving average (the "mean") of the spread for the given period.
    strategy_df['Spread_MA'] = strategy_df['Spread'].rolling(window=ma_period).mean()

    # Calculate the upper and lower bands for trade entry.
    strategy_df['Upper_Band'] = strategy_df['Spread_MA'] + deviation_threshold
    strategy_df['Lower_Band'] = strategy_df['Spread_MA'] - deviation_threshold

    # Drop rows with missing values (NaNs) that are created by the rolling calculations.
    strategy_df.dropna(inplace=True)
    # Reset the index to make looping easier.
    strategy_df.reset_index(drop=True, inplace=True)


    # --- Part 2: Simulate the Trading Logic ---
    # We will loop through the data and simulate being in or out of a trade.

    # This list will store information about each completed trade.
    completed_trades = []
    
    # State variables to keep track of our current position.
    in_trade = False
    trade_type = None # Will be 'Long' or 'Short'
    entry_price = 0
    entry_time = None
    entry_index = 0

    # Loop through each row of our strategy DataFrame.
    # .iterrows() gives us the index and the data for each row.
    for i, row in strategy_df.iterrows():
        # --- Check for ENTRY signals if we are NOT currently in a trade ---
        if not in_trade:
            # Check for a SHORT entry signal (spread is far above the mean).
            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Short'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                entry_index = i
                
            # Check for a LONG entry signal (spread is far below the mean).
            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Long'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                entry_index = i
        
        # --- Check for EXIT signals if we ARE currently in a trade ---
        else:
            # Check for a SHORT trade EXIT (spread crosses back below the mean).
            if trade_type == 'Short' and row['Spread'] < row['Spread_MA']:
                exit_price = row['Spread']
                exit_time = row['timestamp']
                # For a short trade, profit is made when the price goes down.
                profit = entry_price - exit_price
                # Record the completed trade.
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': 'Short', 'profit': profit
                })
                # Reset state to be ready for the next trade.
                in_trade = False

            # Check for a LONG trade EXIT (spread crosses back above the mean).
            elif trade_type == 'Long' and row['Spread'] > row['Spread_MA']:
                exit_price = row['Spread']
                exit_time = row['timestamp']
                # For a long trade, profit is made when the price goes up.
                profit = exit_price - entry_price
                # Record the completed trade.
                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': 'Long', 'profit': profit
                })
                # Reset state.
                in_trade = False
    
    # --- Part 3: Analyze and Print the Results ---
    if not completed_trades:
        print("No trades were completed for this strategy.")
        return # Exit the function if there's nothing to analyze.

    # Convert the list of trades into a pandas DataFrame for easy analysis.
    trades_df = pd.DataFrame(completed_trades)

    # Calculate the duration of each trade.
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']

    # Calculate summary statistics.
    total_profit = trades_df['profit'].sum()
    number_of_trades = len(trades_df)
    average_profit = trades_df['profit'].mean()
    average_duration = trades_df['duration'].mean()
    winning_trades = (trades_df['profit'] > 0).sum()
    win_rate = (winning_trades / number_of_trades) * 100

    # Print the summary in a clear, readable format.
    print("--- Strategy Results ---")
    print(f"Total Trades: {number_of_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_profit:.2f} points")
    print(f"Average Profit/Loss per Trade: {average_profit:.2f} points")
    print(f"Average Time in Trade: {average_duration}")
    print("------------------------")
    
    # --- Part 4: Plot the Trades ---
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style for the plot.
    plt.figure(figsize=(18, 9)) # Create a large figure for the plot.

    # Plot the main lines: Spread, Moving Average, and the Bands.
    plt.plot(strategy_df['timestamp'], strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    plt.plot(strategy_df['timestamp'], strategy_df['Spread_MA'], label=f'{ma_period}-Period MA', color='orange', linewidth=2)
    plt.plot(strategy_df['timestamp'], strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    plt.plot(strategy_df['timestamp'], strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label=f'+/- {deviation_threshold} Bands')

    # Separate long and short trades for plotting with different markers.
    long_trades = trades_df[trades_df['trade_type'] == 'Long']
    short_trades = trades_df[trades_df['trade_type'] == 'Short']

    # Plot LONG entries (green upward triangle) and exits (black 'x').
    plt.scatter(long_trades['entry_time'], long_trades['entry_price'], label='Long Entry', marker='^', color='lime', s=120, edgecolor='black', zorder=5)
    plt.scatter(long_trades['exit_time'], long_trades['exit_price'], label='Exit', marker='x', color='black', s=100, zorder=5)
    
    # Plot SHORT entries (red downward triangle) and exits (black 'x').
    plt.scatter(short_trades['entry_time'], short_trades['entry_price'], label='Short Entry', marker='v', color='red', s=120, edgecolor='black', zorder=5)
    plt.scatter(short_trades['exit_time'], short_trades['exit_price'], marker='x', color='black', s=100, zorder=5)

    # Add title and labels to make the chart understandable.
    plt.title(f"Mean Reversion Strategy Backtest (MA = {ma_period}, Threshold = {deviation_threshold})", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Spread Value", fontsize=12)
    plt.legend()
    
    # Display the plot. The script will pause here until you close the plot window.
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Step 1: Load and Prepare the Data ---
    # Define the path to your data file.
    # !! IMPORTANT: Make sure this path is correct for your system. !!
    # I am using the 'cleaned_futures_data.csv' as the full dataset.
    DATA_FILE_PATH = 'data/clean_data/april_may_june_futures_data.csv'
    
    # Load the data from the CSV file.
    try:
        main_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        print("Please make sure the script is in the same folder as the data file, or update the path.")
        exit() # Stop the script if the file doesn't exist.

    # Calculate the spread. This is done only once.
    main_df_with_spread = calculate_spread(main_df)

    # --- Step 2: Define and Run the Backtests ---
    # A list of the moving average periods we want to test.
    ma_periods_to_test = [20, 50, 100]
    
    # The deviation threshold for our strategy.
    DEVIATION = 40

    # Loop through each moving average period and run the full analysis.
    for period in ma_periods_to_test:
        run_strategy_analysis(main_df_with_spread, ma_period=period, deviation_threshold=DEVIATION)

    print("\nAll analyses complete.")