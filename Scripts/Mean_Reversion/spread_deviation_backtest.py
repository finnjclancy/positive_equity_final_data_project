import pandas as pd
import matplotlib.pyplot as plt

def calculate_spread(df):
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

def run_strategy_analysis(df, ma_period, deviation_threshold):
    print(f"\n--- Running Analysis for {ma_period}-Period Moving Average ---")
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

        if not in_trade:

            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Short'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                entry_index = i

            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Long'
                entry_price = row['Spread']
                entry_time = row['timestamp']
                entry_index = i

        else:

            if trade_type == 'Short' and row['Spread'] < row['Spread_MA']:
                exit_price = row['Spread']
                exit_time = row['timestamp']
                profit = entry_price - exit_price

                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': 'Short', 'profit': profit

                })
                in_trade = False

            elif trade_type == 'Long' and row['Spread'] > row['Spread_MA']:
                exit_price = row['Spread']
                exit_time = row['timestamp']
                profit = exit_price - entry_price

                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': exit_time,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'trade_type': 'Long', 'profit': profit

                })
                in_trade = False

    if not completed_trades:
        print("No trades were completed for this strategy.")
        return

    trades_df = pd.DataFrame(completed_trades)
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
    total_profit = trades_df['profit'].sum()
    number_of_trades = len(trades_df)
    average_profit = trades_df['profit'].mean()
    average_duration = trades_df['duration'].mean()
    winning_trades = (trades_df['profit'] > 0).sum()
    win_rate = (winning_trades / number_of_trades) * 100

    print("--- Strategy Results ---")
    print(f"Total Trades: {number_of_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {total_profit:.2f} points")
    print(f"Average Profit/Loss per Trade: {average_profit:.2f} points")
    print(f"Average Time in Trade: {average_duration}")
    print("------------------------")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 9))
    plt.plot(strategy_df['timestamp'], strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    plt.plot(strategy_df['timestamp'], strategy_df['Spread_MA'], label=f'{ma_period}-Period MA', color='orange', linewidth=2)
    plt.plot(strategy_df['timestamp'], strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    plt.plot(strategy_df['timestamp'], strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label=f'+/- {deviation_threshold} Bands')
    long_trades = trades_df[trades_df['trade_type'] == 'Long']
    short_trades = trades_df[trades_df['trade_type'] == 'Short']

    plt.scatter(long_trades['entry_time'], long_trades['entry_price'], label='Long Entry', marker='^', color='lime', s=120, edgecolor='black', zorder=5)
    plt.scatter(long_trades['exit_time'], long_trades['exit_price'], label='Exit', marker='x', color='black', s=100, zorder=5)
    plt.scatter(short_trades['entry_time'], short_trades['entry_price'], label='Short Entry', marker='v', color='red', s=120, edgecolor='black', zorder=5)
    plt.scatter(short_trades['exit_time'], short_trades['exit_price'], marker='x', color='black', s=100, zorder=5)
    plt.title(f"Mean Reversion Strategy Backtest (MA = {ma_period}, Threshold = {deviation_threshold})", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Spread Value", fontsize=12)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    DATA_FILE_PATH = 'data/clean_data/april_may_june_futures_data.csv'

    try:
        main_df = pd.read_csv(DATA_FILE_PATH)

    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_PATH}' was not found.")
        print("Please make sure the script is in the same folder as the data file, or update the path.")
        exit()

    main_df_with_spread = calculate_spread(main_df)
    ma_periods_to_test = [20, 50, 100]
    DEVIATION = 40

    for period in ma_periods_to_test:
        run_strategy_analysis(main_df_with_spread, ma_period=period, deviation_threshold=DEVIATION)

    print("\nAll analyses complete.")