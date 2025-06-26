import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_spread(df):
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

def run_mean_reversion_backtest(df, ma_period, deviation_threshold):
    print(f"Running backtest for MA={ma_period} and Deviation={deviation_threshold}...")
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

    for i, row in strategy_df.iterrows():

        if in_trade:

            if trade_type == 'Long' and row['Spread'] >= row['Spread_MA']:
                exit_price = row['Spread']
                profit = exit_price - entry_price

                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': row['timestamp'],
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'profit': profit

                })
                in_trade = False

            elif trade_type == 'Short' and row['Spread'] <= row['Spread_MA']:
                exit_price = row['Spread']
                profit = entry_price - exit_price

                completed_trades.append({
                    'entry_time': entry_time, 'exit_time': row['timestamp'],
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'profit': profit

                })
                in_trade = False

        if not in_trade:

            if row['Spread'] > row['Upper_Band']:
                in_trade = True
                trade_type = 'Short'
                entry_price = row['Spread']
                entry_time = row['timestamp']

            elif row['Spread'] < row['Lower_Band']:
                in_trade = True
                trade_type = 'Long'
                entry_price = row['Spread']
                entry_time = row['timestamp']

    trades_df = pd.DataFrame(completed_trades)
    return strategy_df, trades_df

def print_performance_table(trades_df):

    if trades_df.empty:
        print("No trades were executed.")
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

    print("\n--- Strategy Performance Summary ---")
    print("------------------------------------------------------")
    print(f"{'Total PnL:':<25} {total_pnl:>20.2f} points")
    print(f"{'Total Trades:':<25} {total_trades:>20}")
    print(f"{'Win Percentage:':<25} {win_percentage:>19.2f} %")
    print(f"{'Average Winner:':<25} {avg_winner:>20.2f} points")
    print(f"{'Average Loser:':<25} {avg_loser:>20.2f} points")
    print(f"{'Average Time in Trade:':<25} {str(avg_time_in_trade)}")
    print("------------------------------------------------------\n")

def plot_trades_chart_with_dates(strategy_df, trades_df, num_labels=10):

    if trades_df.empty:
        print("Cannot plot trades chart: No trades were made.")
        return

    print("Generating trades chart with custom date labels...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(strategy_df.index, strategy_df['Spread'], label='Spread', color='dodgerblue', alpha=0.7)
    ax.plot(strategy_df.index, strategy_df['Spread_MA'], label='50-Period MA', color='orange', linewidth=2)
    ax.plot(strategy_df.index, strategy_df['Upper_Band'], color='red', linestyle='--', alpha=0.5)
    ax.plot(strategy_df.index, strategy_df['Lower_Band'], color='red', linestyle='--', alpha=0.5, label='+/- 40 Bands')
    entry_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in trades_df['entry_time']]
    exit_indices = [strategy_df.index[strategy_df['timestamp'] == t].values[0] for t in trades_df['exit_time']]

    ax.scatter(entry_indices, trades_df['entry_price'], label='Entry', marker='^', color='lime', s=100, edgecolor='black', zorder=5)
    ax.scatter(exit_indices, trades_df['exit_price'], label='Exit', marker='x', color='black', s=100, zorder=5)
    ax.set_title('Mean Reversion Trades (MA=50, Deviation=40)', fontsize=16)
    ax.set_ylabel('Spread Value', fontsize=12)
    ax.legend()
    tick_positions = np.linspace(0, len(strategy_df) - 1, num_labels, dtype=int)
    tick_labels_ts = strategy_df['timestamp'].iloc[tick_positions]
    tick_labels_str = [ts.strftime('%b-%d') for ts in tick_labels_ts]

    plt.xticks(ticks=tick_positions, labels=tick_labels_str, rotation=45, ha="right")
    ax.set_xlabel('Date', fontsize=12)
    fig.tight_layout()
    plt.savefig('mean_reversion_trades_chart_with_dates.png', dpi=150)
    plt.close(fig)
    print("Saved 'mean_reversion_trades_chart_with_dates.png'")

def process_and_display_equity_curve(trades_df):

    if trades_df.empty:
        print("Cannot process equity curve: No trades were made.")
        return

    print("Generating equity curve chart and table...")
    trades_df.sort_values('exit_time', inplace=True)
    trades_df.reset_index(drop=True, inplace=True)
    trades_df['equity_curve'] = trades_df['profit'].cumsum()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(trades_df.index, trades_df['equity_curve'], label='Equity Curve', color='navy')
    ax.set_title('Strategy Equity Curve', fontsize=16)
    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Cumulative PnL (in points)', fontsize=12)
    ax.grid(True)
    plt.savefig('mean_reversion_equity_curve.png', dpi=150)
    plt.close(fig)
    print("Saved 'mean_reversion_equity_curve.png'\n")
    print("--- Equity Curve Data ---")
    equity_table = trades_df[['exit_time', 'profit', 'equity_curve']].copy()

    equity_table.index.name = "Trade No."
    print(equity_table.to_string())
    print("-------------------------")

if __name__ == "__main__":
    DATA_FILE_PATH = '../../Data/clean_data/april_may_june_futures_data.csv'
    MA_PERIOD = 50
    DEVIATION = 40

    try:
        main_df = pd.read_csv(DATA_FILE_PATH)

    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
        exit()

    df_with_spread = calculate_spread(main_df)
    strategy_data, trades_data = run_mean_reversion_backtest(df_with_spread, MA_PERIOD, DEVIATION)
    print_performance_table(trades_data)
    plot_trades_chart_with_dates(strategy_data, trades_data)
    process_and_display_equity_curve(trades_data)
    print("\nDeep-dive analysis complete.")