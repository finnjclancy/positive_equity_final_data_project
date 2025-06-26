import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_spread_and_indicators(df, ma_period):
    print("Step 1: Calculating spread and all technical indicators...")
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

def run_backtest(df, z_score_threshold):
    print(f"\nStep 2: Running backtest for Z-Score entry at +/- {z_score_threshold}...")
    completed_trades = []
    in_trade = False
    trade_type = None
    entry_price = 0
    entry_time = None

    for i, row in df.iterrows():

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

    return completed_trades

def analyze_and_print_results(trades_list, z_score_threshold):
    print("Step 3: Analyzing and printing results...")

    if not trades_list:
        print("No trades were executed for this Z-score level.")
        return

    trades_df = pd.DataFrame(trades_list)
    trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
    total_trades = len(trades_df)
    total_pnl = trades_df['profit'].sum()
    avg_time_in_trade = trades_df['duration'].mean()
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    win_percentage = (len(winners) / total_trades) * 100 if total_trades > 0 else 0
    avg_winner = winners['profit'].mean() if not winners.empty else 0
    avg_loser = losers['profit'].mean() if not losers.empty else 0

    print("------------------------------------------------------")
    print(f"       Backtest Results for Z-Score: {z_score_threshold}       ")
    print("------------------------------------------------------")
    print(f"{'Total PnL:':<25} {total_pnl:>20.2f} points")
    print(f"{'Total Trades:':<25} {total_trades:>20}")
    print(f"{'Win Percentage:':<25} {win_percentage:>19.2f} %")
    print(f"{'Average Winner:':<25} {avg_winner:>20.2f} points")
    print(f"{'Average Loser:':<25} {avg_loser:>20.2f} points")
    print(f"{'Average Time in Trade:':<25} {str(avg_time_in_trade)}")
    print("------------------------------------------------------")

def plot_and_save_equity_curve(trades_list, z_score_threshold):
    print("Step 4: Generating and saving equity curve...")

    if not trades_list:
        print("Cannot plot equity curve: No trades were made.")
        return

    trades_df = pd.DataFrame(trades_list)
    trades_df.sort_values('exit_time', inplace=True)
    trades_df['equity_curve'] = trades_df['profit'].cumsum()
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(trades_df['exit_time'], trades_df['equity_curve'], label='Equity Curve', color='navy')
    plt.title(f'Equity Curve for Z-Score Entry at {z_score_threshold}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative PnL (in points)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    filename = f'equity_curve_zscore_{z_score_threshold}.png'

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Equity curve saved as '{filename}'")

if __name__ == "__main__":
    DATA_FILE_PATH = '../../../Data/clean_data/april_may_june_futures_data.csv'
    MA_PERIOD = 20
    Z_SCORE_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0]

    try:
        main_df = pd.read_csv(DATA_FILE_PATH)

        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])

    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
        exit()

    prepared_df = calculate_spread_and_indicators(main_df, MA_PERIOD)

    for z_score in Z_SCORE_LEVELS:
        completed_trades = run_backtest(prepared_df.copy(), z_score)

        analyze_and_print_results(completed_trades, z_score)
        plot_and_save_equity_curve(completed_trades, z_score)

    print("\n\nAll backtests are complete.")