import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = '../../Data/clean_data/april_may_june_futures_data.csv'
NUMBER_OF_DATE_LABELS = 10

print(f"Loading data from: {input_file}")
try:
    df = pd.read_csv(input_file)

    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"Error: The file was not found at '{input_file}'")
    print("Please make sure your folder structure is correct.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

def plot_price_comparison_with_date_labels(data):
    print("Generating price comparison chart with gaps removed and date labels...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.set_title('FESX vs. FDXM Price Over Time (Gaps Removed)', fontsize=16)
    ax1.plot(data.index, data['FESX_close'], color='royalblue', label='FESX Close', linewidth=1)
    ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax2 = ax1.twinx()

    ax2.plot(data.index, data['FDXM_close'], color='darkorange', label='FDXM Close', linewidth=1)
    ax2.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    tick_positions = np.linspace(0, len(data) - 1, NUMBER_OF_DATE_LABELS, dtype=int)
    tick_labels_ts = data['timestamp'].iloc[tick_positions]
    tick_labels_str = [ts.strftime('%b-%d') for ts in tick_labels_ts]

    plt.xticks(ticks=tick_positions, labels=tick_labels_str, rotation=45, ha="right")
    ax1.set_xlabel('Date', fontsize=12)
    fig.tight_layout()
    plt.savefig('price_comparison_no_gaps_with_dates.png', dpi=150)
    plt.close(fig)
    print("Saved 'price_comparison_no_gaps_with_dates.png'")

plot_price_comparison_with_date_labels(df)
print("\nVisualization has been generated and saved.")