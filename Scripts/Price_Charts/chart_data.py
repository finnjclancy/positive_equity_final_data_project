import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

input_file = 'Data/clean_data/merged_futures_data.csv'

print(f"Loading data from: {input_file}")
try:
    df = pd.read_csv(input_file)

    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"Error: The file was not found at '{input_file}'")
    print("Please make sure the script is in the correct directory relative to your data.")
    exit()

column_rename_map = {
    'Timestamp': 'timestamp',
    'FESX_Close': 'fesx_close',
    'FDXM_Close': 'fdxm_close',

}
columns_to_keep = [col for col in column_rename_map.keys() if col in df.columns]

if 'Timestamp' not in columns_to_keep or 'FESX_Close' not in columns_to_keep or 'FDXM_Close' not in columns_to_keep:
    print("Error: The input file is missing one of the required columns: 'Timestamp', 'FESX_Close', 'FDXM_Close'.")
    exit()

df = df[columns_to_keep]
df.rename(columns=column_rename_map, inplace=True)
print("Cleaned up column names for easier use.")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)

def plot_price_comparison_with_gaps(data):
    print("Generating price comparison chart with time gaps...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.set_title('FESX vs. FDXM Price Over Time', fontsize=16)
    ax1.set_xlabel('Date and Time', fontsize=12)
    ax1.plot(data['timestamp'], data['fesx_close'], color='royalblue', label='FESX Close', linewidth=1)
    ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.tick_params(axis='x', rotation=45)
    ax2 = ax1.twinx()

    ax2.plot(data['timestamp'], data['fdxm_close'], color='darkorange', label='FDXM Close', linewidth=1)
    ax2.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    plt.savefig('price_comparison_with_gaps.png', dpi=150)
    plt.close(fig)
    print("Saved 'price_comparison_with_gaps.png'")

plot_price_comparison_with_gaps(df)
print("\nVisualization has been generated and saved.")