import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Data/split_data/april_may_june_futures_data.csv'

try:
    df = pd.read_csv(file_path)

    print("Successfully loaded the data file.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])
print("Calculating ATR and Spread...")
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
avg_fesx_atr = df['FESX_ATR'].mean()
avg_fdxm_atr = df['FDXM_ATR'].mean()
price_ratio = avg_fdxm_atr / avg_fesx_atr

print(f"Calculated Price Ratio (FDXM ATR / FESX ATR) = {price_ratio:.4f}")
df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)
df.dropna(inplace=True)
print("Generating plot with time gaps...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))
plt.plot(df['timestamp'], df['Spread'], label='FDXM-FESX Spread', color='purple')
plt.title('ATR-Normalized Spread (FESX vs FDXM)', fontsize=16)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('Spread Value', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
output_image_file = 'spread_analysis_chart_with_gaps.png'

plt.savefig(output_image_file, dpi=150)
print(f"Chart successfully saved as '{output_image_file}'")
plt.show()
print("Analysis complete.")