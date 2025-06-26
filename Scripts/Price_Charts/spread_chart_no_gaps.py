# 16_spread_analysis_no_gaps.py

# This script calculates the ATR-normalized spread between FESX and FDXM
# and then plots the result.
#
# This updated version plots the spread against a sequential index to remove
# the large visual gaps caused by overnight and weekend market closures,
# resulting in a much clearer and more useful chart.

import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Load and Prepare the Data ---
file_path = 'Data/split_data/april_may_june_futures_data.csv'

try:
    df = pd.read_csv(file_path)
    print("Successfully loaded the data file.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Convert to datetime and sort to ensure correct order for calculations.
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)

# *** KEY CHANGE FOR PLOTTING ***
# Create a new column that is a simple sequence of numbers (0, 1, 2...).
# This will be our new x-axis for the chart.
df.reset_index(drop=True, inplace=True)
df['sequence_index'] = df.index


# --- Step 2: Calculate True Range (TR) and Average True Range (ATR) ---
# This part of the logic is correct and remains unchanged. The .shift(1)
# correctly handles the overnight jumps by comparing to the previous close.
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


# --- Step 3: Calculate the Spread using your formula ---
# This also remains unchanged.
avg_fesx_atr = df['FESX_ATR'].mean()
avg_fdxm_atr = df['FDXM_ATR'].mean()
price_ratio = avg_fdxm_atr / avg_fesx_atr
print(f"Calculated Price Ratio (FDXM ATR / FESX ATR) = {price_ratio:.4f}")

df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)

# Drop any rows with NaN values that were created by the rolling calculations.
df.dropna(inplace=True)


# --- Step 4: Plot the Spread (with Gaps Removed) ---
print("Plotting the spread against the sequential index...")

# Set a professional plot style.
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))

# *** KEY PLOTTING CHANGE ***
# Plot the 'Spread' column against the new 'sequence_index' column.
plt.plot(df['sequence_index'], df['Spread'], label='FDXM-FESX Spread', color='teal')

# Update the chart labels to reflect the new x-axis.
plt.title('ATR-Normalized Spread (FESX vs FDXM) - Gaps Removed', fontsize=16)
plt.xlabel('Sequence Index (Represents each 10-min data point)', fontsize=12)
plt.ylabel('Spread Value', fontsize=12)

plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure before showing it.
output_image_file = 'spread_analysis_no_gaps.png'
plt.savefig(output_image_file, dpi=150)
print(f"Chart successfully saved as '{output_image_file}'")

# Display the plot.
plt.show()

print("Analysis complete.")