# 20_spread_analysis_with_date_labels.py

# This script calculates the ATR-normalized spread and plots the result
# against a sequential index to remove visual gaps. It then adds custom
# date labels to the x-axis to provide time context.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy for creating spaced intervals

# --- Configuration ---
file_path = '../../Data/clean_data/april_may_june_futures_data.csv' # Adjusted for simplicity
# How many date labels to show on the x-axis.
NUMBER_OF_DATE_LABELS = 10

# --- Step 1: Load and Prepare the Data ---
try:
    df = pd.read_csv(file_path)
    print("Successfully loaded the data file.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Convert to datetime and sort.
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
# Reset index to create the sequential x-axis.
df.reset_index(drop=True, inplace=True)


# --- Step 2: Calculate True Range (TR) and Average True Range (ATR) ---
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


# --- Step 3: Calculate the Spread ---
avg_fesx_atr = df['FESX_ATR'].mean()
avg_fdxm_atr = df['FDXM_ATR'].mean()
price_ratio = avg_fdxm_atr / avg_fesx_atr
print(f"Calculated Price Ratio (FDXM ATR / FESX ATR) = {price_ratio:.4f}")

df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)
# Drop NaN rows and reset the index again to ensure it's a clean 0, 1, 2... sequence.
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# --- Step 4: Plot the Spread with Custom Date Labels ---
print("Plotting the spread...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 7))

# Plot the 'Spread' column against the DataFrame's new, clean index.
ax.plot(df.index, df['Spread'], label='FDXM-FESX Spread', color='teal')

# --- THIS IS THE KEY CHANGE FOR DATE LABELS ---
# 1. Get evenly spaced positions along the index (0, 1, 2...).
tick_positions = np.linspace(0, len(df) - 1, NUMBER_OF_DATE_LABELS, dtype=int)

# 2. Look up the actual timestamps at these specific positions.
tick_labels_ts = df['timestamp'].iloc[tick_positions]

# 3. Format these timestamps into readable date strings (e.g., "Apr-22").
tick_labels_str = [ts.strftime('%b-%d') for ts in tick_labels_ts]

# 4. Apply the custom positions and labels to the x-axis.
plt.xticks(ticks=tick_positions, labels=tick_labels_str, rotation=45, ha="right")

# Update the chart labels.
ax.set_title('ATR-Normalized Spread (FESX vs FDXM) - Gaps Removed', fontsize=16)
ax.set_xlabel('Date', fontsize=12) # Change the label back to 'Date'.
ax.set_ylabel('Spread Value', fontsize=12)

ax.grid(True)
ax.legend()
fig.tight_layout()

# Save the figure before showing it.
output_image_file = 'spread_analysis_with_date_labels.png'
plt.savefig(output_image_file, dpi=150)
print(f"Chart successfully saved as '{output_image_file}'")

# Display the plot.
plt.show()

print("Analysis complete.")