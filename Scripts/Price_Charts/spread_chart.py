# 18_spread_analysis_with_gaps_save_png.py

# This script calculates the ATR-normalized spread and plots it against
# the original timestamps, which will show the time gaps from inactive
# market periods (overnight/weekends).
#
# It saves the final chart, with gaps, as a PNG file.

import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Load the Data ---
file_path = 'Data/split_data/april_may_june_futures_data.csv' # Make sure this path is correct

try:
    df = pd.read_csv(file_path)
    print("Successfully loaded the data file.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Convert the 'timestamp' column to a proper datetime object.
df['timestamp'] = pd.to_datetime(df['timestamp'])


# --- Step 2: Calculate True Range (TR) and Average True Range (ATR) ---
# This logic is exactly as you provided in your original script.
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


# --- Step 3: Calculate the Final Spread Value ---
# This also remains unchanged.
avg_fesx_atr = df['FESX_ATR'].mean()
avg_fdxm_atr = df['FDXM_ATR'].mean()
price_ratio = avg_fdxm_atr / avg_fesx_atr
print(f"Calculated Price Ratio (FDXM ATR / FESX ATR) = {price_ratio:.4f}")

df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)
# Remove initial rows with NaN values.
df.dropna(inplace=True)


# --- Step 4: Plot the Spread (with Gaps) and Save as PNG ---
print("Generating plot with time gaps...")

# Set a professional plot style.
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 7))

# Plot the 'Spread' column against the original 'timestamp' column.
# This will render the chart with the visible time gaps.
plt.plot(df['timestamp'], df['Spread'], label='FDXM-FESX Spread', color='purple')

# Add clear chart titles and labels.
plt.title('ATR-Normalized Spread (FESX vs FDXM)', fontsize=16)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('Spread Value', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout() # Adjusts plot to ensure everything fits without overlapping.


# --- SAVE THE FIGURE ---
# Define the filename for your output image.
output_image_file = 'spread_analysis_chart_with_gaps.png'

# Save the current figure to a file before showing it.
plt.savefig(output_image_file, dpi=150)

# Print a confirmation message.
print(f"Chart successfully saved as '{output_image_file}'")

# Display the plot on the screen.
plt.show()

print("Analysis complete.")