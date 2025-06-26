# 02_spread_analysis.py

import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Load the Data ---
file_path = 'data/clean_data/april_may_june_futures_data.csv'
df = pd.read_csv(file_path)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Step 2: Calculate True Range (TR) and Average True Range (ATR) ---
# ATR is a measure of volatility. We need it for your spread formula.

# Calculate the range of each candle (High - Low).
df['FESX_h-l'] = df['FESX_high'] - df['FESX_low']
df['FDXM_h-l'] = df['FDXM_high'] - df['FDXM_low']

# Calculate the difference between the current high and the previous close.
# .shift(1) gets the value from the previous row. We take the absolute value (abs).
df['FESX_h-pc'] = abs(df['FESX_high'] - df['FESX_close'].shift(1))
df['FDXM_h-pc'] = abs(df['FDXM_high'] - df['FDXM_close'].shift(1))

# Calculate the difference between the current low and the previous close.
df['FESX_l-pc'] = abs(df['FESX_low'] - df['FESX_close'].shift(1))
df['FDXM_l-pc'] = abs(df['FDXM_low'] - df['FDXM_close'].shift(1))

# The True Range (TR) is the BIGGEST of the three values we just calculated.
# .max(axis=1) finds the maximum value in each row for the given columns.
df['FESX_TR'] = df[['FESX_h-l', 'FESX_h-pc', 'FESX_l-pc']].max(axis=1)
df['FDXM_TR'] = df[['FDXM_h-l', 'FDXM_h-pc', 'FDXM_l-pc']].max(axis=1)

# The Average True Range (ATR) is a moving average of the True Range.
# We use a 14-period window, which is a standard setting.
# .rolling(14) creates a 14-period window, and .mean() calculates the average.
df['FESX_ATR'] = df['FESX_TR'].rolling(window=14).mean()
df['FDXM_ATR'] = df['FDXM_TR'].rolling(window=14).mean()


# --- Step 3: Calculate the Spread using your formula ---

# To get a single Price Ratio for the whole period, we can use the average ATR.
avg_fesx_atr = df['FESX_ATR'].mean()
avg_fdxm_atr = df['FDXM_ATR'].mean()

# Calculate the Price Ratio as defined in the project description.
price_ratio = avg_fdxm_atr / avg_fesx_atr
print(f"Calculated Price Ratio (FDXM ATR / FESX ATR) = {price_ratio:.4f}")

# Calculate the spread for every single point in time.
# Spread = FDXM_close - (FESX_close * Price_Ratio)
df['Spread'] = df['FDXM_close'] - (df['FESX_close'] * price_ratio)


# --- Step 4: Plot the Spread ---
print("Plotting the spread... please close the plot window to continue.")

# Create a new figure for our plot with a specific size for better viewing.
plt.figure(figsize=(15, 7))

# Plot the 'Spread' column against the 'timestamp'.
plt.plot(df['timestamp'], df['Spread'], label='FDXM-FESX Spread')

# Add a title and labels to the axes to make the chart clear.
plt.title('Spread Price (FDXM vs FESX)')
plt.xlabel('Date and Time')
plt.ylabel('Spread Value')

# Add a grid for easier reading of values.
plt.grid(True)

# Add a legend to identify the plotted line.
plt.legend()

# Display the plot on the screen. The script will pause here until you close the plot window.
plt.show()

# The script will end after the plot window is closed.
print("Analysis complete.")