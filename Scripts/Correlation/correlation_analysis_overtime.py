# 19a_rolling_correlation_with_date_labels_FIXED.py

# This script creates a clean, gap-free rolling correlation chart.
# It includes a fix for the IndexError that occurs when creating custom
# date labels after dropping rows with NaN values.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
ROLLING_WINDOW_SIZE = 250
input_csv_file = '../../Data/clean_data/april_may_june_futures_data.csv'
SHADING_THRESHOLD = 0.8
NUMBER_OF_DATE_LABELS = 10

# --- Main Script Logic ---
print(f"Starting analysis for: {input_csv_file}")
print(f"Plotting with gaps removed and adding custom date labels.")

try:
    df = pd.read_csv(input_csv_file)
    print("Successfully loaded the data.")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    # We still need to reset the index here initially to create the sequence.
    df.reset_index(drop=True, inplace=True)
    df['sequence_index'] = df.index

except FileNotFoundError:
    print(f"Error: The file '{input_csv_file}' was not found.")
    exit()

# Step 2: Calculate the rolling correlation.
df['rolling_correlation'] = df['FESX_close'].rolling(window=ROLLING_WINDOW_SIZE).corr(df['FDXM_close'])

# Step 3: Handle NaN values and FIX the index.
# This line removes the first 249 rows, which messes up the original index.
df.dropna(inplace=True)

# *** THIS IS THE FIX ***
# We reset the index AGAIN after dropping the NaN values.
# This creates a new, clean index starting from 0, which prevents the error.
# The original timestamps and sequence_index values are preserved.
df.reset_index(drop=True, inplace=True)


# --- Step 4: Create the Visualization ---
print("Creating the plot...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --- Top Panel (ax1): Prices ---
ax1.set_title('Market Prices Over Time (Gaps Removed)', fontsize=14)
# We plot against the DataFrame's new index, which is now a clean 0, 1, 2... sequence.
ax1.plot(df.index, df['FESX_close'], color='royalblue', label='FESX Close')
ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='royalblue')
ax1_twin = ax1.twinx()
ax1_twin.plot(df.index, df['FDXM_close'], color='darkorange', label='FDXM Close')
ax1_twin.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor='darkorange')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

# --- Bottom Panel (ax2): Rolling Correlation ---
ax2.set_title(f'{ROLLING_WINDOW_SIZE}-Period Rolling Correlation', fontsize=14)
ax2.plot(df.index, df['rolling_correlation'], color='green', label='Rolling Correlation')
ax2.set_ylabel('Correlation Coefficient', fontsize=12)
ax2.set_ylim(-.10, 1.001)

ax2.fill_between(
    df.index, # Use the new clean index
    SHADING_THRESHOLD,
    df['rolling_correlation'],
    where=(df['rolling_correlation'] < SHADING_THRESHOLD),
    color='red',
    alpha=0.3,
    label=f'Correlation < {SHADING_THRESHOLD}'
)
ax2.axhline(y=SHADING_THRESHOLD, color='red', linestyle='-', linewidth=1.5)
ax2.legend()

# --- Step 5: Customize the X-axis labels ---
# This part now works correctly because the index is a clean sequence from 0.
tick_positions = np.linspace(0, len(df) - 1, NUMBER_OF_DATE_LABELS, dtype=int)
tick_labels_ts = df['timestamp'].iloc[tick_positions]
tick_labels_str = [ts.strftime('%b-%d %H:%M') for ts in tick_labels_ts] # Added time for more context

plt.xticks(ticks=tick_positions, labels=tick_labels_str, rotation=45, ha="right")
ax2.set_xlabel('Date and Time', fontsize=12)

# --- Final Touches ---
fig.tight_layout(pad=2.0)
output_image_file = 'rolling_correlation_with_date_labels_FIXED.png'
plt.savefig(output_image_file, dpi=150)
print(f"Chart successfully saved as '{output_image_file}'")
plt.show()