# 13_rolling_correlation_with_shading.py

# This script builds upon the previous one by adding a shaded red area
# to the correlation plot. This visually highlights all periods where the
# rolling correlation drops below a critical threshold (0.8), making it
# very easy to spot significant decorrelation events.

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
ROLLING_WINDOW_SIZE = 250
# The file with the full dataset.
input_csv_file = '../../data/split_data/april_may_june_futures_data.csv'
# The correlation threshold below which we will shade the area red.
SHADING_THRESHOLD = 0.8

# --- Main Script Logic ---
print(f"Starting analysis for: {input_csv_file}")
print(f"Plotting against a sequential index to remove time gaps.")
print(f"Shading area where correlation is below {SHADING_THRESHOLD}.")

try:
    # Step 1: Read and prepare the data.
    df = pd.read_csv(input_csv_file)
    print("Successfully loaded the data.")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['sequence_index'] = df.index

except FileNotFoundError:
    print(f"Error: The file '{input_csv_file}' was not found.")
    exit()

# Step 2: Calculate the rolling correlation.
df['rolling_correlation'] = df['FESX_close'].rolling(window=ROLLING_WINDOW_SIZE).corr(df['FDXM_close'])
df.dropna(inplace=True)

# --- Step 3: Create the Visualization ---
print("Creating the plot with highlighted low-correlation zones...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --- Top Panel (ax1): Prices (Unchanged) ---
ax1.set_title('Market Prices Over Time (Gaps Removed)', fontsize=14)
ax1.plot(df['sequence_index'], df['FESX_close'], color='royalblue', label='FESX Close')
ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='royalblue')
ax1_twin = ax1.twinx()
ax1_twin.plot(df['sequence_index'], df['FDXM_close'], color='darkorange', label='FDXM Close')
ax1_twin.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor='darkorange')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')

# --- Bottom Panel (ax2): Rolling Correlation with Shading ---
ax2.set_title(f'{ROLLING_WINDOW_SIZE}-Period Rolling Correlation', fontsize=14)
ax2.plot(df['sequence_index'], df['rolling_correlation'], color='green', label='Rolling Correlation')
ax2.set_ylabel('Correlation Coefficient', fontsize=12)
ax2.set_xlabel('Sequence Index (Represents each 10-min data point)', fontsize=12)
ax2.set_ylim(0, 1.001)

# Add the 0.99 horizontal line for reference.
ax2.axhline(y=0.99, color='gray', linestyle='--', linewidth=1, label='High Correlation (0.99)')

# *** THIS IS THE NEW CODE FOR SHADING ***
# `fill_between` is a powerful function for shading areas.
# - The first argument is the x-axis data: our sequence index.
# - The second argument is the bottom of the shaded area: our threshold.
# - The third argument is the top of the shaded area: the correlation line itself.
# - `where`: This is the crucial condition. We only shade the area WHERE
#            the rolling correlation is LESS THAN our threshold.
# - `color`, `alpha`: These set the appearance of the shaded area.
# - `label`: Adds an item to the legend.
ax2.fill_between(
    df['sequence_index'],
    SHADING_THRESHOLD,
    df['rolling_correlation'],
    where=(df['rolling_correlation'] < SHADING_THRESHOLD),
    color='red',
    alpha=0.3, # alpha makes the color semi-transparent
    label=f'Correlation < {SHADING_THRESHOLD}'
)
# We also add a solid line at the threshold for clarity.
ax2.axhline(y=SHADING_THRESHOLD, color='red', linestyle='-', linewidth=1.5)

# Update the legend to include all items.
ax2.legend()

# --- Final Touches ---
fig.tight_layout(pad=2.0)
output_image_file = 'rolling_correlation_with_shading.png'
plt.savefig(output_image_file, dpi=150)
print(f"Chart successfully saved as '{output_image_file}'")
plt.show()