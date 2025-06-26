# visualize_price_comparison_with_date_labels.py

# This script creates a clean, gap-free price comparison chart by plotting
# against a sequential index. To provide date context, it then adds custom
# date labels to the x-axis at regular intervals.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Import numpy for creating spaced intervals

# --- Configuration ---
input_file = '../../Data/clean_data/april_may_june_futures_data.csv' # Adjusted path for simplicity
# How many date labels to show on the x-axis.
NUMBER_OF_DATE_LABELS = 10

# --- Main Script Logic ---
print(f"Loading data from: {input_file}")

try:
    # Step 1: Load the data from the CSV file.
    df = pd.read_csv(input_file)
    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"Error: The file was not found at '{input_file}'")
    print("Please make sure your folder structure is correct.")
    exit()

# Step 2: Prepare the data for plotting.
# Convert the 'timestamp' column to sort the data correctly.
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)

# Create a 'sequence_index' column (0, 1, 2, ...) to use for plotting.
df.reset_index(drop=True, inplace=True)


# --- Step 3: Create and Save the Price Comparison Visualization ---

def plot_price_comparison_with_date_labels(data):
    """Creates a gap-free plot with custom date labels on the x-axis."""
    print("Generating price comparison chart with gaps removed and date labels...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(15, 8))

    ax1.set_title('FESX vs. FDXM Price Over Time (Gaps Removed)', fontsize=16)

    # Plot FESX on the left y-axis against the DataFrame's index (which is now 0, 1, 2...).
    ax1.plot(data.index, data['FESX_close'], color='royalblue', label='FESX Close', linewidth=1)
    ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create a second y-axis.
    ax2 = ax1.twinx()
    # Plot FDXM on the right y-axis against the DataFrame's index.
    ax2.plot(data.index, data['FDXM_close'], color='darkorange', label='FDXM Close', linewidth=1)
    ax2.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Create a unified legend.
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- THIS IS THE KEY CHANGE FOR DATE LABELS ---
    # 1. Get evenly spaced positions along the index (0, 1, 2...).
    tick_positions = np.linspace(0, len(data) - 1, NUMBER_OF_DATE_LABELS, dtype=int)

    # 2. Look up the actual timestamps at these specific positions.
    tick_labels_ts = data['timestamp'].iloc[tick_positions]

    # 3. Format these timestamps into readable date strings (e.g., "Apr-22").
    tick_labels_str = [ts.strftime('%b-%d') for ts in tick_labels_ts]

    # 4. Apply the custom positions and labels to the x-axis.
    plt.xticks(ticks=tick_positions, labels=tick_labels_str, rotation=45, ha="right")
    ax1.set_xlabel('Date', fontsize=12) # Change the label back to 'Date'.

    # Ensure all elements fit nicely.
    fig.tight_layout()
    
    # Save the final chart as a PNG file.
    plt.savefig('price_comparison_no_gaps_with_dates.png', dpi=150)
    
    # Close the figure to free up memory.
    plt.close(fig)
    print("Saved 'price_comparison_no_gaps_with_dates.png'")


# --- Main Execution ---
plot_price_comparison_with_date_labels(df)
print("\nVisualization has been generated and saved.")