# visualize_price_comparison_no_gaps.py

# This script loads the futures data and creates a single PNG chart comparing
# the closing prices of FESX and FDXM.
#
# It plots the data against a sequential index to "stitch together" the
# active trading periods, removing the visual gaps from inactive market hours.

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Update this path if your script is located elsewhere relative to the data.
input_file = 'Data/split_data/april_may_june_futures_data.csv'

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

# Create a new 'sequence_index' column (0, 1, 2, ...) to use for the x-axis.
# This is the key step to removing the visual time gaps.
df.reset_index(drop=True, inplace=True)
df['sequence_index'] = df.index


# --- Step 3: Create and Save the Price Comparison Visualization ---

def plot_price_comparison_no_gaps(data):
    """Creates and saves a dual-axis plot comparing FESX and FDXM prices."""
    print("Generating price comparison chart with gaps removed...")
    
    # Set a professional style for the plot.
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create the figure and the primary axis.
    fig, ax1 = plt.subplots(figsize=(15, 8))

    ax1.set_title('FESX vs. FDXM Price Over Time (Gaps Removed)', fontsize=16)
    # Update the x-axis label to reflect the new index.
    ax1.set_xlabel('Sequence Index (Represents each 10-min data point)', fontsize=12)

    # Plot FESX on the left y-axis against the 'sequence_index'.
    ax1.plot(data['sequence_index'], data['FESX_close'], color='royalblue', label='FESX Close', linewidth=1)
    ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create a second y-axis that shares the same x-axis.
    ax2 = ax1.twinx()
    # Plot FDXM on the right y-axis against the 'sequence_index'.
    ax2.plot(data['sequence_index'], data['FDXM_close'], color='darkorange', label='FDXM Close', linewidth=1)
    ax2.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Create a unified legend for both lines.
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Ensure all elements fit nicely within the figure.
    fig.tight_layout()
    
    # Save the final chart as a PNG file.
    plt.savefig('price_comparison_no_gaps.png', dpi=150)
    
    # Close the figure to free up memory.
    plt.close(fig)
    print("Saved 'price_comparison_no_gaps.png'")


# --- Main Execution ---
plot_price_comparison_no_gaps(df)
print("\nVisualization has been generated and saved.")