# visualize_price_comparison.py

# This script loads the merged futures data, cleans up the column names,
# and creates a single PNG chart comparing the closing prices of FESX and FDXM.
#
# It plots against the original timestamps, so the time gaps from inactive
# market periods (overnight/weekends) will be visible on the x-axis.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Import for formatting date labels

# --- Configuration ---
input_file = 'Data/clean_data/merged_futures_data.csv'

# --- Main Script Logic ---
print(f"Loading data from: {input_file}")

try:
    # Step 1: Load the data from the CSV file.
    df = pd.read_csv(input_file)
    print("Data loaded successfully.")

except FileNotFoundError:
    print(f"Error: The file was not found at '{input_file}'")
    print("Please make sure the script is in the correct directory relative to your data.")
    exit()

# Step 2: Clean up the column names.
# This makes the columns easier to reference in the code.
column_rename_map = {
    'Timestamp': 'timestamp',
    'FESX_Close': 'fesx_close',
    'FDXM_Close': 'fdxm_close',
}

# We only need a few columns for this plot.
# First, find which of the columns we want actually exist in the file.
columns_to_keep = [col for col in column_rename_map.keys() if col in df.columns]

# If we don't have the necessary columns, we can't proceed.
if 'Timestamp' not in columns_to_keep or 'FESX_Close' not in columns_to_keep or 'FDXM_Close' not in columns_to_keep:
    print("Error: The input file is missing one of the required columns: 'Timestamp', 'FESX_Close', 'FDXM_Close'.")
    exit()
    
# Select only the columns we need.
df = df[columns_to_keep]
# Rename them to our simple format.
df.rename(columns=column_rename_map, inplace=True)
print("Cleaned up column names for easier use.")


# Step 3: Prepare the data for plotting.
# Convert the 'timestamp' column into a proper datetime object.
df['timestamp'] = pd.to_datetime(df['timestamp'])
# Sort the data by time to ensure the line plot is drawn correctly.
df.sort_values('timestamp', inplace=True)


# --- Step 4: Create and Save the Price Comparison Visualization ---

def plot_price_comparison_with_gaps(data):
    """Creates and saves a dual-axis plot comparing FESX and FDXM prices against time."""
    print("Generating price comparison chart with time gaps...")
    
    # Set a professional style for the plot.
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create the figure and the primary axis.
    fig, ax1 = plt.subplots(figsize=(15, 8))

    ax1.set_title('FESX vs. FDXM Price Over Time', fontsize=16)
    ax1.set_xlabel('Date and Time', fontsize=12)

    # Plot FESX on the left y-axis against the 'timestamp' column.
    ax1.plot(data['timestamp'], data['fesx_close'], color='royalblue', label='FESX Close', linewidth=1)
    ax1.set_ylabel('FESX Price', color='royalblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    
    # Rotate the date labels on the x-axis for better readability.
    ax1.tick_params(axis='x', rotation=45)

    # Create a second y-axis that shares the same x-axis (the timeline).
    ax2 = ax1.twinx()
    # Plot FDXM on the right y-axis against the 'timestamp' column.
    ax2.plot(data['timestamp'], data['fdxm_close'], color='darkorange', label='FDXM Close', linewidth=1)
    ax2.set_ylabel('FDXM Price', color='darkorange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Create a unified legend for both lines.
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Improve the date formatting on the x-axis to be cleaner.
    # This will show dates like "May-20", "May-21", etc.
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Ensure all elements fit nicely within the figure.
    fig.tight_layout()
    
    # Save the final chart as a PNG file.
    plt.savefig('price_comparison_with_gaps.png', dpi=150)
    
    # Close the figure to free up memory.
    plt.close(fig)
    print("Saved 'price_comparison_with_gaps.png'")


# --- Main Execution ---
plot_price_comparison_with_gaps(df)
print("\nVisualization has been generated and saved.")