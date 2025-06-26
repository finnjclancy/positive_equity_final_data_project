# 14_correlation_of_returns.py

# This script provides a more accurate correlation analysis by accounting for
# time gaps (overnight/weekends). Instead of correlating raw price levels,
# we correlate the 10-minute percentage returns. This is the standard
# financial method to measure the co-movement of assets while minimizing
# distortions from price level differences and inactive periods.

import pandas as pd
import matplotlib.pyplot as plt

def calculate_returns_correlation(file_path, description):
    """
    This function reads a CSV file, calculates the percentage returns for
    FESX and FDXM, and then calculates the correlation of those returns.
    """
    print(f"--- Analyzing: {description} ---")
    print("   (Calculating correlation based on 10-minute returns to account for gaps)")

    # Read the CSV data into a pandas DataFrame.
    df = pd.read_csv(file_path)

    # --- THIS IS THE KEY CHANGE ---
    # Instead of using the raw prices, we first calculate the percentage change
    # from one row to the next. The .pct_change() method does this for us.
    # The first row will be NaN (Not a Number) since it has no prior period to compare to.
    fesx_returns = df['FESX_close'].pct_change()
    fdxm_returns = df['FDXM_close'].pct_change()

    # Now, we calculate the correlation between these two series of returns.
    # The .corr() function automatically ignores the NaN value in the first row.
    correlation = fesx_returns.corr(fdxm_returns)

    # Print the more accurate correlation result.
    print(f"The correlation of returns between FESX and FDXM is: {correlation:.4f}")
    print("\n")

    # Return the calculated value for plotting.
    return correlation


# --- Main part of the script ---
# Define the files and their descriptions.
files_to_analyze = [
    '../../data/split_data/futures_data_part1.csv',
    '../../data/split_data/futures_data_part2.csv',
    '../../data/split_data/futures_data_part3.csv',
    '../../data/split_data/april_may_june_futures_data.csv' # The full dataset
]

descriptions = [
    "April 22nd - May 12th",
    "May 12th - May 30th",
    "May 30th - June 19th",
    "Full Period\n(Apr 22nd - Jun 19th)"
]

# Create an empty list to store the correlation results.
correlation_results = []

# Loop through the files, but this time call the new function.
for file, desc in zip(files_to_analyze, descriptions):
    corr_value = calculate_returns_correlation(file, desc)
    correlation_results.append(corr_value)


# --- Visualization Section ---
# This part creates the bar chart from the new, more accurate results.

print("--- Creating correlation chart based on returns ---")

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(descriptions, correlation_results, color='mediumseagreen', edgecolor='black')

# Update the title to reflect the new methodology.
ax.set_title('Correlation of 10-Minute Returns (FESX vs. FDXM)', fontsize=16)
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
ax.set_xlabel('Data Period', fontsize=12)

# Adjust the y-axis to a reasonable range for returns correlation.
# It will likely be lower than the raw price correlation, which is expected.
ax.set_ylim(0.85, 1.0)

# Add the exact correlation value on top of each bar.
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}',
             ha='center', va='bottom', color='black', fontweight='bold')

plt.tight_layout()

# Save the final, corrected chart as a new PNG file.
output_image_file = 'correlation_of_returns.png'
plt.savefig(output_image_file)

print(f"Chart successfully saved as '{output_image_file}'")