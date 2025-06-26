# 01_correlation_analysis.py

import pandas as pd

def calculate_and_print_correlation(file_path, description):
    """
    This function reads a CSV file, calculates the correlation between
    FESX_close and FDXM_close prices, and prints the result.
    """
    print(f"--- Analyzing: {description} ---")

    df = pd.read_csv(file_path)

    # Select the two columns we are interested in: the closing prices for FESX and FDXM.
    fesx_prices = df['FESX_close']
    fdxm_prices = df['FDXM_close']

    # Calculate the correlation between these two sets of prices.
    # The .corr() function does all the math for us.
    correlation = fesx_prices.corr(fdxm_prices)

    # Print the calculated correlation, formatted to 4 decimal places to make it easy to read.
    print(f"The correlation between FESX and FDXM is: {correlation:.4f}")
    print("\n") # Print a blank line for better spacing.


# --- Main part of the script ---
# This is where we will call our function for each data file.

# A list containing the filenames of the data segments.
files_to_analyze = [
    'data/split_data/futures_data_part1.csv',
    'data/split_data/futures_data_part2.csv',
    'data/split_data/futures_data_part3.csv',
    'data/clean_data/cleaned_futures_data.csv' # The full dataset
]

# A list of descriptions that match the filenames.
descriptions = [
    "Segment 1",
    "Segment 2",
    "Segment 3",
    "Full Dataset"
]

# Loop through our lists of files and descriptions and run the analysis for each one.
# The zip() function pairs up the items from our two lists.
for file, desc in zip(files_to_analyze, descriptions):
    calculate_and_print_correlation(file, desc)