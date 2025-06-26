# 15_correlation_returns_table.py

# This script provides an accurate correlation analysis by accounting for
# time gaps (overnight/weekends). It calculates the correlation of 10-minute
# percentage returns, which is the standard method for this type of analysis.
#
# The final output is presented as a clean, text-based table.

import pandas as pd

def calculate_returns_correlation(file_path):
    """
    This function reads a CSV file, calculates the percentage returns for
    FESX and FDXM, and then returns the correlation coefficient of those returns.
    """
    # Use a 'try...except' block to handle cases where a file might be missing.
    try:
        # Read the CSV data into a pandas DataFrame.
        df = pd.read_csv(file_path)

        # Calculate the percentage change from one row to the next.
        # This turns the price series into a returns series.
        fesx_returns = df['FESX_close'].pct_change()
        fdxm_returns = df['FDXM_close'].pct_change()

        # Calculate the correlation between the two series of returns.
        correlation = fesx_returns.corr(fdxm_returns)

        # Return the final calculated value.
        return correlation

    except FileNotFoundError:
        # If the file isn't found, print a warning and return None.
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return None


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
    "\nFull Period\n(Apr 22nd - Jun 19th)"
]

# Create an empty dictionary to store the results.
# A dictionary is great for pairing the description with its value.
correlation_results = {}

print("Calculating correlation of 10-minute returns...")

# Loop through our files and run the analysis for each one.
for file, desc in zip(files_to_analyze, descriptions):
    # Call our function to get the correlation value.
    corr_value = calculate_returns_correlation(file)
    # If the function returned a value (i.e., the file was found), add it to our dictionary.
    if corr_value is not None:
        correlation_results[desc] = corr_value

print("Calculation complete.\n")


# --- Display Results in a Table Format ---
# This part of the script formats and prints the final output.

# Check if we have any results to display.
if correlation_results:
    print("---------------------------------------------------------")
    print("      Correlation of 10-Minute Returns (FESX vs. FDXM)   ")
    print("---------------------------------------------------------")
    # Print the table header with padding to align the columns.
    # The `<25` means "left-align this string in a space 25 characters wide".
    print(f"{'Data Period':<25} | {'Correlation Coefficient':<25}")
    print("---------------------------------------------------------")
    
    # Loop through the items in our results dictionary.
    for period, correlation in correlation_results.items():
        # Print each row, using padding and number formatting for alignment.
        # `:.6f` formats the number to 6 decimal places.
        print(f"{period:<25} | {correlation:<25.6f}")
        
    print("---------------------------------------------------------")
else:
    print("No results to display. Please check that the data files exist in the correct path.")