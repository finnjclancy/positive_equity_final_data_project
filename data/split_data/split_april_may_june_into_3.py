# split_data.py

# This script reads a large CSV file and splits it into three smaller,
# equal-sized CSV files based on the number of rows.

# We need to import the pandas library, which is a powerful tool for working with
# data tables in Python. We give it the shorter name 'pd' by convention.
import pandas as pd

# --- Configuration ---
# You can easily change these filenames if you need to split a different file.
input_csv_file = 'april_may_june_futures_data.csv'
output_file_prefix = 'futures_data_part'
number_of_parts = 3

# --- Main Script Logic ---
print(f"Starting to split the file: {input_csv_file}")

# We use a 'try...except' block to gracefully handle the case where the file might not exist.
try:
    # Step 1: Read the entire CSV file into a pandas DataFrame.
    # A DataFrame is like a spreadsheet or a table in memory.
    df = pd.read_csv(input_csv_file)
    print(f"Successfully loaded the file.")

except FileNotFoundError:
    # If the file isn't found, print a helpful error message and stop the script.
    print(f"Error: The file '{input_csv_file}' was not found.")
    print("Please make sure the script is in the same folder as your data file.")
    exit() # This command stops the script.

# Step 2: Calculate the split points.
# We get the total number of rows in the DataFrame.
total_rows = len(df)
print(f"Total rows found in the file: {total_rows}")

# We use integer division (//) to find the number of rows for each of the first two parts.
# This automatically handles cases where the total isn't perfectly divisible by 3.
rows_per_part = total_rows // number_of_parts
print(f"Each part will have approximately {rows_per_part} rows.")

# Step 3: Split the DataFrame into three smaller DataFrames using slicing.
# We use .iloc which selects rows by their integer position (like row 0, row 1, etc.).

# Part 1: From the beginning (row 0) up to (but not including) `rows_per_part`.
part1 = df.iloc[0:rows_per_part]

# Part 2: From `rows_per_part` up to (but not including) `2 * rows_per_part`.
part2 = df.iloc[rows_per_part : 2 * rows_per_part]

# Part 3: From `2 * rows_per_part` all the way to the end of the DataFrame.
# This clever slicing automatically includes any leftover rows in the last part.
part3 = df.iloc[2 * rows_per_part :]


# Step 4: Save each part to a new CSV file.
# We will create the filenames dynamically, e.g., 'futures_data_part1.csv'.

# Save the first part.
output_filename1 = f"{output_file_prefix}1.csv"
part1.to_csv(output_filename1, index=False)
# `index=False` is very important! It prevents pandas from writing its own
# row numbers (0, 1, 2...) as a new column in our output file.
print(f"Saved {output_filename1} with {len(part1)} rows.")

# Save the second part.
output_filename2 = f"{output_file_prefix}2.csv"
part2.to_csv(output_filename2, index=False)
print(f"Saved {output_filename2} with {len(part2)} rows.")

# Save the third part.
output_filename3 = f"{output_file_prefix}3.csv"
part3.to_csv(output_filename3, index=False)
print(f"Saved {output_filename3} with {len(part3)} rows.")

print("\nSplitting complete!")