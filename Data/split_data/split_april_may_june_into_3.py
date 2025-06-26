import pandas as pd

input_csv_file = 'april_may_june_futures_data.csv'
output_file_prefix = 'futures_data_part'
number_of_parts = 3

print(f"Starting to split the file: {input_csv_file}")
try:
    df = pd.read_csv(input_csv_file)

    print(f"Successfully loaded the file.")

except FileNotFoundError:
    print(f"Error: The file '{input_csv_file}' was not found.")
    print("Please make sure the script is in the same folder as your data file.")
    exit()

total_rows = len(df)
print(f"Total rows found in the file: {total_rows}")
rows_per_part = total_rows // number_of_parts

print(f"Each part will have approximately {rows_per_part} rows.")
part1 = df.iloc[0:rows_per_part]
part2 = df.iloc[rows_per_part : 2 * rows_per_part]
part3 = df.iloc[2 * rows_per_part :]
output_filename1 = f"{output_file_prefix}1.csv"

part1.to_csv(output_filename1, index=False)
print(f"Saved {output_filename1} with {len(part1)} rows.")
output_filename2 = f"{output_file_prefix}2.csv"

part2.to_csv(output_filename2, index=False)
print(f"Saved {output_filename2} with {len(part2)} rows.")
output_filename3 = f"{output_file_prefix}3.csv"

part3.to_csv(output_filename3, index=False)
print(f"Saved {output_filename3} with {len(part3)} rows.")
print("\nSplitting complete!")