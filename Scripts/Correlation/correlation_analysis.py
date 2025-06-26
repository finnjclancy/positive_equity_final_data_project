import pandas as pd

def calculate_returns_correlation(file_path):
    try:
        df = pd.read_csv(file_path)
        fesx_returns = df['FESX_close'].pct_change()
        fdxm_returns = df['FDXM_close'].pct_change()
        correlation = fesx_returns.corr(fdxm_returns)

        return correlation

    except FileNotFoundError:
        print(f"Warning: File not found at '{file_path}'. Skipping.")
        return None

files_to_analyze = [
    '../../data/split_data/futures_data_part1.csv',
    '../../data/split_data/futures_data_part2.csv',
    '../../data/split_data/futures_data_part3.csv',
    '../../data/split_data/april_may_june_futures_data.csv'

]
descriptions = [

    "April 22nd - May 12th",
    "May 12th - May 30th",
    "May 30th - June 19th",
    "\nFull Period\n(Apr 22nd - Jun 19th)"

]
correlation_results = {}

print("Calculating correlation of 10-minute returns...")

for file, desc in zip(files_to_analyze, descriptions):
    corr_value = calculate_returns_correlation(file)

    if corr_value is not None:
        correlation_results[desc] = corr_value

print("Calculation complete.\n")

if correlation_results:
    print("---------------------------------------------------------")
    print("      Correlation of 10-Minute Returns (FESX vs. FDXM)   ")
    print("---------------------------------------------------------")
    print(f"{'Data Period':<25} | {'Correlation Coefficient':<25}")
    print("---------------------------------------------------------")

    for period, correlation in correlation_results.items():
        print(f"{period:<25} | {correlation:<25.6f}")

    print("---------------------------------------------------------")

else:
    print("No results to display. Please check that the data files exist in the correct path.")