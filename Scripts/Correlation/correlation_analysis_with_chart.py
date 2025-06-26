import pandas as pd
import matplotlib.pyplot as plt

def calculate_returns_correlation(file_path, description):
    print(f"--- Analyzing: {description} ---")
    print("   (Calculating correlation based on 10-minute returns to account for gaps)")
    df = pd.read_csv(file_path)
    fesx_returns = df['FESX_close'].pct_change()
    fdxm_returns = df['FDXM_close'].pct_change()
    correlation = fesx_returns.corr(fdxm_returns)

    print(f"The correlation of returns between FESX and FDXM is: {correlation:.4f}")
    print("\n")
    return correlation

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
    "Full Period\n(Apr 22nd - Jun 19th)"

]
correlation_results = []

for file, desc in zip(files_to_analyze, descriptions):
    corr_value = calculate_returns_correlation(file, desc)

    correlation_results.append(corr_value)

print("--- Creating correlation chart based on returns ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(descriptions, correlation_results, color='mediumseagreen', edgecolor='black')

ax.set_title('Correlation of 10-Minute Returns (FESX vs. FDXM)', fontsize=16)
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
ax.set_xlabel('Data Period', fontsize=12)
ax.set_ylim(0.85, 1.0)

for bar in bars:
    yval = bar.get_height()

    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}',
             ha='center', va='bottom', color='black', fontweight='bold')

plt.tight_layout()
output_image_file = 'correlation_of_returns.png'

plt.savefig(output_image_file)
print(f"Chart successfully saved as '{output_image_file}'")