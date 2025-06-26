import pandas as pd
from datetime import datetime

# Read the merged futures data
df = pd.read_csv('../clean_data/cleaned_futures_data.csv')

# Convert timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter data between May 20th and June 20th
start_date = datetime(2025, 4, 20)
end_date = datetime(2025, 6, 20)

filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

# Save the filtered data to a new CSV file
output_file = '../clean_data/april_may_june_futures_data.csv'
filtered_df.to_csv(output_file, index=False)

print(f"Data filtered and saved to {output_file}")
print(f"Number of records in filtered data: {len(filtered_df)}") 