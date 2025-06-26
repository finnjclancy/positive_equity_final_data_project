import pandas as pd

fesx_df = pd.read_csv('../data/FESX_10_min_data.csv', skiprows=1)
fdxm_df = pd.read_csv('../data/FDXM_10_min_data.csv', skiprows=1)

fesx_df['Timestamp'] = pd.to_datetime(fesx_df['Timestamp'])
fdxm_df['Timestamp'] = pd.to_datetime(fdxm_df['Timestamp'])
fesx_df.set_index('Timestamp', inplace=True)
fdxm_df.set_index('Timestamp', inplace=True)
fesx_df = fesx_df.add_prefix('FESX_')
fdxm_df = fdxm_df.add_prefix('FDXM_')
merged_df = pd.merge(fesx_df, fdxm_df, left_index=True, right_index=True)

merged_df.reset_index(inplace=True)
merged_df.to_csv('../data/merged_futures_data.csv', index=False)
print(f"Total rows in FESX data: {len(fesx_df)}")
print(f"Total rows in FDXM data: {len(fdxm_df)}")
print(f"Total rows in merged data (matching timestamps): {len(merged_df)}")