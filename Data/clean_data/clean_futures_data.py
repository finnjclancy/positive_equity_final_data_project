import pandas as pd

df = pd.read_csv('../data/merged_futures_data.csv')
cleaned_df = df[[

    'Timestamp',
    'FESX_Open',
    'FESX_High',
    'FESX_Low',
    'FESX_Close',
    'FDXM_Open',
    'FDXM_High',
    'FDXM_Low',
    'FDXM_Close',
    'FESX_Vol',
    'FDXM_Vol'

]].copy()
cleaned_df.columns = [
    'timestamp',
    'FESX_open',
    'FESX_high',
    'FESX_low',
    'FESX_close',
    'FDXM_open',
    'FDXM_high',
    'FDXM_low',
    'FDXM_close',
    'FESX_vol',
    'FDXM_vol'

]
cleaned_df.to_csv('../data/cleaned_futures_data.csv', index=False)
print(f"Original merged data shape: {df.shape}")
print(f"Cleaned data shape: {cleaned_df.shape}")
print("\nFirst few rows of cleaned data:")
print(cleaned_df.head())