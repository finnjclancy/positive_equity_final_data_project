Positive Equity Data Analysis Project.

June 10 min data from FDXM and FESX taken from Stellar.

Data was initially pasted into Excel, further analysis and data perparation was done using the python scripts in this project.

The Excel sheet containing the copy and pasted data can be found here - https://posequity-my.sharepoint.com/:x:/g/personal/fclancy_positive_ie/EZuvq7Zt2_tCthJFA1fXAgYB2lUnTN5hp6zEnFjOy8NBAQ?e=HX2F35 

The 'Data' directory contains data and python scripts used to clean the data.
    - The 'clean_data' directory contains the raw data that had been cleaned in Excel.
    - The 'split_data' directory contains data from higher vol months, April 22nd - June 19th.
        - It also has the data split into 3 equal length files to do correlation analysis over different timeframes.

The 'Output' directory contains content that was printed to the terminal and graphs produced after running the scripts in the 'Scripts' folder.
    - The 'Part_1_Data_and_Spread_Charts' directory has graphs showing the time series graphs of the FDXM, FESX and the spread.
    - The 'Part_2_Correlation' directory has the correlation over time and a table containing the correlation over the 3 equally split time periods 
    - The 'Part_3_Spread_Analysis has backtests for the Breakout and Mean Reversion strategies
        - For both strategies there are results for the following:
            - Entering when price deviates 40 points from a moving average and then closing when price crosses the moving average
            - Entering when price deviates 40 points from a moving average and then closing when price crosses the moving average and closing after a max time
            - Entering when price deviates 40 points from a moving average and only closing after a certain amount of time has passed

The 'Scripts' directory contains python files that did data analysis
    - There are 4 sub directories:
        - Breakout_Strategy:
            - This folder has the scripts that backtest different versions of the breakout strategy i.e. longing when price deviates 40 ticks from a MA, shorting when price is 40 ticks below a MA
        - Correlation:
            - These scripts do correlation analysis on the data to create tables and graphs
        - Mean_Reversion:
            - This folder has the scripts that backtest different versions of the Mean Reversion strategy i.e. shorting when price deviates 40 ticks from a MA, longing when price is 40 ticks below a MA
        - Price_Charts:
            - This folder creates charts for the price data of FDXM, FESX and the spread. 
    

