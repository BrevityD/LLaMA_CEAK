"""This script filters molecules based on their LCE. 
Only molecules with an LCE greater than 1 are selected, as those with an LCE less than 
or equal to 1 are considered potentially ineffective or unsuitable for further analysis."""
import pandas as pd

origin_file = "data/ceak_experiments_hzx.csv"
# origin_file = "data/ceak_datasets.csv"
target_file = "data/ceak_experiments_hzx_sub.csv"
# target_file = "data/ceak_datasets_sub.csv"

df = pd.read_csv(origin_file)


filtered_df = df[df['lce']>1]

filtered_df.to_csv(target_file, index=False)
