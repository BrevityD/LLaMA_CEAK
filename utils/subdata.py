"""Filter and subset electrolyte data based on LCE (Lithium Coulombic Efficiency) values.

This script:
- Reads raw electrolyte experiment data from CSV
- Filters to only include entries with LCE > 1
- Saves filtered subset to new CSV file

Rationale:
Molecules with LCE <= 1 are considered potentially ineffective or unsuitable 
for further analysis in battery electrolyte applications.
"""
import pandas as pd  # type: ignore
from typing import NoReturn

def main() -> NoReturn:
    """Main function to filter and save electrolyte data."""
    origin_file = "data/ceak_experiments_hzx.csv"
    # origin_file = "data/ceak_datasets.csv"
    target_file = "data/ceak_experiments_hzx_sub.csv" 
    # target_file = "data/ceak_datasets_sub.csv"

    df: pd.DataFrame = pd.read_csv(origin_file)
    filtered_df: pd.DataFrame = df[df['lce'] > 1]
    filtered_df.to_csv(target_file, index=False)

if __name__ == "__main__":
    main()
