"""
EDA placeholder file header...
"""

import pandas as pd
import numpy as np


def parse_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def main():
    data = parse_dataset("./data/csv/train.csv")
    print(data.head())

    first_row = data.loc[0][data.columns != 'label']
    num_matrix = np.array(first_row).reshape(28, 28)
    print(num_matrix)


if __name__ == "__main__":
    main()
