"""
EDA placeholder file header...
"""

import pandas as pd
import numpy as np


def parse_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def main():
    data = parse_dataset("/data/csv/dataset.csv")


if __name__ == "__main__":
    main()