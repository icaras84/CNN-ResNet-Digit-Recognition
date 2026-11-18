"""
Michael Kong
CSE 163
This program implements tests for some functions in pre_process.py
"""

import numpy as np
import pandas as pd

import pre_process

TRAIN_DATA = 'data/csv/train.csv'


def test_csv_to_df():
    """
    Tests the csv_to_df function.
    """
    df = pre_process.csv_to_df(TRAIN_DATA)

    assert isinstance(df, pd.DataFrame)
    assert not len(df) == 0
    assert 'label' in df.columns


def test_extract_images():
    """
    Tests the extract_image function. Resulting arrays should be equal length
    and labels should only contain values 0-9.
    """
    df = pre_process.csv_to_df(TRAIN_DATA)
    images, labels = pre_process.extract_images(df)

    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == len(labels)
    assert (np.all((labels >= 0) & (labels < 10)))


def test_sample_indices():
    """
    Tests the sample_indices function. Should pick 10 (seeded) random images
    for each digit 0-9 from the data without duplicates.
    """
    df = pre_process.csv_to_df(TRAIN_DATA)
    images, labels = pre_process.extract_images(df)

    indices = pre_process.sample_indices(images, labels, 5)
    sampled_labels = labels[indices]

    assert len(indices) == 100
    assert len(set(indices)) == 100

    for digit in range(10):
        assert np.sum(sampled_labels == digit) == 10


def main() -> None:
    test_csv_to_df()
    test_extract_images()
    test_sample_indices()
    print('All tests passed!')


if __name__ == "__main__":
    main()
