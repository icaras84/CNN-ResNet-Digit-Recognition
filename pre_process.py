"""
Michael Kong
William Phomphakdee
CSE 163
This program implements various functions for preprocessing and visualizing
image and label data sourced from a given CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from numpy.typing import NDArray
import os

TRAIN_DATA = 'data/csv/train.csv'


def plt_save(filename: str) -> None:
    """
    A method for saving a matplotlib figure to the test_result directory.
    :param filename: valid file path
    :return: None
    """
    if not os.path.exists('./data/test_result/'):
        os.makedirs('./data/test_result/')
    plt.savefig(f'./data/test_result/{filename}')


def csv_to_df(path: str, detail: bool = False) -> pd.DataFrame:
    """
    Given a file path to a CSV of training data returns a DataFrame with
    the shape and NaN information if detail flag is True.
    """
    df = pd.read_csv(path)
    if detail:
        print(f'(rows, columns): {df.shape}')
        print(f'NaN values: {df.isnull().sum().sum()}')
    return df


def plot_label_counts(data: pd.DataFrame) -> None:
    """
    Given a DataFrame of labels and image data plots the distribution
    of each digit found in the dataset as a barplot.
    """
    count_plot = sns.catplot(data=data, x='label', kind='count')
    count_plot.fig.set_size_inches(8, 6)
    uniform = len(data)/10
    count_plot.ax.axhline(y=uniform, linestyle='--', linewidth=1,
                          label='Expected Count (Balanced)')
    count_plot.ax.set_ylim(0, 5000)
    count_plot.fig.suptitle('Distribution of Digit Labels')
    count_plot.set_axis_labels('Digit', 'Number of Images')
    count_plot.ax.legend()
    plt_save("label_counts.png")
    plt.show()


def extract_images(data: pd.DataFrame) -> tuple:
    """
    Given a DataFrame of labels and image data with dimension n by 785 parses
    the labels into a Numpy array of length n and images into a n by 28 by 28
    array. Returns the resulting arrays as a tuple.
    """
    images = data.loc[:, 'pixel0':'pixel783'].values.reshape(-1, 28, 28)
    labels = data.loc[:, 'label'].values
    return images, labels


def convert_to_tensors(images: NDArray,
                       labels: NDArray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given an array of images and array of corresponding labels, converts each
    into tensors. The image tensor has an additional dimension for a single
    channel representing black to white.
    """
    images = images.astype(np.float32) / 255.0
    images = images[:, np.newaxis, :, :]
    images = torch.from_numpy(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def display_images(images: NDArray, labels: NDArray,
                   indices: list[int]) -> None:
    """
    Given numpy arrays of image data and corresponding labels, displays the
    labelled images for the given list of indices side-by-side in a grid.
    """
    count = len(indices)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(0.8*cols, 0.8*rows))
    for ax, i in zip(axs.flat, indices):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    for ax in axs.flat[count:]:
        ax.axis('off')
    plt.tight_layout()
    plt_save("digit_samples.png")
    plt.show()


def sample_indices(images: NDArray, labels: NDArray,
                   seed: int | None = None) -> list[int]:
    """
    Given numpy arrays of image data and corresponding labels, returns a list
    of indices for 10 random images for each digit 0-9 from the dataset.
    Integer seed can be provided to fix random output, otherwise defaults to
    random rng.
    """
    rand = np.random.default_rng(seed)
    sample_indices = []
    for digit in range(10):
        indices = np.where(labels == digit)[0]
        sample_indices.extend(rand.choice(indices, size=10, replace=False))
    return sample_indices


def main() -> None:
    df = csv_to_df(TRAIN_DATA, True)
    images, labels = extract_images(df)
    im_ten, lab_ten = convert_to_tensors(images, labels)
    plot_label_counts(df)
    display_images(images, labels, sample_indices(images, labels, 40))


if __name__ == "__main__":
    main()
