import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from numpy.typing import NDArray

TRAIN_DATA = 'data/csv/train.csv'


def csv_to_df(path: str, detail: bool = False) -> pd.DataFrame:
    """
    Given a file path to a CSV of training data returns a DataFrame with
    shape and NaN information if detail flag is True.
    """
    df = pd.read_csv(path)
    print(f'(rows, columns): {df.shape}')
    print(f'NaN values: {df.isnull().sum().sum()}')
    return df


def plot_label_counts(data: pd.DataFrame) -> None:
    """
    Given a DataFrame of labels and image data plots the distribution
    of each digit found in the dataset.
    """
    count_plot = sns.catplot(data=data, x='label', kind='count')
    uniform = len(data)/10
    count_plot.ax.axhline(y=uniform, linestyle='--', linewidth=1,
                          label='Expected Count (Balanced)')
    count_plot.ax.set_ylim(0, 5000)
    count_plot.fig.suptitle('Distribution of Digit Labels')
    count_plot.set_axis_labels('Digit', 'Number of Images')
    count_plot.ax.legend()
    plt.show()


def extract_images(data: pd.DataFrame) -> tuple[NDArray, NDArray]:
    images = data.loc[:, 'pixel0':'pixel783'].values.reshape(-1, 28, 28)
    labels = data.loc[:, 'label'].values
    return images, labels


def convert_to_tensors(images, labels):
    images = images.astype(np.float32) / 255.0
    images = images[:, np.newaxis, :, :]
    images = torch.from_numpy(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def display_images(images, labels, indices):
    count = len(indices)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    for ax, i in zip(axs.flat, indices):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    for ax in axs.flat[count:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = csv_to_df(TRAIN_DATA, True)
    # plot_label_counts(df)
    images, labels = extract_images(df)
    print(images.shape, labels.shape)
    # display_images(images, labels, list(range(0, 100, 10)))
    im_ten, lab_ten = convert_to_tensors(images, labels)


if __name__ == "__main__":
    main()
