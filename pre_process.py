import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DATA = 'data/csv/train.csv'


def csv_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_label_counts(data: pd.DataFrame) -> None:
    count_plot = sns.catplot(data=data, x='label', kind='count')
    uniform = len(data)/10
    count_plot.ax.axhline(y=uniform, linestyle='--', linewidth=1,
                          label='Expected Count (Uniform)')
    count_plot.ax.set_ylim(3000, 5000)
    count_plot.fig.suptitle('Distribution of Digit Labels')
    count_plot.set_axis_labels('Digit', 'Number of Images')
    count_plot.ax.legend()
    plt.show()


def main() -> None:
    df = csv_to_df(TRAIN_DATA)
    plot_label_counts(df)


if __name__ == "__main__":
    main()