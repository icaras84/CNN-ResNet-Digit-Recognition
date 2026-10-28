import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch


def test_modules() -> None:
    """
    Run this method to validate that all external libraries are installed
    """

    # test both pandas and numpy through running them
    test_data = {'x': np.linspace(-2, 2, 100)}
    # points along a parabola
    test_data['py'] = test_data['x']**2
    test_data['vy'] = 2 * test_data['x']

    test_data_frame = pd.DataFrame(test_data)
    print(test_data_frame)

    # test both seaborn and matplotlib
    sns.relplot(x='x', y='py', data=test_data_frame, kind='line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Parabola')

    plt.show()

    # test pytorch
    x = torch.rand(5, 3)
    print(x)


def main() -> None:
    test_modules()


if __name__ == "__main__":
    main()
