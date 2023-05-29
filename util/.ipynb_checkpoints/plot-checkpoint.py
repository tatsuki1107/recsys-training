import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()


def plot_train_curve(
    train_loss: list,
    test_loss: list,
):
    plt.subplots(1, figsize=(8, 6))
    plt.plot(
        np.arange(len(train_loss)),
        train_loss,
        label=f"Train RMSE (last: {train_loss[-1]:.2f})",
        linewidth=3
    )
    plt.plot(
        np.arange(len(test_loss)),
        test_loss,
        label=f"Test RMSE, (last: {test_loss[-1]:.2f})",
        linewidth=3
    )

    plt.title("Train/Test Curves", fontdict=dict(size=20))
    plt.xlabel("Number of Epochs", fontdict=dict(size=20))
    plt.ylabel("Root Mean Squared Error", fontdict=dict(size=20))
    plt.tight_layout()
    plt.legend(loc="best", fontsize=20)
    plt.show()


def plot_heatmap(matrix: np.ndarray) -> None:
    """評価値matrixをヒートマップで可視化
    args:
          matrix: 評価値行列
    """
    fig, ax = plt.subplots(figsize=(20, 5))

    my_cmap = plt.cm.get_cmap('Reds')
    heatmap = plt.pcolormesh(matrix.T, cmap=my_cmap)
    plt.colorbar(heatmap)
    ax.grid()
    plt.tight_layout()
    plt.show()


def plot_item_rank(df: pd.DataFrame) -> None:
    """アイテムのランキングを可視化"""

    item_count = df.groupby("movie_id")["movie_id"].count(
    ).sort_values(ascending=False).values
    item_rank = [r for r in range(len(item_count))]

    sns.scatterplot(x=item_rank, y=item_count)


def plot_rec_count(rec_list: np.ndarray) -> None:
    """レコメンドされたアイテムをカウントして可視化"""
    rec_count = {}
    for movie_index in np.unique(rec_list):
        rec_count[movie_index] = np.count_nonzero(rec_list == movie_index)

    plt.scatter([i for i in range((len(rec_count)))],
                list(rec_count.values()),
                color="blue"
                )
    plt.ylabel("recommend count", fontdict=dict(size=20))
    plt.title(
        f'recommend count ({len(rec_count)} unique items)',
        fontdict=dict(size=20)
    )
