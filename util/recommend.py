import pandas as pd
import numpy as np


def generate_rec_list(
    dataset: np.ndarray,
    pred_matrix: np.ndarray,
    top_k: int = 5
) -> np.ndarray:
    """
    Args:
        dataset (np.ndarray): 学習,テストデータセット
        pred_matrix (np.ndarray): 評価予測行列
        top_k (int): 推薦件数

    Returns:
        np.ndarray: 各ユーザーの推薦リスト
    """
    pivot = pd.DataFrame(dataset).pivot(index=0, columns=1, values=2)
    history_matrix = pivot.values.astype(float)
    rec_candidate = np.where(np.isnan(history_matrix), pred_matrix.T, np.nan)
    rec_list = np.argsort(-rec_candidate, axis=1, kind="heapsort")[:, :top_k]

    return rec_list
