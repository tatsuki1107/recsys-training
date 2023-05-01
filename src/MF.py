from dataclasses import dataclass
import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from typing import Tuple, List


@dataclass
class MF:
    """MF class for MF model."""
    n_users: int
    n_items: int
    n_factors: int
    lr: float
    reg: float
    n_epochs: int
    random_state: int = 12345
    rating_range: tuple = (0.5, 5)

    def __post_init__(self):
        """Initialize user and item factors."""
        np.random.seed(self.random_state)
        self.random_ = check_random_state(self.random_state)

        self.P = np.random.normal(
            size=(self.n_users, self.n_factors),
            scale=1/np.sqrt(self.n_factors)
        )
        self.Q = np.random.normal(
            size=(self.n_items, self.n_factors),
            scale=1/np.sqrt(self.n_factors)
        )

    def fit(
        self,
        train: np.ndarray,
        test: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Fit model to training data."""
        train_loss, test_loss = [], []
        for _ in tqdm(range(self.n_epochs)):
            self.random_.shuffle(train)
            for user, item, rating in train:
                user, item = int(user), int(item)
                error = rating - self._predict_pair(user, item)
                grad_P = -2*error * self.Q[item] + self.reg * self.P[user]
                grad_Q = -2*error * self.P[user] + self.reg * self.Q[item]
                self._update_P(user, grad_P)
                self._update_Q(item, grad_Q)

            train_y_hat = self.predict(train[:, :2])
            train_rmse = mean_squared_error(
                train[:, 2], train_y_hat, squared=False
            )
            train_loss.append(train_rmse)

            test_y_hat = self.predict(test[:, :2])
            test_rmse = mean_squared_error(
                test[:, 2], test_y_hat, squared=False
            )
            test_loss.append(test_rmse)

        return train_loss, test_loss

    def _predict_pair(self, u: int, i: int) -> float:
        """Predict rating of user u for item i."""
        return self.P[u].T @ self.Q[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for user/item pairs."""
        return np.array([self._predict_pair(int(u), int(i)) for u, i in X])

    def _update_P(self, user: int, grad: np.ndarray):
        """Update user factors."""
        self.P[user] -= self.lr * grad

    def _update_Q(self, item: int, grad: np.ndarray):
        """Update item factors."""
        self.Q[item] -= self.lr * grad
