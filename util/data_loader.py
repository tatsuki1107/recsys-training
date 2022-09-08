import os
import numpy as np

class DataLoader:
    def load() -> np.array:
        dataset = np.array([
               [5, 3, +1],
               [6, 2, +1],
               [4, 1, +1],
               [8, 5, -1],
               [2, 4, -1],
               [3, 6, -1],
               [7, 6, -1],
               [4, 2, np.nan],
               [5, 1, np.nan],
               [8, 6, np.nan],
               [3, 4, np.nan],
               [4, 7, np.nan],
               [4, 4, np.nan],
        ])
        
        return dataset