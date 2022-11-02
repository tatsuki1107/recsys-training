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
    
    def userItemMatrix() -> np.array:
        dataset = np.array([
              [np.nan, 4,      3,      1,      2,      np.nan],
              [5,      5,      4,      np.nan, 3,      3     ],
              [4,      np.nan, 5,      3,      2,      np.nan],
              [np.nan, 3,      np.nan, 2,      1,      1     ],
              [2,      1,      2,      4,      np.nan, 3     ],
        ])
        
        return dataset
    
    def train_rating() -> list:
        
        dataset = [
    { "user_id": 1, "item_id": 2, "rating": 4},
    { "user_id": 1, "item_id": 3, "rating": 3},
    { "user_id": 1, "item_id": 4, "rating": 1},
    { "user_id": 1, "item_id": 5, "rating": 2},
    { "user_id": 1, "item_id": 7, "rating": 4},
    { "user_id": 2, "item_id": 1, "rating": 5},
    { "user_id": 2, "item_id": 2, "rating": 5},
    { "user_id": 2, "item_id": 3, "rating": 4},
    { "user_id": 2, "item_id": 5, "rating": 3},
    { "user_id": 2, "item_id": 6, "rating": 3},
    { "user_id": 2, "item_id": 7, "rating": 5},
    { "user_id": 2, "item_id": 8, "rating": 4},
    { "user_id": 3, "item_id": 1, "rating": 4},
    { "user_id": 3, "item_id": 3, "rating": 5},
    { "user_id": 3, "item_id": 4, "rating": 3},
    { "user_id": 3, "item_id": 5, "rating": 2},
    { "user_id": 3, "item_id": 8, "rating": 3},
    { "user_id": 4, "item_id": 2, "rating": 3},
    { "user_id": 4, "item_id": 4, "rating": 2},
    { "user_id": 4, "item_id": 5, "rating": 1},
    { "user_id": 4, "item_id": 6, "rating": 1},
    { "user_id": 4, "item_id": 7, "rating": 3},
    { "user_id": 5, "item_id": 1, "rating": 2},
    { "user_id": 5, "item_id": 2, "rating": 1},
    { "user_id": 5, "item_id": 3, "rating": 2},
    { "user_id": 5, "item_id": 4, "rating": 4},
    { "user_id": 5, "item_id": 6, "rating": 3},
    { "user_id": 5, "item_id": 7, "rating": 2},
]
        return dataset
    
    def test_rating() -> list:
        
        # 内容ブーステッド協調フィルタリングの予測値を仮のテストデータとする。
        dataset = [
            { "user_id": 1, "item_id": 1, "rating": 3},
            { "user_id": 1, "item_id": 6, "rating": 2},
            { "user_id": 1, "item_id": 8, "rating": 3},
            { "user_id": 1, "item_id": 9, "rating": 4},
            { "user_id": 2, "item_id": 4, "rating": 3},
            { "user_id": 2, "item_id": 9, "rating": 4},
            { "user_id": 3, "item_id": 2, "rating": 4},
            { "user_id": 3, "item_id": 6, "rating": 2},
            { "user_id": 3, "item_id": 7, "rating": 3},
            { "user_id": 3, "item_id": 9, "rating": 3},
            { "user_id": 4, "item_id": 1, "rating": 3},
            { "user_id": 4, "item_id": 3, "rating": 3},
            { "user_id": 4, "item_id": 8, "rating": 3},
            { "user_id": 4, "item_id": 9, "rating": 3},
            { "user_id": 5, "item_id": 5, "rating": 5},
            { "user_id": 5, "item_id": 8, "rating": 1},
            { "user_id": 5, "item_id": 9, "rating": 1},
        ]
        
        return dataset
    
    def item() -> list:
        
        dataset = [
    {"item_id": 1, "type": "赤身"},
    {"item_id": 2, "type": "赤身"},
    {"item_id": 3, "type": "赤身"},
    {"item_id": 4, "type": "白身"},
    {"item_id": 5, "type": "白身"},
    {"item_id": 6, "type": "白身"},
    {"item_id": 7, "type": "光物"},
    {"item_id": 8, "type": "光物"},
    {"item_id": 9, "type": "光物"},
]
        return dataset