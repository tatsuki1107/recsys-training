import numpy as np
import pandas as pd
import random
from feedBack import user_feedback
from model import base_recommend
from evaluate import evalMap

random.seed(1234)

class train:
    def __init__(self, 
                 train_policy: str, 
                 test_policy: str, 
                 train_pow_true: float = None,
                 test_pow_true: float = None
                ) -> None:
        self.feature = []
        self.item_id = []
        self.populality = []
        
        self.train_policy = train_policy
        self.test_policy = test_policy
        self.train_pow_true = train_pow_true
        self.test_pow_true = test_pow_true

    def make_dataframe(self) -> pd.DataFrame:
        item_dict = {"item_id": self.item_id, "feature": self.feature, "populality": self.populality}
        df = pd.DataFrame(data=item_dict)
        
        return df
    
    def devide_df(self, df) -> list:
        devided_list = []
        for i in range(50):
            clicked_items = df[(df["iter"]==i) & (df["clicked"]==1)]["item_id"].to_list()
            devided_list.append(clicked_items)
    
        return devided_list
        
    def ranking_train(self) -> float:
    
        for i in range(300):
            self.feature.append(round(random.uniform(1,5),2))
            self.populality.append(round(random.uniform(1,5),2))
            self.item_id.append(str(i+1))
        
            if i == 209:
                df_train = self.make_dataframe()
                self.feature, self.item_id, self.populality = [], [], []
        
        df_test = self.make_dataframe()
        
        train_log_df, train_user_df = user_feedback(df_train, self.train_policy, self.train_pow_true)
        test_log_df, test_user_df = user_feedback(df_test, self.test_policy, self.test_pow_true)
        
        for_recommend_df = pd.merge(train_log_df, df_train, on="item_id")
        for_recommend_df = pd.concat([for_recommend_df, df_test], sort=True)[["item_id", "feature", "clicked"]]
        # テストデータのクリックを-1とする
        for_recommend_df = for_recommend_df.fillna({"clicked": -1.0})
        for_recommend_df = for_recommend_df.astype({"clicked": np.int64})
        
        rec_list, profile = base_recommend(for_recommend_df)
        
        y_true = self.devide_df(test_log_df)
        y_pred = rec_list
        
        map_7 = evalMap().mapk(y_true, y_pred)
        
        return map_7, profile
        