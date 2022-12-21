import numpy as np
import random

def base_recommend(df):
    
    profile = round(df[df["clicked"]==1]["feature"].mean(),2)
    test_df = df[df["clicked"]==-1]
    candidate_df = test_df[test_df["feature"]>=profile].reset_index(drop=True)
    rec_list = []
    for i in range(50):
        index = random.sample(list(candidate_df.index.values), k=7)
        rec_i = candidate_df.iloc[index].sort_values('feature', ascending=False)["item_id"].to_list()
        
        rec_list.append(rec_i)
    
    return rec_list, profile