import numpy as np
import random

def base_recommend(df):
    
    profile = round(df[df["clicked"]==1]["feature"].mean(),2)
    test_df = df[df["clicked"]==-1]
    candidate_df = test_df[test_df["feature"]>=profile].reset_index(drop=True)
    #rec_list = []
    #for i in range(50):
        #index = random.sample(list(candidate_df.index.values), k=5)
        #rec_i = candidate_df.iloc[index].sort_values('feature', ascending=False)["item_id"].to_list()
        
        #rec_list.append(rec_i)
    rec_5 = candidate_df.sort_values("feature", ascending=False)["item_id"].to_list()[:5]
    rec_list = [rec_5]*50
    
    return rec_list, profile