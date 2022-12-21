import numpy as np
import pandas as pd
import random

random.seed(1234)

def user_feedback(df, policy, pow_true) -> pd.DataFrame:
    # 人気順orランダム推薦でランキングしてみる（毎月のバッチ処理と仮定）
    # 数値の高い順にすると、ログが同じになるので、少しランダムに
    sorted_df = df.copy()
    if policy == "populality":
        sorted_df = sorted_df[sorted_df["populality"] >= 3.8]
        sorted_df.reset_index(drop=True, inplace=True)
    elif policy == "feature":
        sorted_df = sorted_df[sorted_df["feature"] >= 3.8]
        sorted_df.reset_index(drop=True, inplace=True)
    position = list(range(1,8))
    item_dict = {
        "position": [], "item_id": [], 
        "clicked": [], "iter": []
    }
    
    if pow_true != None:
        position_bias = lambda k: pow(0.9/k, pow_true)
        theta = np.array([position_bias(k) for k in range(1,8)])
    else:
        theta = np.array([1.0]*7)
    
    
    
    for i in range(50):
        recommend_items = random.sample(list(sorted_df.index.values), k=7)
        df = sorted_df.loc[recommend_items,:]
        
        rel_max = 5
        rel = np.array(0.1 + 0.9 * pow(2, df["feature"].values-1)/pow(2, rel_max-1))
        
        click_p = rel*theta
        click_true = np.random.binomial(n=1,p=click_p)
        
        item_dict["position"].extend(position)
        item_dict["item_id"].extend(df["item_id"].values)
        item_dict["clicked"].extend(click_true)
        item_dict["iter"].extend(len(position)*[i+1])
    
    log_df = pd.DataFrame(data=item_dict)
    log_df = log_df.astype({"clicked": np.int64})
    
    count_dict = log_df[log_df["clicked"]==1].groupby("item_id").agg({"clicked": "count"})["clicked"].to_dict()
    history_dict = {"item_id": list(count_dict.keys()), "click count": list(count_dict.values())}
    user_df = pd.DataFrame(data=history_dict)
    
    print(f"クリック率: {log_df[log_df['clicked']==1]['clicked'].count()/350}")
    
    return log_df, user_df