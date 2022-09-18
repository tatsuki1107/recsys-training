import numpy as np

class ItemCollaborativeRecommender():
    def recommend(self, R, K_ITEMS, THETA) -> np.array:
        U = np.arange(R.shape[0])
        I = np.arange(R.shape[1])
        Ui = [U[~np.isnan(R)[:,i]] for i in I]
        Iu = [I[~np.isnan(R)[u,:]] for u in U]
        ru_mean = np.nanmean(R, axis=1)
        R2 = R - ru_mean.reshape((ru_mean.size, 1))
        
        # 調整コサイン類似度のアイテムxアイテム類似度行列計算
        S = np.zeros((I.size, I.size))
        for i in I:
            for j in I:
                Uij = np.intersect1d(Ui[i], Ui[j])
                num = np.sum([R2[u,i]*R2[u,j] for u in Uij])
                den_i = np.sqrt(np.sum([R2[u,i]**2 for u in Uij]))
                den_j = np.sqrt(np.sum([R2[u,j]**2 for u in Uij]))
                # 調整コサイン類似度
                cosine = round(num / (den_i * den_j),3)
                S[i,j] = cosine
        
        # 類似アイテムの選定
        Ii = {i: {j: S[i,j] for j in I if i != j} for i in I}
        # 上位K件抽出
        Ii = {i: dict(sorted(Ii[i].items(), key=lambda x:x[1], reverse=True)[:K_ITEMS]) for i in I}
        # 閾値以上のkeyを抽出
        Ii = {i: np.array([k for k, j in Ii[i].items() if j > THETA]) for i in I}
        
        # 嗜好予測（予測評価値行列をreturn）
        R3 = R.copy()
        for u in U:
            for i in I:
                if ~np.isnan(R3[u,i]):
                    continue
                Iiu = np.intersect1d(Ii[i], Iu[u])
                rui_pred = round((ru_mean[u] + np.sum([S[i,j]*R2[u,j] for j in Iiu]) / np.sum(np.abs([S[i,j] for j in Iiu]))),3)
                R3[u,i] = rui_pred
        
        return R3
        