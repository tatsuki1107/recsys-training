import numpy as np

class UserCollaborativeRecommender():
    def recommend(self, R, K_USERS, THETA) -> np.array:
        U = np.arange(R.shape[0])
        I = np.arange(R.shape[1])
        Ui = [U[~np.isnan(R)[:,i]] for i in I]
        Iu = [I[~np.isnan(R)[u,:]] for u in U]
        ru_mean = np.nanmean(R, axis=1)
        R2 = R - ru_mean.reshape((ru_mean.size, 1))
        
        S = np.zeros((U.size, U.size))
        for u in U:
            for v in U:
                Iuv = np.intersect1d(Iu[u], Iu[v])
                num = np.sum([R2[u,i]*R2[v,i] for i in Iuv])
                den_u = round(np.sqrt(np.sum([R2[u,i]**2 for i in Iuv])), 3)
                den_v = round(np.sqrt(np.sum([R2[v,i]**2 for i in Iuv])), 3)
                prsn = round(num / (den_u * den_v), 3)
                # user-user類似度行列
                S[u,v] = prsn
        
        # 類似ユーザK人選定
        Uu = {u: {v: S[u,v] for v in U if u!=v} for u in U}
        uu = {u: dict(sorted(Uu[u].items(), key=lambda i:i[1], reverse=True)[:K_USERS]) for u in U}
        # 類似度が閾値以上のユーザだけ抽出
        Uu = {u: np.array([k for k, t in Uu[u].items() if t > THETA]) for u in U}
        
        R3 = R.copy()
        for u in U:
            for i in I:
                if ~np.isnan(R3[u,i]):
                    continue
                Uui = np.intersect1d(Ui[i], Uu[u])
                if Uui.size <= 0:
                    R3[u,i] = ru_mean[u]
                    continue
                # 嗜好予測
                rui_pred = round(ru_mean[u] + np.sum([S[u,v]*R2[v,i] for v in Uui]) / np.sum(np.abs([S[u,j] for j in Uui])),3)
                R3[u,i] = rui_pred
        
        return R3