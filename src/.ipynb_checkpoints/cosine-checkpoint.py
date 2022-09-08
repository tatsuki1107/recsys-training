import numpy as np

class CosineRecommender():
    def recommend(self, dataset, k) -> dict():
        I = np.arange(dataset.shape[0])
        Iup = I[dataset[:,-1]==+1]
        x = dataset[:,:-1]
        Iu_not = np.setdiff1d(I, I[~np.isnan(dataset[:,-1])])
        positive = x[Iup, :]
        profile = self._createProfile(positive)
        cosineDesc = self._cosDesc(profile, x, Iu_not)
        rec_list = self._order(cosineDesc, k)
        
        return rec_list
        
    def _createProfile(self, posi) -> list:
        pu = []
        for i in np.arange(posi.shape[1]):
            a = posi[:, i]
            sum_posi = np.sum(a)
            pu.append(sum_posi / posi.shape[0])
        
        return pu
    
    def _cosDesc(self, pu, x, Ino) -> dict():
        den_u = np.linalg.norm(pu)
        scores = {}
        for i in Ino:
            num = pu@x[i, :]
            den_i = np.linalg.norm(x[i,:])
            cosine = num / (den_u * den_i)
            scores.update({i: cosine})
        sorted_scores = sorted(scores.items(), key=lambda i: i[1], reverse=True)
        
        return sorted_scores
    
    def _order(self, cosDesc, k) -> dict():
        rec_list = dict(cosDesc[:k])
        return rec_list
        
        