import numpy as np

class KNNRecommender():
    def recommend(self, dataset, K_ITEMS, TOP_K, THETA) -> dict():
        I = np.arange(dataset.shape[0])
        Iu = I[~np.isnan(dataset[:,-1])]
        Iup = I[dataset[:,-1]==+1]
        Iun = I[dataset[:,-1]==-1]
        x = dataset[:,:-1]
        Iu_not = np.setdiff1d(I, Iu)
        
        D = self._createMatrix(Iu_not, Iu, x)
        near_dist = np.argsort(D[:,Iu])
        near_dist = near_dist[:, np.arange(K_ITEMS)]
        kItems = {Iu_not[i]: near_dist[i,:] for i in np.arange(len(Iu_not))}
        
        pred_score = {}
        for i in Iu_not:
            array = kItems[i]
            positive = array[np.isin(array, Iup)]
            negative = array[np.isin(array, Iun)]
            score = round(((positive.size - negative.size) / K_ITEMS), 3)
            pred_score[i] = score
        
        
        rec_list = self._order(pred_score, TOP_K, THETA)
        
        return rec_list
    
    def _createMatrix(self, Iu_not, Iu, x) -> np.array:
        
        matrix = np.zeros((Iu_not.size, Iu.size))
        for j in np.arange(len(Iu_not)):
            for i in np.arange(len(Iu)):
                x1 = x[Iu_not[j],0]-x[Iu[i],0]
                x2 = x[Iu_not[j],1]-x[Iu[i],1]
                matrix[j,i] = round(np.sqrt(np.sum((x1**2)+(x2**2))), 3)
        
        return matrix
    
        
        
    def _order(self, pred_score, k, THETA) -> dict():
        order_list = dict(sorted(pred_score.items(), key=lambda x:x[1], reverse=True)[:k])
        keys = [k for k, t in order_list.items() if t > THETA]
        rec_list = {keys[i]: order_list[keys[i]] for i in np.arange(len(keys))}
        
        return rec_list
        