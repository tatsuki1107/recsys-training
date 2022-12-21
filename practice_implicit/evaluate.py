import numpy as np

class evalMap():
    # 各レコードの平均 precision
    # 1/min(m,K)*p

    def apk(self,y_i_true, y_i_pred):
        assert (len(y_i_pred) <= 7)
        assert (len(np.unique(y_i_pred)) == len(y_i_pred))
    
        sum_precision = 0.0
        num_hits = 0
    
        for i, p in enumerate(y_i_pred):
            if p in y_i_true:
                num_hits += 1
                precision = num_hits / (i+1)
                sum_precision += precision
        
        if sum_precision == 0.0:
            return 0.0
    
        else :
            return sum_precision / min(len(y_i_true), 7)

    #MAP@K計算用の関数
    def mapk(self, y_true, y_pred) -> float:
        ap = [self.apk(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)]
    
        return np.mean(ap)