from functools import wraps
from time import time


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time()
        result = func(*args, **kargs)
        elapsed_time = time() - start
        print(f'{func.__name__}の処理時間:　{elapsed_time:.2f}秒')
        return result
    
    return wrapper