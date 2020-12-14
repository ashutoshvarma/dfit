from functools import wraps
from time import time

# test module, fitting using MLE
import dfit
from scipy import stats

COUNT = 50

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%r args: %r took: %2.4f sec' % (f.__name__, args, (te-ts)/COUNT))
        return result
    return wrap

def repeat(count):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            rets = []
            for _ in range(count):
                rets.append(f(*args, **kwargs))
            return rets
        return wrapper
    return decorator

@timing
@repeat(count=COUNT)
def run_dfit(size):
    data = stats.gamma.rvs(2, loc=1.5, scale=2, size=int(size))
    f = dfit.DFit(data, distributions=["gamma", "cauchy", "beta"])
    f.fit()

run_dfit(2e4)
run_dfit(4e4)
run_dfit(8e4)
run_dfit(16e4)
# test(8e4)
# test(16e4)
# test(32e4)
# test(64e4)