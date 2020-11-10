import timeit, functools

import numpy as _np
import pyximport; pyximport.install(setup_args={'include_dirs': _np.get_include()})

from ts_dist import dtw_dist as dtw_dist_py
from ts_dist import lcss_dist as lcss_dist_py
from ts_dist import edr_dist as edr_dist_py

from ts_dist_cy import dtw_dist as dtw_dist_cy
from ts_dist_cy import lcss_dist as lcss_dist_cy
from ts_dist_cy import edr_dist as edr_dist_cy

x = _np.random.normal(0, 1, (1000))
y = _np.random.normal(0, 1, (1000))

def benchmark(f, *args, **kwargs):
    t = timeit.Timer(functools.partial(f, *args, **kwargs)) 
    if f.__module__ == "ts_dist":
        fname = f.__name__ + "_py"
    elif f.__module__ == "ts_dist_cy":
        fname = f.__name__ + "_cy"
    else:
        raise ValueError("Invalid function.")
         
    print("Average time taken for {}: ".format(fname)+"{}".format(t.timeit(10)/10))

benchmark(dtw_dist_py, x, y)
benchmark(dtw_dist_cy, x, y)
benchmark(lcss_dist_py, x, y, delta=_np.inf, epsilon=0.5)
benchmark(lcss_dist_cy, x, y, delta=_np.inf, epsilon=0.5)
benchmark(edr_dist_py, x, y, epsilon=0.5)
benchmark(edr_dist_cy, x, y, epsilon=0.5)