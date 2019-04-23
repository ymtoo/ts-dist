# ts-dist
The module contains Python implementations of distance measures between two real time series feature vectors. Faster Cython implementations are also provided.

## Distance measures
The following distance measures have been implemented.
* Dynamic Time Warping (DTW) [1]
* Longest Common Subsequence (LCSS) [2]
* Edit Distance on Real sequence (EDR) [3]

## Example
```python
import numpy as np
import pyximport; pyximport.install()

from dtw_dist import dtw_dist as dtw_dist_py
from ts_dist import lcss_dist as lcss_dist_py
from ts_dist import edr_dist as edr_dist_py

from dtw_dist_cy import dtw_dist as dtw_dist_cy
from ts_dist_cy import lcss_dist as lcss_dist_cy
from ts_dist_cy import edr_dist as edr_dist_cy

x = np.random.normal(0, 1, (1000))
y = np.random.normal(0, 1, (1000))

dtw_py = dtw_dist_py(x, y)
dtw_cy = dtw_dist_cy(x, y)

lcss_py = lcss_dist_py(x, y)
lcss_cy = lcss_dist_cy(x, y)

edr_py = edr_dist_py(x, y)
edr_cy = edr_dist_cy(x, y)
```

## Dependencies
* Python 3
* Cython
* Numpy
* Numba

## References
[1] https://www.cs.unm.edu/~mueen/DTW.pdf

[2] M. Vlachos, G. Kollios and D. Gunopulos, "Discovering Similar Multidimensional Trajectories", 2002.

[3] L Chen, MT Özsu, V Oria, "Robust and fast similarity search for moving object trajectories", 2005.