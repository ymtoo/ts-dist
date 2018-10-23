# dtw-dist
This is a Python implementation of distance measure between two time series feature vectors based on multidimensional Dynamic Time Warping (DTW). A faster Cython implementation is also provided.

## Specifications
The following specifications of the distance measure are implemented.
* Exact and locality constrained
* Dependent and independent warping for the multidimensional case

## Usage
* To import the Python module
```python
from dtw_dist import dtw_dist as dtw_dist_py
```
* To import the Cython module
```python
import pyximport; pyximport.install()
from dtw_dist_cy import dtw_dist as dtw_dist_cy
```

## Example
```python
import numpy as np

from dtw_dist import dtw_dist as dtw_dist_py
import pyximport; pyximport.install()
from dtw_dist_cy import dtw_dist as dtw_dist_cy

x = np.random.normal(0, 1, (1000))
y = np.random.normal(0, 1, (1000))
dist_py = dtw_dist_py(x, y)
dist_cy = dtw_dist_cy(x, y)
```

## Dependencies
* Python 3
* Cython
* Numpy
* Numba

## References
[1] https://www.cs.unm.edu/~mueen/DTW.pdf