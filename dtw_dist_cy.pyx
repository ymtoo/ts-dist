"""Dynamic time warping (DTW) distance."""
cimport cython
import numpy as _np
cimport numpy as _np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _sum_absolute(double[:] x, double[:] y, int n_feature): 
    cdef int i
    cdef double res = 0
    for i in range(n_feature):
        res += abs(x[i]-y[i]) 
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _dtw_dist(double[:, :] X, double[:, :] Y, w): 
    """Compute pairwise distance between two time series feature vectors based 
    on multidimensional DTW with dependent warping.
    
    :param X (2-D array): time series feature vector denoted by X
    :param Y (2-D array): time series feature vector denoted by Y
    :param w (int): window size 
    :returns: distance between X and Y with the best alignment
    :Reference: https://www.cs.unm.edu/~mueen/DTW.pdf
    """
    cdef int i, j
    cdef int n_feature = X.shape[0] 
    cdef int n_frame_X = X.shape[1] 
    cdef int n_frame_Y = Y.shape[1] 
    cdef int n_frame_X_Y = n_frame_X-n_frame_Y

    cdef double[:, :] D = _np.full((n_frame_X+1, n_frame_Y+1), _np.inf)
    D[0, 0] = 0
    w = max(w, abs(n_frame_X_Y))
    for i in range(1, n_frame_X+1):
        for j in range(max(1, i-w), min(n_frame_Y+1, i+w+1)):
            cost = _sum_absolute(X[:, i-1], Y[:, j-1], n_feature) 
            D[i, j] = cost+min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return D[n_frame_X, n_frame_Y]

def dtw_dist(_np.ndarray X, _np.ndarray Y, w=_np.inf, mode='dependent'):
    """Compute pairwise distance between two time series feature vectors based on multidimensional DTW.

    :param X (array): time series feature vector denoted by X
    :param Y (array): time series feature vector denoted by Y
    :param w (int): window size (default=Inf)
    :param mode (string): 'dependent' or 'independent' (default='dependent')
    :returns: distance between X and Y with the best alignment
    """
    X = _np.array(X, dtype=_np.float) 
    Y = _np.array(Y, dtype=_np.float) 
    if X.ndim == 1:
        X = _np.reshape(X, (1, X.size)) 
    if Y.ndim == 1:
        Y = _np.reshape(Y, (1, Y.size)) 
    if mode == 'dependent':
        dist = _dtw_dist(X, Y, w) 
    elif mode == 'independent':
        n_feature = X.shape[0]
        dist = 0
        for i in range(n_feature):
            dist += _dtw_dist(X[[i], :], Y[[i], :], w) 
    else:
        raise ValueError('The mode must be either \'dependent\' or \'independent\'.')
    return dist