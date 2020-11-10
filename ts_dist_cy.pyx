"""Time series distance measures (Cython implementation)."""
cimport cython
from cpython cimport bool
from libc.math cimport log
from libc.stdio cimport printf
import numpy as _np
cimport numpy as _np

from utils import check_arrays, standardization

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
cdef double _sum_log_absolute(double[:] x, double[:] y, int n_feature): 
    cdef int i
    cdef double res = 0
    for i in range(n_feature):
        res += log(abs(x[i]-y[i])) 
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _dtw_dist(double[:, :] X, double[:, :] Y, w): 
    """Compute the DTW distance between X and Y using Dynamic Programming.
    
    :param X (2-D array): time series feature array denoted by X
    :param Y (2-D array): time series feature array denoted by Y
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

def dtw_dist(_np.ndarray X, _np.ndarray Y, w=_np.inf, mode="dependent"):
    """Compute multidimensional DTW distance between X and Y using Dynamic Programming.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :param w (int): window size (default=Inf)
    :param mode (string): "dependent" or "independent" (default="dependent")
    :returns: distance between X and Y with the best alignment
    :Reference: https://www.cs.unm.edu/~mueen/DTW.pdf
    """
    X, Y = check_arrays(X, Y)
    if mode == "dependent":
        dist = _dtw_dist(X, Y, w)
    elif mode == "independent":
        n_feature = X.shape[0]
        dist = 0
        for i in range(n_feature):
            dist += _dtw_dist(X[[i], :], Y[[i], :], w)
    else:
        raise ValueError("The mode must be either \"dependent\" or \"independent\".")
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool abs_diff_less(double[:] x, double[:] y, int n_feature, double epsilon): 
    cdef int i
    cdef bool res = True
    for i in range(n_feature):
        if abs(x[i]-y[i]) > epsilon:
            res = False
            break
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _lcss_dist(double[:, :] X, double[:, :] Y, delta, epsilon):
    """Compute the LCSS distance between X and Y using Dynamic Programming.

    :param X (2-D array): time series feature array denoted by X
    :param Y (2-D array): time series feature array denoted by Y
    :param delta (int): time sample matching threshold
    :param epsilon (float): amplitude matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: M Vlachos et al., "Discovering Similar Multidimensional Trajectories", 2002.
    """
    cdef int i, j
    cdef int n_feature = X.shape[0] 
    cdef int n_frame_X = X.shape[1] 
    cdef int n_frame_Y = Y.shape[1]

    cdef double[:, :] S = _np.zeros((n_frame_X+1, n_frame_Y+1))
    for i in range(1, n_frame_X+1):
        for j in range(1, n_frame_Y+1):
            if abs_diff_less(X[:, i-1], Y[:, j-1], n_feature, epsilon) and (
                abs(i-j) < delta):
                S[i, j] = S[i-1, j-1]+1
            else:
                S[i, j] = max(S[i, j-1], S[i-1, j])
    return 1-S[n_frame_X, n_frame_Y]/min(n_frame_X, n_frame_Y)

def lcss_dist(X, Y, delta, epsilon):
    """Compute the LCSS distance between X and Y using Dynamic Programming.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :param delta (int): time sample matching threshold
    :param epsilon (float): amplitude matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: M Vlachos et al., "Discovering Similar Multidimensional Trajectories", 2002
    """
    X, Y = check_arrays(X, Y)
    dist = _lcss_dist(X, Y, delta, epsilon)
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] initialize_D(int n_frame_X, int n_frame_Y):
    cdef double[:, :] D = _np.full((n_frame_X+1, n_frame_Y+1), _np.inf)
    for i in range(n_frame_X+1):
        D[i, 0] = i
    for j in range(n_frame_Y+1):
        D[0, j] = j
    return D

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _edr_dist(double[:, :] X, double[:, :] Y, epsilon):
    """Compute the EDR distance between X and Y using Dynamic Programming.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :epsilon (float): matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: L. Chen et al., "Robust and Fast Similarity Search for Moving Object Trajectories", 2005.
    """
    cdef int i, j
    cdef int n_feature = X.shape[0] 
    cdef int n_frame_X = X.shape[1]
    cdef int n_frame_Y = Y.shape[1]

    cdef double[:, :] D = initialize_D(n_frame_X, n_frame_Y)
    for i in range(1, n_frame_X+1):
        for j in range(1, n_frame_Y+1):
            if abs_diff_less(X[:, i-1], Y[:, j-1], n_feature, epsilon):
                subcost = 0.0
            else:
                subcost = 1.0
            D[i, j] = min(D[i-1, j-1]+subcost, D[i-1, j]+1, D[i, j-1]+1)
    return D[n_frame_X, n_frame_Y]/max(n_frame_X, n_frame_Y)

def edr_dist(X, Y, epsilon):
    """Compute the EDR distance between X and Y using Dynamic Programming.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :epsilon (float): matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: L. Chen et al., "Robust and Fast Similarity Search for Moving Object Trajectories", 2005.
    """
    X, Y = check_arrays(X, Y)
    X = standardization(X)
    Y = standardization(Y)
    dist = _edr_dist(X, Y, epsilon)
    return dist