"""Dynamic time warping (DTW) distance."""
cimport cython
import numpy as _np
cimport numpy as _np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _sum_absolute(double[:] x_real, double[:] x_imag, double[:] y_real, double[:] y_imag, int n_feature):
    cdef int i
    cdef double res = 0
    for i in range(n_feature):
        res += ((x_real[i]-y_real[i])**2+(x_imag[i]-y_imag[i])**2)**(0.5)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _dtw_dist(double[:, :] X_real, double[:, :] X_imag, double[:, :] Y_real, double[:, :] Y_imag, w):
    """Compute pairwise distance between two time series feature vectors based 
    on multidimensional DTW with dependent warping.
    
    :param X (2-D array): time series feature vector denoted by X
    :param Y (2-D array): time series feature vector denoted by Y
    :param w (int): window size 
    :returns: distance between X and Y with the best alignment
    :Reference: https://www.cs.unm.edu/~mueen/DTW.pdf
    """
    cdef int i, j
    cdef int n_feature = X_real.shape[0]
    cdef int n_frame_X = X_real.shape[1]
    cdef int n_frame_Y = Y_real.shape[1]  
    cdef int n_frame_X_Y = n_frame_X-n_frame_Y

    cdef double[:, :] D = _np.full((n_frame_X+1, n_frame_Y+1), _np.inf)
    D[0, 0] = 0
    w = max(w, abs(n_frame_X_Y))
    for i in range(1, n_frame_X+1):
        for j in range(max(1, i-w), min(n_frame_Y+1, i+w+1)):
            cost = _sum_absolute(X_real[:, i-1], X_imag[:, i-1], Y_real[:, j-1], Y_imag[:, j-1], n_feature)
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
    X = _np.array(X, dtype=_np.complex128)
    Y = _np.array(Y, dtype=_np.complex128)
    X_real = X.real
    X_imag = X.imag
    Y_real = Y.real
    Y_imag = Y.imag
    if X.ndim == 1:
        X_real = _np.reshape(X_real, (1, X_real.size))
        X_imag = _np.reshape(X_imag, (1, X_imag.size))
    if Y.ndim == 1:
        Y_real = _np.reshape(Y_real, (1, Y_real.size))
        Y_imag = _np.reshape(Y_imag, (1, Y_imag.size))
    if mode == 'dependent':
        dist = _dtw_dist(X_real, X_imag, Y_real, Y_imag, w)
    elif mode == 'independent':
        n_feature = X.shape[0]
        dist = 0
        for i in range(n_feature):
            dist += _dtw_dist(X_real[[i], :], X_imag[[i], :], Y_real[[i], :], Y_imag[[i], :], w)
    else:
        raise ValueError('The mode must be either \'dependent\' or \'independent\'.')
    return dist