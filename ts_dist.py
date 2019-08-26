"""Time series distance measures."""
from numba import jit as _jit
import numpy as _np

from utils import check_arrays, standardization

@_jit(nopython=True)
def _dtw_dist(X, Y, w):
    """Compute the DTW distance between X and Y using Dynamic Programming.

    :param X (2-D array): time series feature array denoted by X
    :param Y (2-D array): time series feature array denoted by Y
    :param w (int): window size 
    :returns: distance between X and Y with the best alignment
    :Reference: https://www.cs.unm.edu/~mueen/DTW.pdf
    """
    n_frame_X, n_frame_Y = X.shape[1], Y.shape[1]
    D = _np.full((n_frame_X+1, n_frame_Y+1), _np.inf)
    D[0, 0]= 0
    w = max(w, abs(n_frame_X-n_frame_Y))
    for i in range(1, n_frame_X+1):
        X_vec = X[:, i-1]
        for j in range(max(1, i-w), min(n_frame_Y+1, i+w+1)):
            diff = _np.abs(X_vec-Y[:, j-1])
            cost = _np.sum(diff)
            D[i, j] = cost+min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return D[n_frame_X, n_frame_Y]

def dtw_dist(X, Y, w=_np.inf, mode="dependent"):
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

@_jit(nopython=True)
def _lcss_dist(X, Y, delta, epsilon):
    """Compute the LCSS distance between X and Y using Dynamic Programming.

    :param X (2-D array): time series feature array denoted by X
    :param Y (2-D array): time series feature array denoted by Y
    :param delta (int): time sample matching threshold
    :param epsilon (float): amplitude matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: M Vlachos et al., "Discovering Similar Multidimensional Trajectories", 2002.
    """
    n_frame_X, n_frame_Y = X.shape[1], Y.shape[1]
    S = _np.zeros((n_frame_X+1, n_frame_Y+1))
    for i in range(1, n_frame_X+1):
        for j in range(1, n_frame_Y+1):
            if _np.all(_np.abs(X[:, i-1]-Y[:, j-1]) < epsilon) and (
                _np.abs(i-j) < delta):
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

@_jit(nopython=True)
def _edr_dist(X, Y, epsilon):
    """Compute the EDR distance between X and Y using Dynamic Programming.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :param epsilon (float): matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: L. Chen et al., "Robust and Fast Similarity Search for Moving Object Trajectories", 2005.
    """
    n_frame_X, n_frame_Y = X.shape[1], Y.shape[1]
    D = _np.full((n_frame_X+1, n_frame_Y+1), _np.inf)
    D[:, 0] = _np.arange(n_frame_X+1)
    D[0, :] = _np.arange(n_frame_Y+1)
    for i in range(1, n_frame_X+1):
        for j in range(1, n_frame_Y+1):
            if _np.all(_np.abs(X[:, i-1]-Y[:, j-1]) < epsilon):
                subcost = 0
            else:
                subcost = 1
            D[i, j] = min(D[i-1, j-1]+subcost, D[i-1, j]+1, D[i, j-1]+1)
    return D[n_frame_X, n_frame_Y]/max(n_frame_X, n_frame_Y)

def edr_dist(X, Y, epsilon):
    """Compute the EDR distance between X and Y using Dynamic Programming.

    :param X (array): time series feature array denoted by X
    :param Y (array): time series feature array denoted by Y
    :param epsilon (float): matching threshold
    :returns: distance between X and Y with the best alignment
    :Reference: L. Chen et al., "Robust and Fast Similarity Search for Moving Object Trajectories", 2005.
    """
    X, Y = check_arrays(X, Y)
    X = standardization(X)
    Y = standardization(Y)
    dist = _edr_dist(X, Y, epsilon)
    return dist
