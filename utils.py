"""Utility functions."""
import numpy as _np

def check_arrays(X, Y):
	"""Set X and Y appropriately.
	
	:param X (array): time series feature array denoted by X
	:param Y (array): time series feature array denoted by Y
	:returns: X and Y in 2D numpy arrays
	"""
	X = _np.array(X, dtype=_np.float)
	Y = _np.array(Y, dtype=_np.float)
	if X.ndim == 1:
		X = _np.reshape(X, (1, X.size))
	if Y.ndim == 1:
		Y = _np.reshape(Y, (1, Y.size))
	return X, Y

def standardization(X):
	"""Transform X to have zero mean and one standard deviation."""
	return (X-X.mean(axis=1, keepdims=True))/X.std(axis=1, keepdims=True)