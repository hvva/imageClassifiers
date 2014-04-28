from PIL import Image
from numpy import *

def pca(X):
	""" Principal Component Analysis
	  input: X, matrix with training data stored as flattened arrays in rows
	  return: projection matrix, variance and mean
	"""

	# det dimensions
	num_data,dim = X.shape

	# center data
	mean_X = X.mean(axis=0)
	X = X - mean_X

	if dim>num_data:
	  # PCA - compact trick used
	  M = dot(X,X.T)
	  e,EV = linalg.eigh(M)
	  tmp = dot(X.T,EV).T
	  V = tmp[::-1]
	  S = sqrt(e)[::-1]
	  for i in range(V.shape[1]):
	    V[:,i] /= S
	else:
	  # PCA - SVD used
	  U,S,V = linalg.svd(X)
	  V = V[:num_data]

	# return projection matrix, variance and mean
	return V,S,mean_X
