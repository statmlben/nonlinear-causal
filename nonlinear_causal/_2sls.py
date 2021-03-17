import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sliced import SlicedInverseRegression

class _2SLS(object):
	"""Two-stage least square

	Parameters
	----------

	"""
	def __init__(self, theta=None, beta=None, normalize=True):
		self.theta = theta
		self.beta = beta
		self.normalize = normalize

	def fit(self, LD_Z, cor_ZX, cor_ZY):
		self.theta = np.dot(np.linalg.inv( LD_Z ), cor_ZX)
		if self.normalize == True:
			self.theta = normalize(self.theta.reshape(1, -1))[0]
		LD_X = self.theta.T.dot(LD_Z).dot(self.theta)
		if LD_X.ndim == 0:
			self.beta = self.theta.T.dot(cor_ZY) / LD_X
		else:
			self.beta = np.linalg.inv(LD_X).dot(self.theta.T).dot(cor_ZY)


class SIR_LS(object):
	"""Sliced inverse regression + least sqaure

	Parameters
	----------

	"""
	def __init__(self, theta=None, beta=None, sir=None, n_directions=1):
		self.theta = theta
		self.beta = beta
		self.n_directions = n_directions
		self.sir = None

	def fit(self, Z, X, cor_ZY):
		self.sir = SlicedInverseRegression(n_directions=self.n_directions)
		self.sir.fit(Z, X)
		self.theta = self.sir.directions_
		X_sir = self.sir.transform(Z)
		LD_X_sir = np.dot(X_sir.T, X_sir)
		if LD_X_sir.ndim == 0:
			self.beta = np.dot(self.theta, cor_ZY) / LD_X_sir
		else:
			self.beta = np.linalg.inv(LD_X_sir).dot(np.dot(self.theta, cor_ZY))
		


