import numpy as np
from scipy.special import lambertw
from sklearn.preprocessing import OneHotEncoder
from random import choices

def sim(n, p, theta0, beta0, alpha0=0., case='linear', feat='normal', range=1., prob=.3, return_phi=False):
	if feat == 'normal':
		Z = np.random.randn(n, p)
	elif feat == 'uniform':
		Z = np.random.uniform(low=-range, high=range, size=(n,p))
	elif feat == 'cate':
		Z = np.random.choice(2, size=(n, p), p=[1-prob, prob]) + np.random.choice(2, size=(n, p), p=[1-prob, prob])
	else:
		print('Wrong feature distribution!')
	# normalize the feature
	U = np.random.randn(n)
	eps = np.random.randn(n)
	gamma = np.random.randn(n)

	alpha0 = alpha0*np.ones(p)

	# simulate X and Y
	if case == 'linear':
		X = np.dot(Z, theta0) + U + eps
		phi = X
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'exp':
		X = np.exp( np.dot(Z, theta0) + U**2 + eps )
		phi = np.log(X)
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'cubic':
		X = (np.dot(Z, theta0) + U**2 + eps)**3
		phi = np.sign(X)*(abs(X)**(1./3))
		y = beta0*phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse':
		X = 1. / (np.dot(Z, theta0) + U**2 + eps)
		phi = 1. / X
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma
	
	elif case == 'sigmoid':
		X = 1 / (1 + np.exp( - np.dot(Z, theta0) - U**2 - eps ))
		phi = np.log( X / (1 - X) )
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	else:
		print('Wrong case!')
	if return_phi:
		return Z, X, y, phi	
	else:
		return Z, X, y

def center(X):
	p = X.ndim
	if p == 1:
		X -= np.mean(X)
	for j in range(p):
		X[:,j] -= np.mean(X[:,j])
	return X
		
