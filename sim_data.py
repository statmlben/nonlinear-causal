import numpy as np
from scipy.special import lambertw
from sklearn.preprocessing import OneHotEncoder
from random import choices

def sim(n, p, theta0, beta0, case='linear', feat='normal', prob=.3):
	if feat == 'normal':
		Z = np.random.randn(n, p)
	elif feat == 'cate':
		Z = np.random.choice(2, size=(n, p), p=[1-prob, prob]) + np.random.choice(2, size=(n, p), p=[1-prob, prob])
	else:
		print('Wrong feature distribution!')
	# normalize the feature
	U = np.random.randn(n)
	eps = np.random.randn(n)
	gamma = np.random.randn(n)

	# simulate X and Y
	if case == 'linear':
		X = np.dot(Z, theta0) + U**2 + eps
		y = beta0 * X + U + gamma

	elif case == 'exp':
		X = np.exp( np.dot(Z, theta0) + U**2 + eps )
		y = beta0 * np.log(X) + U + gamma

	elif case == 'cubic':
		X = (np.dot(Z, theta0) + U**2 + eps)**3
		y = beta0*np.sign(X)*(abs(X)**(1./3)) + U + gamma

	elif case == 'inverse':
		X = .1 / np.dot(Z, theta0) + U**2 + eps
		y = beta0 / X + U + gamma
	else:
		print('Wrong case!')
	return Z, X, y
