import numpy as np
from scipy.special import lambertw

def sim(n, p, theta0, beta0, case='1'):
	if case == '1':
		Z = np.random.randn(n, p)
		U = np.random.randn(n)
		eps = np.random.randn(n)
		gamma = np.random.randn(n)

		# simulate X and Y
		X = np.dot(Z, theta0) + U**2 + eps
		y = beta0 * X + U + gamma

	elif case == '2':
		Z = np.random.randn(n, p)
		U = np.random.randn(n)
		eps = np.random.randn(n)
		gamma = np.random.randn(n)

		# simulate X and Y
		X = np.exp( np.dot(Z, theta0) + U**2 + eps )
		y = beta0 * np.log(X) + U + gamma

	else:
		Z = np.random.randn(n, p)
		U = np.random.randn(n)
		eps = np.random.randn(n)
		gamma = np.random.randn(n)

		# simulate X and Y
		X = (np.dot(Z, theta0) + U**2 + eps)**3
		# X = ( np.dot(Z, theta0) + U**2 + eps ) ** 2
		y = beta0*np.sign(X)*(abs(X)**(1./3)) + U + gamma

	return Z, X, y
