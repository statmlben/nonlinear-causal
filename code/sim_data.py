import numpy as np
from scipy.special import lambertw
from sklearn.preprocessing import OneHotEncoder
from random import choices

def sim(n, p, theta0, beta0, alpha0=0., xi0=0., IoR=None, case='linear', feat='normal', effect='indiv', u_range=1., prob=.3, dominant_factor=.3, return_phi=False):
	if feat == 'normal':
		Z = np.random.randn(n, p)
	elif feat == 'AP-normal':
		cov = np.zeros((p,p))
		for i in range(p):
			for j in range(p):
				cov[i,j] = (.5)**abs(i-j)
		Z = np.random.multivariate_normal(np.zeros(p), cov, n)
	elif feat == 'laplace':
		Z = np.random.laplace(size = (n, p))
	elif feat == 'uniform':
		Z = np.random.uniform(low=-u_range, high=u_range, size=(n,p))
	elif feat in ['cate', 'dominant_cate']:
		Z = np.random.choice(2, size=(n, p), p=[1-prob, prob]) + np.random.choice(2, size=(n, p), p=[1-prob, prob])
	else:
		print('Wrong feature distribution!')
	# normalize the feature
	U = np.random.randn(n)
	eps = np.random.randn(n)
	gamma = np.random.randn(n)
	alpha0 = alpha0*np.ones(p)
	Z_obs = Z.copy()
	## add dominant effect
	if feat == 'dominant_cate':
		Z[Z==2] = 1 + dominant_factor
	if effect == 'epistasis':
		p_epi = len(xi0)
		xi0 = xi0*np.ones(p_epi)
		Z_inter = Z[:,np.random.randint(p, size=p_epi)] * Z[:,np.random.randint(p, size=p_epi)]
		epi_effect = np.dot(Z_inter, xi0)
	else:
		epi_effect = 0.
	# simulate X and Y
	if case == 'linear':
		X = np.dot(Z, theta0) + epi_effect + U**2 + eps
		phi = X
		if IoR is not None:
			phi_ior = IoR
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'quad':
		right = np.dot(Z, theta0) + epi_effect + U**2 + eps + np.e
		X = np.sign(np.random.randn(len(Z))) * np.sqrt(np.abs(right))
		phi = X**2
		if IoR is not None:
			phi_ior = IoR**2
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'log':
		X = np.exp( np.dot(Z, theta0) + epi_effect + U**2 + eps )
		phi = np.log(X)
		if IoR is not None:
			phi_ior = np.log(IoR)
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'cube-root':
		X = (np.dot(Z, theta0) + epi_effect + U**2 + eps)**3
		phi = np.sign(X)*(abs(X)**(1./3))
		if IoR is not None:
			phi_ior = np.sign(IoR)*(abs(IoR)**(1./3))
		y = beta0*phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse':
		X = 1. / (np.dot(Z, theta0) + epi_effect + U**2 + eps)
		phi = 1. / X
		if IoR is not None:
			phi_ior = 1. / IoR
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma
	
	elif case == 'sigmoid':
		X = 1 / (1 + np.exp( - np.dot(Z, theta0) - epi_effect - U**2 - eps ))
		phi = np.log( X / (1 - X) )
		if IoR is not None:
			phi_ior = np.log( IoR / (1 - IoR) )
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

	elif case == 'piecewise_linear':
		tmp = np.dot(Z, theta0) + epi_effect + U**2 + eps
		X = 1.*(tmp<=0.)*tmp + 2.*tmp*(tmp>0.)
		phi = 1.*X*(X<=0) + .5*X*(X>0)
		if IoR is not None:
			phi_ior = 1.*(IoR<=0.)*IoR + 2*IoR*(IoR>0.)
		y = beta0 * phi + np.dot(Z, alpha0) + U + gamma
	
	elif case == 'quad+exp':
		right = np.dot(Z, theta0) + epi_effect + U**2 + eps + np.e
		X = np.sign(np.random.randn(len(Z))) * np.sqrt(np.abs(right))
		phi = X**2
		if IoR is not None:
			phi_ior = IoR**2
		y = beta0 * np.exp(X) + np.dot(Z, alpha0) + U + gamma

	elif case == 'quad+linear':
		right = np.dot(Z, theta0) + epi_effect + U**2 + eps + np.e
		X = np.sign(np.random.randn(len(Z))) * np.sqrt(np.abs(right))
		phi = X**2
		if IoR is not None:
			phi_ior = IoR**2
		y = beta0 * X + np.dot(Z, alpha0) + U + gamma

	elif case == 'quad+log':
		right = np.dot(Z, theta0) + epi_effect + U**2 + eps + np.e
		X = np.sign(np.random.randn(len(Z))) * np.sqrt(np.abs(right))
		phi = X**2
		if IoR is not None:
			phi_ior = IoR**2
		y = beta0 * np.log(np.abs(X)) + np.dot(Z, alpha0) + U + gamma

	elif case == 'quad+inverse':
		right = np.dot(Z, theta0) + U**2 + epi_effect + eps + np.e
		X = np.sign(np.random.randn(len(Z))) * np.sqrt(np.abs(right))
		phi = X**2
		if IoR is not None:
			phi_ior = IoR**2
		y = beta0 * 1 / X + np.dot(Z, alpha0) + U + gamma

	elif case == 'quad+abs':
		right = np.dot(Z, theta0) + epi_effect + U**2 + eps + np.e
		X = np.sign(np.random.randn(len(Z))) * np.sqrt(np.abs(right))
		phi = X**2
		if IoR is not None:
			phi_ior = IoR**2
		y = beta0 * np.abs(X) + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse+linear':
		X = 1. / (np.dot(Z, theta0) + epi_effect + U**2 + eps)
		phi = 1. / X
		if IoR is not None:
			phi_ior = 1. / IoR
		y = beta0 * X + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse+exp':
		X = 1. / (np.dot(Z, theta0) + epi_effect + U**2 + eps)
		phi = 1. / X
		Z = Z[X<10]
		U = U[X<10]
		gamma = gamma[X<10]
		X = X[X<10]
		if IoR is not None:
			phi_ior = 1. / IoR
		y = beta0 * np.exp(X) + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse+quad':
		X = 1. / (np.dot(Z, theta0) + epi_effect + U**2 + eps)
		phi = 1. / X
		if IoR is not None:
			phi_ior = 1. / IoR
		y = beta0 * X**2 + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse+abs':
		X = 1. / (np.dot(Z, theta0) + epi_effect + U**2 + eps)
		phi = 1. / X
		if IoR is not None:
			phi_ior = 1. / IoR
		y = beta0 * np.abs(X) + np.dot(Z, alpha0) + U + gamma

	elif case == 'inverse+log':
		X = 1. / (np.dot(Z, theta0) + epi_effect + U**2 + eps)
		phi = 1. / X
		if IoR is not None:
			phi_ior = 1. / IoR
		y = beta0 * np.log(np.abs(X)) + np.dot(Z, alpha0) + U + gamma

	else:
		raise NameError('Sorry, no build-in case.')
	
	if IoR is None:
		return Z_obs, X, y, phi
	
	else:
		return Z_obs, X, y, phi, phi_ior - np.mean(phi)

def sim_phi(X, case='linear'):
	if case == 'linear':
		return X
	elif case == 'log':
		return np.log(X)
	elif case == 'cube-root':
		return np.sign(X)*(abs(X)**(1./3))
	elif case == 'inverse':
		return 1. / X
	elif case == 'sigmoid':
		return np.log( X / (1 - X) )
	elif case == 'piecewise_linear':
		return 1.*X*(X<=0) + .5*X*(X>0)
	elif case == 'quad':
		return X**2
	else:
		raise NameError('Sorry, no build-in case.')

def center(X):
	p = X.ndim
	if p == 1:
		X -= np.mean(X)
	for j in range(p):
		X[:,j] -= np.mean(X[:,j])
	return X
		
