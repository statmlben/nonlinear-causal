## test for elanet.py
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.datasets import make_regression

n, d = 1000, 10
X, y, beta_true = make_regression(n, d, coef=True)
X = np.hstack((X, X[:,:1]))
LD_X = np.dot(X.T, X)

from nonlinear_causal import _2SCausal
cov = np.sum(X*y[:,None], axis=0)
elasnet = _2SCausal.elasticSUM(lam=10.)
elasnet.fit(LD_X, cov)

print(elasnet.beta)
print(beta_true)

## test SCAD and MCP
import numpy as np
from scipy import stats
from sklearn.datasets import make_regression
import pycasso
n, d = 1000, 10
X, y, beta_true = make_regression(n, d, coef=True)
X = np.hstack((X, X[:,:1]))
s = pycasso.Solver(X, y, penalty='scad', lambdas=10**np.arange(-3,3,.1))
s.train()
s.plot()

## Test Haoran's issue
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nonlinear_causal import _2SCausal
from sklearn.preprocessing import power_transform, quantile_transform

n, p = 2000, 10
# theta0 = np.random.randn(p)
p_value = []
n_sim = 1
for i in range(n_sim):
	theta0, beta0 = np.ones(p), 1.0
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	Z, X, y, phi = sim(n, p, theta0, beta0, case='linear', feat='normal', range=.01)
	## normalize Z, X, y
	center = StandardScaler(with_std=False)
	mean_X, mean_y = X.mean(), y.mean()
	Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
	y_scale = y.std()
	y = y / y_scale
	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.5, random_state=42)
	n1, n2 = len(Z1), len(Z2)
	# LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
	LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
	LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
	# np.cov( np.dot(Z, theta0), X )
	# print('True beta: %.3f' %beta0)

	LD_pop = np.diag(np.ones(p))*n2
	## solve by 2sls
	LS = _2SCausal._2SLS(reg='scad')
	## Stage-1 fit theta
	LS.fit_theta(LD_Z1, cor_ZX1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cor_ZY2, n2=n2)
	# LS.fit_beta(LD_pop, cor_ZY2)
	## generate CI for beta
	# LS.test_effect(n2, LD_Z2, cor_ZY2, if_abs=False)
	LS.test_effect(n2, LD_pop, cor_ZY2, if_abs=False)
	p_value.append(LS.p_value)
p_value = np.array(p_value)
print('Rejection: 2sls: %.3f' %(len(p_value[p_value<.05])/n_sim))

## Test invalid IVs
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nonlinear_causal import _2SCausal
from sklearn.preprocessing import power_transform, quantile_transform

n, p = 2000, 10
# theta0 = np.random.randn(p)
p_value = []
n_sim = 1
for i in range(n_sim):
	theta0, beta0 = np.ones(p), 1.0
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	Z, X, y, phi = sim(n, p, theta0, beta0, case='linear', feat='normal', range=.01)
	## normalize Z, X, y
	center = StandardScaler(with_std=False)
	mean_X, mean_y = X.mean(), y.mean()
	Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
	y_scale = y.std()
	y = y / y_scale
	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.5, random_state=42)
	n1, n2 = len(Z1), len(Z2)
	# LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
	LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
	LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
	# np.cov( np.dot(Z, theta0), X )
	# print('True beta: %.3f' %beta0)

	LD_pop = np.diag(np.ones(p))*n2
	## solve by 2sls
	LS = _2SCausal._2SLS(reg='scad')
	## Stage-1 fit theta
	LS.fit_theta(LD_Z1, cor_ZX1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cor_ZY2, n2=n2)
	# LS.fit_beta(LD_pop, cor_ZY2)
	## generate CI for beta
	# LS.test_effect(n2, LD_Z2, cor_ZY2, if_abs=False)
	LS.test_effect(n2, LD_pop, cor_ZY2, if_abs=False)
	p_value.append(LS.p_value)
p_value = np.array(p_value)
print('Rejection: 2sls: %.3f' %(len(p_value[p_value<.05])/n_sim))