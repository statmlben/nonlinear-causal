import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.datasets import make_regression

n, d = 1000, 10
X, y, beta_true = make_regression(n, d, coef=True)
LD_X = np.dot(X.T, X)

from nonlinear_causal import _2SCausal
cov = np.sum(X*y[:,None], axis=0)
elasnet = _2SCausal.elasticSUM(lam=.01)
elasnet.fit(LD_X, cov)

print(elasnet.beta)
print(beta_true)