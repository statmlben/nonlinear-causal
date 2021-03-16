## simulate two-stage dataset

# sim data
import numpy as np
n, p = 1000, 2
Z = np.random.randn(n, p)
U = np.random.randn(n)
eps = np.random.randn(n)
gamma = np.random.randn(n)

# simulate X and Y
theta0 = np.array([1,1])
theta0 = theta0 / np.sqrt(np.sum(theta0**2))
beta0 = 1.
X = np.exp( np.dot(Z, theta0) + U**2 + eps )
y = beta0 * np.log(X) + U + gamma

print('True beta: %.3f' %beta0)
## solve by 2SLS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
# theta_LS = np.dot(np.linalg.inv( np.dot(Z.T, Z) ), np.dot(Z.T, X))
reg = LinearRegression(fit_intercept=False).fit(Z, X)
theta_LS = normalize(reg.coef_.reshape(1, -1))[0]
# theta_LS = theta_LS / np.sqrt(np.sum(theta_LS**2))
X_hat = np.dot(Z, theta_LS)
reg2 = LinearRegression(fit_intercept=False).fit(X_hat.reshape(-1, 1), y)
beta_LS = reg2.coef_
print('est beta based on 2SLS: %.3f' %beta_LS)

## solve by SIR+LS
from sliced import SlicedInverseRegression
sir = SlicedInverseRegression(n_directions=1)
sir.fit(Z, X)
X_sir = sir.transform(Z)
reg2 = LinearRegression(fit_intercept=False).fit(X_sir, y)
beta_sir = reg2.coef_
print('est beta based on SIR+LS: %.3f' %beta_sir)
