## simulate two-stage dataset

# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

n, p = 2000, 10

beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
mse_air, mse_mean, ue_air, ue_mean = [], [], [], []
n_sim = 100
for i in range(n_sim):
	# theta0 = np.random.randn(p)
	theta0 = np.ones(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	Z, X, y, phi = sim(n, p, theta0, beta0, case='linear', feat='normal', range=1., return_phi=True)
	a, b = np.quantile(X, 0.1), np.quantile(X, 0.9)
	IoR = np.arange(a, b, (b-a)/100)
	# link function for IoR
	phi_ior = IoR - np.mean(phi)
	# phi_ior = np.sign(IoR)*(abs(IoR)**(1./3)) - np.mean(phi)
	# phi_ior = np.log( IoR / (1 - IoR) ) - np.mean(phi)
	# link function for training data
	phi = phi - np.mean(phi)
	y_scale = y.std()
	y = y / y_scale
	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.5, random_state=42)
	n1, n2 = len(Z1), len(Z2)
	# LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
	LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
	LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
	# np.cov( np.dot(Z, theta0), X )
	print('True beta: %.3f' %beta0)

	## solve by SIR+LS
	from nonlinear_causal import _2SCausal
	echo = _2SCausal._2SIR(sparse_reg=None)		
	echo.cond_mean = KernelRidge(kernel='rbf', alpha=.001, gamma=.1)
	# echo.cond_mean = KNeighborsRegressor(n_neighbors=100)
	## stage-1: fit sir
	echo.fit_sir(Z1, X1)
	## stage-2: fit regression
	echo.fit_reg(LD_Z2, cor_ZY2)
	## fit link function
	echo.fit_air(Z1, X1)

	# echo.fit(Z, X, cor_ZY)
	print('est beta based on 2SIR: %.3f' %(echo.beta*y_scale))
	pred_phi = echo.link(X=X[:,None]).flatten()
	pred_mean = echo.cond_mean.predict(X=X[:,None]).flatten()

	pred_ior = echo.link(X=IoR[:,None]).flatten()
	pred_mean_ior = echo.cond_mean.predict(X=IoR[:,None]).flatten()
	# beta_LS.append(abs(LS.beta))
	# beta_RT_LS.append(abs(RT_LS.beta))
	# beta_LS_SIR.append(abs(echo.beta[0]))
	mse_air.append( np.mean( (pred_phi - phi)**2 ) )  
	mse_mean.append( np.mean( (pred_mean - phi)**2 ) ) 
	ue_air.append( np.max(abs(pred_ior - phi_ior)) )
	ue_mean.append( np.max(abs(pred_mean_ior - phi_ior)) )

print('MSE: AIR: %.3f(%.3f); mean: %.3f(%.3f)' 
	%( np.mean(mse_air), np.std(mse_air)/np.sqrt(n_sim), np.mean(mse_mean), np.std(mse_mean)/np.sqrt(n_sim) ))
print('UE: AIR: %.3f(%.3f); mean: %.3f(%.3f)' 
	%( np.mean(ue_air), np.std(ue_air)/np.sqrt(n_sim), np.mean(ue_mean), np.std(ue_mean)/np.sqrt(n_sim) ))

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)
sns.set_theme(style="whitegrid")
d = {'x': list(X)*3, 'phi':list(pred_phi)+list(phi)+list(pred_mean), 
		'type':['pred']*n+['true']*n+['cond_mean']*n}
# d = {'x': list(IoR)*3, 'phi':list(pred_ior)+list(phi_ior)+list(pred_mean_ior), 
# 		'type':['pred']*100+['true']*100+['cond_mean']*100}
sns.scatterplot(data=d, x="x", y="phi", hue="type", s=10, alpha=.5)
plt.show()