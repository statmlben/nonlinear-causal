## simulate two-stage dataset

# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim, sim_phi
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import power_transform, quantile_transform
from sklearn.preprocessing import PowerTransformer
from sklearn.isotonic import IsotonicRegression
n, p = 2000, 300

beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
mse_air, mse_mean, ue_air, ue_mean = [], [], [], []
mse_RT_LS, mse_LS, ue_RT_LS, ue_LS = [], [], [], []

link_plot = { 'x': [], 'phi': [], 'method': [] }

# 'linear', 'log', 'cube-root', 'inverse', 'sigmoid', 
n_sim, case = 100, 'sigmoid'
for i in range(n_sim):
	# theta0 = np.random.randn(p)
	theta0 = np.ones(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	Z, X, y, phi = sim(n, p, theta0, beta0, case=case, feat='normal', range=1., return_phi=True)
	## generate true phi function
	if i == 0:
		a, b = np.quantile(X, 0.05), np.quantile(X, 0.95)
		IoR = np.arange(a, b, (b-a)/100)
		phi_ior_org = sim_phi(X=IoR, case=case)
	# link function for training data
	phi_ior = phi_ior_org - np.mean(phi)
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
	# echo.cond_mean = KernelRidge(kernel='rbf', alpha=.001, gamma=.1)
	# echo.cond_mean = KNeighborsRegressor(n_neighbors=100)
	echo.cond_mean = IsotonicRegression(increasing='auto',out_of_bounds='clip')
	## stage-1: fit sir
	echo.fit_sir(Z1, X1)
	## stage-2: fit regression
	echo.fit_reg(LD_Z2, cor_ZY2)
	## fit link function
	echo.fit_air(Z1, X1)

	# echo.fit(Z, X, cor_ZY)
	print('est beta based on 2SIR: %.3f' %(echo.beta*y_scale))
	pred_phi = echo.link(X=X[:,None]).flatten()
	pred_mean = echo.cond_mean.predict(X[:,None]).flatten()

	pred_ior = echo.link(X=IoR[:,None]).flatten()
	pred_mean_ior = echo.cond_mean.predict(IoR[:,None]).flatten()

	## solve by RT-2SLS
	pt = PowerTransformer()
	RT_X1 = pt.fit_transform(X1.reshape(-1,1)).flatten()
	# RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
	# RT_LS = _2SCausal._2SLS(sparse_reg=None)
	# ## Stage-1 fit theta
	# RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
	# ## Stage-2 fit beta
	# RT_LS.fit_beta(LD_Z2, cor_ZY2)
	pred_ior_RT_2SLS = pt.transform(IoR.reshape(-1,1)).flatten()
	pred_RT_2SLS = pt.transform(X.reshape(-1,1)).flatten()
	pred_ior_RT_2SLS = pred_ior_RT_2SLS - np.mean(pred_RT_2SLS)
	pred_RT_2SLS = pred_RT_2SLS - np.mean(pred_RT_2SLS)
	# beta_LS.append(abs(LS.beta))
	# beta_RT_LS.append(abs(RT_LS.beta))
	# beta_LS_SIR.append(abs(echo.beta[0]))
	mse_air.append( np.mean( (pred_phi - phi)**2 ) )  
	mse_mean.append( np.mean( (pred_mean - phi)**2 ) )
	mse_RT_LS.append( np.mean( (pred_RT_2SLS - phi)**2 ) )
	mse_LS.append( np.mean( (X - X.mean() - phi)**2 ) )
	
	ue_air.append( np.max(abs(pred_ior - phi_ior)) )
	ue_mean.append( np.max(abs(pred_mean_ior - phi_ior)) )
	ue_RT_LS.append( np.max(abs(pred_ior_RT_2SLS - phi_ior)) )
	ue_LS.append( np.max(abs(IoR - np.mean(X) - phi_ior)) )
	## append plot dict
	link_plot['x'].extend(IoR)
	link_plot['phi'].extend(list(phi_ior))
	link_plot['method'].extend(['True']*len(IoR))
	link_plot['x'].extend(IoR)
	link_plot['phi'].extend(list(pred_ior_RT_2SLS))
	link_plot['method'].extend(['PT-2SLS']*len(IoR))
	link_plot['x'].extend(IoR)
	link_plot['phi'].extend(list(pred_ior))
	link_plot['method'].extend(['2SIR+AIR']*len(IoR))
	link_plot['x'].extend(IoR)
	link_plot['phi'].extend(list(IoR-np.mean(X)))
	link_plot['method'].extend(['2SLS']*len(IoR))
	link_plot['x'].extend(IoR)
	link_plot['phi'].extend(list(pred_mean_ior))
	link_plot['method'].extend(['Cond-mean']*len(IoR))
	

print('MSE: AIR: %.3f(%.3f); mean: %.3f(%.3f); RT-LS: %.3f(%.3f); LS: %.3f(%.3f)' 
	%( np.mean(mse_air), np.std(mse_air)/np.sqrt(n_sim), 
	   np.mean(mse_mean), np.std(mse_mean)/np.sqrt(n_sim),
	   np.mean(mse_RT_LS), np.std(mse_RT_LS)/np.sqrt(n_sim),
	   np.mean(mse_LS), np.std(mse_LS)/np.sqrt(n_sim) ))
print('UE: AIR: %.3f(%.3f); mean: %.3f(%.3f), RT-LS: %.3f(%.3f); LS: %.3f(%.3f)' 
	%( np.mean(ue_air), np.std(ue_air)/np.sqrt(n_sim), 
	   np.mean(ue_mean), np.std(ue_mean)/np.sqrt(n_sim),
	   np.mean(ue_RT_LS), np.std(ue_RT_LS)/np.sqrt(n_sim),
	   np.mean(ue_LS), np.std(ue_LS)/np.sqrt(n_sim) ))

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)
sns.set_theme(style="whitegrid")
sns.lineplot(data=link_plot, x="x", y="phi", hue="method",
			style="method", alpha=.7)
plt.show()