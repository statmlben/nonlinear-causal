## Test invalid IVs
import numpy as np
# from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nl_causal.ts_models import _2SLS, _2SIR
from sklearn.preprocessing import power_transform, quantile_transform
from scipy.linalg import sqrtm
from nl_causal.linear_reg import L0_IC
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, LassoLarsIC, LassoCV

n, p = 10000, 50
df = {'case': [], 'pred_K': []}

# theta0 = np.random.randn(p)
for beta0 in [.05]:
    for case in ['linear', 'log', 'cube-root', 'inverse', 'piecewise_linear', 'quad']:
        K_lst = []
        bad_select = 0
        p_value = []
        n_sim = 1000
        if beta0 > 0.:
            n_sim = 100
        for i in range(n_sim):
            theta0 = np.ones(p)
            theta0 = theta0 / np.sqrt(np.sum(theta0**2))
            alpha0 = np.zeros(p)
            alpha0[:5] = 1
            Z, X, y, phi = sim(n, p, theta0, beta0, alpha0=alpha0, case=case, feat='AP-normal')
            if abs(X).max() > 1e+8:
                continue
            ## normalize Z, X, y
            center = StandardScaler(with_std=False)
            mean_X, mean_y = X.mean(), y.mean()
            Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
            Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=.5, random_state=42)
            ## scale y
            y_scale = y2.std()
            y1 = y1 / y_scale
            y2 = y2 / y_scale
            y1 = y1 - y2.mean()
            y2 = y2 - y2.mean()
            # summary data
            n1, n2 = len(Z1), len(Z2)
            LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
            LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
            # print('True beta: %.3f' %beta0)

            Ks = range(int(p/2))

            ## solve by SIR+LS
            reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3), 
                    Ks=Ks, max_iter=50000, refit=False, find_best=False)
            echo = _2SIR(sparse_reg=reg_model)
            ## Stage-1 fit theta
            echo.fit_theta(Z1, X1)
            ## Stage-2 fit beta
            echo.fit_beta(LD_Z2, cor_ZY2, n2=n2)
            ## generate CI for beta
            echo.test_effect(n2, LD_Z2, cor_ZY2)
            # print('est beta based on 2SIR: %.3f; p-value: %.5f' %(echo.beta*y_scale, echo.p_value))
            if sorted(echo.best_model_) != sorted([0,1,2,3,4,50]):
                bad_select += 1
            df['case'].append(case)
            K_lst.append(len(echo.best_model_)-1)
            df['pred_K'].append(len(echo.best_model_)-1)

for beta0 in [.05]:
    for case in ['linear', 'log', 'cube-root', 'inverse', 'piecewise_linear', 'quad']:
        print('case: %s; beta0: %.3f, n: %d, p: %d' %(case, beta0, n, p))
        tmp = df[df['case'] ==case]['pred_K']
        print('pred_K: mean(std) %.3f(%.3f); (Q1, Q2, Q3): %s' %(tmp.mean(), tmp.std()/np.sqrt(len(df)), tmp.quantile([.25, .50, .75]).values))

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set_theme(style="whitegrid", palette="muted")
# df = pd.DataFrame(df)
# sns.stripplot(data=df, x="case", y="pred_K")
# plt.show()