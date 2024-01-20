import numpy as np
from scipy.special import lambertw
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from random import choices

def sim(n, p, theta0, beta0, alpha0=0., case='log', feat='normal', IoR=None):
    r"""Simulate data for the nonlinear causal IV model (see [1]).

    Parameters
    ----------

    n : int
        Number of samples.

    p : int
        Number of features (IVs).
    
    theta0 : array_like
        True coefficients for the IVs to exposure.

    beta0 : float
        True coefficient for the causal effect from exposure to outcome.
    
    alpha0 : float, optional
        True coefficients for invalid IVs (default is 0.).

    case : str, optional
        Type of nonlinear causal transformation ('linear', 'log', 'cube-root', 'inverse', 'sigmoid', 'piecewise_linear'), (default is 'log').
    
    feat : str, optional
        Type of feature distribution ('normal', 'AP-normal', 'laplace', 'uniform', 'cate'). (default is 'normal').
    
    IoR : array_like or None, optional
        The region of interest (default is None): checking the nonlinear causal transformation.

    Returns
    -------
    Z : {array-like} of shape (n, p)
        n simluated data of IVs.

    X : {array-like} of shape (n, )
        n simluated data of exposure.

    y : {array-like} of shape (n, )
        n simluated data of outcome.
    
    phi : {array-like} of shape (n, )
        transformed exposure based on the transformation `case`.

    phi_ior : {array-like} of shape (n, ) (if IoR is not `None`)
        transformed region of intere (IoR) based on the transformation `case`. 

    References
    ----------

    .. [1] `Dai, B., Li, C., Xue, H., Pan, W., & Shen, X. (2024). Inference of nonlinear causal effects with GWAS summary data. In Conference on Causal Learning and Reasoning. PMLR.
        <https://openreview.net/pdf?id=cylRvJYxYI>`_

    """

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
        Z = np.random.uniform(low=-1.0, high=1.0, size=(n,p))
    elif feat == 'cate':
        Z = np.random.choice(2, size=(n, p), p=[0.7, 0.3]) + np.random.choice(2, size=(n, p), p=[0.7, 0.3])
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
        X = X - np.mean(X)
        phi = X
        if IoR is not None:
            phi_ior = IoR
        y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

    elif case == 'log':
        X = np.exp( np.dot(Z, theta0) + U + eps )
        X = X - np.mean(X)
        phi = np.log(X)
        if IoR is not None:
            phi_ior = np.log(IoR)
        y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

    elif case == 'cube-root':
        X = (np.dot(Z, theta0) + U + eps)**3
        X = X - np.mean(X)
        phi = np.sign(X)*(abs(X)**(1./3))
        if IoR is not None:
            phi_ior = np.sign(IoR)*(abs(IoR)**(1./3))
        y = beta0*phi + np.dot(Z, alpha0) + U + gamma

    elif case == 'inverse':
        X = 1. / (np.dot(Z, theta0) + U + eps)
        X = X - np.mean(X)
        phi = 1. / X
        if IoR is not None:
            phi_ior = 1. / IoR
        y = beta0 * phi + np.dot(Z, alpha0) + U + gamma
    
    elif case == 'sigmoid':
        X = 1 / (1 + np.exp( - np.dot(Z, theta0) - U - eps ))
        X = X - np.mean(X)
        phi = np.log( X / (1 - X) )
        if IoR is not None:
            phi_ior = np.log( IoR / (1 - IoR) )
        y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

    elif case == 'piecewise_linear':
        tmp = np.dot(Z, theta0) + U + eps
        X = 1.*(tmp<=0.)*tmp + 2.*tmp*(tmp>0.)
        X = X - np.mean(X)
        phi = 1.*X*(X<=0) + .5*X*(X>0)
        if IoR is not None:
            phi_ior = 1.*(IoR<=0.)*IoR + 2*IoR*(IoR>0.)
        y = beta0 * phi + np.dot(Z, alpha0) + U + gamma
    
    elif case == 'quad':
        tmp = np.dot(Z, theta0) + U + 2.0 + eps
        # ensure the phi is positive
        Z = Z[tmp>0]
        U = U[tmp>0]
        gamma = gamma[tmp>0]
        tmp = tmp[tmp>0]

        X = np.sign(np.random.randn(len(tmp))) * np.sqrt(tmp)
        X = X - np.mean(X)
        phi = X**2
        if IoR is not None:
            phi_ior = IoR**2
        y = beta0 * phi + np.dot(Z, alpha0) + U + gamma

    else:
        raise NameError('Sorry, no build-in case.')
    
    ## normalize the dataset
    center = StandardScaler(with_std=False)
    mean_y = np.mean(y)
    Z, y = center.fit_transform(Z), y - mean_y
    phi = phi - np.mean(phi)

    y_scale = y.std()
    y = y / y_scale
    Z = Z / y_scale
    phi = phi / y_scale

    m = "ψ(x) = z^T θ + ω; \n" "y = β ψ(x) + z^T α + ε. \n"
    msg = m + \
         "--- \n" \
         "β: causal effect from x to y. \n" \
         "ψ(x): causal link among (z, x, y). \n" \
         "--- \n" \
         "True β : %.3f \n" \
         "True ψ(x) : %s" %(beta0, case)

    print_msg_box(msg, indent=1, title='True Model')

    if IoR is None:
        return Z, X, y, phi
    else:
        return Z, X, y, phi, phi_ior - np.mean(phi)

def sim_phi(X, case='linear'):
    r"""Apply a transformation to the input based on the specified case.

    Parameters
    ----------

    X : array_like
        Input data to be transformed.

    case : str, optional
        Type of transformation to be applied (default is 'linear'). Supported cases are: 'linear', 'log', 'cube-root', 'inverse', 'sigmoid', and 'piecewise_linear'.

    Returns
    -------

    array_like
        The transformed data based on the specified case.
    """

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
    else:
        raise NameError('Sorry, no build-in case.')

def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)