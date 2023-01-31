import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def unique_columns(X):
    """
    Find unique columns for numpy array.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Feature matrix
    
    Returns
    -------
    index: The index set for unique subset

    """
    ind = np.lexsort(X)
    diff = np.any(X.T[ind[1:]] != X.T[ind[:-1]], axis=1)
    edges = np.where(diff)[0] + 1
    result = np.split(ind, edges)
    # result = [group for group in result if len(group) >= minoccur]
    index = np.array([tmp[0] for tmp in result])
    return index

def calculate_cor_(X, thresh=0.8, verbose=0):
    """
    Remove low-correlated features.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Feature matrix

    Returns
    -------
    X: return feature matrix by removing low-correlated features.
    """
    cols = X.columns
    variables = np.array(range(X.shape[1]))
    for ix in range(len(variables)):
        if ix >= len(variables):
            break
        cor_tmp = np.zeros(len(variables))
        cor_tmp[ix+1:] = X.iloc[:,variables].iloc[:,ix+1:].corrwith(X.iloc[:,variables].iloc[:,ix]).values
        if max(cor_tmp) > thresh:
            variables = variables[cor_tmp<thresh]
            # print(len(variables))
    if verbose:
        print('Remaining variables:')
        print(X.columns[variables])
    cols_new = cols[variables]
    return X.iloc[:, variables], cols_new


def calculate_vif_(X, thresh=2.5, verbose=0, method='best'):
    """
    Remove multicollinearity features.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Feature matrix

    Returns
    -------
    X: return feature matrix by removing multicollinearity features.
    """
    cols = X.columns
    variables = list(range(X.shape[1]))
    dropped = True
    if method == 'best':
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                    for ix in range(X.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                if verbose:
                    print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                            '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
    elif method == 'greedy':
        while dropped:
            dropped = False
            for ix in range(X.iloc[:, variables].shape[1]):
                if ix >= X.iloc[:, variables].shape[1]:
                    break
                vif_tmp = variance_inflation_factor(X.iloc[:, variables].values, ix)
                if vif_tmp > thresh:
                    dropped = True
                    # print("drop col: %s" %variables[ix])
                    del variables[ix]
    if verbose:
        print('Remaining variables:')
        print(X.columns[variables])
    cols_new = cols[variables]
    return X.iloc[:, variables], cols_new


