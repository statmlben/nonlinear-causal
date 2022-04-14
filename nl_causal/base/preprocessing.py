import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X, thresh=2.5, verbose=0):
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
    if verbose:
        print('Remaining variables:')
        print(X.columns[variables])
    cols_new = cols[variables]
    return X.iloc[:, variables], cols_new
