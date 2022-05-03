import pandas as pd
import numpy as np
from scipy.stats import hmean, gmean

def combine_pvalues(p_values, method='cauchy', weights=None):
    """
    Methods for combining the p-values of independent/dependent tests bearing upon the same hypothesis.
    
    Parameters
    ----------

    p_values: {array-like} of shape (num_test)
        Array of p-values assumed to come from the same hypothesis.

    method: {geometric, bonferroni, median, hmean, hommel, cauchy}
        Name of method to use to combine p-values.
    
    Returns
    -------

    p_value: float
        The combined p-value.

    Reference
    ---------
    Vovk, V., & Wang, R. (2020). Combining p-values via averaging. Biometrika, 107(4), 791-808.
    """

    p_values = np.array(p_values)
    n_tests = len(p_values)
    if n_tests == 1:
        return p_values[0]

    if method == 'gmean':
        p_value_cp = np.e*gmean(p_values, 0)
    elif method == 'median':
        p_value_cp = 2*np.median(p_values, 0)
    elif method == 'bonferroni':
        p_value_cp = n_tests*np.min(p_values, 0)
    elif method == 'hmean':
        p_value_cp = np.e * np.log(n_tests) * hmean(p_values, 0)
    elif method == 'hommel':
        const = np.sum(1. / (np.arange(n_tests) + 1.))
        order_const = const*(n_tests/(np.arange(n_tests) + 1.))
        p_value_cp = np.sort(p_values)*order_const
        p_value_cp = np.min(p_value_cp)
    elif method == 'cauchy':
        t0 = np.mean(np.tan((.5 - p_values)*np.pi))
        p_value_cp = .5 - np.arctan(t0)/np.pi
    else:
        raise NameError("cp method should be {geometric, bonferroni, median, hmean, hommel, cauchy}")
    p_value_cp = np.minimum(p_value_cp, 1.)
    return p_value_cp