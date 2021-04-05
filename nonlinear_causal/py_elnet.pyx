import numpy as np
cimport cython

cdef extern from "elnet.h"
    cdef int elnet(double lambda1, double lambda2, const arma::vec& diag, const arma::mat& X, const arma::vec& r, double thr, arma::vec& x, arma::vec& yhat, int trace, int maxiter)

cdef void run_elnet(int n, int dim, double[::1] r, double[::1] diag, double lambda1, double lambda2, double thr, int trace, int maxiter):
    cdef double[::1] beta = np.zeros(dim)
    cdef double[::1] yhat = np.zeros(n)
    return elnet(&lambda1, &lambda2, &diag, &beta, &r, &thr, &x, &yhat, &trace, &maxiter)