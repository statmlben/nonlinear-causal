import  numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf, abs
from scipy.linalg.cython_blas cimport sdot, ddot
from scipy.linalg.cython_blas cimport snrm2, dnrm2
from scipy.linalg.cython_blas cimport saxpy, daxpy
from scipy.linalg.cython_blas cimport scopy, dcopy

# np.import_array()

cdef double _dot(int n, double[::1] x, double[::1] y):
	"""xy"""
	cdef int inc = 1
	return ddot(&n, &x[0], &inc, &y[0], &inc)

cdef void _copy(int dim, double[::1] x, double[::1] y):
	""""y:=x"""
	cdef int inc = 1
	dcopy(&dim, &x[0], &inc, &y[0], &inc)

cdef double _nrm2(int n, double[::1] x):
	"""sqrt(sum((x_i)^2))"""
	cdef int inc = 1
	return dnrm2(&n, &x[0], &inc)

cdef void _axpy(int n, double alpha, double[::1] x, double[::1] y):
	"""y := alpha * x + y"""
	cdef int inc = 1
	daxpy(&n, &alpha, &x[0], &inc, &y[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def elastCD_LD(double[:,::1] LD_X, double[::1] cor, double lam1, double lam2, int max_iter, double eps, int print_step):
	cdef int d = len(LD_X)
	cdef double diff = 1.0
	cdef double[::1] beta = np.zeros(d)
	cdef double[::1] beta_old = np.zeros(d)
	cdef double[::1] delta_beta = np.zeros(d)
	cdef double u = 0.0

	for ite in xrange(max_iter):
		if diff < eps:
			break
		_copy(d, beta, beta_old)
		for j in xrange(d):
			u = LD_X[j,j] * beta[j] + cor[j] - _dot(d, LD_X[j], beta)
			if abs(u) > lam1:
				if u >= 0:
					beta[j] = abs( u - lam1 ) / ( LD_X[j,j] + lam2 )
				else:
					beta[j] = - abs( u - lam1 ) / ( LD_X[j,j] + lam2 )
			else:
				beta[j] = 0
			delta_beta[j] = beta[j] - beta_old[j]
		diff = _nrm2(d, delta_beta) / (_nrm2(d, beta_old) + 1e-8)
		if print_step==1:
			printf('ite %d: coordinate descent with diff: %10.3f. \n', ite, diff)
	if ite == (max_iter-1):
		printf('The algo did not convergence, pls increase max_iter')
	return np.array(beta)
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def elastCD_HD(double[:,::1] X, double[::1] diag, double[::1] cor, double lam1, double lam2, int max_iter, double eps, int print_step):
	cdef int n = X.shape[0]
	cdef int d = X.shape[1]
	cdef double diff = 1.0
	cdef double[::1] beta = np.zeros(d)
	cdef double[::1] beta_old = np.zeros(d)
	cdef double[::1] y_hat = np.zeros(n)
	cdef double[::1] delta_beta = np.zeros(d)
	cdef double u = 0.0

	for ite in xrange(max_iter):
		if diff < eps:
			break
		_copy(d, beta, beta_old)
		for j in xrange(d):
			u = diag[j] * beta[j] + cor[j] - _dot(n, X[j], y_hat)
			if abs(u) > lam1:
				if u >= 0:
					beta[j] = abs( u - lam1 ) / ( diag[j] + lam2 )
				else:
					beta[j] = - abs( u - lam1 ) / ( diag[j] + lam2 )
			else:
				beta[j] = 0
			# y = y + (beta[j] - beta_old[j]) * X[j]
			delta_beta[j] = beta[j] - beta_old[j]
			_axpy(n, delta_beta[j], X[j], y_hat)
		diff = _nrm2(d, delta_beta) / (_nrm2(d, beta_old) + 1e-8)
		# diff = max(abs(delta_beta))
		if print_step==1:
			printf('ite %d: coordinate descent with diff: %10.3f. \n', ite, diff)
	if ite == (max_iter-1):
		printf('The algo did not convergence, pls increase max_iter')
	return np.array(beta)