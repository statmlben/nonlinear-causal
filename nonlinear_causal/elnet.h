#ifndef __ELNET_H__
#define __ELNET_H__

extern int elnet(double lambda1, double lambda2, const arma::vec& diag, const arma::mat& X, const arma::vec& r, double thr, arma::vec& x, arma::vec& yhat, int trace, int maxiter);

#endif //__ELNET_H__