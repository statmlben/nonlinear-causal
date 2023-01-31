import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from os.path import join
from pandas_plink import read_plink1_bin, read_plink
from pandas_plink import get_data_folder
import argparse
from nl_causal.ts_models import _2SLS, _2SIR
from nl_causal.linear_reg import L0_IC, SCAD_IC
from scipy import stats
from sklearn.preprocessing import power_transform, quantile_transform
from nl_causal.base.preprocessing import calculate_vif_, unique_columns, calculate_cor_
import os
import glob
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


np.random.seed(1)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='nl-causal')
    parser.add_argument('-f', '--file', default='data/both.CTSS.3181.50.2',type=str,
                        help='Path to the data file (default: cis.MSMB.10620.21.3)')
    # parser.add_argument('-e', '--eps', default=1e-4, type=float,
    #                     help='Diag-eps to make LD matrix to be PD.')
    # both.ATF6.11277.23.3; both.CTSS.3181.50.2
    # parser.add_argument('-mi', '--MI_cut', default=.005, type=float,
    #                     help='The threshold for marginal dependece to screen the independent features (default: 0.005)')
    # parser.add_argument('-cor', '--cor_cut', default=0.8, type=float,
    #                 help='The threshold for correlation to screen the highly correlated features, (default: 0.8)')
    # parser.add_argument('-vif', '--vif_cut', default=10, type=float,
    #                 help='The threshold for VIF to screen the colinear features, (default: 10)')
    args = parser.parse_args()

    root_path = os.getcwd()
    target = args.file
    data_folder = os.path.join(root_path, target)

    os.chdir(data_folder)
    bed_file = glob.glob('*.bed')[0]
    exp_file = glob.glob('*pheno')[0]
    sum_stat_file = glob.glob('*.sum.stats')[0]

    exp = pd.read_csv(os.path.join(data_folder, exp_file), '\t', engine='python')
    sum_stat = pd.read_csv(os.path.join(data_folder, sum_stat_file), sep=" |\t", index_col='rsID', engine='python')
    ## drop sum_stat with missing Effect
    sum_stat = sum_stat.dropna(subset=['Effect'])
    # assume a sample size in the sum_stat
    # potential issue #1
    sum_stat['size'] = 140306
    # compute the normalized effect coeff
    sum_stat['cor_effect'] = sum_stat['Effect'] / np.sqrt((sum_stat['size'] - 2)*(sum_stat['StdErr']**2) + sum_stat['Effect']**2)

    ## Load Stage 1 data
    (bim, fam, bed) = read_plink(os.path.join(data_folder, bed_file), verbose=False)
    snp_data = bed.compute()

    ## impute missing values by snp-wise mean
    ind_miss = np.isnan(snp_data)
    snp_data[ind_miss] = np.nanmean(snp_data, 1)[np.where(ind_miss)[0]]

    ## find the shared snps btw sum_stat and SNPs
    shared_snp = list(set(sum_stat.index).intersection(set(bim['snp'])))

    ## select shared snps from SNP data
    bed_idx = np.array([snp in shared_snp for snp in bim['snp']])
    Z1 = snp_data[bed_idx].T

    ## select shared snps from exposure
    X1 = exp['protein_abundance'].values
    cor_ZY2 = sum_stat.loc[bim['snp'][bed_idx]]['cor_effect'].values

    ## center the data
    from sklearn import preprocessing
    Z1 = preprocessing.StandardScaler().fit_transform(Z1)
    X1 = X1 - X1.mean()

    ## remove duplicate columns
    unique_idx = unique_columns(Z1)
    Z1 = Z1[:,unique_idx]
    cor_ZY2 = cor_ZY2[unique_idx]

    print('\n##### Causal inference of %s #####' %target)
    print('\n##### n1=%d; n2=%d p=%d #####' %(len(X1), sum_stat['size'][0], Z1.shape[1]))

    ## remove independent features based on mutual_info_regression (Stage 1)
    # print('-'*20)
    # print("MI-Screening to remove low-dependent features")
    # MI_cut = args.MI_cut
    # from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_regression
    # MI1 = mutual_info_regression(Z1, X1, random_state=0)
    # Z1 = Z1[:,MI1>MI_cut]
    # cor_ZY2 = cor_ZY2[MI1>MI_cut]
    # print("MI-Screening is Done! Remaining %d feats" %Z1.shape[1])

    ## remove high correlated features
    # print('-'*20)
    # print('remove high correlated features')
    # Z1, valid_cols = calculate_cor_(pd.DataFrame(Z1), thresh=args.cor_cut)
    # Z1, valid_cols = Z1.values, np.array(valid_cols)
    # cor_ZY2 = cor_ZY2[valid_cols]
    # print("high correlated features is Done! Remaining %d feats" %Z1.shape[1])

    # ## remove colinear columns
    # print('-'*20)
    # print("VIF to remove colinear features")
    # Z1, valid_cols = calculate_vif_(pd.DataFrame(Z1), thresh=args.vif_cut, method='greedy')
    # Z1, valid_cols = Z1.values, np.array(valid_cols)
    # cor_ZY2 = cor_ZY2[valid_cols]
    # print("VIF is Done! Remaining %d feats" %Z1.shape[1])

    ## n1 and n2 is pre-given
    n1, n2, p = len(X1), 140306, Z1.shape[1]
    LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)

    ## add diagonal eps to make the matrix PD.
    for power_tmp in np.arange(-5, -2, .2):
        LD_Z1 += np.linalg.norm(LD_Z1) * (10**power_tmp) *np.eye(p)
        if np.linalg.cond(LD_Z1) < 1 / np.finfo('float32').eps:
            break
    
    ## may can change the LD matrix in stage 2
    LD_Z2, cov_ZY2 = LD_Z1/n1*n2, cor_ZY2*n2

    Ks = range(int(p/2)-1)

    ## 2SLS
    try:
        reg_model = L0_IC(fit_intercept=False, alphas=5**np.arange(-2,3,.3),
                        Ks=Ks, max_iter=10000, find_best=False, refit=False, var_res=1.)
        LS = _2SLS(sparse_reg=reg_model)
        ## if you do not want a selection
        ## LS = _2SLS(sparse_reg=None)
        ## Stage-1 fit theta
        LS.fit_theta(LD_Z1, cov_ZX1)
        ## Stage-2 fit beta
        LS.fit_beta(LD_Z2, cov_ZY2, n2)
        ## produce p_value and CI for beta
        LS.test_effect(n2, LD_Z2, cov_ZY2)
        print('-'*20)
        print('LS beta with selection: %.3f' %LS.beta)
        print('p-value based on 2SLS with selection: %.5f' %LS.p_value)
    except:
        print('-'*20)
        print('LS (with selection) failed, since the est variance is negative')
        pass

    ## 2SLS without selection
    try:
        LS_no_select = _2SLS(sparse_reg=None)
        ## Stage-1 fit theta
        LS_no_select.fit_theta(LD_Z1, cov_ZX1)
        ## Stage-2 fit beta
        LS_no_select.fit_beta(LD_Z2, cov_ZY2, n2)
        ## produce p_value and CI for beta
        LS_no_select.test_effect(n2, LD_Z2, cov_ZY2)
        print('-'*20)
        print('LS beta without selection: %.3f' %LS_no_select.beta)
        print('p-value based on 2SLS without selection: %.5f' %LS_no_select.p_value)
    except:
        print('-'*20)
        print('LS (without selection) failed, since the est variance is negative')
        pass

    ## 2SIR
    try:
        reg_model = L0_IC(fit_intercept=False, alphas=5**np.arange(-2,3,.3),
                        Ks=Ks, max_iter=10000, find_best=False, refit=False)
        SIR = _2SIR(sparse_reg=reg_model, data_in_slice=0.1*n1)
        ## Stage-1 fit theta
        SIR.fit_theta(Z1=Z1, X1=X1)
        ## Stage-2 fit beta
        SIR.fit_beta(LD_Z2, cov_ZY2, n2)
        ## generate CI for beta
        SIR.test_effect(n2, LD_Z2, cov_ZY2)
        print('-'*20)
        print('2SIR beta with selection: %.3f' %SIR.beta)
        print('p-value based on 2SIR with selection: %.5f' %SIR.p_value)
        
        ## (Optional) fit the nonlinear link function
        SIR.fit_link(Z1=Z1, X1=X1)
        # predict for fitted link
        IoR = np.arange(0, 1, 1./100)
        link_IoR = SIR.link(X = IoR[:,None])
    except:
        print('-'*20)
        print('SIR (without selection) failed, since the est variance is negative')
        pass


    try:
        ## 2SIR without selection
        SIR_no_select = _2SIR(sparse_reg=None, data_in_slice=0.1*n1)
        ## Stage-1 fit theta
        SIR_no_select.fit_theta(Z1=Z1, X1=X1)
        ## Stage-2 fit beta
        SIR_no_select.fit_beta(LD_Z2, cov_ZY2, n2)
        ## generate CI for beta
        SIR_no_select.test_effect(n2, LD_Z2, cov_ZY2)
        print('-'*20)
        print('2SIR beta without selection: %.3f' %SIR_no_select.beta)
        print('p-value based on 2SIR without selection: %.5f' %SIR_no_select.p_value)
    except:
        print('-'*20)
        print('SIR (without selection) failed, since the est variance is negative')
        pass

    ## Comb-2SIR
    try:
        data_in_slice_lst = [.1*n1, .2*n1, .3*n1, .5*n1]
        comb_pvalue, comb_beta, comb_eigenvalue = [], [], []
        for data_in_slice_tmp in data_in_slice_lst:
            # print('data_in_slice: %.3f' %data_in_slice_tmp)
            reg_model = L0_IC(fit_intercept=False, alphas=5**np.arange(-2,3,.3),
                            Ks=Ks, max_iter=10000, find_best=False, refit=False, var_res=1.)
            SIR = _2SIR(sparse_reg=reg_model, data_in_slice=data_in_slice_tmp)
            ## Stage-1 fit theta
            SIR.fit_theta(Z1=Z1, X1=X1)
            ## Stage-2 fit beta
            SIR.fit_beta(LD_Z2, cov_ZY2, n2)
            ## generate CI for beta
            SIR.test_effect(n2, LD_Z2, cov_ZY2)
            comb_beta.append(SIR.beta)
            comb_pvalue.append(SIR.p_value)
        comb_T = np.tan((0.5 - np.array(comb_pvalue))*np.pi).mean()
        correct_pvalue = min( max(.5 - np.arctan(comb_T)/np.pi, np.finfo(np.float64).eps), 1.0)
        # correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
        print('-'*20)
        print('Comb-2SIR beta with selection: %.3f' %np.mean(comb_beta))
        print('p-value based on Comb-2SIR with selection: %.5f' %correct_pvalue)
    except:
        print('-'*20)
        print('Comb-SIR (with selection) failed, since the est variance is negative')
        pass


    ## Comb-2SIR without selection
    try:
        data_in_slice_lst = [.1*n1, .2*n1, .3*n1, .5*n1]
        comb_pvalue, comb_beta, comb_eigenvalue = [], [], []
        for data_in_slice_tmp in data_in_slice_lst:
            # print('data_in_slice: %.3f' %data_in_slice_tmp)
            SIR_no_select = _2SIR(sparse_reg=None, data_in_slice=data_in_slice_tmp)
            ## Stage-1 fit theta
            SIR_no_select.fit_theta(Z1=Z1, X1=X1)
            ## Stage-2 fit beta
            SIR_no_select.fit_beta(LD_Z2, cov_ZY2, n2)
            ## generate CI for beta
            SIR_no_select.test_effect(n2, LD_Z2, cov_ZY2)
            comb_beta.append(SIR_no_select.beta)
            comb_pvalue.append(SIR_no_select.p_value)
        comb_T = np.tan((0.5 - np.array(comb_pvalue))*np.pi).mean()
        correct_pvalue = min( max(.5 - np.arctan(comb_T)/np.pi, np.finfo(np.float64).eps), 1.0)
        # correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
        print('-'*20)
        print('Comb-2SIR beta without selection: %.3f' %np.mean(comb_beta))
        print('p-value based on Comb-2SIR without selection: %.5f' %correct_pvalue)
    except:
        print('-'*20)
        print('Comb-SIR (without selection) failed, since the est variance is negative')
        pass