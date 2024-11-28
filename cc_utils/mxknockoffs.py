from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import scipy.linalg

import knockpy
from knockpy.knockoff_filter import KnockoffFilter

import copy
from collections import *

from .abstract_test_setting import abc_setting
from utils.p2e import p2e 

#####

class knockoffs(abc_setting):
    """
    The general knockoffs CC object.
    """
    def __init__(             # this is where one should determine all aspects of the e-value function
        self, 
        m,
        kn_filter,    # knockpy object (already instantiated with ksampler, fstat, etc)
        mu=None,
        Sigma=None
    ):
        super(knockoffs,self).__init__(f'{kn_filter.ksampler} knockoffs')
        
        self.m = m
        
        self.kn_filter = kn_filter
        self.ksampler = kn_filter.ksampler
        self.S = None
        if not isinstance(kn_filter.S, type(None)):
            self.S = kn_filter.S
            self.ksampler.S = self.S
            
        self.mu = mu
        self.Sigma = Sigma
        
    def forward(self, X, y, fdr):
        assert not isinstance(self.ksampler.S, type(None))   # make sure that the S-matrix has been inherited
        self.ksampler.X = X   # update the ksampler with the new data matrix
        
        Xtilde = self.ksampler.sample_knockoffs()
        
        return self.kn_filter.forward(
            fdr=fdr, 
            X=X, y=y, Xk=Xtilde,
            mu=self.mu,
            Sigma=self.Sigma
            # **kn_kwargs
        )
        
    def e_function(self, X, y, derand, alpha_kn,):
        """
        Computes the derandomized knockoff e-values as constructed in <CITE HERE>.
        
            Parameters:
                derand: the number of knockoffs to sample for derandomization
                a_kn: the FDR level at which to run the knockoff sampler
                kn_kwargs: kwargs for the knockpy knockoff forward method
            
            Returns:
                e_avg: the derandomized knockoff e-values
        """
        
        assert not isinstance(self.ksampler.S, type(None))   # make sure that the S-matrix has been inherited
        self.ksampler.X = X   # update the ksampler with the new data matrix
        
        e_list = []
        for i in range(derand):
            # for each derandomization run, sample new knockoffs
            Xtilde = self.ksampler.sample_knockoffs()
            
            # kn_filter = KnockoffFilter(ksampler=ksampler, fstat=fstat, knockoff_kwargs={'method': kmethod})

            # forward 
            rej = self.kn_filter.forward(
                fdr=alpha_kn, 
                X=X, y=y, Xk=Xtilde,
                mu=self.mu,
                Sigma=self.Sigma
                # **kn_kwargs
            )   # we directly sample Xtilde, and put it into our forward function
            
            # important components
            T = self.kn_filter.threshold
            W = self.kn_filter.W
            p = self.m    # number of hypotheses tested
            
            # alternative kn e-values
            if sum(W >= T) < 1/alpha_kn:
                # we use a different T
                for Tstar in ([0] + sorted(W[(W<T)*(W>0)]) + [T]):
                    if sum(W >= Tstar) < 1/alpha_kn:
                        T = Tstar    # inf(t: sum(W>=t) < 1/a_kn)
                        break

            denom = 1 + sum(W <= (-1 * T))
            evalues = p * (W >= T) / denom
            
            e_list.append(evalues)
        
        # now, combine the evalues
        e_avg = np.mean(e_list, axis=0); assert (len(e_avg) == p)
        return e_avg
    
    def get_suff_stat(self, idx, X, y):
        """
        Given the data, compute the sufficient statistic S_j for H_j.

        Here we write a function that simply returns (X_{-j}, y), which is the sufficient
        statistic for H_j in the conditional independence testing problem when we use MX
        knockoffs. There may be alternate sufficient statistics that can be overwritten 
        by the user.
        
        For fixed-X knockoffs, this has been overwritten as (X_{-j}, y) is no longer the 
        sufficient statistc; see the specifc class implementation below.
        """
        neg_idx = np.array(range(self.m))!=idx
        return (X[:,neg_idx], y)

    @abstractmethod
    def resample_data(self, idx, suff_stat):
        """
        Given the sufficient statistic for H_j, resample (X, y).

        Must be implemented by the user as usually this depends on distributional assumptions.
        """
        pass
        
    
    def cond_e_sampling(self, idx, suff_stat, derand, alpha_kn, n_smpl=1, **kn_kwargs):
        neg_idx = np.array(range(self.m))!=idx
        
        e_samples = []
        
        for b in range(n_smpl):
            X_new, y_new = self.resample_data(idx, suff_stat)   # resample (conditionally on S_j) so that the knockoffs can be run
            e = self.e_function(X=X_new, y=y_new, derand=derand, alpha_kn=alpha_kn, **kn_kwargs)
            e_samples.append(e) 
            
        return np.array(e_samples)
    

class fxknockoffs(knockoffs):
    """
    The fixed-X knockoffs CC object.
    """
    
    def __init__(
        self, 
        X,
        kn_filter,    # knockpy object
    ):
        n, m = X.shape
        super(fxknockoffs,self).__init__(m, kn_filter)
        assert self.ksampler == 'fx', f'the knockpy filter object must have "fx" as its ksampler.'
        
        self.X = X    # design matrix is fixed in the fixed-X setting and even in the CC step
        self.n = n
        
    def get_suff_stat(self, idx, y):
        """
        Uses the sufficient statistic given in Luo et al. (2022)
        """
        neg_idx = np.array(range(self.m))!=idx
        X_neg = self.X[:,neg_idx]
        
        proj = X_neg @ np.linalg.inv(X_neg.T @ X_neg) @ X_neg.T 
        orthog_proj = np.eye(proj.shape[0]) - proj
        
        suff_stat = (proj @ y, np.square(orthog_proj @ y).sum())
        return suff_stat
        
    def resample_data(self, idx, suff_stat):
        neg_idx = np.array(range(self.m))!=idx
        X_neg = self.X[:,neg_idx]    # (n x m-1) matrix
        
        # find orthonormal basis of column space of X_neg
        basis = scipy.linalg.orth(X_neg)  # (n x k) matrix, k effective rank (should be m-1)
        rank = basis.shape[1]
        
        # find orthonormal basis for subspace orthog to span of X_{-j}
        u, s, v = np.linalg.svd(basis)
        ortho_subspace_basis = u[:, rank:]  # (n x n-k) matrix, so (n x n-m+1) as expected
        
        # now, sample uniformly from unit sphere of dim n-m, which must be (n-m+1)-size vector
        u_sphere = np.random.normal(size=self.n-self.m+1)
        
        # now sample y with these components
        suff_stat_1, suff_stat_2 = suff_stat
        y_cond = suff_stat_1 + np.sqrt(suff_stat_2) * (ortho_subspace_basis @ u_sphere) 
        return (self.X, y_cond)
        

class gaussian_mxknockoffs(knockoffs):
    def __init__(             # this is where one should determine all aspects of the e-value function
        self, 
        kn_filter,    # knockpy object
        mu=None,
        Sigma=None,
    ):
        m = len(mu)
        super(gaussian_mxknockoffs,self).__init__(m, kn_filter, mu, Sigma)  # should define things like S-matrix
        
        # distributional properties
        # self.mu = mu
        # self.Sigma = Sigma
        
    def resample_data(self, idx, suff_stat):
        """
        Uses the fact that the conditional distribution of X_j | X_{-k} is known when 
        X is a multivariate Gaussian with mean mu and covariance Sigma.
        """
        X_neg, y = suff_stat # suff stat should be X[:,neg_idx], y
        n = len(y)
        
        neg_idx = np.array(range(self.m))!=idx
        
        inverse = np.linalg.inv(self.Sigma[neg_idx][:,neg_idx])
        cond_Sigma = self.Sigma[idx,idx] - self.Sigma[idx,neg_idx] @ inverse @ self.Sigma[neg_idx,idx] # 1x1
        cond_mu_vec = self.Sigma[idx,neg_idx] @ inverse @ (X_neg.T)
        
        X_new = np.zeros(shape=(n, self.m))
        X_new[:,idx] = np.random.normal(
            loc=cond_mu_vec, scale=np.sqrt(cond_Sigma), size=(n,)  # setting the slice of nx1 to new copies under the null
        )  
        X_new[:,neg_idx] = X_neg
        
        return (X_new, y)
    
from knockpy.knockoff_stats import data_dependent_threshhold

    
class ghostknockoffs(abc_setting):
    """
    The general knockoffs CC object.
    """
    def __init__(             # this is where one should determine all aspects of the e-value function
        self, 
        mu,
        Sigma,
        S,
        groups=None,
        feature_statistic='L1'
    ):
        super(ghostknockoffs,self).__init__(f'ghost knockoffs')
        
        self.mu = mu
        self.Sigma = Sigma
        self.S = S
        
        self.m = len(mu)
        
        self.groups == groups
        self.feat_stat == feature_statistic
        assert self.feat_stat in ['L1', 'chi_squared'], f"feature_statistic must be one of {['L1', 'chi_squared']}"
        
        # ghost knockoff params
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.P = np.eye(self.m) - self.S @ self.Sigma_inv
        self.V = 2 * self.S - (self.S @ self.Sigma_inv @ self.S)
        self.cho_V = np.linalg.cholesky(self.V)
        
    def sample_knockoffs(self, Z):
        return (self.P @ Z) + (self.cho_V @ np.random.normal(size=self.m))
    
    def get_features(self, Z, Ztilde):
        if self.groups == None:
            return np.divide((Z**2 - Ztilde**2), np.diag(self.Sigma))
        else:
            W = []
            group_ids = np.unique(self.groups)
            for g in group_ids:
                where_g = self.groups==g    # boolean indexer
                
                Z_g = Z[where_g]; Ztilde_g = Z_tilde[where_g]
                Sigma_gg_inv = np.linalg.inv(self.Sigma[where_g,:][:,where_g])
                
                W.append( Z_g.T@Sigma_gg_inv@Z - Ztilde_g.T@Sigma_gg_inv@Ztilde_g )
                
            return np.array(W)
        
    def forward(self, W, fdr):
        # borrowing the code in the knockpy make_selections function
        assert len(W.shape)==1 and len(W) == len(self.groups)   # check W is 1-D and has 1 stat for each group
        T = data_dependent_threshhold(W=W, fdr=fdr)    # set threshold T after the forward run
        selected_flags = (W >= T).astype("float32")
        self.T = T
        return selected_flags, T
        
    def e_function(self, Z, derand, alpha_kn, early_stopping=True):
        """
        Computes the derandomized knockoff e-values as constructed in <CITE HERE>.
        
            Parameters:
                derand: the number of knockoffs to sample for derandomization
                a_kn: the FDR level at which to run the knockoff sampler
                kn_kwargs: kwargs for the knockpy knockoff forward method
            
            Returns:
                e_avg: the derandomized knockoff e-values
        """
        p = self.m    # number of hypotheses tested
        
        e_list = []
        noises = np.random.multivariate_normal(mean=np.zeros(self.m), cov=self.V, size=derand) 
        # (self.cho_V @ np.random.normal(size=(self.m, derand))).T
        for i in range(derand):
            # for each derandomization run, sample new knockoffs
            # kn_filter = KnockoffFilter(ksampler=ksampler, fstat=fstat, knockoff_kwargs={'method': kmethod})
            
            # sample ghost knockoffs easily (and calculate feature stat)
            Ztilde = self.P @ Z + noises[i]
            W = (Z**2 - Ztilde**2)   #np.abs(Z) - np.abs(Ztilde) #(Z**2 - Ztilde**2) 

            # forward 
            selected_flags, T = self.forward(fdr=alpha_kn, W=W)  
            
            # alternative kn e-values
            if early_stopping:
                if sum(W >= T) < 1/alpha_kn:
                    # we use a different T
                    for Tstar in ([0] + sorted(W[(W<T)*(W>0)]) + [T]):
                        if sum(W >= Tstar) < 1/alpha_kn:
                            T = Tstar    # inf(t: sum(W>=t) < 1/a_kn)
                            break

            denom = 1 + sum(W <= (-1 * T))
            evalues = p * (W >= T) / denom
            
            e_list.append(evalues)
        
        # now, combine the evalues
        e_avg = np.mean(e_list, axis=0); assert (len(e_avg) == p)
        return e_avg
    
    def get_suff_stat(self, idx, Z):
        """
        Given the data, compute the sufficient statistic S_j for H_j.
        
        In the case of ghost knockoffs, (X_{-j}, y) is a valid sufficient statistic in 
        theory. However, this defeats its own purpose, as it is advertised as a MX
        variable selection procedure which only requires Z, the Z-scores of the Pearson
        correlations. We instead take the sufficient statistic as Z, and sample
        Z_j | Z_{-j} by taking its knockoff Ztilde_j and reconstructing the data under H_j
        as (Ztilde_j, Z_{-j}).  
        """
        # neg_idx = np.array(range(self.m))!=idx
        return Z

    def resample_data(self, idx, suff_stat):
        """
        We instead take the sufficient statistic as Z_{-j}, and sample
        Z_j | Z_{-j} by taking its knockoff Ztilde_j and reconstructing the data under H_j
        as (Ztilde_j, Z_{-j}).
        """
        Z = suff_stat    # reminder that the suff stat is simply Z in our implementation, though Z_j is never used
        Ztilde_j = np.dot(self.P[idx], Z) + np.random.normal(loc=0.0, scale=np.sqrt(self.V[idx,idx]))
        # (self.P @ Z)[idx] + np.random.normal(loc=0.0, scale=np.sqrt(self.V[idx,idx]))
        
        Z_copy = copy.deepcopy(Z)
        Z_copy[idx] = Ztilde_j     # replace Z_j with Ztilde_j, which now satisfies the null H_j
        
        return Z_copy
    
    def cond_e_sampling(self, idx, suff_stat, derand, alpha_kn, n_smpl=1, **kn_kwargs):        
        e_samples = []
        
        for b in range(n_smpl):
            Z_new = self.resample_data(idx, suff_stat)   # resample (conditionally on S_j) so that the knockoffs can be run
            e = self.e_function(Z_new, derand, alpha_kn)
            e_samples.append(e) 
            
        return np.array(e_samples)