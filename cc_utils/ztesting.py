import numpy as np
import pandas as pd

import scipy.stats
import scipy.linalg
import scipy.optimize as opt

import copy
from collections import *

from .abstract_test_setting import abc_setting
from utils.p2e import p2e 

#####

class ztesting(abc_setting):
    def __init__(             # this is where one should determine all aspects of the e-value function
        self, 
        Sigma, 
        alt_type='twosided',
        alt = None,
        method = 'lrt',
        calibrator = None,
        aon_threshold = None
    ):
        super(ztesting,self).__init__('ztesting')
        
        # initialize based on attributes that don't change (e.g., distributional attributes)
        self.m = np.array(Sigma).shape[0]
        self.Sigma = Sigma
        
        self.method = method
        
        self.alt_type = alt_type
        assert self.alt_type in ['onesided', 'twosided'], f"alt type must be one of {['onesided', 'twosided']}"
        
        self.alt = alt
        if (self.alt_type == 'onesided') and (self.method != 'lrt'):
            if alt in ['positive', 'negative']:
                self.alt = np.ones(self.m) * (1 if alt=='positive' else -1)
        
        self.calibrator = calibrator
        self.aon_threshold = aon_threshold
        
    def p_function(self, Z, alt=None):
        # generate p-values
        if self.alt_type == 'twosided':
            p = scipy.stats.norm.sf(np.abs(Z / np.sqrt(np.diag(self.Sigma))))*2
        else:
            # where self.alt is positive, use positive one-sided p-values
            # where self.alt is negative, use negative one-sided p-values 
            # use np.where 
            
            if alt:
                # this is used to enforce the p-values to be all positive or all negative
                if alt in ['+', 'positive']: 
                    p = scipy.stats.norm.sf(Z / np.sqrt(np.diag(self.Sigma)))
                elif alt == ['-', 'negative']:
                    p = 1-scipy.stats.norm.sf(Z / np.sqrt(np.diag(self.Sigma)))
                else:
                    assert False, f'a specified {alt} must be "+"/"positive" or "-"/"negative"'
            else: 
                p_pos = scipy.stats.norm.sf(Z / np.sqrt(np.diag(self.Sigma))) # 1 - this is p_neg
                p = np.where(np.array(self.alt) > 0, p_pos, 1-p_pos)  
        return p
        
    def e_function(self, Z, initial_boost=1, **kwargs):
        vars(self).update(kwargs)
        
        method = self.method
        
        if method=='lrt':
            # the likelihood ratio
            e = np.exp((self.alt*Z - (self.alt**2)/2)/np.diag(self.Sigma))  
            e = np.multiply(initial_boost, e)
        elif method=='p2e_cal':
            calibrator = self.calibrator
            # p2e calibration
            e = p2e(self.p_function(Z), kappa=calibrator)
        elif method=='p2e_aon':
            aon_threshold = self.aon_threshold
            e = 1 * (self.p_function(Z) <= self.aon_threshold) / self.aon_threshold  
        
        return e    # if there's a preexisting boost factor, should return boosted e-values 
    
    def get_suff_stat(self, idx, Z):
        neg_idx = np.array(range(self.m))!=idx
        suff_stat = Z[neg_idx] - (self.Sigma[neg_idx, idx] / self.Sigma[idx, idx]) * Z[idx]
        return suff_stat
    
    def cond_e_sampling(self, idx, suff_stat, n_smpl=1, initial_boost=1):
        neg_idx = np.array(range(self.m))!=idx
        
        new_Z_smpl = np.zeros(shape=(self.m, n_smpl))   # matrix of new draws for CC        
        Z_idx_smpl = np.random.normal(loc=0.0, scale=np.sqrt(self.Sigma[idx,idx]), size=n_smpl)   # draw Z_j
        
        # populate new Z matrix sampled conditionally on S_j
        new_Z_smpl[idx,:] = Z_idx_smpl   
        new_Z_smpl[neg_idx,] = (np.matmul(
            self.Sigma[neg_idx, idx].reshape(-1,1), new_Z_smpl[idx,:].reshape(-1,1).T
        )/self.Sigma[idx,idx]) + suff_stat.reshape(-1,1)  # (m-1 x 1) * (1 x n_smpl) + (m-1 x 1)
        new_Z_smpl = new_Z_smpl.T
        
        return np.apply_along_axis(self.e_function, 1, new_Z_smpl, initial_boost=initial_boost)
    
    def get_initial_boost_factor(self, alpha, alt): 
        # Example 3 in https://arxiv.org/pdf/2009.02824.pdf, the original e-BH paper
        # this is used for boosting LRT e-values
        def f(b, a, d):
            return (b * scipy.stats.norm.cdf( d/2 + np.log(a * b)/d ) -1)
        
        return opt.fsolve(f, 1, args=(alpha, alt))[0] 
    