import numpy as np
import pandas as pd

import scipy.stats

import copy
from collections import *

from .abstract_test_setting import abc_setting
from utils.p2e import p2e 

#####

class ttesting(abc_setting):
    def __init__(             # this is where one should determine all aspects of the e-value function
        self, 
        Psi, 
        df,
        alt_type='twosided',
        alt = None,
        method = 'lrt',
        calibrator = None,
        aon_threshold = None
    ):
        super(ttesting,self).__init__('ttesting')
        
        # initialize based on attributes that don't change (e.g., distributional attributes)
        self.m = np.array(Psi).shape[0]
        self.df = df
        
        self.Psi = Psi
        self.method = method
        
        self.alt_type = alt_type
        assert self.alt_type in ['onesided', 'twosided'], f"alt type must be one of {['onesided', 'twosided']}"
        
        self.alt = alt
        if (self.alt_type == 'onesided') and (self.method != 'lrt'):
            if alt in ['positive', 'negative']:
                self.alt = np.ones(self.m) * (1 if alt=='positive' else -1)    # enforces self.alt to be numerical
        
        self.calibrator = calibrator
        self.aon_threshold = aon_threshold
        
    def p_function(self, T, alt=None):
        # generate p-values
        if self.alt_type == 'twosided':
            p = scipy.stats.t.sf(np.abs(T), df=self.df)*2
        else:
            # where self.alt is positive, use positive one-sided p-values
            # where self.alt is negative, use negative one-sided p-values 
            # use np.where 
            
            if alt:
                # this is used to enforce the p-values to be all positive or all negative
                if alt in ['+', 'positive']: 
                    p = scipy.stats.t.sf(T, df=self.df)
                elif alt == ['-', 'negative']:
                    p = 1-scipy.stats.t.sf(T, df=self.df)
                else:
                    assert False, f'a specified {alt} must be "+"/"positive" or "-"/"negative"'
            else: 
                p_pos = scipy.stats.t.sf(T, df=self.df) # 1 - this is p_neg
                p = np.where(np.array(self.alt) > 0, p_pos, 1-p_pos)

        return p
        
    def e_function(self, T, **kwargs):
        vars(self).update(kwargs)
        
        method = self.method
        
        if method=='lrt':
            # the likelihood ratio
            if self.alt_type == 'twosided':
                magn_alt = np.abs(self.alt)
                e_neg = scipy.stats.nct.pdf(T, self.df, nc=-1*magn_alt) / scipy.stats.t.pdf(T, self.df) 
                e_pos = scipy.stats.nct.pdf(T, self.df, nc=magn_alt) / scipy.stats.t.pdf(T, self.df) 
                e = 0.5 * (e_neg + e_pos)   # average of the positive-alternative and negative-alternative LR
            else:
                e = scipy.stats.nct.pdf(T, self.df, nc=self.alt) / scipy.stats.t.pdf(T, self.df) 
        elif method=='p2e_cal':
            # p2e calibration
            e = p2e(self.p_function(T), kappa=self.calibrator)
        elif method=='p2e_aon':
            aon_threshold = self.aon_threshold
            e = 1 * (self.p_function(T) <= self.aon_threshold) / self.aon_threshold
        else:
            assert False, f'{method} is not an implemented e-value method'
        return e
    
    def get_suff_stat(self, idx, Z, W):
        neg_idx = np.array(range(self.m))!=idx
        
        suff_stat_1 = Z[neg_idx] - (self.Psi[neg_idx,idx] / self.Psi[idx,idx]) * Z[idx]  
        suff_stat_2 = sum(W**2) + Z[idx]**2/self.Psi[idx,idx]
    
        return (suff_stat_1, suff_stat_2)
    
    def cond_e_sampling(self, idx, suff_stat, n_smpl=1):
        neg_idx = np.array(range(self.m))!=idx
        suff_stat_1, suff_stat_2 = suff_stat
        
        # draw T_j
        T_new = np.random.standard_t(df=self.df, size=n_smpl)
        
        samples = []
        for b in range(n_smpl):
            new_T_smpl = np.zeros(self.m)
            new_T_smpl[idx] = T_new[b]
            new_T_smpl[neg_idx] = (suff_stat_1 * np.sqrt((self.df + new_T_smpl[idx]**2) / (np.diag(self.Psi)[neg_idx]*suff_stat_2)) +
                          new_T_smpl[idx] * self.Psi[neg_idx,idx] / self.Psi[idx,idx])
            
            # add the sample
            samples.append(new_T_smpl)
        # resulting samples is a (n_smpl x m) matrix
        samples = np.array(samples)
            
        return np.apply_along_axis(self.e_function, 1, samples)
    