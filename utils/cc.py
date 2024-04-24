import numpy as np
import scipy as sp
import pandas as pd

import copy
import warnings
from collections import *

from .multiple_testing import eBH

class CC():
    def __init__(self, alpha_cc, rej_function, budget_type=None, guarantee=False, alpha_fdr=None, **rej_kwargs):
        self.alpha_cc = alpha_cc
        
        self.budget_type = budget_type if budget_type != None else 'default'; assert self.budget_type in ['default', 'static', 'truncated']
        
        self.guarantee = guarantee
        self.rej_function = lambda e : rej_function(e, idx=None, alpha=self.alpha_cc, **rej_kwargs)
        self.rej_function_for_idx = lambda e, idx : rej_function(e, idx=idx, alpha=self.alpha_cc, **rej_kwargs)
        
        self.alpha_fdr = alpha_cc
        if not isinstance(alpha_fdr, type(None)):
            self.alpha_fdr = alpha_fdr   # if the FDR is different, this should be specified and override the previous value
        
        self.mc_samples = defaultdict(list)    # this will keep the samples used to estimated phi(c;S_j)
        
    def compute_fast_coefs(self, e_original):
        """
        For each idx, computes the fast-evaluation coefficient.
        """
        m = len(e_original)
        rej_vec = self.rej_function(e_original)
        
        # fast-evaluation coefficient calculation
        ratio = np.divide(m/self.alpha_cc, np.maximum(rej_vec,1))
        coefs = np.divide(ratio, e_original, out=(np.ones(m)*np.inf), where=(e_original!=0))  
        
        return coefs
        
    def cc_bootstrapping(self, e_samples, fast_coef, idx, **budget_kwargs):
        """
        For each sample of e | S_j, computes the term inside the expectation of phi(fast_coef; S_j)
        
            Parameters:
                idx: the index of the hypothesis being tested
                e_samples: the samples of e | S_j under H_j (usually constructed through some setting
                           object's cond_e_sampling method)
            
            Returns:
                bootstrap_values: a vector of values of the computed term inside phi(fast_coef; S_j)
        """
        
        shape = np.array(e_samples).shape
        if len(shape) == 1:
            # only one sample of e | S_j
            e_samples = e_sample.reshape(-1,1).T

        bootstrap_values = []
        for e_sample in e_samples:
            m = len(e_sample.flatten())
            R = self.rej_function_for_idx(e_sample,  idx=idx)
            
            if self.budget_type == 'default':
                budget = e_sample[idx]
            elif self.budget_type == 'static':
                budget = 1
            elif self.budget_type == 'truncated':
                trunc_func = budget_kwargs['trunc_func']
                budget = trunc_func(e_sample[idx])    # T(alpha * boost_factor * e_j)

            ratio = (m/self.alpha_cc)/ max(R, 1)    # scalar
            # indic = True if (fast_coef == np.inf) else (fast_coef * e_sample[idx] >= ratio)    # if fast_coef is infinite, then reject
            indic = (fast_coef * e_sample[idx] >= ratio)
            bootstrap_values.append(ratio * indic * (R > 0) - budget)  
            # note: we want the expectation of the above to be less than 0
            
        self.mc_samples[idx].extend(list(bootstrap_values))
            
        return np.array(bootstrap_values)
    
    def boost(self, e_original, idx_list, rej=None):
        """
        Given the original e-values and set of indices, boost at those indices.
        """
        m = len(e_original)
        R = self.rej_function(e_original)
        ratio = np.divide((m/self.alpha_cc), np.maximum(R, 1))    # vector
        
        # initialize boosted e-values
        e_boosted = np.zeros(m)
        
        boost_set = copy.deepcopy(idx_list)
        
        e_boosted[boost_set] = (ratio * (R > 0))[boost_set]  # boost set gets the boost
        
        if self.guarantee:
            # for some procedures, we can guarantee rejecting significant e-values for e_original
            if isinstance(rej, type(None)):
                ebh_rej_set = eBH(e_original, self.alpha_fdr)  # not actually correct since alpha_cc != alpha_fdr
            else: 
                ebh_rej_set = rej 
                
            if len(ebh_rej_set) > 0:
                e_boosted[ebh_rej_set] = e_original[ebh_rej_set] 
        
        return e_boosted
    
    def clear_mc_samples(self, idx):
        if idx == None:
            self.mc_samples = defaultdict(list)
        else:
            self.mc_samples[idx] = []
            
def truncation_function(x, m):
    # returns the truncation function T(x); Eqn (7) in Wang and Ramdas (2021) 
    # https://arxiv.org/pdf/2009.02824.pdf
    denom_ceil = m//x+1
    
    return (x >= 1) * m/denom_ceil  
    
    
    