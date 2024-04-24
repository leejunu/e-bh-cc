import numpy as np
import pandas as pd

import scipy.special

import copy
from collections import *

#####
    
class hedged_cs():
    """
    For a set of i.i.d. observations from some distribution P with bounded
    support, constructs an exact anytime-valid sequence of confidence
    intervals. Uses Theorem 3 in https://arxiv.org/pdf/2010.09686.pdf.
    """
    def __init__(self, m, theta, alpha, c, interval=True):
        self.data = [] # empty data
        self.T = 0  # starts out having seen zero data
        
        self.m = m
        self.alpha = alpha
        self.theta = theta
        self.c = c
        
        self.mu_list = [1/2]
        self.var_list = [1/4]
        
        self.K_plus_list = [1]
        self.K_minus_list = [1]
        
        self.use_interval = interval
        
    def update(self, data):
        self.data.extend(list(data))
        
        t_range = np.arange(len(self.mu_list)+1, len(data) + len(self.mu_list)+1)
        
        # calculate running mean/variance
        previous_sum = self.mu_list[-1] * len(self.mu_list)
        new_mean_range = (previous_sum + np.cumsum(data)) / t_range
        previous_sum_of_sq = self.var_list[-1] * len(self.var_list)
        new_var_range = ( previous_sum_of_sq + np.cumsum((np.array(data)-new_mean_range)**2) ) / t_range
        
        self.mu_list.extend(list(new_mean_range))
        self.var_list.extend(list(new_var_range))
        
        # calculate lambda parameter
        new_lam_range = np.sqrt( 2 * np.log(2/self.alpha) / (self.var_list[-1-len(t_range):-1] * np.log(t_range) * (t_range-1)) )
        
        # calculate lambda_p/m
        lam_plus_list = np.minimum(self.c/self.m, np.abs(new_lam_range)) if self.m !=0 else np.abs(new_lam_range)
        lam_minus_list = np.minimum(self.c/(1-self.m), np.abs(new_lam_range)) if self.m !=1 else np.abs(new_lam_range)
        
        # calculate K_p/m
        new_K_plus = np.cumprod(1 + (np.array(data) - self.m) * lam_plus_list) * self.K_plus_list[-1]
        new_K_minus = np.cumprod(1 - (np.array(data) - self.m) * lam_minus_list) * self.K_minus_list[-1]
        if self.use_interval:
            new_K_combo = np.maximum(self.theta * new_K_plus, (1-self.theta) * new_K_minus)
        else:
            new_K_combo = (self.theta * new_K_plus + (1-self.theta) * new_K_minus)
        
        # update
        self.K_plus_list.append(new_K_plus[-1])
        self.K_minus_list.append(new_K_minus[-1])
        self.T = len(self.data)
        
        # return sum(new_K_combo < 1/self.alpha) == len(new_K_combo)
        return sum(new_K_combo >= 1/self.alpha)==0
    
    def reset(self):
        self.data = [] # empty data
        self.T = 0  # starts out having seen zero data
        
#         self.m = m
#         self.alpha = alpha
#         self.theta = theta
#         self.c = c
        
        self.mu_list = [1/2]
        self.var_list = [1/4]
        
        self.K_plus_list = [1]
        self.K_minus_list = [1]
        
        # self.use_interval = interval
         

class asy_cs():
    """
    For a set of i.i.d. observations from some distribution P of finite variance, 
    computes an asymptotic anytime-valid sequence of confidence intervals (i.e., 
    the coverage probability for the mean approaches 1-alpha with more and more
    samples. Uses Theorem 2.2 in https://arxiv.org/pdf/2103.06476.pdf.
    """
    def __init__(self, rho, alpha, m=None):
        self.data = []
        self.T = 0
        self.m = m
        
        self.rho = rho
        self.alpha = alpha
        
        self.interval = [None, None]
        
        self.mu_list = [0]
        self.var_list = [0]
        
    def update(self, data):
        self.data.extend(list(data))
        t = len(self.data) + self.T
        
        new_mu = (self.mu_list[-1] * self.T + sum(data)) / (t)
        new_var = np.mean( np.array(self.data)**2)-new_mu**2
        self.mu_list.append(new_mu)
        self.var_list.append(new_var)
        
        B = np.sqrt(new_var * 2*(t*self.rho**2+1)/(t**2*self.rho**2) * np.log(np.sqrt(t*self.rho**2)/self.alpha))
        self.interval = [new_mu - B, new_mu + B]
        
        # update length
        self.T = t
        
        if self.m != None:
            return self.interval[0] <= self.m and self.interval[1] >= self.m
        
        else:
            return None
    
    def reset(self):
        self.data = []
        self.T = 0
        self.interval = [None, None]
        
        self.mu_list = [0]
        self.var_list = [0]
        
        
class asy_log_cs():
    """
    A version of asy_cs with log-log convergence rate. Uses 
    Proposition 2.3 in https://arxiv.org/pdf/2103.06476.pdf.
    """
    def __init__(self, alpha, m=None):
        self.data = []
        self.T = 0
        self.m = m

        self.alpha = alpha
        
        self.interval = [None, None]
        
        self.mu_list = [0]
        self.var_list = [0]
        
    def update(self, data):
        self.data.extend(list(data))
        t = len(self.data) + self.T
        
        new_mu = (self.mu_list[-1] * self.T + sum(data)) / (t)
        new_var = np.mean( np.array(self.data)**2)-new_mu**2
        self.mu_list.append(new_mu)
        self.var_list.append(new_var)
        
        B = 1.7 * np.sqrt(new_var / t * (0.72*np.log(10.4/self.alpha) + np.log(np.log(2*t))))
        self.interval = [new_mu - B, new_mu + B]
        
        # update length
        self.T = t
        
        if self.m != None:
            return self.interval[0] <= self.m and self.interval[1] >= self.m
        
        else:
            return None
        
    def reset(self):
        self.data = []
        self.T = 0
        self.interval = [None, None]
        
        self.mu_list = [0]
        self.var_list = [0]
            
# util for correct rho for asymptotic CS
def get_rho(t_star, alpha):
    '''
    A method to choose the parameter rho for asy_cs such that the 
    resulting CS will be tightest at time t_star.
    '''
    z = np.exp(-1)*alpha**2 * (-1)
    return np.sqrt((-1 * scipy.special.lambertw(z, k=-1) - 1)/t_star)

# util for getting correct alpha_cs level
def get_alpha_cs(alpha_fdr, rel_mc_error, n_filter, n_prev_rej, power_guarantee):
    if n_filter==0:
        return 1000    # value doesn't matter, never will be used
    
    alpha_corrected = alpha_fdr * rel_mc_error
    
    # if the filter only consists of previous rejections and power is guaranteed, then we won't run CC since there is no point
    if ((n_filter == n_prev_rej) and power_guarantee) or n_filter==0:
        return 2000    # value doesn't matter, never will be used
    
    # otherwise, we want to return alpha_corrected * |R| / n_tested
    # the n_tested does not include prev rejections when power is guaranteed
    factor = max(1, n_prev_rej) / (n_filter -  (n_prev_rej if power_guarantee else 0) )
    return (factor * alpha_corrected)