import numpy as np
import pandas as pd
from numba import jit

import copy
from collections import *

from .abstract_test_setting import abc_setting 

#####

class conformal(abc_setting):
    def __init__(             # this is where one should determine all aspects of the e-value function
        self, 
        alpha_conf
    ):
        super(conformal,self).__init__('conformal')
        self.alpha_conf = alpha_conf
        
    def p_function(self, cal_set, test_set):
        # generate p-values
        p = None
        return p
        
    # def e_function(self, cal_weights, cal_scores, test_weights, test_scores):
    def e_function(self, cal_set, test_set):
        # unpack
        cal_array = np.array(cal_set).T
        test_array = np.array(test_set).T
        cal_weights = cal_array[0]; cal_scores = cal_array[1]
        test_weights = test_array[0]; test_scores = test_array[1] 
        
        self.n = len(cal_weights)
        self.m = len(test_weights)
        
        sum_cal_weights = sum(cal_weights)
        
        # find stopping time for each j
        T_array = np.ones(self.m) * np.inf
        scores = np.concatenate((cal_scores, test_scores)).flatten()   # all scores
        decreasing_thresholds = copy.deepcopy( scores[np.argsort(-scores)] )  
        
#         for t in decreasing_thresholds:
#             numer_1 = test_weights + sum(cal_weights * (cal_scores>=t))  # broadcasted to length m
#             denom_1 = max(1, sum(test_scores>=t))     
#             denom_2 = test_weights + sum_cal_weights
            
#             ratio = np.divide(self.m * numer_1 , denom_1 * denom_2)
#             where_less_than_alpha = np.nonzero(ratio <= self.alpha_conf)[0]  # j in [m] where t is valid
#             T_array[where_less_than_alpha] = t    # where the t is valid, set T_j = t

#         # use fast numba function for finding T_j, j \in [m]
#         T_array = Tj_numbahelper(
#             T_array, decreasing_thresholds, 
#             sum_cal_weights, cal_weights, cal_scores, test_weights, test_scores, 
#             self.alpha_conf
#         )
            
#         # now construct the e-values
#         assert len(T_array) == len(test_set), 'there should be a T_j for each Z_j in the test set'
#         numer = (test_weights + sum_cal_weights)  * (test_scores >= T_array)    # length m
#         denom = test_weights + (cal_weights * (cal_scores>=T_array.reshape(self.m, 1))).sum(axis=1)
#         e = numer/denom

        e = e_numbahelper(
            T_array, decreasing_thresholds, 
            sum_cal_weights, cal_weights, cal_scores, test_weights, test_scores, 
            self.alpha_conf
        )
        
        return e 
    
    def get_suff_stat(self, idx, cal_set, test_set):
        unordered_set = copy.deepcopy(cal_set)
        unordered_set.append(test_set[idx])   # this shouldn't be a numpy array in the first place
        test_set_with_hole = copy.deepcopy(test_set)
        test_set_with_hole[idx] = [None, None]
        
        # essentially, test_set_with_hole has a None in spot idx, which we will fill through random choice in unordered
        return unordered_set, test_set_with_hole    
    
    def cond_e_sampling(self, idx, suff_stat, n_smpl=1):
        unordered_set, test_set_with_hole = suff_stat
        
        unordered_array = np.array(unordered_set).T
        unordered_weights = unordered_array[0]   # get the weights in the unordered collection w_1, ..., w_n, w_{n+j}
        normalized_unordered_weights = unordered_weights / sum(unordered_weights)    # these are the probabilities
        
        # choose what fills the hole in test_set_with_hole based on the normalized weights as probabilities
        resamples_idxs = np.random.choice(len(unordered_set), size=n_smpl, replace=True, p=normalized_unordered_weights)   
        
        e_samples = []
        for b in resamples_idxs:
            recons_cal_set = copy.deepcopy(unordered_set)
            recons_test_set = copy.deepcopy(test_set_with_hole)
            recons_test_set[idx] = recons_cal_set[b]
            recons_cal_set.pop(b)
            
            e = self.e_function(recons_cal_set, recons_test_set)
            e_samples.append(e)
            
        return np.array(e_samples)
    
@jit(nopython=True)
def Tj_numbahelper(T_array, decr_thresholds, sum_cal_weights, cal_weights, cal_scores, test_weights, test_scores, alpha):
    m = len(T_array)
    for t in decr_thresholds:
        numer_1 = test_weights + sum(cal_weights * (cal_scores>=t))  # broadcasted to length m
        denom_1 = max(1, sum(test_scores>=t))     
        denom_2 = test_weights + sum_cal_weights

        ratio = np.divide(m * numer_1 , denom_1 * denom_2)
        where_less_than_alpha = np.nonzero(ratio <= alpha)[0]  # j in [m] where t is valid
        T_array[where_less_than_alpha] = t 
        
    return T_array

@jit(nopython=True)
def e_numbahelper(T_array, decr_thresholds, sum_cal_weights, cal_weights, cal_scores, test_weights, test_scores, alpha):
    m = len(T_array)
    for t in decr_thresholds:
        numer_1 = test_weights + sum(cal_weights * (cal_scores>=t))  # broadcasted to length m
        denom_1 = max(1, sum(test_scores>=t))     
        denom_2 = test_weights + sum_cal_weights

        ratio = np.divide(m * numer_1 , denom_1 * denom_2)
        where_less_than_alpha = np.nonzero(ratio <= alpha)[0]  # j in [m] where t is valid
        T_array[where_less_than_alpha] = t 
        
    # after T_array made, can construct e-values
    numer = (test_weights + sum_cal_weights)  * (test_scores >= T_array)    # length m
    denom = np.zeros(m)
    for j in range(m):
        Tj = T_array[j]
        denom[j] = test_weights[j] + sum(cal_weights * (cal_scores>=Tj))
    e = numer/denom
        
    return e

@jit(nopython=True)
def storey_numbahelper(m, n, decr_thresholds, cal_weights, cal_scores, test_weights, test_scores, alpha, weighted=True, idx=None):
    # only implemented unweighted case
    # follows construction of bashari et al
    
    if idx != None:
        w_j = test_weights[idx]
    else:
        w_j = 0 if weighted else 1
    
    inf_t = np.inf
    for t in decr_thresholds:
        # fdp = m/n * sum(cal_weights *(cal_scores>=t)) / max(1, sum(test_weights *(test_scores>=t)))
        fdp = m / (np.sum(cal_weights)+w_j) * sum(w_j+cal_weights *(cal_scores>=t)) / max(1, sum(test_scores>=t))
        if fdp <= alpha:
            inf_t = t
            
    pi_hat_inverse = m/(np.sum(cal_weights)+w_j) * (w_j+sum(cal_weights *(cal_scores<=t))) / max(1, sum(test_scores<=t))
    return pi_hat_inverse

def storey_correction(m, n, cal_weights, cal_scores, test_weights, test_scores, alpha, weighted=True, idx=None):
    scores = np.concatenate((cal_scores, test_scores)).flatten()    
    decr_thresholds = copy.deepcopy( scores[np.argsort(-scores)] ) 
    
    return storey_numbahelper(m, n, decr_thresholds, cal_weights, cal_scores, test_weights, test_scores, alpha, weighted, idx)

# @jit(nopython=True)
# def storey_numbahelper(m, n, decr_thresholds, cal_weights, cal_scores, test_weights, test_scores, alpha):
#     # only implemented unweighted case
#     # follows construction of bashari et al

#     # scores = np.concatenate((cal_scores, test_scores)).flatten()    
#     # decreasing_thresholds = copy.deepcopy( scores[np.argsort(-scores)] )  
#     inf_t = np.inf
#     for t in decr_thresholds:
#         fdp = m/n * sum((cal_scores>=t)) / max(1, sum((test_scores>=t)))
#         if fdp <= alpha:
#             inf_t = t
            
#     pi_hat_inverse = m/(n+1) * (1+sum((cal_scores<=t))) /  max(1, sum((test_scores<=t)))
#     return pi_hat_inverse

# def storey_correction(m, n, cal_weights, cal_scores, test_weights, test_scores, alpha):
#     scores = np.concatenate((cal_scores, test_scores)).flatten()    
#     decr_thresholds = copy.deepcopy( scores[np.argsort(-scores)] ) 
    
#     return storey_numbahelper(m, n, decr_thresholds, cal_weights, cal_scores, test_weights, test_scores, alpha)