import numpy as np
import pandas as pd

import scipy.stats

import copy
from collections import *

#####

def eBH(e, alpha):
    '''
    Runs the eBH procedure to control FDR at given level alpha.
        
        Parameters:
            e : The e-values on which to run the procedure.
            alpha : The level at which to control FDR.
            
        Returns:
            rej: The rejection set; the selected indices that have rejected nulls.
        
    '''
    m = len(e)
    e_sort = np.sort(e)[::-1] # descending order
    khat = 0
    for k in range(m,0,-1):
        if e_sort[k-1] >= m/(alpha*k):
            khat = k
            break
    if khat == 0:
        return np.array([])
    else:
        return np.nonzero(np.array(e) >= m/(alpha*khat))[0]
    
def eBH_infty(e, alpha, idx=None):
    '''
    Given e-values e and for each index j, runs the eBH procedure on e' at level alpha, where
    e' is e with e_j replaced by infinity. 
        
        Parameters:
            e : The e-values on which to run the procedure.
            alpha : The level at which to run eBH.
            idx : (Optional) The specific index at which to run the eBH_infty procedure.
            
        Returns:
            n_rej: A vector such that the jth component contains |eBH(e', alpha)|, where e' is e with e_j replaced by infinity.
        
    '''  
    m = len(e)
    if (idx != None):
        # eBH_infty for a specific idx
        e_copy = copy.deepcopy(e)
        e_copy[idx] = np.inf
        return len(eBH(e_copy, alpha))
    
    e_sort = np.sort(e)[::-1] # descending order
    ranks = m - np.argsort(e).argsort() # ranks (1, 2, ..., m)
    
    # initial eBH
    khat = 0
    for k in range(m,0,-1):
        if e_sort[k-1] >= m/(alpha*k):
            khat = k
            break
    
    # now, solve for what happens in the case where e_j --> infty
    S = 1+np.nonzero(   e_sort >= m/(alpha * (np.array(range(m))+2)  ))[0]    # {k : e_(k) >= m/(a(k+1))}; add 1 because it's zero indexed
    s_hat_list = 1+np.array([max(S[S<=rank-1], default=0) for rank in ranks])  # list[j] = max in S less than rank(e_j), +1 for e_j=inf 
    
    # if khat >= rank, then regular eBH threshold is preserved
    n_rej = np.where(khat >= ranks, khat*np.ones(m), s_hat_list) # list[j] = R_alpha^{e_j-->infty)    
    return n_rej
    
def pBH(p, alpha):
    '''
    Runs the BH procedure to control FDR at given level alpha.
        
        Parameters:
            p : The p-values on which to run the procedure.
            alpha : The level at which to control FDR.
            
        Returns:
            rej: The rejection set; the selected indices that have rejected nulls.
        
    '''
    
    # scipy_rej = np.nonzero(sp.stats.false_discovery_control(ps=p) < alpha)[0]
    # return scipy_rej
    
    m = len(p)
    p_sort = np.sort(p) # ascending order
    khat = 0
    for k in range(m, 0, -1):
        if (p_sort[k-1] * m / k) <= alpha:
            khat = k
            break
    if khat == 0:
        return np.array([])
    else:
        return np.nonzero(np.array(p) <= (alpha*khat/m))[0]
    
#####

# "inverse" operations
def eBH_min_alpha(e):
    m = len(e)
    e_sort = np.sort(e)[::-1]
    # return the minimum value of m/(ke_k), which is the min alpha needed for >0 discoveries
    return min(m / (e_sort * np.array(range(1,1+m))))  

#####

# evaluation operations
def evaluate(rejections, H_1):
    """
    Given the rejections and true nonnulls (H_1), returns the FDP and power of the procedure.
    """
    m_1 = len(H_1)
    R = len(rejections)
    
    if R == 0:
        return {'fdp': 0, 'power': 0}
    
    td = len(set(rejections) & set(H_1))
    return {'fdp': 1 - td/max(R,1), 'power': td/max(m_1,1)}