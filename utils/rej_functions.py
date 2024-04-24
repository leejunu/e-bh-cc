import numpy as np
import pandas as pd

import copy
from collections import *

from .multiple_testing import eBH, eBH_infty

#####

def ebh_rej_function(e, idx, alpha):
    """
    Returns a vector of length len(e) with each element equal to |eBH(e, alpha)|
    """
    R = len(eBH(e, alpha))
    if idx != None:
        return R
    else:
        return np.ones_like(e) * R
    
def ebh_infty_rej_function(e, idx, alpha):
    """
    Returns a vector of length len(e) with each element equal to |eBH_infty(e, alpha, idx)|
    """
    if idx != None:
        e_copy = copy.deepcopy(e)
        e_copy[idx] = np.inf
        return len(eBH(e_copy, alpha))
    else:
        return eBH_infty(e, alpha, idx=None)
    
def ebh_union_rej_function(e, idx, alpha):
    """
    Returns a vector of length len(e) with each element equal |eBH(e, alpha) U {idx}|
    """
    m = len(e)
    rej_set = eBH(e, alpha)
    if idx != None:
        return len(rej_set) + (idx not in rej_set)
    else:
        return np.ones_like(e) * len(rej_set) + ~np.isin(range(m), rej_set)
    
def ebh_union_rej_function_clipped(e, idx, alpha, lower=None, upper=None):
    m = len(e)
    rej_set = eBH(e, alpha)
    
    lb = 0 if not lower else lower
    ub = len(e) if not upper else upper
    
    if idx != None:
        return min(max(len(rej_set) + (idx not in rej_set), lb), ub)
    else:
        a = np.ones_like(e) * len(rej_set) + ~np.isin(range(m), rej_set)
        return np.clip(a, lb, ub)