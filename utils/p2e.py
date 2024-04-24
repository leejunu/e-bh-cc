import numpy as np 
import pandas as pd

import copy
from collections import *

#####

def p2e(pvals, kappa=0.5):
    """
        p2e calibrators defined as in Vovk and Wang (2022)
    """
    p = np.array(pvals)
    if kappa=='ramdas':
        return (1 - p + p*np.log(p)) / (p * (-np.log(p))**2)
    else:
        assert (kappa > 0 ) and (kappa < 1), f'{kappa} must be in (0,1)'
        return kappa * (p**(kappa-1))
    
def e2p(evals):
    e = np.array(evals)
    return np.minimum(1/e, 1)