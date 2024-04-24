import numpy as np
import pandas as pd

import scipy.stats
import sklearn as skl

import copy
import warnings
from collections import *

#####

def lasso_mask(X, y, **LassoCV_kwargs):
    '''
    Masking indices based on whether their corresponding Lasso coefficient is nonzero.
    '''
    model = skl.linear_model.LassoCV(**LassoCV_kwargs).fit(X,y)
    mask = (model.coef_ != 0)
    
    return mask, model

def regression_pvalue_mask(X, y, alpha, **Linear_kwargs):
    '''
    Masking indices based on whether their corresponding t-stat p-value is significant at alpha.
    '''
    fit_intercept = True
    if 'fit_intercept' in Linear_kwargs:
        fit_intercept = Linear_kwargs['fit_intercept']
    
    n, m = X.shape
    model = skl.linear_model.LinearRegression(**Linear_kwargs).fit(X,y)
    if fit_intercept == True:
        # in this case, need to augment X with all ones
        X_temp = np.hstack((np.ones((n,1)), X))
        params = np.append(model.intercept_, model.coef_)
    else:
        X_temp = X
        params = model.coef_
        
    # make predictions and calculate the MSE
    pred = model.predict(X)
    mse = (sum((y-pred)**2))/(n-len(params))
    
    sd_array = np.sqrt(copy.deepcopy( mse*np.diag( np.linalg.inv(np.dot(X_temp.T, X_temp)) ) ))
    tstat_array = params / sd_array
    
    p = np.array(    # twosided p-value array
        [2*(1-scipy.stats.t.cdf(np.abs(tstat), df=(n-len(params)))) for tstat in tstat_array]
    ) 
    p = p[-m:]    # return only the p-values associated with non-intercept
    mask = (p <= alpha) 
    
    return mask, p   # return the mask = 1 where p is significant at alpha
