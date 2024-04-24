import numpy as np

"""
All functions have been directly copy and pasted from the Github repository of Jin and Candes (2023), titled 
`Model-free selective inference under covariate shift via weighted conformal p-values`.
Specifically, this is the collection of data-generating functions used for their outlier detection
simulation.
"""

def gen_data(Wset, n, a): 
    Wi = Wset[np.random.choice(range(50), n),:]
    Vi = np.random.normal(size=n*50).reshape((n,50))
    Xi = np.sqrt(a) * Vi + Wi
    return(Xi)
    
def gen_weight(Xi, theta):
    linx = Xi @ theta   
    wx = np.exp(linx) / (1+ np.exp(linx))    
    return(wx)
    
def gen_data_weighted(Wset, n, a, theta):
    Wi = Wset[np.random.choice(range(50), n),:]
    Vi = np.random.normal(size=n*50).reshape((n,50))
    Xi = np.sqrt(a) * Vi + Wi
    wx = gen_weight(Xi, theta)
    sel = np.random.binomial(n=1, p=wx[:,0])
    X_sel = Xi[sel==1,:]
    X = X_sel
    
    while X.shape[0] < n:
        Wi = Wset[np.random.choice(range(50), n),:]
        Vi = np.random.normal(size=n*50).reshape((n,50))
        Xi = np.sqrt(a) * Vi + Wi
        wx = gen_weight(Xi, theta)
        sel = np.random.binomial(n=1, p=wx[:,0])
        X_sel = Xi[sel==1,:]
        X = np.concatenate((X, X_sel))
    
    X = X[range(n),:]
    
    return(X)