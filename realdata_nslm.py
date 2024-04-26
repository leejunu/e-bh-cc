import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from quantile_forest import RandomForestQuantileRegressor
import argparse
from cc_utils.conformal import conformal
from utils import CC
from utils.multiple_testing import pBH, eBH, eBH_infty, evaluate
from utils.rej_functions import ebh_rej_function, ebh_infty_rej_function, ebh_union_rej_function
from utils.ci_sequence import get_alpha_cs, get_rho, hedged_cs, asy_cs, asy_log_cs 
from collections import *

"""experiment configurations"""
parser = argparse.ArgumentParser('')
parser.add_argument('--task_id', type=int, default=500)
args = parser.parse_args()
task_id = args.task_id - 1 
params = {'alpha_fdr': [0.1, 0.2, 0.3, 0.4, 0.5], 'seed': range(1,101)}
params_grid = list(ParameterGrid(params))
alpha_fdr = params_grid[task_id]['alpha_fdr']
seed = params_grid[task_id]['seed']
np.random.seed(seed)

confidence_required = True
estimand = "atc"
cc_factor = 11
power_guarantee = False
alpha_cc = alpha_fdr * cc_factor / 10

"""mc details"""
mc_max_samples = 2000 
mc_batch_size = 100 
mc_switchpoint = int(0.6 * mc_max_samples // mc_batch_size * mc_batch_size)

"""read the data"""
data = pd.read_csv('./data/acic_data.csv')

"""preprocess the data"""
X = data.drop(['Z','Y'], axis = 1).values
Z = data['Z'].values
Y = data['Y'].values
n = X.shape[0]

"""the threshold"""
if estimand == "atc": # infer on Y(1) in the control group
    threshold = 0.3
if estimand == "att": # infer on Y(0) in the treated group
    threshold = -0.3
"""split the data into training, calibration and testing sets"""
reind = np.random.permutation(n)

X_train = X[reind[:8000],:]
Z_train = Z[reind[:8000]]
Y_train = Y[reind[:8000]]

Z_cal = Z[reind[8000:9000]]
if estimand == "atc": 
    X_cal = X[reind[8000:9000],:][Z_cal==1,:]
    Y_cal = Y[reind[8000:9000]][Z_cal==1]

if estimand == "att":
    X_cal = X[reind[8000:9000],:][Z_cal==0,:]
    Y_cal = Y[reind[8000:9000]][Z_cal==0]

Z_test = Z[reind[9000:10000]]
if estimand == "atc":
    X_test = X[reind[9000:10000],:][Z_test==0,:]
    Y_test = Y[reind[9000:10000]][Z_test==0]
if estimand == "att":
    X_test = X[reind[9000:10000],:][Z_test==1,:]
    Y_test = Y[reind[9000:10000]][Z_test==1]

m = len(Y_test)

"""fit the models"""
ps_rf = RandomForestClassifier(max_depth = 10)
ps_rf.fit(X_train, Z_train)
reg_rf = RandomForestRegressor()
if estimand == "atc":
    reg_rf.fit(X_train[Z_train == 1,:], Y_train[Z_train ==1])
if estimand == "att":
    reg_rf.fit(X_train[Z_train == 0,:], Y_train[Z_train ==0])

"""compute the nonconformity scores"""
if estimand == "atc":
    score_cal = reg_rf.predict(X_cal)
    score_cal = score_cal[Y_cal <= threshold] #only use the inliers
    score_test = reg_rf.predict(X_test) 
if estimand == "att":
    score_cal = -reg_rf.predict(X_cal)
    score_cal = score_cal[Y_cal >= threshold]
    score_test = -reg_rf.predict(X_test)

"""compute the weights"""
pp = np.mean(Z_train)
ps_cal = ps_rf.predict_proba(X_cal)[:,1]
ps_test = ps_rf.predict_proba(X_test)[:,1]

if estimand == "atc":
    weight_cal = pp * (1-ps_cal) / (1-pp) / ps_cal
    weight_cal = weight_cal[Y_cal <= threshold]
    weight_test = pp * (1-ps_test) / (1-pp) / ps_test

if estimand == "att":
    weight_cal = 1/(pp * (1-ps_cal) / (1-pp) / ps_cal)
    weight_cal = weight_cal[Y_cal >= threshold]
    weight_test = 1/(pp * (1-ps_test) / (1-pp) / ps_test)

cal_set = []
for i in range(len(weight_cal)):
    cal_set.append([weight_cal[i], score_cal[i]])


test_set = []
for i in range(len(weight_test)):
    test_set.append([weight_test[i], score_test[i]])

"""compute the e-values"""
evalue_setting = conformal(alpha_conf=alpha_fdr)
e_original = evalue_setting.e_function(cal_set, test_set)

"""eBH baseline"""
U = np.random.uniform(size=m)
randomized_rej = eBH(e_original/U, alpha_fdr)
rej = eBH(e_original, alpha_fdr)

"""make new CC objects every seed"""

key = 'ebh-CC-union'
CC_dict = {}
CC_dict[key] = defaultdict(list)
CC_dict[key]['CC'] = CC(alpha_cc=alpha_cc, rej_function=ebh_union_rej_function, guarantee=power_guarantee)
cc_rej = eBH(e_original, alpha_cc)

"""make masks"""
mask = (e_original != 0) 
#mask = np.array(mask)
#if len(rej) > 0:
#    mask[rej] = 1
print(f'mask size: {sum(mask)}')


CC_dict[key]['fast_coefs'] = CC_dict[key]['CC'].compute_fast_coefs(e_original)    # differs due to different rej-functions
n_filter = sum(mask)
alpha_cs = get_alpha_cs(
    alpha_fdr, 0.0001,  
    n_filter=n_filter, 
    n_prev_rej=len(cc_rej), 
    power_guarantee=power_guarantee
)                      
rho_cs = get_rho(mc_switchpoint, alpha_cs)

# loop through each hypothesis
for j in range(m):

    if (j in rej and power_guarantee):
        # if j has already been rejected (by the e-values), then it will be rejected as long as we are guaranteed strict improvement
        CC_dict[key]['boost_set'].append(j)
        continue
    elif mask[j] == 0:        
        continue    # filtered out
    
            
    # make sufficient statistic
    suff_stat = evalue_setting.get_suff_stat(idx=j, cal_set=cal_set, test_set=test_set)  
    unordered_set, _ = suff_stat
    unordered_set_array = np.array(unordered_set).T 
    unordered_set_weights = unordered_set_array[0]
    
    # adaptively get mc samples for evaluating phi(;S_j)
    # if we have made a decision for each CC method
    # terminate the sampling early
    #made_decision_set = []
    made_decision = False
    up_transf = m/alpha_cc
    lo_transf = sum(unordered_set_weights)/min(unordered_set_weights)   # mc samples originally lie in [-lo, up]
                
    CC_dict[key]['exact_CS'] = hedged_cs(m=lo_transf/(lo_transf+up_transf), theta=0.5, alpha=alpha_cs, c=0.5, interval=True)
    CC_dict[key]['asymp_CS'] = asy_cs(rho_cs, alpha_cs, m=lo_transf/(lo_transf+up_transf))

    CC_dict[key]['conf_array'] = np.zeros(m)   # keeps track of which are confident (1 for confident)
    CC_dict[key]['mc_samples_array'] = np.zeros(m)
        
    for k in range(0, mc_max_samples // mc_batch_size):
        #if len(made_decision_set) == len(CC_KEYS):
        #    break    # we're done here
        if made_decision:
            break

        # sample conditionally given S_j under H_j
        e_samples = evalue_setting.cond_e_sampling(
            idx=j, suff_stat=suff_stat, n_smpl=mc_batch_size,
        )

        # for each CC method, get the e-samples and supplement the existing mc samples for estimating phi
        #for key in CC_KEYS:
        #    if key in made_decision_set:
        #        continue    # we already made a decision for this method, continue to the next one
        cc_object = CC_dict[key]['CC']
        fast_coefs = CC_dict[key]['fast_coefs']
        if ((k+1) * mc_batch_size) >= mc_switchpoint:
            cs = CC_dict[key]['asymp_CS']
            if len(cs.data) == 0:
                # the CS needs to see all the data up to this point
                cs.update(CC_dict[key]['exact_CS'].data)
        else:
            cs = CC_dict[key]['exact_CS']

        mc_samples = cc_object.cc_bootstrapping(e_samples, fast_coefs[j], j)    # mc samples generated by CC object's bootstrapping fn

        # test inclusion of zero (transformed) in the the anytime CI
        not_included = not cs.update( (mc_samples + lo_transf) / (lo_transf+up_transf))   # scale the data so it lies in [0,1] (0 is transformed to alpha/m)

        # if the CS does not contain at this point, we can accept/reject with confidence
        if not_included:
            made_decision = True
            CC_dict[key]['conf_array'][j] = 1
            if np.mean(cc_object.mc_samples[j]) < 0:
                # since this case implies mean is less than 0, boost
                CC_dict[key]['boost_set'].append(j)
            CC_dict[key]['mc_samples_array'][j] = ((k+1) * mc_batch_size)
        # instead, if we run out of mc sample budget, forced to make a decision
        elif (k == mc_max_samples//mc_batch_size - 1):
            made_decision = True
            CC_dict[key]['mc_samples_array'][j] = ((k+1) * mc_batch_size)
            # when confidence is required to boost index j, we are forced to do nothing at this point
            # otherwise, we can boost j when mean is negative, but without confidence
            if (np.mean(cc_object.mc_samples[j]) < 0):
                CC_dict[key]['no_conf_boost_set'].append(j)
                if (not confidence_required):
                    CC_dict[key]['boost_set'].append(j)

"""store the results"""
results = pd.DataFrame([len(rej), len(CC_dict[key]['boost_set']), len(CC_dict[key]['no_conf_boost_set']), len(randomized_rej), m])
results.to_csv("results/fixed_" + estimand + "_cc_" + str(cc_factor) + "_fdr_"+str(int(alpha_fdr*10))+"_seed_"+str(seed)+".csv", index=False)
