import numpy as np
import pandas as pd

import scipy.linalg
from sklearn import svm

import argparse
import importlib
import time
import datetime
import os
import sys
import copy
from collections import *

from utils import CC
from utils.multiple_testing import pBH, eBH, eBH_infty, evaluate
from utils.p2e import p2e
from utils.rej_functions import ebh_rej_function, ebh_infty_rej_function, ebh_union_rej_function
from utils.ci_sequence import get_alpha_cs, get_rho, hedged_cs, asy_cs, asy_log_cs 
from utils.generating_data import *

from cc_utils.conformal import conformal

#####
CC_KEYS = [
    'ebh-CC', 
    'ebh-CC-infty', 
    'ebh-CC-union'
]

### arguments
parser = argparse.ArgumentParser(description='Script for eBH and eBH-CC z-testing experiments.')

# experiment config
parser.add_argument('--name', type=str, required=True, help='the name associated with the experiment.')
parser.add_argument('--n_exp', type=int, default=100, help='number of experiments (seeds) to run.')
parser.add_argument('--start_seed', type=int, default=0, help='the seed to with which to start when running the total experiment.')
parser.add_argument('--end_seed', type=int, default=None, help='the last seed in the total experiment.')

# experiment details
parser.add_argument('--m', type=int, default=500, help='number of observations in the test dataset.')
parser.add_argument('--n', type=int, default=1000, help='number of observations in the calibration dataset.') 
parser.add_argument('--n_train', type=int, default=None, help='number of observations in the training dataset.') 
parser.add_argument('--prop_outliers', type=float, default=0.1, help='frequency of outliers in the test dataset.') 

parser.add_argument('--amp', type=float, required=True, help='the signal for the true alternatives.')  
parser.add_argument('--theta_type', type=int, required=True, help='how to design the weight w(X)= sigmoid(theta^T X)')

# alpha details
parser.add_argument('--alpha_fdr', type=float, default=0.1, help='the level at which to control FDR.')  

# conformal details
parser.add_argument('--weight_scores', default=False, action='store_true', help='turn on to weight the classifier scores by w(x)')

# mc details
parser.add_argument('--mc_max_samples', type=int, default=1000, help='max number of MC iterations to estimate sign(phi(c; S_j)).')
parser.add_argument('--mc_batch_size', type=int, default=0, help='batch size for batched MC samples used in confdeince sequence')
parser.add_argument('--rel_mc_error', type=float, default=None, help='specifies q where if a is the desired FDR control, \
                                                                           a + qa is the theoretical FDR control after \
                                                                           estimating sign(phi(c; S_j)).')
parser.add_argument('--exact_cs_switchpoint', type=float, default=1.0, help='proportion of mc_samples to use for exact_cs before switching \
                                                                             to asymp_cs, which is tighter but only guarantees asymptotic coverage.')
parser.add_argument('--confidence_required', default=False, action='store_true', help='turn on to require the CS to be confident before boosting an index.')

# cc details
parser.add_argument('--power_guarantee', default=False, action='store_true', help='turn on to return the set R U R_CC, which may not always \
                                                                                   control the FDR theoretically.')
parser.add_argument('--acc_factor', type=float, default=1, help='determines the alpha parameter to use in the boosted e-value construction.')

args = parser.parse_args()
print(args)
                          
#####

### make directories
filepath = f'results/{args.name}/'
os.makedirs(filepath, exist_ok=1)    # name should be the results filepath



### process the arguments
# config
seeds = []
if args.end_seed == None:
    seeds = list(range(args.start_seed, args.start_seed+args.n_exp))
else:
    assert args.start_seed < args.end_seed, 'start seed must be less than end seed'
    seeds = list(range(args.start_seed, args.end_seed+1))
    # filepath += f'seeds_{args.start_seed}-{args.end_seed}/'
    # os.makedirs(filepath, exist_ok=1) 
n_exp = len(seeds)

# details
m = args.m        # number of hypotheses
n = args.n        # length of data matrix
amp = args.amp    # (unnormalized) signal strength

n_train = n if args.n_train == None else args.n_train    # can specify training amount 

prop_outliers = args.prop_outliers

THETA = np.zeros(50) 
if args.theta_type==1:
    # the original setting
    p_theta = 5
    THETA[:p_theta,] = 0.1 
elif args.theta_type==2:
    # p_theta = 20
    # THETA[:p_theta,] = 0.1 # np.linspace(1.0, 0.1, num=p_theta)
    # THETA[:p_theta//2,] += 0.1
    # THETA[:p_theta//4,] += 0.1
    THETA = np.linspace(0, 1, len(THETA)+1)[1:]
elif args.theta_type==3:
    p_theta = 6 
    THETA[:p_theta,] = np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1])   
else:
    assert False, f'{args.theta_type} is not a valid argument for theta_type'
THETA = THETA.reshape((50,1))

# Wset
np.random.seed(369)   # Wset should be same for all
Wset = np.random.uniform(size=(50,50)) * 6 - 3

n_outliers = int(np.ceil(m * prop_outliers))
nonzero = np.array(range(m))[:n_outliers]    # the indices of the outliers
nulls = np.array(range(m))[n_outliers:]      # the indices of the inliers 

print(f'true nonnulls: {sorted(nonzero)}')


# alpha details
alpha_fdr = args.alpha_fdr    # fdr target
alpha_cc = alpha_fdr  *  args.acc_factor   # default is alpha_fdr times 1
use_cs = True if args.rel_mc_error != None else False

print(f'fdr {alpha_fdr} - alpha_cc {alpha_cc}')

# mc details
mc_max_samples = args.mc_max_samples
mc_batch_size = args.mc_max_samples if args.mc_batch_size<=0 else args.mc_batch_size
mc_switchpoint = int(args.exact_cs_switchpoint * mc_max_samples // mc_batch_size * mc_batch_size)

### experiment script

for_concat = []    # for recording data
timer_list = []
power_list = defaultdict(list)
for i, seed in enumerate(seeds):
    start_time = time.time()
    print(f'amp {amp} - seed {seed}') 
    
    # random seed
    np.random.seed(seed) 
    
    # generate data
    Xtrain = gen_data(Wset, n_train, 1)
    Xcalib = gen_data(Wset, n, 1)
 
    Xtest0 = gen_data_weighted(Wset, m-n_outliers, 1, THETA)    # inliers 
    Xtest1 = gen_data_weighted(Wset, n_outliers, amp, THETA)    # outliers

    Xtest = np.zeros((m, Xtest0.shape[1]))
    Xtest[nonzero,] = Xtest1
    Xtest[nulls,] = Xtest0
    
    # scoring function; training phase
    classifier = svm.OneClassSVM(nu=0.004, kernel="rbf", gamma=0.1)
    classifier.fit(Xtrain)

    # compute calibration scores
    # scores should be that larger values <=> more likely outliers
    calib_weights = gen_weight(Xcalib, THETA).flatten() 
    calib_scores = -1 * classifier.score_samples(Xcalib).flatten() * (calib_weights if args.weight_scores else 1)
    cal_set = []
    for i in range(len(calib_weights)):
        cal_set.append( [calib_weights[i], calib_scores[i]] )

    # compute test scores 
    test_weights = gen_weight(Xtest, THETA).flatten()
    test_scores = -1 * classifier.score_samples(Xtest).flatten() * (test_weights if args.weight_scores else 1)
    test_set = []
    for i in range(len(test_weights)):
        test_set.append( [test_weights[i], test_scores[i]] )
        
    print(f"max cw: {max(calib_weights)} - min cw: {min(calib_weights)}")
    print(f"max tw: {max(test_weights)} - min tw: {min(test_weights)}")
        
    evalue_setting = conformal(alpha_conf=alpha_fdr)
    e_original = evalue_setting.e_function(cal_set, test_set)
    
    # eBH baseline
    rej = eBH(e_original, alpha_fdr)
    for_concat.append(  
        {'seed': seed, 'amp':amp, 'procedure': 'ebh', 
             'alpha_fdr': alpha_fdr, 'acc_factor': args.acc_factor,
        } | evaluate(rej, nonzero)
    ) 
    
    # make new CC objects every seed
    CC_dict = {}
    for key in CC_KEYS:
        CC_dict[key] = defaultdict(list)
    
    CC_dict['ebh-CC']['CC'] = CC(
        alpha_cc = alpha_cc, 
        alpha_fdr = alpha_fdr,
        rej_function=ebh_rej_function,
        guarantee=args.power_guarantee   # user specified
    )
    CC_dict['ebh-CC-infty']['CC'] = CC(
        alpha_cc=alpha_cc, alpha_fdr = alpha_fdr,rej_function=ebh_infty_rej_function, guarantee=args.power_guarantee
    )
    CC_dict['ebh-CC-union']['CC'] = CC(
        alpha_cc=alpha_cc, alpha_fdr = alpha_fdr,rej_function=ebh_union_rej_function, guarantee=args.power_guarantee
    )
    
    # make masks
    mask = (e_original != 0) 
    # make sure to also consider all previous rejections (if power is guaranteed, then we will boost without any computation anyway)
    mask = np.array(mask)
    if len(rej) > 0:
        mask[rej] = 1
    print(f'mask size: {sum(mask)}')
    
    # initialize fast coefs, confidence sequences (including parameters), and confidence counters
    for key in CC_KEYS:
        CC_dict[key]['fast_coefs'] = CC_dict[key]['CC'].compute_fast_coefs(e_original)    # differs due to different rej-functions
        if use_cs:
            # denominator: number of things mc tested; numerator: number of rejections made by eBH
            n_filter = sum(mask)
            alpha_cs = get_alpha_cs(
                alpha_fdr, args.rel_mc_error,  
                n_filter=n_filter, 
                n_prev_rej=len(rej), 
                power_guarantee=args.power_guarantee
            )                      
            rho_cs = get_rho(mc_switchpoint, alpha_cs)
            
            # two types of CS, we switch between them
            CC_dict[key]['exact_CS'] = None #hedged_cs(m=(alpha_fdr/(1+alpha_fdr)), theta=0.5, alpha=alpha_cs, c=0.5, interval=True)
            CC_dict[key]['asymp_CS'] = None #asy_cs(rho_cs, alpha_cs, m=(alpha_fdr/(1+alpha_fdr)))
            # for outlier detection, we do per-Sj confidence sequences

        CC_dict[key]['conf_array'] = np.zeros(m)   # keeps track of which are confident (1 for confident)
        CC_dict[key]['mc_samples_array'] = np.zeros(m)
        
    # loop through each hypothesis
    for j in range(m):
        if (j in rej) and args.power_guarantee:
            # if j has already been rejected (by the e-values), then it will be rejected as long as we are guaranteed strict improvement
            for key in CC_KEYS:
                CC_dict[key]['boost_set'].append(j)
            continue
        elif mask[j] == 0:
            continue    # filtered out
            
        # make sufficient statistic
        suff_stat = evalue_setting.get_suff_stat(idx=j, cal_set=cal_set, test_set=test_set)  
        unordered_set, _ = suff_stat
        unordered_set_array = np.array(unordered_set).T 
        unordered_set_weights = unordered_set_array[0]
        unordered_set_scores = unordered_set_array[1]
            
        if not use_cs:
            # in this case, we just take the max # of mc samples and estimate phi(c;S_j) 
            e_samples = evalue_setting.cond_e_sampling(
                idx=j, suff_stat=suff_stat, n_smpl=mc_max_samples,
            )
            for key in CC_KEYS:
                cc_object = CC_dict[key]['CC']
                fast_coefs = CC_dict[key]['fast_coefs']
                mc_samples = cc_object.cc_bootstrapping(e_samples, fast_coefs[j], j) 
                
                if (np.mean(mc_samples) < 0):
                    # since this case implies mean is less than 0, boost
                    CC_dict[key]['boost_set'].append(j)
            continue
        else:

            # adaptively get mc samples for evaluating phi(;S_j)
            # if we have made a decision for each CC method
            # terminate the sampling early
            made_decision_set = []
            for key in CC_KEYS:
                up_transf = m/alpha_cc
                lo_transf = sum(unordered_set_weights)/min(unordered_set_weights)   # mc samples originally lie in [-lo, up]
                
                CC_dict[key]['exact_CS'] = hedged_cs(m=lo_transf/(lo_transf+up_transf), theta=0.5, alpha=alpha_cs, c=0.5, interval=True)
                CC_dict[key]['asymp_CS'] = asy_cs(rho_cs, alpha_cs, m=lo_transf/(lo_transf+up_transf))

            for k in range(0, mc_max_samples//mc_batch_size):
                if len(made_decision_set) == len(CC_KEYS):
                    break    # we're done here

                # sample conditionally given S_j under H_j
                e_samples = evalue_setting.cond_e_sampling(
                    idx=j, suff_stat=suff_stat, n_smpl=mc_batch_size,
                )

                # for each CC method, get the e-samples and supplement the existing mc samples for estimating phi
                for key in CC_KEYS:
                    if key in made_decision_set:
                        continue    # we already made a decision for this method, continue to the next one
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
                        made_decision_set.append(key)
                        CC_dict[key]['conf_array'][j] = 1
                        if np.mean(cc_object.mc_samples[j]) < 0:
                            # since this case implies mean is less than 0, boost
                            CC_dict[key]['boost_set'].append(j)
                        CC_dict[key]['mc_samples_array'][j] = ((k+1) * mc_batch_size)
                    # instead, if we run out of mc sample budget, forced to make a decision
                    elif (k == mc_max_samples//mc_batch_size - 1):
                        made_decision_set.append(key) 
                        CC_dict[key]['mc_samples_array'][j] = ((k+1) * mc_batch_size)
                        # when confidence is required to boost index j, we are forced to do nothing at this point
                        # otherwise, we can boost j when mean is negative, but without confidence
                        if (np.mean(cc_object.mc_samples[j]) < 0):
                            CC_dict[key]['no_conf_boost_set'].append(j)
                            if (not args.confidence_required):
                                CC_dict[key]['boost_set'].append(j)
                
    # now, with the boost set, boost e-values for each CC 
    power_list['ebh'].append(evaluate(rej, nonzero)['power'])
    print(f"num of discoveries by eBH: {len(rej)}")
    for key in CC_KEYS:
        cc_object = CC_dict[key]['CC']
        boost_set = CC_dict[key]['boost_set']
        e_boosted = cc_object.boost(e_original, boost_set)
        boosted_rej = np.array(eBH(e_boosted, alpha_fdr), dtype=int)
        improve = np.array(np.setdiff1d(boosted_rej, rej), dtype=int)    # elements in rej but not in nonzero, i.e. false discoveries
        
        nc_boost_set = list(set(CC_dict[key]['no_conf_boost_set']) | set(boost_set))    # include the boost set coming from ignoring the CS
        nc_e_boosted = cc_object.boost(e_original, nc_boost_set)
        nc_boosted_rej = np.array(eBH(nc_e_boosted, alpha_fdr), dtype=int)
        nc_improve = np.array(np.setdiff1d(nc_boosted_rej, rej), dtype=int)
        
        # calculate confidence metrics
        confidence_array = CC_dict[key]['conf_array']
        no_conf_prop = np.nan if len(nc_improve)==0 else np.mean((1-confidence_array)[nc_improve])
        
        # calculate mc metrics
        mc_samples_array = CC_dict[key]['mc_samples_array']
        avg_samples_improve = np.nan if len(improve)==0 else np.mean(mc_samples_array[improve]) 
        
        tested_H0_idx = [(idx not in nonzero and mask[idx]==1) for idx in range(m)]    # mask of idx that are in H0 and have not been filtered out
        avg_samples_true_nulls = np.nan if sum(tested_H0_idx)==0 else np.mean(mc_samples_array[tested_H0_idx])
        
        # record data
        evaluated = evaluate(boosted_rej, nonzero)
        for_concat.append(
            {'seed': seed, 'amp':amp, 'procedure': key, 
             'alpha_fdr': alpha_fdr, 'acc_factor': args.acc_factor,
             'max_mc_samples': mc_max_samples, 
             'no_conf_prop': no_conf_prop,
             'avg_samples_improve': avg_samples_improve,
             'avg_samples_H0_tested': avg_samples_true_nulls,
             'nc_fdp': evaluate(nc_boosted_rej, nonzero)['fdp'],
             'nc_power': evaluate(nc_boosted_rej, nonzero)['power'],
             'mask_size': sum(mask),
            } | evaluated
        )
        power_list[key].append(evaluated['power'])
        
        
    # end of seed 
    # for dk in power_list:
        # print(f'running power {(dk+":").ljust(15)}{np.mean(power_list[dk])}')
    timer_list.append(time.time()-start_time) 
    df_results = pd.DataFrame.from_dict(for_concat)
    print(df_results.groupby(['procedure'], sort=False)[['fdp', 'power',]].agg(['mean', 'sem']))
    
    print(f'seed({seed}) time: {(timer_list[-1]):.2f} seconds')
    print('----------')

df_results = pd.DataFrame.from_dict(for_concat)
print(df_results.groupby(['procedure'], sort=False)[['fdp', 'power', 'no_conf_prop']].agg(['mean', 'sem']))
print(df_results.groupby(['procedure'], sort=False)[['nc_fdp', 'nc_power',]].agg(['mean', 'sem']))
print(df_results.groupby(['procedure'], sort=False)[['avg_samples_improve', 'avg_samples_H0_tested']].agg(['mean', 'sem']))    
print(df_results.groupby(['procedure'], sort=False)[['mask_size']].agg(['mean', 'sem']))    
    
# save results 
results_path = f'results-amp_{args.amp}-prop_{prop_outliers}-seeds_{seeds[0]}-{seeds[-1]}.csv'
argparse_path = f'args-seeds_{seeds[0]}-{seeds[-1]}.csv'

# save the dataframe and argparse history
df_results.to_csv(os.path.join(filepath, results_path))    
if 0 >= seeds[0] and 0 <= seeds[-1]:
    pd.DataFrame.from_dict(vars(args), orient='index').to_csv(os.path.join(filepath, argparse_path)) # only need to save once per set of experiments

print(f'saved csv to {results_path}')

print(f'total time: {str(datetime.timedelta(seconds = int(sum(timer_list)) ))}')
print(f'maximum exp time: {max(timer_list):.2f} seconds') 