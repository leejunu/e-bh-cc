import numpy as np
import pandas as pd

import scipy.linalg

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
from utils.cc import truncation_function

from cc_utils import ztesting    # hardcoding the use of of the ztesting setting

#####
# TEST_SETTING = 'ztesting'
# SETTING = getattr(importlib.import_module('cc_utils'), TEST_SETTING) 
CC_KEYS = ['ebh-CC', 'ebh-CC-infty', 'ebh-CC-union']

### arguments
parser = argparse.ArgumentParser(description='Script for eBH and eBH-CC z-testing experiments.')

# experiment config
parser.add_argument('--name', type=str, required=True, help='the name associated with the experiment.')
parser.add_argument('--n_exp', type=int, default=100, help='number of experiments (seeds) to run.')
parser.add_argument('--start_seed', type=int, default=0, help='the seed to with which to start when running the total experiment')
parser.add_argument('--end_seed', type=int, default=None, help='the last seed in the total experiment')

# experiment details
parser.add_argument('--m', type=int, default=100, help='dimension of the Z-distribution.')
parser.add_argument('--tp', type=int, default=10, help='the number of true non-nulls; must be less than m.')
parser.add_argument('--tp_first', default=False, action='store_true', help='turn on to make the first tp hypotheses true alternatives.') 

parser.add_argument('--cov_type', type=str, default='toeplitz', help='type of covariance matrix of the Z-distribution.')
parser.add_argument('--rho', type=float, default=0.5, help='rho used to construct the covariance matrix.')
parser.add_argument('--amp', type=float, required=True, help='the signal for the true alternatives.')

# alpha details
parser.add_argument('--alpha_fdr', type=float, default=0.05, help='the level at which to control FDR.')

# e-value details
parser.add_argument('--e_method', type=str, default='lrt', help='which type of e-value to use; options: ["lrt", "p2e_cal", "p2e_aon"].')
parser.add_argument('--e_alt', type=str, default='exact', help='which alternative to use when constructing the "lrt" e-value.')
parser.add_argument('--alt_type', type=str, default='onesided', help='whether the alternative hypotheses are "onesided" or "twosided".')
parser.add_argument('--p2e_calibrator', type=str, default='ramdas', help='which p-to-e calibrator to use when method is "p2e_cal". \
                                                                        options are either "ramdas" or a float in (0,1).')
parser.add_argument('--p2e_aon_threshold', type=str, default='ren', help='which method to use the all-or-nothiing threshold when the method is \
                                                                         "p2e_aon". options are either "ren" or a float in (0,1).')
parser.add_argument('--initial_boost', default=False, action='store_true', help='turn on to use the initial boosting mechanism outlined in \
                                                                                 Ramdas and Wang (2021), where the e-value is multiplied by a factor.')

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
parser.add_argument('--mask', default=None, help='type of mask. default is no mask (other than nonzero e-values). other options include \
                                                  ["none", "pvalues"]')

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
amp = args.amp    # signal strength
if args.tp_first:
    nonzero = np.array(range(int(args.tp)))
else:
    nonzero = np.array(range(0,m,int(m/args.tp))) 

# covariance matrix
if args.cov_type=='toeplitz':
    Sigma = scipy.linalg.toeplitz(args.rho ** np.arange(0,m,1))    # rho^|i-j|
    cho = np.linalg.cholesky(Sigma)                             # cholesky decomposition
else:
    assert False, f'{args.cov_type} is not a valid covariance matrix construction'

# alpha details
alpha_fdr = args.alpha_fdr    # fdr target
alpha_cc = alpha_fdr 
use_cs = True if args.rel_mc_error != None else False

# e-value details
if args.e_alt == 'exact':
    e_alt = amp
else:
    e_alt = float(args.e_alt)
    
evalue_setting = ztesting(
    Sigma=Sigma, method=args.e_method, alt_type=args.alt_type, alt=e_alt, 
    calibrator=args.p2e_calibrator, aon_threshold=args.p2e_aon_threshold
)
initial_boost = 1; budget_type = 'static'
if args.initial_boost:
    initial_boost = evalue_setting.get_initial_boost_factor(alpha_fdr, e_alt)
    # budget_type = 'truncated'
# trunc_func =  lambda e: 1  #lambda e: truncation_function(e * alpha_fdr * initial_boost, m)
    
print(f"marginal boosting factor: {initial_boost}")

# p-value specific setting object
pvalue_setting = ztesting(
    Sigma=Sigma, alt_type=args.alt_type, alt=e_alt
)

# mc details
mc_max_samples = args.mc_max_samples
mc_batch_size = args.mc_max_samples if args.mc_batch_size==0 else args.mc_batch_size
mc_switchpoint = int(args.exact_cs_switchpoint * mc_max_samples // mc_batch_size * mc_batch_size)



### experiment script
start_time = time.time()

for_concat = []    # for recording data
for i, seed in enumerate(seeds):
    if i % 10 == 0:
        print(seed)
    
    # random seed
    np.random.seed(seed) 
    
    # set alternative signal strengths
    mu_vec = np.zeros(m)
    mu_vec[nonzero] = amp 

    # generate z statistics
    Z = (cho @ np.random.normal(size=m)) + mu_vec
    
    # make new CC objects every seed
    CC_dict = {}
    for key in CC_KEYS:
        CC_dict[key] = defaultdict(list)
    
    CC_dict['ebh-CC']['CC'] = CC(
        alpha_cc = alpha_cc, 
        rej_function=ebh_rej_function, 
        budget_type=budget_type, 
        guarantee=args.power_guarantee   # user specified
    )
    CC_dict['ebh-CC-infty']['CC'] = CC(alpha_cc=alpha_cc, rej_function=ebh_infty_rej_function, budget_type=budget_type, guarantee=args.power_guarantee)
    CC_dict['ebh-CC-union']['CC'] = CC(alpha_cc=alpha_cc, rej_function=ebh_union_rej_function, budget_type=budget_type, guarantee=args.power_guarantee)
    # we can use the static budget and guarantee power as the e-value is has expectation 1 conditioned on S_j (due to independence)
    
    # p-values baseline
    p_original = pvalue_setting.p_function(Z)
    p_rej = pBH(p_original, alpha_fdr)
    for_concat.append(  
        {'seed': seed, 'procedure': f'bh', 
         'rho': args.rho,
        } | evaluate(p_rej, nonzero)
    )    # record data
    
    # e-values
    e_original = evalue_setting.e_function(Z, initial_boost=initial_boost)
    rej = eBH(e_original, alpha_fdr) 
    for_concat.append(  
        {'seed': seed, 'procedure': f'ebh', 
         'rho': args.rho,
        } | evaluate(rej, nonzero)
    )    # record data
    
     # make filter
    mask = e_original != 0
    if args.mask == "pvalues":
        mask *= (p_original <= 3 * alpha_fdr)
    # make sure to also consider all previous rejections (if power is guaranteed, then we will boost without any computation anyway)
    mask = np.array(mask)
    if len(rej) > 0:
        mask[rej] = 1
    # print(f'mask: {np.nonzero(mask)[0]}')

    # initialize fast coefs, confidence sequences (including parameters), and confidence counters
    for key in CC_KEYS:
        CC_dict[key]['fast_coefs'] = CC_dict[key]['CC'].compute_fast_coefs(e_original)    # differs due to different rej-functions
        if use_cs:
            # denominator: number of things mc tested; numerator: number of rejections made by eBH
            alpha_cs = get_alpha_cs(
                alpha_fdr, args.rel_mc_error,  
                n_filter=sum(mask), 
                n_prev_rej=len(rej), 
                power_guarantee=args.power_guarantee
            )
            rho_cs = get_rho(mc_switchpoint, alpha_cs)
            
            # two types of CS, we switch between them
            CC_dict[key]['exact_CS'] = hedged_cs(m=(alpha_fdr/m), theta=0.5, alpha=alpha_cs, c=0.5, interval=True)
            CC_dict[key]['asymp_CS'] = asy_cs(rho_cs, alpha_cs, m=alpha_fdr/m)

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
            
        suff_stat = evalue_setting.get_suff_stat(idx=j, Z=Z)   # get sufficient statistic
        
        if not use_cs:
            # in this case, we just take the max # of mc samples and estimate phi(c;S_j) 
            e_samples = evalue_setting.cond_e_sampling(idx=j, suff_stat=suff_stat, n_smpl=mc_max_samples, initial_boost=initial_boost)
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
                CC_dict[key]['exact_CS'].reset()   # reset CS
                CC_dict[key]['asymp_CS'].reset()   # reset CS

            for k in range(0, mc_max_samples//mc_batch_size):
                if len(made_decision_set) == len(CC_KEYS):
                    break    # we're done here

                # sample conditionally given S_j under H_j
                e_samples = evalue_setting.cond_e_sampling(idx=j, suff_stat=suff_stat, n_smpl=mc_batch_size,initial_boost=initial_boost)

                # for each CC method, get the e-samples and supplement the existing mc samples for estimating phi
                for key in CC_KEYS:
                    if key in made_decision_set:
                        continue    # we already made a decision for this method, continue to the next one
                    cc_object = CC_dict[key]['CC']
                    fast_coefs = CC_dict[key]['fast_coefs']
                    if ((k+1) * mc_batch_size) >= mc_switchpoint:
                        cs = CC_dict[key]['asymp_CS']
                        cs.update(CC_dict[key]['exact_CS'].data)
                    else:
                        cs = CC_dict[key]['exact_CS']

                    mc_samples = cc_object.cc_bootstrapping(e_samples, fast_coefs[j], j)    # mc samples generated by CC object's bootstrapping fn

                    # test inclusion of zero (transformed) in the the anytime CI
                    not_included = not cs.update(alpha_fdr/m * (1+mc_samples))    # scale the data so it lies in [0,1] (0 is transformed to alpha/m)

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
    for key in CC_KEYS:
        cc_object = CC_dict[key]['CC']
        boost_set = CC_dict[key]['boost_set']
        e_boosted = cc_object.boost(e_original, boost_set)
        boosted_rej = np.array(eBH(e_boosted, alpha_cc), dtype=int)
        improve = np.array(np.setdiff1d(boosted_rej, rej), dtype=int)    # elements in rej but not in nonzero, i.e. false discoveries
        
        nc_boost_set = list(set(CC_dict[key]['no_conf_boost_set']) | set(boost_set))
        nc_e_boosted = cc_object.boost(e_original, nc_boost_set)
        nc_boosted_rej = np.array(eBH(nc_e_boosted, alpha_cc), dtype=int)
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
            {'seed': seed, 
             'procedure': key, 
             'rho': args.rho,
             'max_mc_samples': mc_max_samples, 
             'no_conf_prop': no_conf_prop,
             'avg_samples_improve': avg_samples_improve,
             'avg_samples_H0_tested': avg_samples_true_nulls,
             'nc_fdp': evaluate(nc_boosted_rej, nonzero)['fdp'],
             'nc_power': evaluate(nc_boosted_rej, nonzero)['power'],
             'mask_size': sum(mask)
            } | evaluated
        ) 

df_results = pd.DataFrame.from_dict(for_concat)
print(df_results.groupby(['procedure'], sort=False)[['fdp', 'power', 'no_conf_prop']].agg(['mean', 'sem']))
print(df_results.groupby(['procedure'], sort=False)[['nc_fdp', 'nc_power',]].agg(['mean', 'sem']))
print(df_results.groupby(['procedure'], sort=False)[['avg_samples_improve', 'avg_samples_H0_tested']].agg(['mean', 'sem']))    
print(df_results.groupby(['procedure'], sort=False)[['mask_size']].agg(['mean', 'sem']))    


# save results 
results_path = f'results-amp_{args.amp}-alt_{args.e_alt}-{"init_boost-" if args.initial_boost else ""}seeds_{seeds[0]}-{seeds[-1]}.csv'
argparse_path = f'args-seeds_{seeds[0]}-{seeds[-1]}.csv'
# results_path = f'fdr_{alpha_fdr}-amp_{amp}-rho_{args.rho}'
# if args.e_method=='lrt':
#     results_path +=  f'-method_lrt_{evalue_setting.alt}_{evalue_setting.alt_type}'
# elif args.e_method=='p2e_aon:
#     results_path +=  f'-method_p2e_aon_{evalue_setting.alt}_{evalue_setting.alt_type}_{evalue_setting.aon_threshold}'
# elif args.e_method=='p2e_cal:
#     results_path +=  f'-method_p2e_cal_{evalue_setting.alt}_{evalue_setting.alt_type}_{evalue_setting.calibrator}'
# else:
#     assert False
# results_path += f'-mc_max_{mc_max_samples}-mc_error_{args.relative_mc_error}-cc_comparison.csv'

# save the dataframe and argparse history
df_results.to_csv(os.path.join(filepath, results_path))    
pd.DataFrame.from_dict(vars(args), orient='index').to_csv(os.path.join(filepath, argparse_path)) 

print(f'saved csv to {results_path}')
print(str(datetime.timedelta(seconds = int(time.time() - start_time))))