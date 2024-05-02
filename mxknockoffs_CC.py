import numpy as np
import pandas as pd

import scipy.linalg
import sklearn as skl

import knockpy
from knockpy.knockoff_filter import KnockoffFilter

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
from utils.filters import regression_pvalue_mask, lasso_mask

from cc_utils.mxknockoffs import knockoffs, gaussian_mxknockoffs

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
parser.add_argument('--m', type=int, default=400, help='dimension of the Z-distribution.')
parser.add_argument('--n', type=int, default=500, help='number of observations in the design matrix.')
# parser.add_argument('--tp', type=int, default=20, help='the number of true non-nulls; must be less than m.')
parser.add_argument('--tp_freq', type=int, default=1, help='frequency of nonzero elements in the beta vector.')
parser.add_argument('--tp_alt_sign', default=False, action='store_true', help='makes the nonzero elements in beta alternate in sign.')

parser.add_argument('--cov_type', type=str, default='toeplitz', help='type of covariance matrix of the Z-distribution.')
parser.add_argument('--rho', type=float, default=0.5, help='rho used to construct the covariance matrix.')
parser.add_argument('--amp', type=float, required=True, help='the signal for the true alternatives.')

parser.add_argument('--n_estimate', type=int, default=None, help='number of samples to use to estimate the Lasso CV alpha.')

parser.add_argument('--model', type=str, default='linear', help='model which Y|X follows. must be one of ["linear", "logistic"].')

# alpha details
parser.add_argument('--alpha_fdr', type=float, default=0.1, help='the level at which to control FDR.')

# e-value details
parser.add_argument('--akn_factor', type=float, default=1.0, help='the level at which to run derandomized knockoffs.')
parser.add_argument('--derand', type=int, default=1, help='the number of knockoffs to sample for derandomization.')


parser.add_argument('--ksampler', type=str, default='gaussian', help='the type of knockoffs to sample. currently only "gaussian" is implemented.')
parser.add_argument('--fstat', type=str, default='lcd', help='the type of feature statistic. default is "lcd", the lasso coefficient difference. \
                                                              see knockpy for more details.')
parser.add_argument('--kmethod', type=str, default='sdp', help='the method of sampling the "ksampler" knockoff. only "sdp" and "mvr" knockoffs are implemented.')


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
                                                  ["none", "power_only", "fdr_only", "feature_selection"]')

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

# covariance matrix
if args.cov_type=='toeplitz':
    Sigma = scipy.linalg.toeplitz(args.rho ** np.arange(0,m,1))    # rho^|i-j|
    cho = np.linalg.cholesky(Sigma)                             # cholesky decomposition
else:
    assert False, f'{args.cov_type} is not a valid covariance matrix construction'
    
# beta
nonzero = np.arange(args.tp_freq-1, args.m, args.tp_freq).astype(int)    # every tp_freq elements is B: (0, .., 0, B, ...)
print(nonzero)
signs = np.resize([1, (-1 if args.tp_alt_sign else 1)], len(nonzero))
beta_vec = np.zeros(m)
beta_vec[nonzero] = signs * amp / np.sqrt(n)    # populate the beta vector with +/- amp/sqrt n

print(f'true nonnulls: {nonzero}')

# alpha details
alpha_fdr = args.alpha_fdr    # fdr target
alpha_cc = alpha_fdr 
use_cs = True if args.rel_mc_error != None else False

# knockoff details (and instantiating the knockoff filter to pre-compute S-matrix)
ksampler = args.ksampler; fstat = args.fstat; kmethod = args.kmethod
derand = args.derand
akn_factor = args.akn_factor
alpha_kn = akn_factor * alpha_fdr    # used in derandomized knockoff construction
print(alpha_kn)

# mc details
mc_max_samples = args.mc_max_samples
mc_batch_size = args.mc_max_samples if args.mc_batch_size<=0 else args.mc_batch_size
mc_switchpoint = int(args.exact_cs_switchpoint * mc_max_samples // mc_batch_size * mc_batch_size)

# compute S matrix (although the code uses X, y, in fact S is derived independent of the draw of X, y)
np.random.seed(369)
X = np.random.multivariate_normal(
    mean=np.zeros(m), cov=Sigma, size=10
)
if args.model == 'linear':
    y = np.dot(X, beta_vec) + np.random.randn(10)
elif args.model == 'logistic':
    bern_probs = 1 / (1 + np.exp(-1 * np.dot(X, beta_vec)))   # expit i.e. logistic sigmoid
    y = (np.random.uniform(size=len(bern_probs)) < bern_probs) * 1
else:
    assert False, f'{args.model} is not implemented for Y|X'

# compute the S-matrix by running the kn_filter forward function
kn_filter = KnockoffFilter(ksampler=ksampler, fstat=fstat, knockoff_kwargs={'method': kmethod})
kn_filter.forward(  
    X=X,
    y=y,
    mu=np.zeros(m),
    Sigma=Sigma,
    fdr=alpha_fdr # desired level of false discovery rate control
)
S = kn_filter.S                      # the computed S matrix
del X; del y; kn_filter.Xk = None    # clear the filter
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
    set_aside = n if args.n_estimate == None else args.n_estimate
    X_total = np.random.multivariate_normal(mean=np.zeros(m), cov=Sigma, size=(n+set_aside,))
    X = X_total[:n,]; X_aside = X_total[n:,]    # split data
    
    if args.model == 'linear':
        y = np.dot(X, beta_vec) + np.random.randn(len(X))
        y_aside = np.dot(X_aside, beta_vec) + np.random.randn(len(X_aside))
    elif args.model == 'logistic':
        bern_probs = 1 / (1 + np.exp(-1 * np.dot(X, beta_vec)))   # expit i.e. logistic sigmoid
        y = (np.random.uniform(size=len(X)) < bern_probs) * 1
        
        bern_probs_aside = 1 / (1 + np.exp(-1 * np.dot(X_aside, beta_vec)))   
        y_aside = (np.random.uniform(size=len(X_aside)) < bern_probs_aside) * 1
    # so now, (X,y) and (X_aside, y_aside) are two separate (independent) datasets
        
    # precompute lambda_cv
    _, lasso_model = lasso_mask(X_aside, y_aside, cv=5)     # fit_intercept default true here, since LCD also uses it
    lam_cv = lasso_model.alpha_                             # the regularization param selected using CV 
    
    # new kn filter and e-object
    kn_filter = KnockoffFilter(
        ksampler=ksampler, fstat=fstat, knockoff_kwargs={'method': kmethod}, fstat_kwargs={'alphas' : lam_cv}
    )
    kn_filter.S = S    # load precomputed S matrix
    
    # make new CC objects every seed
    CC_dict = {}
    for key in CC_KEYS:
        CC_dict[key] = defaultdict(list)
    
    CC_dict['ebh-CC']['CC'] = CC(
        alpha_cc = alpha_cc, 
        rej_function=ebh_rej_function,
        guarantee=args.power_guarantee   # user specified
    )
    CC_dict['ebh-CC-infty']['CC'] = CC(alpha_cc=alpha_cc, rej_function=ebh_infty_rej_function, guarantee=args.power_guarantee)
    CC_dict['ebh-CC-union']['CC'] = CC(alpha_cc=alpha_cc, rej_function=ebh_union_rej_function, guarantee=args.power_guarantee)
    
    # knockoffs baseline 
    kn_rej = np.nonzero(
        kn_filter.forward(  
            X=X,
            y=y,
            mu=np.zeros(m),
            Sigma=Sigma,
            fdr=alpha_fdr
        )
    )[0]
    
    for_concat.append(  
        {'seed': seed, 'amp':amp, 'procedure': f'mxkn',
         'alpha_fdr': alpha_fdr, 'akn_factor': 1.0,
         # 'm': args.m, 'n': args.n, 'rho': args.rho,
         # 'tp': args.tp, 'tp_freq': args.tp_freq, 'tp_alt_signs': args.tp_alt_sign
        } | evaluate(kn_rej, nonzero)
    )    # record data
        
    # e-values
    assert not isinstance(kn_filter.ksampler, type(None)), "make sure kn_filter has its sampler instantiated (by calling forward once)"
    # assert np.isclose(kn_filter.ksampler.S, S), "these should be the exact same S-matrix"
    evalue_setting = gaussian_mxknockoffs(
        mu=np.zeros(m), Sigma=Sigma, kn_filter=kn_filter   
    )   
    e_original = evalue_setting.e_function(X, y, derand=derand, alpha_kn=alpha_kn)
    rej = eBH(e_original, alpha_fdr)
    
    for_concat.append(  
        {'seed': seed, 'amp':amp, 'procedure': 'ebh',
         'alpha_fdr': alpha_fdr, 'akn_factor': akn_factor,
        } | evaluate(rej, nonzero)
    )    
    
    # make masks
    if args.mask == None or args.mask == 'none':
        mask = (e_original != 0) 
    elif args.mask == 'power_only':
        mask = 1*np.array([(idx in nonzero) for idx in range(m)]) * (e_original != 0) 
    elif args.mask == 'fdr_only':
        mask = 1*np.array([(idx not in nonzero) for idx in range(m)]) * (e_original != 0) 
    elif args.mask == 'feature_selection':
        # mask[j] = 1 <=> e_original[j] > 0 AND (j selected by LASSO OR linear model p-value of j is significant at alpha_fdr)
        mask_lasso, _ = lasso_mask(X, y,)
        mask_reg_p, _ = regression_pvalue_mask(X, y, alpha=alpha_fdr, fit_intercept=True, )
        mask_nonzero = (e_original != 0)   
        mask = ((mask_lasso + mask_reg_p) > 0) * mask_nonzero
    else:
        assert False, f'{args.mask} not a valid type of filter'
    # make sure to also consider all previous rejections (if power is guaranteed, then we will boost without any computation anyway)
    mask = np.array(mask)
    if len(rej) > 0:
        mask[rej] = 1
    print(f'mask: {np.nonzero(mask)[0]}')
    
    # initialize fast coefs, confidence sequences (including parameters), and confidence counters
    for key in CC_KEYS:
        CC_dict[key]['fast_coefs'] = CC_dict[key]['CC'].compute_fast_coefs(e_original)    # differs due to different rej-functions
        if use_cs:
            # denominator: number of things mc tested; numerator: number of rejections made by eBH
            if args.mask in ['power_only', 'fdr_only']:
                n_filter = m    # pretend as though we were running no-mask
            else:
                n_filter = sum(mask)
            alpha_cs = get_alpha_cs(
                alpha_fdr, args.rel_mc_error,  
                n_filter=n_filter, 
                n_prev_rej=len(rej), 
                power_guarantee=args.power_guarantee
            )                      
            rho_cs = get_rho(mc_switchpoint, alpha_cs)
            
            # two types of CS, we switch between them
            CC_dict[key]['exact_CS'] = hedged_cs(m=(alpha_fdr/(1+alpha_fdr)), theta=0.5, alpha=alpha_cs, c=0.5, interval=True)
            CC_dict[key]['asymp_CS'] = asy_cs(rho_cs, alpha_cs, m=(alpha_fdr/(1+alpha_fdr)))

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
        suff_stat = evalue_setting.get_suff_stat(idx=j, X=X, y=y)    
            
        if not use_cs:
            # in this case, we just take the max # of mc samples and estimate phi(c;S_j) 
            e_samples = evalue_setting.cond_e_sampling(
                idx=j, suff_stat=suff_stat, n_smpl=mc_max_samples,
                derand=derand, alpha_kn=alpha_kn
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
                CC_dict[key]['exact_CS'].reset()   # reset CS
                CC_dict[key]['asymp_CS'].reset()   # reset CS

            for k in range(0, mc_max_samples//mc_batch_size):
                if len(made_decision_set) == len(CC_KEYS):
                    break    # we're done here

                # sample conditionally given S_j under H_j
                e_samples = evalue_setting.cond_e_sampling(
                    idx=j, suff_stat=suff_stat, n_smpl=mc_batch_size,
                    derand=derand, alpha_kn=alpha_kn
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
                    not_included = not cs.update( (mc_samples + m) / (m + m/alpha_fdr))   # scale the data so it lies in [0,1] (0 is transformed to (alpha/(1+alpha)) [THIS SCALING IS ONLY CORRECT WHEN alpha_fdr = alpha_cc]

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
    power_list['mxkn'].append(evaluate(kn_rej, nonzero)['power'])
    power_list['ebh'].append(evaluate(rej, nonzero)['power'])
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
             'alpha_fdr': alpha_fdr, 'akn_factor': akn_factor,  
             'max_mc_samples': mc_max_samples, 
             'no_conf_prop': no_conf_prop,
             'avg_samples_improve': avg_samples_improve,
             'avg_samples_H0_tested': avg_samples_true_nulls,
             'nc_fdp': evaluate(nc_boosted_rej, nonzero)['fdp'],
             'nc_power': evaluate(nc_boosted_rej, nonzero)['power'],
             'mask_size': sum(mask)
            } | evaluated
        )
        power_list[key].append(evaluated['power'])
        
        
    # end of seed 
    for dk in power_list:
        print(f'running power {(dk+":").ljust(15)}{np.mean(power_list[dk])}')
    timer_list.append(time.time()-start_time)
    print(f'seed({seed}) time: {(timer_list[-1]):.2f} seconds')
    print('----------')

df_results = pd.DataFrame.from_dict(for_concat)
print(df_results.groupby(['procedure'], sort=False)[['fdp', 'power', 'no_conf_prop']].agg(['mean', 'sem']))
print(df_results.groupby(['procedure'], sort=False)[['nc_fdp', 'nc_power',]].agg(['mean', 'sem']))
print(df_results.groupby(['procedure'], sort=False)[['avg_samples_improve', 'avg_samples_H0_tested']].agg(['mean', 'sem']))    
print(df_results.groupby(['procedure'], sort=False)[['mask_size']].agg(['mean', 'sem']))    
    
# save results 
results_path = f'results-amp_{args.amp}-derand_{derand}-aknf_{args.akn_factor}-{ksampler}-{args.model}-seeds_{seeds[0]}-{seeds[-1]}.csv'
argparse_path = f'args-seeds_{seeds[0]}-{seeds[-1]}.csv'

# save the dataframe and argparse history
df_results.to_csv(os.path.join(filepath, results_path))    
if 0 >= seeds[0] and 0 <= seeds[-1]:
    pd.DataFrame.from_dict(vars(args), orient='index').to_csv(os.path.join(filepath, argparse_path)) # only need to save once per set of experiments

print(f'saved csv to {results_path}')

print(f'total time: {str(datetime.timedelta(seconds = int(sum(timer_list)) ))}')
print(f'maximum exp time: {max(timer_list):.2f} seconds') 
