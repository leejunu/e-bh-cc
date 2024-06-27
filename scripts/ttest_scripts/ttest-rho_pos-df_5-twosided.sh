#!/bin/bash
#$ -N ttest-rho_pos-df_5-twosided-04302024
#$ -t 1-60
#$ -o job_output/$JOB_NAME/$JOB_ID-$TASK_ID.log
#$ -j y

module load python/3.11.5
source venv_ebhcc/bin/activate

# user set this before-hand
n_exp=1000
n_machines=6
n_seeds_per=$(( ($n_exp + $n_machines - 1) / $n_machines ))    # rounds up n_exp/n_machines

n_amps=10    # 1, 1.5, ..., 5.5

# there should be n_amps * n_machines tasks

whichseed=$(( ($SGE_TASK_ID-1) /  $n_amps))   
whichamp=$(( ($SGE_TASK_ID-1) % $n_amps))    

amp=$(bc <<<"scale=3;$whichamp/2+2") # so that amp starts at 2

startseed=$(( $whichseed * $n_seeds_per ))
temp=$(( $startseed + $n_seeds_per )) 
endseed=$(( ($temp < $n_exp ? $temp : $n_exp) - 1))   # make sure we do exactly n_exp experiments

python -u ttesting_CC.py --name '04302024-ttest-rho_pos-df_5-twosided' --start_seed $startseed --end_seed $endseed --m 100 --df 5 --tp 10 --tp_first --amp $amp --alpha_fdr 0.05 --rho 0.9 --true_sigma_sq 1.1 --rel_mc_error 0.1 --mc_max_samples 5000 --mc_batch_size 100 --exact_cs_switchpoint 0.6 --confidence_required --power_guarantee --e_method 'lrt' --e_alt 'exact' --mask 'pvalues' --alt_type 'twosided' --tp_random_sign 

